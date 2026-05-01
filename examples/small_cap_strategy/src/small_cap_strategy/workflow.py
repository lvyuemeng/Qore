from __future__ import annotations

import asyncio
import logging
from argparse import ArgumentParser
from dataclasses import dataclass, replace
from datetime import date

import polars as pl
from qore_backtest import BacktestSettings, DateColumnDayFrameSource, TradingCalendar
from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_backtest.metrics import compute_metrics
from qore_data import DataSettings, StockPipeline
from qore_factor.fundamental.quality import DebtToAssetRatioFactor
from qore_factor.pipeline import FactorPipeline
from qore_runner import RebalanceSchedule, RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer

logger = logging.getLogger("small_cap_strategy")


@dataclass(frozen=True, slots=True)
class _StoreMarketDataSource:
    pipe: StockPipeline

    def frame_for_day(self, trading_day: date) -> pl.DataFrame | None:
        frame = pl.DataFrame(
            self.pipe.store.read(
                "stock_ohlcv", filters={"date": trading_day}, backend="duckdb"
            ).collect()
        )
        return frame if not frame.is_empty() else None


@dataclass(frozen=True, slots=True)
class StrategySpec:
    benchmark: str
    start: date
    end: date
    top_n: int
    primary_factor: str
    primary_ascending: bool
    min_listing_days: int
    max_single_position: float
    liquidity_lookback_days: int
    capacity_ratio_limit: float
    min_daily_amount_cny: float
    rebalance_schedule: RebalanceSchedule


DEFAULT_DATA_SETTINGS = DataSettings()
DEFAULT_BACKTEST_SETTINGS = BacktestSettings()
DEFAULT_TRADING_CALENDAR = TradingCalendar()
DEFAULT_RUNNER_SETTINGS = RunnerSettings(
    max_single=0.10,
    drawdown_stop=DEFAULT_BACKTEST_SETTINGS.drawdown_stop,
)


def run_small_cap_workflow(
    *,
    data_settings: DataSettings = DEFAULT_DATA_SETTINGS,
    runner_settings: RunnerSettings = DEFAULT_RUNNER_SETTINGS,
    backtest_settings: BacktestSettings = DEFAULT_BACKTEST_SETTINGS,
    calendar: TradingCalendar = DEFAULT_TRADING_CALENDAR,
) -> BacktestResult:
    spec = StrategySpec(
        benchmark="000852.SH",
        start=date(2010, 1, 1),
        end=date(2026, 4, 21),
        top_n=20,
        primary_factor="total_market_cap",
        primary_ascending=True,
        min_listing_days=60,
        max_single_position=0.10,
        liquidity_lookback_days=20,
        capacity_ratio_limit=0.10,
        min_daily_amount_cny=10_000_000.0,
        rebalance_schedule=RebalanceSchedule(
            frequency="monthly", buy_delay=1, sell_delay=2
        ),
    )
    logger.info("workflow_start benchmark=%s", spec.benchmark)

    pipe = StockPipeline.from_settings(data_settings)
    try:
        rebalance_dates = (
            spec.rebalance_schedule.schedule(
                trading_days=pl.DataFrame(
                    {"date": calendar.trading_days_between(spec.start, spec.end)},
                    schema={"date": pl.Date},
                )
            )
            .get_column("date")
            .unique()
            .sort()
            .to_list()
        )
        logger.info("rebalance dates=%d", len(rebalance_dates))

        constituents = asyncio.run(pipe.resolve(spec.benchmark, spec.start))
        symbol_list = constituents.to_list()
        selection = (
            pipe.market_corpus(
                symbols=symbol_list,
                start=spec.start,
                end=spec.end,
                include_fundamentals=True,
            )
            .filter(pl.col("date").is_in(rebalance_dates))
            .with_columns(
                pl.lit(None, dtype=pl.Float64).alias(
                    f"position_to_amount_{spec.liquidity_lookback_days}d_ratio"
                ),
                pl.lit(None, dtype=pl.Float64).alias(
                    f"min_amount_{spec.liquidity_lookback_days}d"
                ),
            )
        )

        factor_lf = (
            FactorPipeline()
            .add(DebtToAssetRatioFactor(produces="debt_to_asset_ratio"))
            .run(selection)
        )
        filtered = factor_lf.filter(
            pl.col("roe").fill_null(0.0) > 0.0,
            pl.col("debt_to_asset_ratio").fill_null(0.0) < 0.60,
            pl.col("operating_cashflow").fill_null(0.0) > 0.0,
            pl.col("pe_ttm").fill_null(0.0).is_between(0.0, 50.0),
            pl.col("pb").fill_null(0.0).is_between(0.0, 3.0),
            pl.col("is_st").fill_null(False).not_(),
            pl.col("is_suspended").fill_null(False).not_(),
            pl.col("listing_days").fill_null(0) >= 60,
        )

        sign = -1.0 if spec.primary_ascending else 1.0
        signal_frame = pl.DataFrame(
            filtered.select(
                "date",
                "symbol",
                (pl.col(spec.primary_factor).cast(pl.Float64) * pl.lit(sign)).alias(
                    "signal"
                ),
            ).collect()
        )
        logger.info("signal rows=%d", len(signal_frame))

        decision_frame = pl.DataFrame(
            filtered.select("date", "symbol")
            .unique()
            .sort(
                "date",
                spec.primary_factor,
                descending=[False, not spec.primary_ascending],
            )
            .group_by("date", maintain_order=True)
            .head(spec.top_n)
            .with_columns(
                pl.lit(True).alias("selected"),
                pl.lit(None, dtype=pl.String).alias("exclude_reason"),
            )
            .collect()
        )
        logger.info("decision rows=%d", len(decision_frame))
        if decision_frame.is_empty():
            raise ValueError(
                "Small-cap workflow produced no rebalance selection snapshots."
            )

        runner = StrategyRunner.from_settings(
            runner_settings,
            EqualWeightSizer(top_k=spec.top_n, max_weight=spec.max_single_position),
        )
        engine = BacktestEngine.from_settings(
            replace(backtest_settings, start=spec.start, end=spec.end),
            runner,
            calendar,
            signal_source=DateColumnDayFrameSource(frame=signal_frame),
            market_data_source=_StoreMarketDataSource(pipe),
            decision_source=DateColumnDayFrameSource(frame=decision_frame),
        )
        result = engine.run()
        m = compute_metrics(result)
        logger.info(
            "backtest_done sharpe=%.3f ret=%.4f dd=%.4f",
            m.get("sharpe_ratio", float("nan")),
            m.get("annualized_return", float("nan")),
            m.get("max_drawdown", float("nan")),
        )
        return result
    finally:
        asyncio.run(pipe.close())
        if pipe.store is not None:
            pipe.store.close()


async def prepare_small_cap_data(
    *, data_settings: DataSettings = DEFAULT_DATA_SETTINGS
) -> None:
    spec = StrategySpec(
        benchmark="000852.SH",
        start=date(2010, 1, 1),
        end=date(2026, 4, 21),
        top_n=20,
        primary_factor="total_market_cap",
        primary_ascending=True,
        min_listing_days=60,
        max_single_position=0.10,
        liquidity_lookback_days=20,
        capacity_ratio_limit=0.10,
        min_daily_amount_cny=10_000_000.0,
        rebalance_schedule=RebalanceSchedule(
            frequency="monthly", buy_delay=1, sell_delay=2
        ),
    )
    logger.info("prepare_start benchmark=%s", spec.benchmark)
    pipe = StockPipeline.from_settings(data_settings)
    try:
        symbols = (await pipe.resolve(spec.benchmark, spec.end)).to_list()
        logger.info("resolve_done symbols=%d", len(symbols))
        if not symbols:
            raise ValueError(f"No constituents for '{spec.benchmark}'.")
        await pipe.stock_profiles(symbols, spec.end)
        await pipe.stock_daily(symbols, spec.start, spec.end)
        await pipe.fundamentals(symbols, spec.end)
        await pipe.audit_opinions(symbols, spec.start, spec.end)
        await pipe.analyst_forecasts(symbols, spec.end)
        await pipe.announcements(symbols, spec.start, spec.end)
        logger.info("prepare_done symbols=%d", len(symbols))
    finally:
        await pipe.close()


def build_parser() -> ArgumentParser:
    p = ArgumentParser()
    for a in (
        "--db-path",
        "--parquet-root",
        "--initial-capital",
        "--commission",
        "--slippage",
        "--drawdown-stop",
    ):
        p.add_argument(a)
    p.add_argument("--prepare-data", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    ds = DataSettings(db_path=args.db_path, parquet_root=args.parquet_root)
    bs = BacktestSettings(
        initial_capital=args.initial_capital or 10_000_000.0,
        commission=args.commission or 0.0003,
        slippage=args.slippage or 0.0005,
        drawdown_stop=args.drawdown_stop or 0.15,
    )
    rs = RunnerSettings(max_single=0.10, drawdown_stop=bs.drawdown_stop)
    if args.prepare_data:
        asyncio.run(prepare_small_cap_data(data_settings=ds))
        return 0
    result = run_small_cap_workflow(
        data_settings=ds, runner_settings=rs, backtest_settings=bs
    )
    print(result.nav)
    print(result.diagnostics)
    result.view().with_drawdown().plot().overview()
    return 0


def cli() -> None:
    raise SystemExit(main())
