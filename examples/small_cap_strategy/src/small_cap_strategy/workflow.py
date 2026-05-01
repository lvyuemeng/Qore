from __future__ import annotations

import asyncio
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date

import polars as pl
from qore_backtest import BacktestEngine, BacktestSettings, TradingCalendar
from qore_data import DataSettings, StockPipeline
from qore_factor.fundamental.quality import DebtToAssetRatioFactor
from qore_factor.pipeline import FactorPipeline
from qore_runner.sizer import EqualWeightSizer

logger = logging.getLogger("small_cap_strategy")


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
    liquidity_lookback_days: int = 20
    capacity_ratio_limit: float = 0.10
    min_daily_amount_cny: float = 10_000_000.0


DEFAULT_DATA_SETTINGS = DataSettings()
DEFAULT_BACKTEST_SETTINGS = BacktestSettings()
DEFAULT_CALENDAR = TradingCalendar()
DEFAULT_SPEC = StrategySpec(
    benchmark="000852.SH",
    start=date(2010, 1, 1),
    end=date(2026, 4, 21),
    top_n=20,
    primary_factor="total_market_cap",
    primary_ascending=True,
    min_listing_days=60,
    max_single_position=0.10,
)


# ── phase 1: fetch ──────────────────────────────────────────────────────────


async def prepare_data(
    settings: DataSettings = DEFAULT_DATA_SETTINGS,
    spec: StrategySpec = DEFAULT_SPEC,
) -> StockPipeline:
    """Fetch index constituents, OHLCV, fundamentals, profiles into the store."""
    pipe = StockPipeline.from_settings(settings)
    symbols = (await pipe.resolve(spec.benchmark, spec.start)).to_list()
    logger.info("resolve symbols=%d", len(symbols))
    if not symbols:
        raise ValueError(f"No constituents for '{spec.benchmark}'.")
    await pipe.stock_profiles(symbols, spec.end)
    await pipe.stock_daily(symbols, spec.start, spec.end)
    await pipe.fundamentals(symbols, spec.end)
    await pipe.analyst_forecasts(symbols, spec.end)
    logger.info("fetch_done symbols=%d", len(symbols))
    return pipe


# ── phase 2: build signals ──────────────────────────────────────────────────


def build_signals(
    pipe: StockPipeline,
    spec: StrategySpec = DEFAULT_SPEC,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Read from store, apply factor pipeline, return signal and market data."""
    selection = pipe.market_corpus(
        symbols=[],
        start=spec.start,
        end=spec.end,
        include_fundamentals=True,
    ).filter(
        pl.col("roe").fill_null(0.0) > 0.0,
        pl.col("debt_to_asset_ratio").fill_null(0.0) < 0.60,
        pl.col("operating_cashflow").fill_null(0.0) > 0.0,
        pl.col("pe_ttm").fill_null(0.0).is_between(0.0, 50.0),
        pl.col("pb").fill_null(0.0).is_between(0.0, 3.0),
        pl.col("is_st").fill_null(False).not_(),
        pl.col("is_suspended").fill_null(False).not_(),
        pl.col("listing_days").fill_null(0) >= spec.min_listing_days,
    )
    factor_lf = (
        FactorPipeline()
        .add(
            DebtToAssetRatioFactor(produces="debt_to_asset_ratio"),
        )
        .run(selection)
    )

    sign = -1.0 if spec.primary_ascending else 1.0
    signal_lf = factor_lf.select(
        "date",
        "symbol",
        (pl.col(spec.primary_factor).cast(pl.Float64) * sign).alias("signal"),
    ).drop_nulls(subset=["signal"])

    market_lf = pipe.read("stock_ohlcv", dates=(spec.start, spec.end))
    return signal_lf, market_lf


# ── phase 3: backtest ───────────────────────────────────────────────────────


def run_backtest(
    signal_lf: pl.LazyFrame,
    market_lf: pl.LazyFrame,
    spec: StrategySpec = DEFAULT_SPEC,
    settings: BacktestSettings = DEFAULT_BACKTEST_SETTINGS,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
) -> pl.DataFrame:
    """Run the backtest and return a metrics dict."""
    engine = BacktestEngine(
        config=BacktestSettings(
            initial_capital=settings.initial_capital,
            commission=settings.commission,
            slippage=settings.slippage,
            buy_delay=1,
            sell_delay=2,
            start=spec.start,
            end=spec.end,
        ),
        calendar=calendar,
        signals=signal_lf,
        market_data=market_lf,
        sizer=EqualWeightSizer(max_weight=spec.max_single_position),
        top_k=spec.top_n,
    )
    result = engine.run()
    m = result.metrics()
    logger.info(
        "backtest sharpe=%.3f ret=%.4f dd=%.4f",
        m.get("sharpe_ratio", float("nan")),
        m.get("annualized_return", float("nan")),
        m.get("max_drawdown", float("nan")),
    )
    return result


# ── orchestrator ────────────────────────────────────────────────────────────


def run_small_cap_workflow(
    data_settings: DataSettings = DEFAULT_DATA_SETTINGS,
    spec: StrategySpec = DEFAULT_SPEC,
    backtest_settings: BacktestSettings = DEFAULT_BACKTEST_SETTINGS,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
):
    pipe = asyncio.run(prepare_data(data_settings, spec))
    try:
        signals, market = build_signals(pipe, spec)
        return run_backtest(signals, market, spec, backtest_settings, calendar)
    finally:
        asyncio.run(pipe.close())
        pipe.store.close()


async def prepare_small_cap_data(
    data_settings: DataSettings = DEFAULT_DATA_SETTINGS,
) -> None:
    pipe = await prepare_data(data_settings)
    await pipe.close()


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> ArgumentParser:
    p = ArgumentParser()
    for a in (
        "--db-path",
        "--parquet-root",
        "--initial-capital",
        "--commission",
        "--slippage",
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
    )
    if args.prepare_data:
        asyncio.run(prepare_small_cap_data(data_settings=ds))
        return 0
    result = run_small_cap_workflow(data_settings=ds, backtest_settings=bs)
    print(result.nav)
    result.view().with_drawdown().plot().overview()
    return 0


def cli() -> None:
    raise SystemExit(main())
