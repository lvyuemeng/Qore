from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl
from qore_backtest import (
    BacktestSettings,
    MappingDayFrameSource,
    NullSignalOverlaySource,
    StoreMarketDataSource,
    TradingCalendar,
)
from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_backtest.view import BacktestView
from qore_data import DataSettings, Universe
from qore_data.store.duckdb import QoreStore
from qore_data.universe import (
    CandidateFilter,
    CandidateSort,
    StockCandidateSpec,
    StockSelectionPipeline,
)
from qore_factor.fundamental.quality import DebtToAssetRatioFactor
from qore_factor.pipeline import FactorPipeline
from qore_runner import RebalanceSchedule, RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener


@dataclass(frozen=True, slots=True)
class StrategySpec:
    benchmark: str
    start: date
    end: date
    suggested_max_aum_cny: float
    top_n: int
    primary_factor: str
    primary_ascending: bool
    min_listing_days: int
    max_single_position: float
    liquidity_lookback_days: int
    capacity_ratio_limit: float
    min_daily_amount_cny: float
    audit_exclusion_days: int
    filters: tuple[CandidateFilter, ...]
    rebalance_schedule: RebalanceSchedule


def _strategy_spec() -> StrategySpec:
    return StrategySpec(
        benchmark="8841431.WI",
        start=date(2010, 1, 1),
        end=date(2026, 4, 21),
        suggested_max_aum_cny=50_000_000.0,
        top_n=20,
        primary_factor="total_market_cap",
        primary_ascending=True,
        min_listing_days=60,
        max_single_position=0.10,
        liquidity_lookback_days=20,
        capacity_ratio_limit=0.10,
        min_daily_amount_cny=10_000_000.0,
        audit_exclusion_days=360,
        rebalance_schedule=RebalanceSchedule(
            frequency="monthly", buy_delay=1, sell_delay=2
        ),
        filters=(
            CandidateFilter("roe", "gt", 0.0),
            CandidateFilter("debt_to_asset_ratio", "lt", 0.60),
            CandidateFilter("operating_cash_flow", "gt", 0.0),
            CandidateFilter("pe_ttm", "between", (0.0, 50.0)),
            CandidateFilter("pb", "between", (0.0, 3.0)),
        ),
    )


def run_small_cap_workflow(
    *,
    data_settings: DataSettings | None = None,
    runner_settings: RunnerSettings | None = None,
    backtest_settings: BacktestSettings | None = None,
    calendar: TradingCalendar | None = None,
) -> BacktestResult:
    spec = _strategy_spec()
    resolved_data_settings = data_settings or DataSettings()
    resolved_backtest_settings = backtest_settings or BacktestSettings()
    resolved_calendar = calendar or TradingCalendar()
    resolved_runner_settings = runner_settings or RunnerSettings(
        max_single=spec.max_single_position,
        drawdown_stop=resolved_backtest_settings.drawdown_stop,
    )

    store = QoreStore.from_settings(resolved_data_settings)
    selection_by_day = _selection_frames_by_day(store, resolved_calendar, spec)
    overlays = _build_decision_overlays(selection_by_day, spec)
    universe = _universe_for_backtest(store, spec)

    strategy = CrossSectionalScreener(
        {spec.primary_factor: -1.0 if spec.primary_ascending else 1.0},
        rebalance_schedule=spec.rebalance_schedule,
    )
    runner = StrategyRunner.from_settings(
        resolved_runner_settings,
        strategy,
        EqualWeightSizer(top_k=spec.top_n, max_weight=spec.max_single_position),
    )
    engine = BacktestEngine.from_settings(
        resolved_backtest_settings,
        runner,
        resolved_calendar,
        factor_source=MappingDayFrameSource(selection_by_day),
        market_data_source=StoreMarketDataSource(store=store),
        decision_overlays_by_day=overlays,
        signal_overlay_source=NullSignalOverlaySource(),
    )
    return engine.run(universe, spec.start, spec.end)


def build_stock_category_report(
    *,
    data_settings: DataSettings | None = None,
) -> pl.DataFrame:
    spec = _strategy_spec()
    store = QoreStore.from_settings(data_settings or DataSettings())
    return (
        StockSelectionPipeline.from_index(
            store,
            index_symbol=spec.benchmark,
            as_of=spec.end,
        )
        .with_category_inputs()
        .category_report()
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run small-cap quality enhanced strategy")
    parser.add_argument("--db-path", default="data/qore.duckdb")
    parser.add_argument("--parquet-root", default="data/raw")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--drawdown-stop", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_settings = DataSettings(db_path=args.db_path, parquet_root=args.parquet_root)
    backtest_settings = BacktestSettings(
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
        drawdown_stop=args.drawdown_stop,
    )
    runner_settings = RunnerSettings(
        max_single=_strategy_spec().max_single_position,
        drawdown_stop=backtest_settings.drawdown_stop,
    )
    category_report = build_stock_category_report(
        data_settings=data_settings,
    )
    result = run_small_cap_workflow(
        data_settings=data_settings,
        runner_settings=runner_settings,
        backtest_settings=backtest_settings,
    )
    print(category_report)
    print(result.nav)
    print(result.diagnostics)
    result.view().with_drawdown().plot().overview()
    _plot_primary_factor_series(data_settings=data_settings)
    return 0


def cli() -> None:
    raise SystemExit(main())


def _plot_primary_factor_series(*, data_settings: DataSettings | None = None) -> None:
    spec = _strategy_spec()
    store = QoreStore.from_settings(data_settings or DataSettings())
    selection_by_day = _selection_frames_by_day(store, TradingCalendar(), spec)
    overlays = _build_decision_overlays(selection_by_day, spec)
    rows: list[dict[str, object]] = []
    for rebalance_day, selection in selection_by_day.items():
        if spec.primary_factor not in selection.columns:
            continue
        overlay = overlays.get(rebalance_day)
        if overlay is None or overlay.is_empty():
            continue
        selected_symbols = (
            overlay.filter(pl.col("selected")).get_column("symbol").to_list()
        )
        if not selected_symbols:
            continue
        value_frame = pl.DataFrame(
            selection.lazy()
            .filter(pl.col("symbol").is_in(selected_symbols))
            .select(pl.col(spec.primary_factor).cast(pl.Float64, strict=False))
            .drop_nulls()
            .collect()
        )
        if value_frame.is_empty():
            continue
        value = value_frame.get_column(spec.primary_factor).mean()
        if value is not None:
            rows.append({"date": rebalance_day, "avg_selected_signal": float(value)})
    if not rows:
        return
    series = pl.DataFrame(
        rows, schema={"date": pl.Date, "avg_selected_signal": pl.Float64}
    ).sort("date")
    BacktestView(nav=pl.DataFrame()).plot().timeseries(
        series,
        date_col="date",
        value_col="avg_selected_signal",
        title=f"Selected average {spec.primary_factor}",
        ylabel=spec.primary_factor,
    )


def _build_decision_overlays(
    selection_by_day: Mapping[date, pl.DataFrame],
    spec: StrategySpec,
) -> dict[date, pl.DataFrame]:
    overlays: dict[date, pl.DataFrame] = {}
    for rebalance_day, selection in selection_by_day.items():
        base_spec = StockCandidateSpec(
            filters=spec.filters,
            sort_by=(
                CandidateSort(
                    spec.primary_factor, descending=not spec.primary_ascending
                ),
            ),
            top_n=None,
            min_listing_days=spec.min_listing_days,
            exclude_limit_up=True,
            exclude_limit_down=True,
        )
        base_eligible = pl.DataFrame(base_spec.apply(selection.lazy()).collect())
        capacity_eligible = base_eligible.filter(
            pl.col("capacity_pass").fill_null(False)
            & ~pl.col("active_audit_exclusion").fill_null(False)
        )
        ranked = capacity_eligible.sort(
            spec.primary_factor,
            descending=not spec.primary_ascending,
            nulls_last=True,
        )
        selected_symbols = set(ranked.head(spec.top_n).get_column("symbol").to_list())
        base_symbols = set(base_eligible.get_column("symbol").to_list())
        overlays[rebalance_day] = pl.DataFrame(
            selection.lazy()
            .with_columns(
                pl.col("symbol").is_in(list(selected_symbols)).alias("selected"),
                pl.when(pl.col("symbol").is_in(list(selected_symbols)))
                .then(pl.lit(None, dtype=pl.String))
                .when(pl.col("active_audit_exclusion").fill_null(False))
                .then(pl.lit("audit"))
                .when(~pl.col("capacity_pass").fill_null(False))
                .then(pl.lit("capacity"))
                .when(pl.col("symbol").is_in(list(base_symbols)))
                .then(pl.lit("rank_cutoff"))
                .otherwise(pl.lit("quality_or_valuation"))
                .alias("exclude_reason"),
            )
            .select("symbol", "selected", "exclude_reason")
            .collect()
        )
    return overlays


def _selection_frames_by_day(
    store: QoreStore,
    calendar: TradingCalendar,
    spec: StrategySpec,
) -> dict[date, pl.DataFrame]:
    frames: dict[date, pl.DataFrame] = {}
    for rebalance_day in _scheduled_rebalance_days(
        calendar,
        spec.start,
        spec.end,
        spec.rebalance_schedule,
    ):
        selection = _selection_frame(store, spec, rebalance_day)
        if not selection.is_empty():
            frames[rebalance_day] = selection
    return frames


def _selection_frame(store: QoreStore, spec: StrategySpec, as_of: date) -> pl.DataFrame:
    pipeline = (
        StockSelectionPipeline.from_index(
            store,
            index_symbol=spec.benchmark,
            as_of=as_of,
        )
        .with_profiles()
        .with_statuses()
        .with_fundamentals()
        .with_daily_market()
        .with_liquidity_capacity(lookback_days=spec.liquidity_lookback_days)
        .with_audit_opinion_state(max_age_days=spec.audit_exclusion_days)
    )
    base = pl.DataFrame(
        FactorPipeline()
        .add(DebtToAssetRatioFactor())
        .run(
            pipeline.collect()
            .lazy()
            .with_columns(pl.col("operating_cashflow").alias("operating_cash_flow"))
        )
        .collect()
    )
    return pl.DataFrame(
        base.lazy()
        .with_columns(
            (
                (
                    pl.col(
                        f"position_to_amount_{spec.liquidity_lookback_days}d_ratio"
                    ).is_null()
                    | (
                        pl.col(
                            f"position_to_amount_{spec.liquidity_lookback_days}d_ratio"
                        )
                        <= spec.capacity_ratio_limit
                    )
                )
                & (
                    pl.col(f"min_amount_{spec.liquidity_lookback_days}d").is_null()
                    | (
                        pl.col(f"min_amount_{spec.liquidity_lookback_days}d")
                        >= spec.min_daily_amount_cny
                    )
                )
            ).alias("capacity_pass")
        )
        .collect()
    )


def _universe_for_backtest(store: QoreStore, spec: StrategySpec) -> Universe:
    universe = (
        StockSelectionPipeline.from_index(
            store,
            index_symbol=spec.benchmark,
            as_of=spec.end,
        )
        .with_profiles()
        .with_statuses()
        .universe(
            StockCandidateSpec(
                exclude_st=False,
                exclude_suspended=False,
                exclude_limit_up=False,
                exclude_limit_down=False,
            )
        )
    )
    if universe.collect().is_empty():
        msg = (
            "No universe data found for benchmark/as_of. "
            "Prepare index_constituents/stock_profiles/stock_ohlcv datasets first."
        )
        raise ValueError(msg)
    return universe


def _monthly_rebalance_days(
    calendar: TradingCalendar,
    start: date,
    end: date,
) -> list[date]:
    return _scheduled_rebalance_days(
        calendar,
        start,
        end,
        RebalanceSchedule(frequency="monthly", buy_delay=1, sell_delay=2),
    )


def _scheduled_rebalance_days(
    calendar: TradingCalendar,
    start: date,
    end: date,
    schedule: RebalanceSchedule,
) -> list[date]:
    trading_days = calendar.trading_days_between(start, end)
    days: list[date] = []
    seen: set[tuple[int, int, int] | tuple[int, int] | date] = set()
    for day in trading_days:
        key = schedule.bucket(day)
        if key in seen:
            continue
        seen.add(key)
        days.append(day)
    return days
