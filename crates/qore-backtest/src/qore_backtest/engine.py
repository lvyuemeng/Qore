from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Protocol

import polars as pl
from qore_data.store.duckdb import QoreStore
from qore_data.universe import Universe
from qore_runner import RunnerSettings
from qore_runner.calendar import TradingCalendar
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import PositionSizer
from qore_runner.strategy import Strategy, StrategyContext, StrategyProviderFrames

from qore_backtest import BacktestSettings

if TYPE_CHECKING:
    from qore_backtest.view import BacktestView


class DayFrameSource(Protocol):
    def frame_for_day(
        self, trading_day: date
    ) -> pl.DataFrame | pl.LazyFrame | None: ...


class DatasetDayFrameSource(Protocol):
    def frame_for_day(
        self,
        dataset: str,
        trading_day: date,
    ) -> pl.DataFrame | pl.LazyFrame | None: ...


FactorSource = DayFrameSource
SignalOverlaySource = DayFrameSource
MarketDataSource = DatasetDayFrameSource


@dataclass(frozen=True, slots=True)
class MappingDayFrameSource:
    by_day: Mapping[date, pl.DataFrame | pl.LazyFrame | None]

    def frame_for_day(
        self,
        trading_day: date,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        return self.by_day.get(trading_day)


@dataclass(frozen=True, slots=True)
class StoreFactorSource:
    store: QoreStore
    dataset: str = "strategy_factors"
    date_column: str = "date"
    symbol_column: str = "symbol"
    factor_columns: tuple[str, ...] | None = None
    base_filters: Mapping[str, object] = field(default_factory=dict)

    def frame_for_day(
        self,
        trading_day: date,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        filters: dict[str, object] = dict(self.base_filters)
        filters[self.date_column] = trading_day
        selected_columns = (
            [self.symbol_column, *self.factor_columns] if self.factor_columns else None
        )
        frame = self.store.read(
            self.dataset,
            filters=filters,
            columns=selected_columns,
            backend="duckdb",
        ).collect()
        if frame.is_empty():
            return None
        keep_columns = [
            column
            for column in frame.columns
            if column not in {self.date_column, self.symbol_column}
        ]
        return pl.DataFrame(
            frame.lazy()
            .select(
                pl.col(self.symbol_column)
                .cast(pl.String, strict=False)
                .alias("symbol"),
                *[
                    pl.col(column).cast(pl.Float64, strict=False).alias(column)
                    for column in keep_columns
                ],
            )
            .filter(pl.col("symbol").is_not_null())
            .unique(subset=["symbol"], keep="last")
            .collect()
        )


@dataclass(frozen=True, slots=True)
class StoreMarketDataSource:
    store: QoreStore

    def frame_for_day(
        self,
        dataset: str,
        trading_day: date,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        return self.store.read(
            dataset,
            filters={"date": trading_day},
            backend="duckdb",
        ).collect()


@dataclass(frozen=True, slots=True)
class StoreSignalOverlaySource:
    store: QoreStore
    dataset: str = "news_scores"
    value_column: str = "score"

    def frame_for_day(
        self,
        trading_day: date,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        return self.store.read(
            self.dataset,
            filters={"date": trading_day},
            columns=["symbol", self.value_column],
            backend="duckdb",
        ).select(
            pl.col("symbol"),
            pl.col(self.value_column).alias("overlay"),
        )


@dataclass(frozen=True, slots=True)
class NullSignalOverlaySource:
    def frame_for_day(
        self,
        trading_day: date,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        del trading_day
        return None


@dataclass(slots=True)
class BacktestResult:
    nav: pl.DataFrame
    positions: pl.DataFrame
    turnover: pl.DataFrame
    fills: pl.DataFrame
    diagnostics: pl.DataFrame

    def view(self) -> BacktestView:
        from qore_backtest.view import BacktestView

        return BacktestView(
            nav=self.nav,
            diagnostics=self.diagnostics,
            trades=self.fills,
        )


@dataclass(slots=True)
class BacktestRunState:
    current_weights_frame: pl.DataFrame
    nav_value: float
    nav_frame: pl.DataFrame
    positions_frame: pl.DataFrame
    turnover_frame: pl.DataFrame
    fills_frame: pl.DataFrame
    diagnostics_frame: pl.DataFrame
    previous_selected_symbols: set[str]

    @classmethod
    def initialize(cls, *, initial_capital: float) -> BacktestRunState:
        return cls(
            current_weights_frame=pl.DataFrame(
                schema={"symbol": pl.String, "weight": pl.Float64}
            ),
            nav_value=initial_capital,
            nav_frame=pl.DataFrame(
                schema={"date": pl.Date, "nav": pl.Float64, "return": pl.Float64}
            ),
            positions_frame=pl.DataFrame(
                schema={"date": pl.Date, "symbol": pl.String, "weight": pl.Float64}
            ),
            turnover_frame=pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "turnover": pl.Float64,
                    "commission": pl.Float64,
                    "risk_flag": pl.Boolean,
                }
            ),
            fills_frame=pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "symbol": pl.String,
                    "status": pl.String,
                    "fill_date": pl.Date,
                    "fill_price": pl.Float64,
                    "quantity": pl.Float64,
                    "reason": pl.String,
                }
            ),
            diagnostics_frame=pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "nav": pl.Float64,
                    "daily_return": pl.Float64,
                    "turnover": pl.Float64,
                    "commission_cost": pl.Float64,
                    "fill_request_count": pl.Int64,
                    "filled_count": pl.Int64,
                    "pending_count": pl.Int64,
                    "rejected_count": pl.Int64,
                    "candidate_count": pl.Int64,
                    "signal_count": pl.Int64,
                    "selected_count": pl.Int64,
                    "drawdown_blocked": pl.Boolean,
                    "force_exit_count": pl.Int64,
                    "decision_non_selected_count": pl.Int64,
                    "forced_liquidation_symbols": pl.String,
                    "decision_selected_count": pl.Int64,
                    "decision_new_symbols": pl.String,
                    "decision_dropped_symbols": pl.String,
                    "decision_snapshot_as_of": pl.Date,
                }
            ),
            previous_selected_symbols=set(),
        )


@dataclass(slots=True)
class BacktestEngine:
    runner: StrategyRunner
    config: BacktestSettings
    calendar: TradingCalendar
    factor_source: FactorSource
    market_data_source: MarketDataSource
    signal_overlay_source: SignalOverlaySource = field(
        default_factory=NullSignalOverlaySource
    )
    _market_data_cache: dict[tuple[str, date], pl.DataFrame] = field(
        default_factory=dict
    )
    _factor_frame_cache: dict[date, pl.DataFrame] = field(default_factory=dict)
    _signal_overlay_cache: dict[date, pl.DataFrame | None] = field(default_factory=dict)
    _next_trading_day_cache: dict[tuple[date, int], date] = field(default_factory=dict)
    decision_overlays_by_day: Mapping[date, pl.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        settings: BacktestSettings,
        runner: StrategyRunner,
        calendar: TradingCalendar,
        *,
        factor_source: FactorSource,
        market_data_source: MarketDataSource,
        decision_overlays_by_day: Mapping[date, pl.DataFrame] | None = None,
        signal_overlay_source: SignalOverlaySource | None = None,
    ) -> BacktestEngine:
        return cls(
            runner=runner,
            config=settings,
            calendar=calendar,
            decision_overlays_by_day=decision_overlays_by_day or {},
            factor_source=factor_source,
            market_data_source=market_data_source,
            signal_overlay_source=signal_overlay_source or NullSignalOverlaySource(),
        )

    @classmethod
    def from_components(
        cls,
        settings: BacktestSettings,
        *,
        strategy: Strategy,
        sizer: PositionSizer,
        runner_settings: RunnerSettings,
        calendar: TradingCalendar,
        factor_source: FactorSource,
        market_data_source: MarketDataSource,
        decision_overlays_by_day: Mapping[date, pl.DataFrame] | None = None,
        signal_overlay_source: SignalOverlaySource | None = None,
    ) -> BacktestEngine:
        runner = StrategyRunner.from_settings(runner_settings, strategy, sizer)
        return cls.from_settings(
            settings,
            runner,
            calendar,
            factor_source=factor_source,
            market_data_source=market_data_source,
            decision_overlays_by_day=decision_overlays_by_day,
            signal_overlay_source=signal_overlay_source,
        )

    def run(
        self,
        universe: Universe,
        start: date,
        end: date,
    ) -> BacktestResult:
        step_keys = self._execution_step_keys(start, end)
        execution_metadata = universe.execution_metadata()
        state = BacktestRunState.initialize(initial_capital=self.config.initial_capital)

        for trading_day in step_keys:
            factor_lf = self._factor_frame_for_day(trading_day)
            strategy_context = self._strategy_context_for_day(
                trading_day,
                factor_lf,
                universe,
            )
            nav_series = (
                state.nav_frame.get_column("nav")
                if not state.nav_frame.is_empty()
                else pl.Series("nav", [state.nav_value])
            )
            target = self.runner.step(
                context=strategy_context,
                nav=nav_series,
                current_weights=state.current_weights_frame,
            )
            selected_symbols = set(
                target.weights_frame.get_column("symbol").to_list()
                if not target.weights_frame.is_empty()
                else []
            )
            selected_new_symbols: list[str] = []
            selected_dropped_symbols: list[str] = []
            current_symbols = set(
                state.current_weights_frame.get_column("symbol").to_list()
                if not state.current_weights_frame.is_empty()
                else []
            )
            forced_liquidation_symbols = sorted(
                set(target.decision.force_exit_symbols).intersection(current_symbols)
            )
            effective_target_weights_frame = _apply_forced_liquidations(
                target.weights_frame,
                forced_liquidation_symbols,
            )
            fill_requests = self._fill_requests_frame(target.decision_signals)
            day_fills = self._fills_for_requests(
                fill_requests,
                trading_day,
                execution_metadata,
            )
            turnover = self._turnover(fill_requests)
            commission_cost = turnover * self.config.commission * state.nav_value
            daily_return = self._portfolio_return(
                day_fills,
                effective_target_weights_frame,
                trading_day,
                execution_metadata,
            )
            state.nav_value = state.nav_value * (1.0 + daily_return) - commission_cost

            counts = (
                day_fills.lazy()
                .select(
                    pl.col("status")
                    .eq("filled")
                    .sum()
                    .fill_null(0)
                    .alias("filled_count"),
                    pl.col("status")
                    .eq("pending")
                    .sum()
                    .fill_null(0)
                    .alias("pending_count"),
                    pl.col("status")
                    .eq("rejected")
                    .sum()
                    .fill_null(0)
                    .alias("rejected_count"),
                )
                .collect()
            )
            filled_count = counts.get_column("filled_count").item()
            pending_count = counts.get_column("pending_count").item()
            rejected_count = counts.get_column("rejected_count").item()

            state.nav_frame = state.nav_frame.vstack(
                pl.DataFrame(
                    {
                        "date": [trading_day],
                        "nav": [state.nav_value],
                        "return": [daily_return],
                    },
                    schema={"date": pl.Date, "nav": pl.Float64, "return": pl.Float64},
                ),
                in_place=False,
            )
            state.positions_frame = state.positions_frame.vstack(
                _positions_for_day(trading_day, effective_target_weights_frame),
                in_place=False,
            )
            state.turnover_frame = state.turnover_frame.vstack(
                pl.DataFrame(
                    {
                        "date": [trading_day],
                        "turnover": [turnover],
                        "commission": [commission_cost],
                        "risk_flag": [target.diagnostics.drawdown_blocked],
                    },
                    schema={
                        "date": pl.Date,
                        "turnover": pl.Float64,
                        "commission": pl.Float64,
                        "risk_flag": pl.Boolean,
                    },
                ),
                in_place=False,
            )
            state.fills_frame = state.fills_frame.vstack(day_fills, in_place=False)
            state.diagnostics_frame = state.diagnostics_frame.vstack(
                pl.DataFrame(
                    {
                        "date": [trading_day],
                        "nav": [state.nav_value],
                        "daily_return": [daily_return],
                        "turnover": [turnover],
                        "commission_cost": [commission_cost],
                        "fill_request_count": [fill_requests.height],
                        "filled_count": [
                            int(filled_count) if isinstance(filled_count, int) else 0
                        ],
                        "pending_count": [
                            int(pending_count) if isinstance(pending_count, int) else 0
                        ],
                        "rejected_count": [
                            int(rejected_count)
                            if isinstance(rejected_count, int)
                            else 0
                        ],
                        "candidate_count": [target.diagnostics.candidate_count],
                        "signal_count": [target.diagnostics.signal_count],
                        "selected_count": [target.diagnostics.selected_count],
                        "drawdown_blocked": [target.diagnostics.drawdown_blocked],
                        "force_exit_count": [len(target.decision.force_exit_symbols)],
                        "decision_non_selected_count": [
                            target.diagnostics.non_selected_count
                        ],
                        "forced_liquidation_symbols": [
                            "|".join(forced_liquidation_symbols)
                        ],
                        "decision_selected_count": [len(selected_symbols)],
                        "decision_new_symbols": ["|".join(selected_new_symbols)],
                        "decision_dropped_symbols": [
                            "|".join(selected_dropped_symbols)
                        ],
                        "decision_snapshot_as_of": [trading_day],
                    },
                    schema=state.diagnostics_frame.schema,
                ),
                in_place=False,
            )
            state.current_weights_frame = effective_target_weights_frame
            state.previous_selected_symbols = selected_symbols

        return BacktestResult(
            nav=state.nav_frame,
            positions=state.positions_frame,
            turnover=state.turnover_frame,
            fills=state.fills_frame,
            diagnostics=state.diagnostics_frame,
        )

    def _execution_step_keys(self, start: date, end: date) -> list[date]:
        if self.config.cadence == "daily":
            return self.calendar.trading_days_between(start, end)
        return self.calendar.trading_days_between(start, end)

    def _factor_frame_for_day(self, trading_day: date) -> pl.LazyFrame:
        cached = self._factor_frame_cache.get(trading_day)
        if cached is not None:
            return cached.lazy()
        source_frame = self.factor_source.frame_for_day(trading_day)
        frame = self._materialize_source_frame(source_frame)
        if frame.is_empty():
            frame = pl.DataFrame(schema={"symbol": pl.String})
        schema_names = set(frame.columns)
        if "factor_name" in schema_names:
            msg = (
                "Factor provider long-frame contracts using 'factor_name' are no longer supported "
                f"(date={trading_day.isoformat()}). Provide a wide symbol-indexed frame."
            )
            raise ValueError(msg)
        if "symbol" in schema_names:
            frame = pl.DataFrame(
                frame.lazy()
                .with_columns(pl.col("symbol").cast(pl.String, strict=False))
                .filter(pl.col("symbol").is_not_null())
                .unique(subset=["symbol"], keep="last")
                .collect()
            )
        self._factor_frame_cache[trading_day] = frame
        return frame.lazy()

    def _strategy_context_for_day(
        self,
        trading_day: date,
        factor_lf: pl.LazyFrame,
        universe: Universe,
    ) -> StrategyContext:
        return StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=trading_day,
            calendar=self.calendar,
            providers=StrategyProviderFrames(
                signal_overlay=self._signal_overlay_frame_for_day(trading_day),
                decision_overlay=self._decision_overlay_for_day(trading_day),
            ),
        )

    def _decision_overlay_for_day(self, trading_day: date) -> pl.DataFrame | None:
        return self.decision_overlays_by_day.get(trading_day)

    def _signal_overlay_frame_for_day(self, trading_day: date) -> pl.DataFrame | None:
        if trading_day in self._signal_overlay_cache:
            return self._signal_overlay_cache[trading_day]
        source_frame = self.signal_overlay_source.frame_for_day(trading_day)
        overlay = self._materialize_source_frame(source_frame)
        if overlay.is_empty():
            self._signal_overlay_cache[trading_day] = None
            return None
        normalized = pl.DataFrame(
            overlay.lazy()
            .select(
                pl.col("symbol").cast(pl.String, strict=False),
                pl.col("overlay").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("symbol").is_not_null())
            .unique(subset=["symbol"], keep="last")
            .collect()
        )
        self._signal_overlay_cache[trading_day] = normalized
        return normalized

    def _portfolio_return(
        self,
        fills: pl.DataFrame,
        weights_frame: pl.DataFrame,
        trading_day: date,
        execution_metadata: pl.DataFrame,
    ) -> float:
        if (
            fills.is_empty()
            or weights_frame.is_empty()
            or execution_metadata.is_empty()
        ):
            return 0.0
        filled_symbols = (
            fills.lazy().filter(pl.col("status") == "filled").select("symbol")
        )
        if filled_symbols.limit(1).collect().is_empty():
            return 0.0
        eligible_weights = pl.DataFrame(
            weights_frame.lazy()
            .join(filled_symbols, on="symbol", how="semi")
            .join(
                execution_metadata.lazy().select("symbol", "session", "dataset"),
                on="symbol",
                how="inner",
            )
            .select("symbol", "weight", "session", "dataset")
            .collect()
        )
        if eligible_weights.is_empty():
            return 0.0
        total_weighted_return = 0.0
        datasets = (
            eligible_weights.get_column("dataset").drop_nulls().unique().to_list()
            if "dataset" in eligible_weights.columns
            else []
        )
        for dataset in datasets:
            dataset_value = str(dataset)
            dataset_weights = pl.DataFrame(
                eligible_weights.lazy()
                .filter(pl.col("dataset") == dataset_value)
                .select("symbol", "weight", "session")
                .collect()
            )
            if dataset_weights.is_empty():
                continue
            market = self._market_frame_for_day(dataset_value, trading_day)
            if market.is_empty():
                continue
            daily_return_expr: pl.Expr = pl.lit(0.0)
            if "daily_return" in market.columns:
                daily_return_expr = (
                    pl.col("daily_return").cast(pl.Float64, strict=False).fill_null(0.0)
                )
            open_expr: pl.Expr = pl.lit(0.0)
            if "open" in market.columns:
                open_expr = pl.col("open").cast(pl.Float64, strict=False)
            close_expr: pl.Expr = pl.lit(0.0)
            if "close" in market.columns:
                close_expr = pl.col("close").cast(pl.Float64, strict=False)
            weighted_return = pl.DataFrame(
                dataset_weights.lazy()
                .join(market.lazy(), on="symbol", how="left")
                .with_columns(
                    (
                        pl.when(pl.col("session") == "nav")
                        .then(daily_return_expr)
                        .otherwise(
                            pl.when(open_expr > 0.0)
                            .then(close_expr / open_expr - 1.0)
                            .otherwise(0.0)
                        )
                    ).alias("day_return")
                )
                .select((pl.col("weight") * pl.col("day_return").fill_null(0.0)).sum())
                .collect()
            )
            value = weighted_return.item()
            total_weighted_return += (
                float(value) if isinstance(value, (int, float)) else 0.0
            )
        return total_weighted_return

    def _fills_for_requests(
        self,
        requests: pl.DataFrame,
        trading_day: date,
        execution_metadata: pl.DataFrame,
    ) -> pl.DataFrame:
        empty_fills = pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "status": pl.String,
                "fill_date": pl.Date,
                "fill_price": pl.Float64,
                "quantity": pl.Float64,
                "reason": pl.String,
            }
        )
        if requests.is_empty():
            return empty_fills
        execution_plan = self._execution_plan(requests, trading_day, execution_metadata)
        joined_chunks: list[pl.DataFrame] = []
        for dataset, fill_date in (
            execution_plan.select("dataset", "fill_date").unique().iter_rows()
        ):
            market_slice = self._market_frame_for_day(str(dataset), fill_date)
            joined = execution_plan.filter(
                (pl.col("dataset") == dataset) & (pl.col("fill_date") == fill_date)
            ).join(market_slice, on="symbol", how="left")
            joined_chunks.append(joined)
        if not joined_chunks:
            return empty_fills
        base = pl.concat(joined_chunks, rechunk=False)
        schema = set(base.columns)
        open_expr: pl.Expr = (
            pl.col("open").cast(pl.Float64, strict=False)
            if "open" in schema
            else pl.lit(None, dtype=pl.Float64)
        )
        nav_expr: pl.Expr = (
            pl.col("nav").cast(pl.Float64, strict=False)
            if "nav" in schema
            else pl.lit(None, dtype=pl.Float64)
        )
        suspended_expr: pl.Expr = (
            pl.col("is_suspended").cast(pl.Boolean, strict=False).fill_null(False)
            if "is_suspended" in schema
            else pl.lit(False, dtype=pl.Boolean)
        )
        limit_up_expr: pl.Expr = (
            pl.col("limit_up").cast(pl.Boolean, strict=False).fill_null(False)
            if "limit_up" in schema
            else pl.lit(False, dtype=pl.Boolean)
        )
        limit_down_expr: pl.Expr = (
            pl.col("limit_down").cast(pl.Boolean, strict=False).fill_null(False)
            if "limit_down" in schema
            else pl.lit(False, dtype=pl.Boolean)
        )

        slippage_buy = 1.0 + self.config.slippage
        slippage_sell = 1.0 - self.config.slippage
        fee_buy = 1.0 + self.config.commission
        fee_sell = 1.0 - self.config.commission

        filled_price_expr = (
            pl.when(pl.col("session") == "nav")
            .then(
                pl.when(pl.col("direction") == "buy")
                .then(pl.col("nav") * fee_buy)
                .otherwise(pl.col("nav") * fee_sell)
            )
            .otherwise(
                pl.when(pl.col("direction") == "buy")
                .then(pl.col("open") * slippage_buy)
                .otherwise(pl.col("open") * slippage_sell)
            )
        )

        return pl.DataFrame(
            base.lazy()
            .select(
                pl.lit(trading_day).cast(pl.Date).alias("date"),
                pl.col("symbol").cast(pl.String, strict=False),
                pl.col("session").cast(pl.String, strict=False),
                pl.col("direction").cast(pl.String, strict=False),
                pl.col("fill_date").cast(pl.Date, strict=False),
                pl.col("quantity").cast(pl.Float64, strict=False),
                open_expr.alias("open"),
                nav_expr.alias("nav"),
                suspended_expr.alias("is_suspended"),
                limit_up_expr.alias("limit_up"),
                limit_down_expr.alias("limit_down"),
            )
            .with_columns(
                (
                    pl.col("open").is_not_null()
                    | pl.col("nav").is_not_null()
                    | pl.col("is_suspended")
                    | pl.col("limit_up")
                    | pl.col("limit_down")
                ).alias("has_market_row")
            )
            .with_columns(
                pl.when(pl.col("session") == "nav")
                .then(
                    pl.when(~pl.col("has_market_row"))
                    .then(pl.lit("rejected"))
                    .when(pl.col("nav") <= 0.0)
                    .then(pl.lit("rejected"))
                    .otherwise(pl.lit("filled"))
                )
                .when(pl.col("session") == "continuous")
                .then(
                    pl.when(~pl.col("has_market_row"))
                    .then(pl.lit("rejected"))
                    .when(pl.col("open") <= 0.0)
                    .then(pl.lit("rejected"))
                    .otherwise(pl.lit("filled"))
                )
                .otherwise(
                    pl.when(~pl.col("has_market_row"))
                    .then(pl.lit("rejected"))
                    .when(pl.col("is_suspended"))
                    .then(pl.lit("rejected"))
                    .when((pl.col("direction") == "buy") & pl.col("limit_up"))
                    .then(pl.lit("pending"))
                    .when((pl.col("direction") == "sell") & pl.col("limit_down"))
                    .then(pl.lit("pending"))
                    .when(pl.col("open") <= 0.0)
                    .then(pl.lit("rejected"))
                    .otherwise(pl.lit("filled"))
                )
                .alias("status"),
            )
            .with_columns(
                pl.when(pl.col("session") == "nav")
                .then(
                    pl.when(~pl.col("has_market_row"))
                    .then(pl.lit("missing nav data"))
                    .when(pl.col("nav") <= 0.0)
                    .then(pl.lit("invalid nav"))
                    .otherwise(pl.lit(None, dtype=pl.String))
                )
                .when(pl.col("session") == "continuous")
                .then(
                    pl.when(~pl.col("has_market_row"))
                    .then(pl.lit("missing derivative data"))
                    .when(pl.col("open") <= 0.0)
                    .then(pl.lit("invalid open price"))
                    .otherwise(pl.lit(None, dtype=pl.String))
                )
                .otherwise(
                    pl.when(~pl.col("has_market_row"))
                    .then(pl.lit("missing price data"))
                    .when(pl.col("is_suspended"))
                    .then(pl.lit("suspended"))
                    .when((pl.col("direction") == "buy") & pl.col("limit_up"))
                    .then(pl.lit("limit up"))
                    .when((pl.col("direction") == "sell") & pl.col("limit_down"))
                    .then(pl.lit("limit down"))
                    .when(pl.col("open") <= 0.0)
                    .then(pl.lit("invalid open price"))
                    .otherwise(pl.lit(None, dtype=pl.String))
                )
                .alias("reason"),
                pl.when(pl.col("status") == "filled")
                .then(filled_price_expr)
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("fill_price"),
                pl.when(pl.col("has_market_row"))
                .then(pl.col("fill_date"))
                .otherwise(pl.lit(None, dtype=pl.Date))
                .alias("fill_date"),
            )
            .select(
                "date",
                "symbol",
                "status",
                "fill_date",
                "fill_price",
                "quantity",
                "reason",
            )
            .collect()
        )

    def _execution_plan(
        self,
        requests: pl.DataFrame,
        trading_day: date,
        execution_metadata: pl.DataFrame,
    ) -> pl.DataFrame:
        if requests.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                    "dataset": pl.String,
                    "fill_date": pl.Date,
                }
            )
        schedule = self.runner.strategy.strategy_rebalance_schedule()
        buy_delay_default = max(0, schedule.buy_delay)
        sell_delay_default = max(0, schedule.sell_delay)
        plan = pl.DataFrame(
            requests.lazy()
            .join(execution_metadata.lazy(), on="symbol", how="left")
            .with_columns(
                pl.col("buy_delay")
                .cast(pl.Int64, strict=False)
                .fill_null(pl.lit(buy_delay_default)),
                pl.col("sell_delay")
                .cast(pl.Int64, strict=False)
                .fill_null(pl.lit(sell_delay_default)),
            )
            .with_columns(
                pl.when(pl.col("direction") == "buy")
                .then(pl.col("buy_delay"))
                .otherwise(pl.col("sell_delay"))
                .alias("fill_delay"),
            )
            .collect()
        )
        invalid_session_symbols = (
            pl.DataFrame(
                plan.lazy()
                .filter(
                    pl.col("session").is_null()
                    | (~pl.col("session").is_in(["auction", "continuous", "nav"]))
                    | pl.col("dataset").is_null()
                )
                .select(pl.col("symbol").cast(pl.String, strict=False).alias("symbol"))
                .filter(pl.col("symbol").is_not_null())
                .unique()
                .collect()
            )
            .get_column("symbol")
            .to_list()
        )
        if invalid_session_symbols:
            missing = ", ".join(sorted(invalid_session_symbols))
            msg = (
                "Execution metadata is required for every symbol "
                f"(missing or invalid session/dataset: {missing})."
            )
            raise ValueError(msg)
        unique_delays = {
            int(delay)
            for delay in pl.DataFrame(
                plan.lazy()
                .select(pl.col("fill_delay").cast(pl.Int64, strict=False).alias("d"))
                .filter(pl.col("d").is_not_null())
                .unique(subset=["d"])
                .collect()
            )
            .get_column("d")
            .to_list()
        }
        if not unique_delays:
            return plan.with_columns(pl.lit(trading_day).alias("fill_date")).drop(
                "fill_delay"
            )
        delay_map = pl.DataFrame(
            {
                "fill_delay": sorted(unique_delays),
                "fill_date": [
                    trading_day
                    if delay == 0
                    else self._next_trading_day(trading_day, delay)
                    for delay in sorted(unique_delays)
                ],
            },
            schema={"fill_delay": pl.Int64, "fill_date": pl.Date},
        )
        return plan.join(delay_map, on="fill_delay", how="left").drop("fill_delay")

    def _next_trading_day(self, trading_day: date, delay: int) -> date:
        key = (trading_day, delay)
        cached = self._next_trading_day_cache.get(key)
        if cached is not None:
            return cached
        resolved = self.calendar.next_trading_day(trading_day, delay)
        self._next_trading_day_cache[key] = resolved
        return resolved

    def _fill_requests_frame(
        self,
        decision_signals: pl.DataFrame,
    ) -> pl.DataFrame:
        if decision_signals.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                }
            )
        return pl.DataFrame(
            decision_signals.lazy()
            .filter(pl.col("signal").is_in(["buy", "sell"]))
            .with_columns(
                pl.col("signal").cast(pl.String, strict=False).alias("direction"),
                pl.col("weight_delta")
                .cast(pl.Float64, strict=False)
                .abs()
                .alias("quantity"),
            )
            .filter(pl.col("quantity") > 1e-12)
            .with_columns(
                pl.when(pl.col("direction") == "buy")
                .then(pl.lit("buy"))
                .when(pl.col("direction") == "sell")
                .then(pl.lit("sell"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("direction"),
            )
            .filter(pl.col("direction").is_not_null())
            .select("symbol", "direction", "quantity")
            .collect()
        )

    def _market_frame_for_day(self, dataset: str, trading_day: date) -> pl.DataFrame:
        cache_key = (dataset, trading_day)
        cached = self._market_data_cache.get(cache_key)
        if cached is not None:
            return cached
        source_frame = self.market_data_source.frame_for_day(dataset, trading_day)
        frame = self._materialize_source_frame(source_frame)
        self._market_data_cache[cache_key] = frame
        return frame

    def _materialize_source_frame(
        self,
        source_frame: pl.DataFrame | pl.LazyFrame | None,
    ) -> pl.DataFrame:
        if isinstance(source_frame, pl.LazyFrame):
            return source_frame.collect()
        if isinstance(source_frame, pl.DataFrame):
            return source_frame
        return pl.DataFrame()

    def _turnover(self, requests: pl.DataFrame) -> float:
        if requests.is_empty():
            return 0.0
        turnover = requests.select(pl.col("quantity").sum()).item()
        return float(turnover) if isinstance(turnover, (int, float)) else 0.0


def _positions_for_day(trading_day: date, weights_frame: pl.DataFrame) -> pl.DataFrame:
    if weights_frame.is_empty():
        return pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.String, "weight": pl.Float64}
        )
    return pl.DataFrame(
        weights_frame.lazy()
        .with_columns(pl.lit(trading_day).cast(pl.Date).alias("date"))
        .select("date", "symbol", "weight")
        .collect()
    )


def _apply_forced_liquidations(
    weights_frame: pl.DataFrame,
    symbols: list[str],
) -> pl.DataFrame:
    if weights_frame.is_empty() or not symbols:
        return weights_frame
    forced = pl.DataFrame(
        {
            "symbol": symbols,
            "forced_weight": [0.0] * len(symbols),
        },
        schema={"symbol": pl.String, "forced_weight": pl.Float64},
    )
    return pl.DataFrame(
        weights_frame.lazy()
        .join(forced.lazy(), on="symbol", how="full", coalesce=True)
        .with_columns(pl.coalesce("forced_weight", pl.col("weight")).alias("weight"))
        .drop("forced_weight")
        .with_columns(
            pl.col("weight")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("weight")
        )
        .collect()
    )
