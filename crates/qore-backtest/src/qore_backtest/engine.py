from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal, cast

import polars as pl
from qore_data.store.duckdb import QoreStore
from qore_data.universe import Universe
from qore_runner import RunnerSettings
from qore_runner.calendar import TradingCalendar
from qore_runner.runner import StrategyRunner, TargetPortfolio
from qore_runner.sizer import PositionSizer
from qore_runner.strategy import Strategy, StrategyContext, StrategyProviderFrames

from qore_backtest import BacktestSettings
from qore_backtest.simulate import (
    TradingSession,
    fill_order_from_market_row,
)

if TYPE_CHECKING:
    from qore_backtest.view import BacktestView


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
            current_weights_frame=_empty_weights_frame(),
            nav_value=initial_capital,
            nav_frame=_empty_nav_frame(),
            positions_frame=_empty_positions_frame(),
            turnover_frame=_empty_turnover_frame(),
            fills_frame=_empty_fills_frame(),
            diagnostics_frame=_empty_diagnostics_frame(),
            previous_selected_symbols=set(),
        )


@dataclass(frozen=True, slots=True)
class BacktestDaySummary:
    trading_day: date
    target: TargetPortfolio
    fill_requests: pl.DataFrame
    fills: pl.DataFrame
    turnover: float
    commission_cost: float
    daily_return: float
    nav_value: float
    force_exit_count: int
    decision_non_selected_count: int
    forced_liquidation_symbols: str
    decision_selected_count: int
    decision_new_symbols: str
    decision_dropped_symbols: str
    decision_snapshot_as_of: date


@dataclass(slots=True)
class BacktestEngine:
    runner: StrategyRunner
    store: QoreStore
    config: BacktestSettings
    calendar: TradingCalendar
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
        store: QoreStore,
        calendar: TradingCalendar,
        decision_overlays_by_day: Mapping[date, pl.DataFrame] | None = None,
    ) -> BacktestEngine:
        return cls(
            runner=runner,
            store=store,
            config=settings,
            calendar=calendar,
            decision_overlays_by_day=decision_overlays_by_day or {},
        )

    @classmethod
    def from_components(
        cls,
        settings: BacktestSettings,
        *,
        strategy: Strategy,
        sizer: PositionSizer,
        runner_settings: RunnerSettings,
        store: QoreStore,
        calendar: TradingCalendar,
        decision_overlays_by_day: Mapping[date, pl.DataFrame] | None = None,
    ) -> BacktestEngine:
        runner = StrategyRunner.from_settings(runner_settings, strategy, sizer)
        return cls.from_settings(
            settings,
            runner,
            store,
            calendar,
            decision_overlays_by_day=decision_overlays_by_day,
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
            )
            selected_symbols = set(target.decision.selected_symbols)
            selected_new_symbols = sorted(
                selected_symbols - state.previous_selected_symbols
            )
            selected_dropped_symbols = sorted(
                state.previous_selected_symbols - selected_symbols
            )
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
            fill_requests = self._fill_requests_frame(
                effective_target_weights_frame,
                state.current_weights_frame,
            )
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

            summary = BacktestDaySummary(
                trading_day=trading_day,
                target=target,
                fill_requests=fill_requests,
                fills=day_fills,
                turnover=turnover,
                commission_cost=commission_cost,
                daily_return=daily_return,
                nav_value=state.nav_value,
                force_exit_count=len(target.decision.force_exit_symbols),
                decision_non_selected_count=(
                    target.decision.frame.height - len(target.decision.selected_symbols)
                ),
                forced_liquidation_symbols="|".join(forced_liquidation_symbols),
                decision_selected_count=len(selected_symbols),
                decision_new_symbols="|".join(selected_new_symbols),
                decision_dropped_symbols="|".join(selected_dropped_symbols),
                decision_snapshot_as_of=trading_day,
            )
            state.nav_frame = state.nav_frame.vstack(
                pl.DataFrame(
                    {
                        "date": [summary.trading_day],
                        "nav": [summary.nav_value],
                        "return": [summary.daily_return],
                    },
                    schema={"date": pl.Date, "nav": pl.Float64, "return": pl.Float64},
                ),
                in_place=False,
            )
            state.positions_frame = state.positions_frame.vstack(
                _positions_for_day(summary.trading_day, effective_target_weights_frame),
                in_place=False,
            )
            state.turnover_frame = state.turnover_frame.vstack(
                pl.DataFrame(
                    {
                        "date": [summary.trading_day],
                        "turnover": [summary.turnover],
                        "commission": [summary.commission_cost],
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
            state.fills_frame = state.fills_frame.vstack(summary.fills, in_place=False)
            state.diagnostics_frame = state.diagnostics_frame.vstack(
                _diagnostics_frame_from_summary(summary),
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
        factor_scores = self.store.read(
            "factor_scores",
            filters={"date": trading_day},
            columns=["symbol", "factor_name", "z_score"],
            backend="duckdb",
        )
        frame = factor_scores.collect()
        if frame.is_empty():
            empty = pl.DataFrame({"symbol": []}, schema={"symbol": pl.String})
            self._factor_frame_cache[trading_day] = empty
            return empty.lazy()
        pivoted = frame.pivot(on="factor_name", index="symbol", values="z_score")
        self._factor_frame_cache[trading_day] = pivoted
        return pivoted.lazy()

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
        news = self.store.read(
            "news_scores",
            filters={"date": trading_day},
            columns=["symbol", "score"],
            backend="duckdb",
        ).collect()
        if news.is_empty():
            self._signal_overlay_cache[trading_day] = None
            return None
        normalized = pl.DataFrame(
            news.lazy()
            .select(
                pl.col("symbol").cast(pl.String, strict=False),
                pl.col("score").cast(pl.Float64, strict=False).alias("overlay"),
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
            market = self._market_data_for_date(dataset_value, trading_day)
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
        if requests.is_empty():
            return _empty_fills_frame()
        execution_plan = self._execution_plan(requests, trading_day, execution_metadata)
        joined_chunks: list[pl.DataFrame] = []
        for dataset, fill_date in (
            execution_plan.select("dataset", "fill_date").unique().iter_rows()
        ):
            market_slice = self._market_data_for_date(str(dataset), fill_date)
            joined = execution_plan.filter(
                (pl.col("dataset") == dataset) & (pl.col("fill_date") == fill_date)
            ).join(market_slice, on="symbol", how="left")
            joined_chunks.append(joined)
        if not joined_chunks:
            return _empty_fills_frame()
        open_expr: pl.Expr = pl.lit(None, dtype=pl.Float64)
        if "open" in joined_chunks[0].columns:
            open_expr = pl.col("open").cast(pl.Float64, strict=False)
        nav_expr: pl.Expr = pl.lit(None, dtype=pl.Float64)
        if "nav" in joined_chunks[0].columns:
            nav_expr = pl.col("nav").cast(pl.Float64, strict=False)
        joined_rows = (
            pl.concat(joined_chunks, rechunk=False)
            .lazy()
            .select(
                pl.col("symbol").cast(pl.String, strict=False),
                pl.col("session").cast(pl.String, strict=False),
                pl.col("direction").cast(pl.String, strict=False),
                pl.col("quantity").cast(pl.Float64, strict=False),
                pl.col("buy_delay").cast(pl.Int64, strict=False).fill_null(1),
                pl.col("sell_delay").cast(pl.Int64, strict=False).fill_null(2),
                open_expr.alias("open"),
                nav_expr.alias("nav"),
                pl.col("is_suspended").cast(pl.Boolean, strict=False).fill_null(False),
                pl.col("limit_up").cast(pl.Boolean, strict=False).fill_null(False),
                pl.col("limit_down").cast(pl.Boolean, strict=False).fill_null(False),
            )
            .collect()
        )
        return self._fills_from_frame(joined_rows, trading_day)

    def _fills_from_frame(
        self,
        frame: pl.DataFrame,
        trading_day: date,
    ) -> pl.DataFrame:
        if frame.is_empty():
            return _empty_fills_frame()

        def build_fill(
            row: tuple[object, ...],
        ) -> tuple[str, str, date | None, float | None, float, str | None] | None:
            (
                symbol,
                session,
                direction,
                quantity,
                buy_delay,
                sell_delay,
                open_,
                nav,
                suspended,
                limit_up,
                limit_down,
            ) = row
            if (
                not isinstance(symbol, str)
                or not isinstance(session, str)
                or not isinstance(direction, str)
                or not isinstance(quantity, (int, float))
                or not isinstance(buy_delay, int)
                or not isinstance(sell_delay, int)
            ):
                return None
            fill = fill_order_from_market_row(
                symbol,
                cast(TradingSession, session),
                trading_day,
                cast(Literal["buy", "sell"], direction),
                float(quantity),
                {
                    "open": open_ if isinstance(open_, (int, float)) else None,
                    "nav": nav if isinstance(nav, (int, float)) else None,
                    "is_suspended": bool(suspended),
                    "limit_up": bool(limit_up),
                    "limit_down": bool(limit_down),
                },
                self.config,
                self.calendar,
                buy_delay=buy_delay,
                sell_delay=sell_delay,
            )
            return (
                fill.symbol,
                fill.status,
                fill.fill_date,
                fill.fill_price,
                fill.quantity,
                fill.reason,
            )

        def to_frame(
            rows: list[tuple[str, str, date | None, float | None, float, str | None]],
        ) -> pl.DataFrame:
            if not rows:
                return _empty_fills_frame()
            symbols = [row[0] for row in rows]
            statuses = [row[1] for row in rows]
            fill_dates = [row[2] for row in rows]
            fill_prices = [row[3] for row in rows]
            quantities = [row[4] for row in rows]
            reasons = [row[5] for row in rows]
            return pl.DataFrame(
                {
                    "date": [trading_day] * len(rows),
                    "symbol": symbols,
                    "status": statuses,
                    "fill_date": fill_dates,
                    "fill_price": fill_prices,
                    "quantity": quantities,
                    "reason": reasons,
                },
                schema={
                    "date": pl.Date,
                    "symbol": pl.String,
                    "status": pl.String,
                    "fill_date": pl.Date,
                    "fill_price": pl.Float64,
                    "quantity": pl.Float64,
                    "reason": pl.String,
                },
            )

        row_iter = frame.iter_rows(named=False)
        if frame.height < 64:
            records = [
                row_fill
                for row in row_iter
                if (row_fill := build_fill(row)) is not None
            ]
            return to_frame(records)

        max_workers = max(1, min(8, (os.cpu_count() or 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            ordered = pool.map(build_fill, row_iter)
            records = [row_fill for row_fill in ordered if row_fill is not None]
            return to_frame(records)

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
        plan = pl.DataFrame(
            requests.lazy()
            .join(execution_metadata.lazy(), on="symbol", how="left")
            .with_columns(
                pl.col("buy_delay").cast(pl.Int64, strict=False).fill_null(1),
                pl.col("sell_delay").cast(pl.Int64, strict=False).fill_null(2),
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
            for delay in plan.get_column("fill_delay").to_list()
            if isinstance(delay, int)
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
        target_weights: pl.DataFrame,
        current_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        if target_weights.is_empty() and current_weights.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                }
            )
        return pl.DataFrame(
            target_weights.lazy()
            .rename({"weight": "target_weight"})
            .join(
                current_weights.lazy().rename({"weight": "current_weight"}),
                on="symbol",
                how="full",
                coalesce=True,
            )
            .with_columns(
                pl.col("target_weight").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("current_weight").cast(pl.Float64, strict=False).fill_null(0.0),
            )
            .with_columns(
                (pl.col("target_weight") - pl.col("current_weight")).alias("delta")
            )
            .filter(pl.col("delta").abs() > 1e-12)
            .with_columns(
                pl.when(pl.col("delta") > 0.0)
                .then(pl.lit("buy"))
                .otherwise(pl.lit("sell"))
                .alias("direction"),
                pl.col("delta").abs().alias("quantity"),
            )
            .select("symbol", "direction", "quantity")
            .collect()
        )

    def _market_data_for_date(self, dataset: str, trading_day: date) -> pl.DataFrame:
        cache_key = (dataset, trading_day)
        cached = self._market_data_cache.get(cache_key)
        if cached is not None:
            return cached
        frame = self.store.read(
            dataset,
            filters={"date": trading_day},
            backend="duckdb",
        ).collect()
        self._market_data_cache[cache_key] = frame
        return frame

    def _turnover(self, requests: pl.DataFrame) -> float:
        if requests.is_empty():
            return 0.0
        turnover = requests.select(pl.col("quantity").sum()).item()
        return float(turnover) if isinstance(turnover, (int, float)) else 0.0


def _empty_weights_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"symbol": pl.String, "weight": pl.Float64})


def _empty_nav_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"date": pl.Date, "nav": pl.Float64, "return": pl.Float64}
    )


def _empty_positions_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"date": pl.Date, "symbol": pl.String, "weight": pl.Float64}
    )


def _empty_turnover_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.Date,
            "turnover": pl.Float64,
            "commission": pl.Float64,
            "risk_flag": pl.Boolean,
        }
    )


def _empty_fills_frame() -> pl.DataFrame:
    return pl.DataFrame(
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


def _positions_for_day(trading_day: date, weights_frame: pl.DataFrame) -> pl.DataFrame:
    if weights_frame.is_empty():
        return _empty_positions_frame()
    return pl.DataFrame(
        weights_frame.lazy()
        .with_columns(pl.lit(trading_day).cast(pl.Date).alias("date"))
        .select("date", "symbol", "weight")
        .collect()
    )


def _empty_diagnostics_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_diagnostics_schema())


def _diagnostics_schema() -> dict[str, pl.DataType]:
    return {
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


def _diagnostics_frame_from_summary(summary: BacktestDaySummary) -> pl.DataFrame:
    counts = (
        summary.fills.lazy()
        .select(
            pl.col("status").eq("filled").sum().fill_null(0).alias("filled_count"),
            pl.col("status").eq("pending").sum().fill_null(0).alias("pending_count"),
            pl.col("status").eq("rejected").sum().fill_null(0).alias("rejected_count"),
        )
        .collect()
    )
    filled_count = counts.get_column("filled_count").item()
    pending_count = counts.get_column("pending_count").item()
    rejected_count = counts.get_column("rejected_count").item()
    return pl.DataFrame(
        {
            "date": [summary.trading_day],
            "nav": [summary.nav_value],
            "daily_return": [summary.daily_return],
            "turnover": [summary.turnover],
            "commission_cost": [summary.commission_cost],
            "fill_request_count": [summary.fill_requests.height],
            "filled_count": [int(filled_count) if isinstance(filled_count, int) else 0],
            "pending_count": [
                int(pending_count) if isinstance(pending_count, int) else 0
            ],
            "rejected_count": [
                int(rejected_count) if isinstance(rejected_count, int) else 0
            ],
            "candidate_count": [summary.target.diagnostics.candidate_count],
            "signal_count": [summary.target.diagnostics.signal_count],
            "selected_count": [summary.target.diagnostics.selected_count],
            "drawdown_blocked": [summary.target.diagnostics.drawdown_blocked],
            "force_exit_count": [summary.force_exit_count],
            "decision_non_selected_count": [summary.decision_non_selected_count],
            "forced_liquidation_symbols": [summary.forced_liquidation_symbols],
            "decision_selected_count": [summary.decision_selected_count],
            "decision_new_symbols": [summary.decision_new_symbols],
            "decision_dropped_symbols": [summary.decision_dropped_symbols],
            "decision_snapshot_as_of": [summary.decision_snapshot_as_of],
        },
        schema=_diagnostics_schema(),
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
        .join(forced.lazy(), on="symbol", how="left")
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
