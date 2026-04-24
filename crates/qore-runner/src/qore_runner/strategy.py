from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, runtime_checkable

import polars as pl
from qore_data.universe import TradingSession, Universe

from qore_runner.calendar import TradingCalendar


@dataclass(frozen=True, slots=True)
class StrategySelectionSpec:
    top_k: int | None = None
    score_column: str = "signal"
    descending: bool = True
    tie_break_columns: tuple[str, ...] = ("symbol",)


@dataclass(frozen=True, slots=True)
class StrategyProviderFrames:
    signal_overlay: pl.DataFrame | pl.LazyFrame | None = None
    decision_overlay: pl.DataFrame | pl.LazyFrame | None = None


@dataclass(frozen=True, slots=True)
class StrategyContext:
    factor_lf: pl.LazyFrame
    universe: Universe | None
    date: date
    calendar: TradingCalendar
    providers: StrategyProviderFrames = field(default_factory=StrategyProviderFrames)
    selection: StrategySelectionSpec = field(default_factory=StrategySelectionSpec)


@runtime_checkable
class Strategy(Protocol):
    name: str
    compatible_sessions: frozenset[TradingSession]
    signal_freq: Literal["event", "daily", "weekly", "monthly"]

    @property
    def required_columns(self) -> frozenset[str]: ...

    def generate(self, context: StrategyContext) -> pl.LazyFrame:
        """Returns a LazyFrame with `symbol` and `signal` columns."""
        ...


def tradeable_universe_frame(
    universe: Universe | None,
    signal_date: date,
    calendar: TradingCalendar,
    compatible_sessions: frozenset[TradingSession],
) -> pl.DataFrame:
    del signal_date, calendar
    if universe is None:
        return pl.DataFrame(schema={"symbol": pl.String, "tradeable": pl.Boolean})
    return _tradeable_from_frame(universe, compatible_sessions)


def align_signals_to_universe(
    signals: pl.LazyFrame,
    universe_frame: pl.DataFrame,
) -> pl.LazyFrame:
    return (
        universe_frame.lazy()
        .join(signals, on="symbol", how="left")
        .with_columns(
            pl.when(pl.col("tradeable"))
            .then(pl.col("signal").cast(pl.Float64, strict=False))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("signal")
        )
        .drop("tradeable")
    )


def _tradeable_from_frame(
    universe_frame: Universe,
    compatible_sessions: frozenset[TradingSession],
) -> pl.DataFrame:
    schema = universe_frame.frame.collect_schema()
    symbol_col = universe_frame.symbol_col
    if symbol_col not in schema:
        return pl.DataFrame(schema={"symbol": pl.String, "tradeable": pl.Boolean})

    select_exprs = [pl.col(symbol_col).cast(pl.String).alias("symbol")]
    if (
        universe_frame.tradeable_col is not None
        and universe_frame.tradeable_col in schema
    ):
        select_exprs.append(
            pl.col(universe_frame.tradeable_col).alias("_tradeable_src")
        )
    if (
        universe_frame.suspended_col is not None
        and universe_frame.suspended_col in schema
    ):
        select_exprs.append(
            pl.col(universe_frame.suspended_col).alias("_suspended_src")
        )
    if universe_frame.session_col is not None and universe_frame.session_col in schema:
        select_exprs.append(pl.col(universe_frame.session_col).alias("_session_src"))
    lf = universe_frame.frame.with_row_index("_order").select(
        pl.col("_order"), *select_exprs
    )

    if (
        universe_frame.tradeable_col is not None
        and universe_frame.tradeable_col in schema
    ):
        lf = lf.with_columns(
            pl.col("_tradeable_src")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
            .alias("tradeable")
        )
    elif (
        universe_frame.suspended_col is not None
        and universe_frame.suspended_col in schema
    ):
        lf = lf.with_columns(
            (~pl.col("_suspended_src").fill_null(False)).alias("tradeable")
        )
    else:
        lf = lf.with_columns(pl.lit(True, dtype=pl.Boolean).alias("tradeable"))

    if universe_frame.session_marker is not None:
        lf = lf.with_columns(
            pl.when(pl.lit(universe_frame.session_marker in compatible_sessions))
            .then(pl.col("tradeable"))
            .otherwise(pl.lit(False, dtype=pl.Boolean))
            .alias("tradeable")
        )
    elif (
        universe_frame.session_col is not None and universe_frame.session_col in schema
    ):
        lf = lf.with_columns(
            pl.when(pl.col("_session_src").is_in(list(compatible_sessions)))
            .then(pl.col("tradeable"))
            .otherwise(pl.lit(False, dtype=pl.Boolean))
            .alias("tradeable")
        )

    return pl.DataFrame(
        lf.filter(pl.col("symbol").is_not_null())
        .group_by("symbol")
        .agg(
            pl.col("_order").min().alias("_order"),
            pl.col("tradeable").last().alias("tradeable"),
        )
        .sort("_order")
        .select("symbol", "tradeable")
        .collect()
    )
