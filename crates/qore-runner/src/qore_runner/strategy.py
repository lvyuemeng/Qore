from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, TypeVar, runtime_checkable

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.instrument import SessionInstrument, TradingSession
from qore_core.universe import Universe

TInstrument = TypeVar("TInstrument", bound=SessionInstrument)


@dataclass(frozen=True, slots=True)
class StrategyContext[TInstrument: SessionInstrument]:
    factor_lf: pl.LazyFrame
    universe: Universe[TInstrument]
    date: date
    calendar: TradingCalendar
    inputs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategyResult:
    signals: pl.LazyFrame


@runtime_checkable
class Strategy[TInstrument: SessionInstrument](Protocol):
    name: str
    compatible_sessions: frozenset[TradingSession]
    signal_freq: Literal["event", "daily", "weekly", "monthly"]
    required_columns: frozenset[str]

    def generate(self, context: StrategyContext[TInstrument]) -> StrategyResult:
        """Returns a LazyFrame with `symbol` and `signal` columns."""
        ...


def tradeable_universe_frame[TInstrument: SessionInstrument](
    universe: Universe[TInstrument],
    signal_date: date,
    calendar: TradingCalendar,
    compatible_sessions: frozenset[TradingSession],
) -> pl.DataFrame:
    rows = [
        {
            "symbol": inst.symbol,
            "tradeable": _is_tradeable(
                universe, inst, signal_date, calendar, compatible_sessions
            ),
        }
        for inst in universe
    ]
    if not rows:
        return pl.DataFrame(schema={"symbol": pl.String, "tradeable": pl.Boolean})
    return pl.DataFrame(rows).with_columns(
        pl.col("symbol").cast(pl.String),
        pl.col("tradeable").cast(pl.Boolean),
    )


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


def _is_tradeable[TInstrument: SessionInstrument](
    universe: Universe[TInstrument],
    inst: TInstrument,
    signal_date: date,
    calendar: TradingCalendar,
    compatible_sessions: frozenset[TradingSession],
) -> bool:
    if inst.session not in compatible_sessions:
        return False
    return not universe.is_suspended(
        inst.symbol,
        calendar.fill_date(signal_date, inst),
    )
