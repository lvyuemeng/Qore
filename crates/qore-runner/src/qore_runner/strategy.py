from __future__ import annotations

from datetime import date
from typing import Literal, Protocol, runtime_checkable

import polars as pl

from qore_core.calendar import TradingCalendar
from qore_core.instrument import TradingSession
from qore_core.universe import Universe


@runtime_checkable
class Strategy(Protocol):
    name: str
    compatible_sessions: frozenset[TradingSession]
    signal_freq: Literal["event", "daily", "weekly", "monthly"]
    required_columns: frozenset[str]

    def generate(
        self,
        lf: pl.LazyFrame,
        universe: Universe,
        date: date,
        calendar: TradingCalendar,
    ) -> pl.Series:
        """Returns Series(name='signal') indexed by row order."""
