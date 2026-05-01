# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass(frozen=True, slots=True)
class BacktestSettings:
    initial_capital: float = 10_000_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    cadence: Literal["daily", "intraday"] = "daily"
    buy_delay: int = 1
    sell_delay: int = 2
    start: date = field(default_factory=lambda: date(2010, 1, 1))
    end: date = field(default_factory=date.today)


from qore_backtest.calendar import TradingCalendar
from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_backtest.view import BacktestView

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestSettings",
    "BacktestView",
    "TradingCalendar",
]
