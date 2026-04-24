# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BacktestSettings:
    initial_capital: float = 10_000_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    drawdown_stop: float = 0.15
    cadence: Literal["daily", "intraday"] = "daily"


from qore_runner.calendar import TradingCalendar

from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_backtest.metrics import compute_metrics
from qore_backtest.simulate import Fill, fill_order
from qore_backtest.view import BacktestView

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestSettings",
    "BacktestView",
    "Fill",
    "TradingCalendar",
    "compute_metrics",
    "fill_order",
]
