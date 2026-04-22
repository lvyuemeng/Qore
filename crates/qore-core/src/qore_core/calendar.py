from __future__ import annotations

import json
from datetime import date, timedelta
from functools import singledispatch
from importlib.resources import files
from typing import TypeVar

from qore_core.config import QoreConfig
from qore_core.instrument import (
    DerivativeInstrument,
    FundInstrument,
    SessionInstrument,
    StockInstrument,
)

TInstrument = TypeVar("TInstrument", bound=SessionInstrument)


@singledispatch
def execution_delay(inst: SessionInstrument) -> int:
    raise TypeError(f"No execution delay logic for {type(inst).__name__}")


@execution_delay.register(StockInstrument)
def _stock_execution_delay(inst: StockInstrument) -> int:
    del inst
    return 1


@execution_delay.register(FundInstrument)
def _fund_execution_delay(inst: FundInstrument) -> int:
    return inst.subscription_delay


@execution_delay.register(DerivativeInstrument)
def _derivative_execution_delay(inst: DerivativeInstrument) -> int:
    del inst
    return 0


class TradingCalendar:
    def __init__(self, holidays: set[date] | None = None) -> None:
        self._holidays = holidays or set()

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays

    def next_trading_day(self, d: date, n: int = 1) -> date:
        return self._shift_trading_day(d, n, 1)

    def prev_trading_day(self, d: date, n: int = 1) -> date:
        return self._shift_trading_day(d, n, -1)

    def trading_days_between(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        return [
            current
            for current in _date_range(start, end)
            if self.is_trading_day(current)
        ]

    def fill_date(self, signal_date: date, inst: TInstrument) -> date:
        delay = execution_delay(inst)
        if delay == 0:
            return signal_date
        return self.next_trading_day(signal_date, delay)

    def _shift_trading_day(self, d: date, n: int, direction: int) -> date:
        if n < 0:
            msg = f"n must be >= 0, got {n}"
            raise ValueError(msg)
        current = d
        remaining = n
        step = timedelta(days=direction)
        while remaining > 0:
            current += step
            if self.is_trading_day(current):
                remaining -= 1
        return current

    @classmethod
    def from_config(cls, config: QoreConfig) -> TradingCalendar:
        del config
        holiday_file = files("qore_core").joinpath("holidays.json")
        data = json.loads(holiday_file.read_text(encoding="utf-8"))
        holidays = {date.fromisoformat(item) for item in data.get("holidays", [])}
        return cls(holidays=holidays)


def _date_range(start: date, end: date) -> list[date]:
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]
