from __future__ import annotations

import json
from datetime import date, timedelta
from importlib.resources import files

from qore_core.config import QoreConfig
from qore_core.instrument import (
    DerivativeInstrument,
    FundInstrument,
    Instrument,
    StockInstrument,
)


class TradingCalendar:
    def __init__(self, holidays: set[date] | None = None) -> None:
        self._holidays = holidays or set()

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays

    def next_trading_day(self, d: date, n: int = 1) -> date:
        if n < 0:
            msg = f"n must be >= 0, got {n}"
            raise ValueError(msg)
        current = d
        remaining = n
        while remaining > 0:
            current += timedelta(days=1)
            if self.is_trading_day(current):
                remaining -= 1
        return current

    def prev_trading_day(self, d: date, n: int = 1) -> date:
        if n < 0:
            msg = f"n must be >= 0, got {n}"
            raise ValueError(msg)
        current = d
        remaining = n
        while remaining > 0:
            current -= timedelta(days=1)
            if self.is_trading_day(current):
                remaining -= 1
        return current

    def trading_days_between(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        days: list[date] = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def fill_date(self, signal_date: date, inst: Instrument) -> date:
        match inst:
            case StockInstrument():
                return self.next_trading_day(signal_date, 1)
            case FundInstrument(subscription_delay=delay):
                return self.next_trading_day(signal_date, delay)
            case DerivativeInstrument():
                return signal_date

    @classmethod
    def from_config(cls, config: QoreConfig) -> TradingCalendar:
        del config
        holiday_file = files("qore_core").joinpath("holidays.json")
        data = json.loads(holiday_file.read_text(encoding="utf-8"))
        holidays = {date.fromisoformat(item) for item in data.get("holidays", [])}
        return cls(holidays=holidays)
