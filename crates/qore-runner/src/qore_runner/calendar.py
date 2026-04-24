from __future__ import annotations

from datetime import date, timedelta


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


def _date_range(start: date, end: date) -> list[date]:
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]
