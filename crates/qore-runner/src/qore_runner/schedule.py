from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

RebalanceFrequency = Literal["event", "daily", "weekly", "monthly"]


@dataclass(frozen=True, slots=True)
class RebalanceSchedule:
    frequency: RebalanceFrequency = "daily"
    buy_delay: int = 1
    sell_delay: int = 2

    def bucket(self, as_of: date) -> tuple[int, int, int] | tuple[int, int] | date:
        if self.frequency in {"event", "daily"}:
            return as_of
        if self.frequency == "weekly":
            iso_year, iso_week, _ = as_of.isocalendar()
            return (iso_year, iso_week)
        return (as_of.year, as_of.month)
