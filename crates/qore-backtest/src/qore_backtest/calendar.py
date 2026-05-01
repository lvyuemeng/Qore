from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    _holidays: set[date] = field(default_factory=set, hash=False, compare=False)
    _fill_next_day_cache: dict[tuple[date, int], date] = field(
        default_factory=dict, hash=False, compare=False
    )

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays

    def next_trading_day(self, d: date, n: int = 1) -> date:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        current = d
        for _ in range(n):
            current += timedelta(days=1)
            while not self.is_trading_day(current):
                current += timedelta(days=1)
        return current

    def prev_trading_day(self, d: date, n: int = 1) -> date:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        current = d
        for _ in range(n):
            current -= timedelta(days=1)
            while not self.is_trading_day(current):
                current -= timedelta(days=1)
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

    def _next_trading_day(self, trading_day: date, delay: int) -> date:
        key = (trading_day, delay)
        cached = self._fill_next_day_cache.get(key)
        if cached is not None:
            return cached
        resolved = self.next_trading_day(trading_day, delay)
        self._fill_next_day_cache[key] = resolved
        return resolved

    def fill_requests_from_signals(
        self, decision_signals: pl.DataFrame
    ) -> pl.DataFrame:
        if decision_signals.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                }
            )
        return pl.DataFrame(
            decision_signals.lazy()
            .filter(pl.col("signal").is_in(["buy", "sell"]))
            .with_columns(
                pl.col("signal").cast(pl.String).alias("direction"),
                pl.col("weight_delta").cast(pl.Float64).abs().alias("quantity"),
            )
            .filter(pl.col("quantity") > 1e-12)
            .select("symbol", "direction", "quantity")
            .collect()
        )

    def fill_plan(
        self,
        requests: pl.DataFrame,
        trading_day: date,
        buy_delay: int,
        sell_delay: int,
    ) -> pl.DataFrame:
        if requests.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                    "fill_date": pl.Date,
                }
            )
        plan = pl.DataFrame(
            requests.lazy()
            .with_columns(
                pl.when(pl.col("direction") == "buy")
                .then(pl.lit(buy_delay))
                .otherwise(pl.lit(sell_delay))
                .alias("fill_delay"),
            )
            .collect()
        )
        delays = {int(d) for d in plan.get_column("fill_delay").to_list()}
        if not delays:
            return plan.with_columns(pl.lit(trading_day).alias("fill_date")).drop(
                "fill_delay"
            )
        delay_map = pl.DataFrame(
            {
                "fill_delay": sorted(delays),
                "fill_date": [
                    trading_day if d == 0 else self._next_trading_day(trading_day, d)
                    for d in sorted(delays)
                ],
            },
            schema={"fill_delay": pl.Int64, "fill_date": pl.Date},
        )
        return plan.join(delay_map, on="fill_delay", how="left").drop("fill_delay")
