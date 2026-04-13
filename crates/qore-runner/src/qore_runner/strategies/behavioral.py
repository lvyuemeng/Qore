from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.instrument import TradingSession
from qore_core.universe import Universe

from qore_runner.strategy import Strategy


class RegimeDetector(Protocol):
    def scale(self, lf: pl.LazyFrame, as_of: date) -> float: ...


@dataclass(slots=True)
class BehavioralGatedStrategy:
    base: Strategy
    regime_detector: RegimeDetector | None = None
    vol_lookback: int = 20
    min_scale: float = 0.5
    name: str = "behavioral_gated"

    @property
    def compatible_sessions(self) -> frozenset[TradingSession]:
        return self.base.compatible_sessions

    @property
    def signal_freq(self) -> str:
        return self.base.signal_freq

    @property
    def required_columns(self) -> frozenset[str]:
        return self.base.required_columns | {"realized_vol_20d"}

    def generate(
        self,
        lf: pl.LazyFrame,
        universe: Universe,
        date: date,
        calendar: TradingCalendar,
    ) -> pl.Series:
        base_signal = self.base.generate(lf, universe, date, calendar)
        df = lf.collect()
        regime_scale = self._regime_scale(lf, date)
        vol_scale = self._vol_scale(df)
        scale = max(self.min_scale, min(1.0, regime_scale * vol_scale))
        return pl.Series(
            name="signal",
            values=[
                None if value is None else float(value) * scale
                for value in base_signal.to_list()
            ],
        )

    def _regime_scale(self, lf: pl.LazyFrame, as_of: date) -> float:
        if self.regime_detector is None:
            return 1.0
        return min(max(float(self.regime_detector.scale(lf, as_of)), 0.0), 1.0)

    def _vol_scale(self, df: pl.DataFrame) -> float:
        if "realized_vol_20d" not in df.columns:
            return 1.0
        mean_vol = float(df.get_column("realized_vol_20d").fill_null(0.0).mean() or 0.0)
        if mean_vol <= 0.0:
            return 1.0
        return min(1.0, 1.0 / (1.0 + mean_vol))
