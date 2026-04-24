from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import polars as pl
from qore_data.universe import TradingSession

from qore_runner.strategy import Strategy, StrategyContext


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
    def signal_freq(self) -> Literal["event", "daily", "weekly", "monthly"]:
        return self.base.signal_freq

    @property
    def required_columns(self) -> frozenset[str]:
        return self.base.required_columns | {"realized_vol_20d"}

    def generate(self, context: StrategyContext) -> pl.LazyFrame:
        base_signal = self.base.generate(context)
        regime_scale = self._regime_scale(context.factor_lf, context.date)
        vol_scale = self._vol_scale(context.factor_lf)
        scale = max(self.min_scale, min(1.0, regime_scale * vol_scale))
        return base_signal.with_columns(
            (pl.col("signal") * pl.lit(scale)).alias("signal")
        )

    def _regime_scale(self, lf: pl.LazyFrame, as_of: date) -> float:
        if self.regime_detector is None:
            return 1.0
        return min(max(float(self.regime_detector.scale(lf, as_of)), 0.0), 1.0)

    def _vol_scale(self, lf: pl.LazyFrame) -> float:
        schema = lf.collect_schema()
        if "realized_vol_20d" not in schema:
            return 1.0
        summary = pl.DataFrame(
            lf.select(
                pl.col("realized_vol_20d").fill_null(0.0).mean().alias("mean_vol")
            ).collect()
        )
        mean_value = summary.get_column("mean_vol").item()
        mean_vol = float(mean_value) if isinstance(mean_value, (int, float)) else 0.0
        if mean_vol <= 0.0:
            return 1.0
        return min(1.0, 1.0 / (1.0 + mean_vol))
