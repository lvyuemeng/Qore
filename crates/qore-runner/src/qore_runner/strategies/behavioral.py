from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl

from qore_runner.strategy import Strategy


class RegimeDetector(Protocol):
    def scale(self, lf: pl.LazyFrame) -> float: ...


@dataclass(slots=True)
class BehavioralGatedStrategy:
    base: Strategy
    regime_detector: RegimeDetector | None = None
    vol_column: str = ""
    min_scale: float = 0.5
    name: str = "behavioral_gated"

    @property
    def required_columns(self) -> frozenset[str]:
        extra = frozenset({self.vol_column}) if self.vol_column else frozenset()
        return self.base.required_columns | extra

    def generate(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        base_signal = self.base.generate(factor_lf)
        regime_scale = self._regime_scale(factor_lf)
        vol_scale = self._vol_scale(factor_lf)
        scale = max(self.min_scale, min(1.0, regime_scale * vol_scale))
        return base_signal.with_columns(
            (pl.col("signal") * pl.lit(scale)).alias("signal")
        )

    def _regime_scale(self, lf: pl.LazyFrame) -> float:
        if self.regime_detector is None:
            return 1.0
        return min(max(float(self.regime_detector.scale(lf)), 0.0), 1.0)

    def _vol_scale(self, lf: pl.LazyFrame) -> float:
        if not self.vol_column:
            return 1.0
        schema = lf.collect_schema()
        if self.vol_column not in schema:
            return 1.0
        summary = pl.DataFrame(
            lf.select(
                pl.col(self.vol_column).fill_null(0.0).mean().alias("mean_vol")
            ).collect()
        )
        mean_value = summary.get_column("mean_vol").item()
        mean_vol = float(mean_value) if isinstance(mean_value, (int, float)) else 0.0
        if mean_vol <= 0.0:
            return 1.0
        return min(1.0, 1.0 / (1.0 + mean_vol))
