from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class ROEStabilityFactor(FundamentalFactor):
    window: int = 8
    name: str = "roe_stability"
    produces: str = "roe_stability"
    requires: frozenset[str] = frozenset({"symbol", "roe"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                1.0 / (pl.col("roe").rolling_std(self.window).over("symbol") + 1e-8)
            ).alias(self.produces)
        )
