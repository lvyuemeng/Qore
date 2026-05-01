from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(slots=True)
class CrossSectionalScreener:
    factor_weights: dict[str, float]
    name: str = "cross_sectional_screener"

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset(self.factor_weights)

    def generate(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        if not self.factor_weights:
            return factor_lf.select(
                "symbol", pl.lit(None, dtype=pl.Float64).alias("signal")
            )
        return factor_lf.select(
            "symbol",
            pl.sum_horizontal(
                *(pl.col(c) * w for c, w in self.factor_weights.items())
            ).alias("signal"),
        )
