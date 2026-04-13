from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(slots=True)
class CarryFactor:
    name: str = "carry"
    produces: str = "carry"
    requires: frozenset[str] = frozenset({"near_price", "far_price", "days_to_roll"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                (pl.col("near_price") - pl.col("far_price"))
                / (pl.col("near_price") + 1e-8)
                / (pl.col("days_to_roll") + 1e-8)
            ).alias(self.produces)
        )
