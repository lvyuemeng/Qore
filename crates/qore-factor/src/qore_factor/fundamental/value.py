from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class BookToPriceFactor(FundamentalFactor):
    name: str = "bp"
    produces: str = "bp"
    requires: frozenset[str] = frozenset({"pb"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns((1.0 / (pl.col("pb") + 1e-8)).alias(self.produces))
