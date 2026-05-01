from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class BookToPriceFactor(FundamentalFactor):
    pb_column: str = "pb"
    name: str = "bp"
    produces: str = "bp"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset({self.pb_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (1.0 / (pl.col(self.pb_column) + 1e-8)).alias(self.produces)
        )
