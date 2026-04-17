from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import polars as pl

from qore_factor.base import OHLCVFactor


@dataclass(slots=True)
class RealizedVolatilityFactor(OHLCVFactor):
    window: int = 20
    annualization: int = 252
    name: str = field(init=False)
    produces: str = field(init=False)
    requires: frozenset[str] = frozenset({"symbol", "close"})

    def __post_init__(self) -> None:
        self.name = f"realized_volatility_{self.window}d"
        self.produces = f"realized_vol_{self.window}d"

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        scale = sqrt(self.annualization)
        log_return = (
            (pl.col("close") / pl.col("close").shift(1).over("symbol")).log()
        ).alias("_log_return")
        return (
            lf.with_columns(log_return)
            .with_columns(
                (
                    pl.col("_log_return").rolling_std(self.window).over("symbol")
                    * scale
                ).alias(self.produces)
            )
            .drop("_log_return")
        )
