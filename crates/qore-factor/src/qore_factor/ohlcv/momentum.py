from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qore_factor.base import OHLCVFactor


@dataclass(slots=True)
class MomentumFactor(OHLCVFactor):
    close_column: str = "close"
    lookback: int = 252
    skip: int = 21
    name: str = field(init=False)
    produces: str = field(init=False)
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"momentum_{self.lookback}d_skip{self.skip}"
        self.produces = f"mom_{self.lookback}d_skip{self.skip}"
        self.requires = frozenset({"date", "symbol", self.close_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lag = self.lookback + self.skip
        return lf.with_columns(
            (
                pl.col(self.close_column)
                / pl.col(self.close_column).shift(lag).over("symbol")
                - 1.0
            ).alias(self.produces)
        )
