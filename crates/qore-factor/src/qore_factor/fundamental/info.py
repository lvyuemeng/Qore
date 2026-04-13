from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class SUEFactor(FundamentalFactor):
    name: str = "sue"
    produces: str = "sue"
    requires: frozenset[str] = frozenset({"actual_eps", "consensus_eps", "eps_std"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                (pl.col("actual_eps") - pl.col("consensus_eps"))
                / (pl.col("eps_std") + 1e-8)
            ).alias(self.produces)
        )
