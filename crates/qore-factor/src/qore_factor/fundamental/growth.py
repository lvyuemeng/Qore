from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class ProfitGrowthPremiumFactor(FundamentalFactor):
    net_profit_growth_column: str = "net_profit_yoy"
    revenue_growth_column: str = "revenue_growth_yoy"
    name: str = "profit_growth_premium"
    produces: str = "profit_growth_premium"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset(
            {self.net_profit_growth_column, self.revenue_growth_column}
        )

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                pl.col(self.net_profit_growth_column)
                - pl.col(self.revenue_growth_column)
            ).alias(self.produces)
        )
