from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class RevenueGrowthFactor(FundamentalFactor):
    name: str = "revenue_growth_yoy"
    produces: str = "revenue_growth_yoy"
    requires: frozenset[str] = frozenset({"revenue_growth_yoy"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("revenue_growth_yoy").cast(pl.Float64).alias(self.produces)
        )


@dataclass(slots=True)
class NetProfitGrowthFactor(FundamentalFactor):
    name: str = "net_profit_growth_yoy"
    produces: str = "net_profit_growth_yoy"
    requires: frozenset[str] = frozenset({"net_profit_growth_yoy"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("net_profit_growth_yoy").cast(pl.Float64).alias(self.produces)
        )


@dataclass(slots=True)
class ProfitGrowthPremiumFactor(FundamentalFactor):
    name: str = "profit_growth_premium"
    produces: str = "profit_growth_premium"
    requires: frozenset[str] = frozenset(
        {"net_profit_growth_yoy", "revenue_growth_yoy"}
    )

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("net_profit_growth_yoy") - pl.col("revenue_growth_yoy")).alias(
                self.produces
            )
        )
