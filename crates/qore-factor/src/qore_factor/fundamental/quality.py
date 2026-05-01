from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class AssetTurnoverFactor(FundamentalFactor):
    revenue_column: str = "revenue"
    assets_column: str = "total_assets"
    name: str = "asset_turnover"
    produces: str = "asset_turnover"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset({self.revenue_column, self.assets_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col(self.revenue_column) / (pl.col(self.assets_column) + 1e-8)).alias(
                self.produces
            )
        )


@dataclass(slots=True)
class CFOYieldFactor(FundamentalFactor):
    cfo_column: str = "operating_cashflow"
    assets_column: str = "total_assets"
    name: str = "cfo_yield"
    produces: str = "cfo_yield"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset({self.cfo_column, self.assets_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col(self.cfo_column) / (pl.col(self.assets_column) + 1e-8)).alias(
                self.produces
            )
        )


@dataclass(slots=True)
class AccrualRatioFactor(FundamentalFactor):
    net_income_column: str = "net_income"
    cfo_column: str = "operating_cashflow"
    assets_column: str = "total_assets"
    name: str = "accrual_ratio"
    produces: str = "accrual_ratio"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset(
            {self.net_income_column, self.cfo_column, self.assets_column}
        )

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                (pl.col(self.net_income_column) - pl.col(self.cfo_column))
                / (pl.col(self.assets_column) + 1e-8)
            ).alias(self.produces)
        )


@dataclass(slots=True)
class DebtToAssetRatioFactor(FundamentalFactor):
    liabilities_column: str = "total_liabilities"
    assets_column: str = "total_assets"
    name: str = "debt_to_asset_ratio"
    produces: str = "debt_to_asset_ratio"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset({self.liabilities_column, self.assets_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.when(pl.col(self.assets_column).abs() > 1e-8)
            .then(
                pl.col(self.liabilities_column).cast(pl.Float64)
                / pl.col(self.assets_column).cast(pl.Float64)
            )
            .otherwise(None)
            .alias(self.produces)
        )


@dataclass(slots=True)
class ROEStabilityFactor(FundamentalFactor):
    roe_column: str = "roe"
    window: int = 8
    name: str = "roe_stability"
    produces: str = "roe_stability"
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.requires = frozenset({"symbol", self.roe_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                1.0
                / (
                    pl.col(self.roe_column).rolling_std(self.window).over("symbol")
                    + 1e-8
                )
            ).alias(self.produces)
        )
