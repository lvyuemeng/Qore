from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qore_factor.base import FundamentalFactor


@dataclass(slots=True)
class GrossMarginFactor(FundamentalFactor):
    name: str = "gross_margin"
    produces: str = "gross_margin"
    requires: frozenset[str] = frozenset({"revenue", "gross_margin"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("gross_margin").cast(pl.Float64).alias(self.produces)
        )


@dataclass(slots=True)
class AssetTurnoverFactor(FundamentalFactor):
    name: str = "asset_turnover"
    produces: str = "asset_turnover"
    requires: frozenset[str] = frozenset({"revenue", "total_assets"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("revenue") / (pl.col("total_assets") + 1e-8)).alias(self.produces)
        )


@dataclass(slots=True)
class CFOYieldFactor(FundamentalFactor):
    name: str = "cfo_yield"
    produces: str = "cfo_yield"
    requires: frozenset[str] = frozenset({"cfo", "total_assets"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("cfo") / (pl.col("total_assets") + 1e-8)).alias(self.produces)
        )


@dataclass(slots=True)
class AccrualRatioFactor(FundamentalFactor):
    name: str = "accrual_ratio"
    produces: str = "accrual_ratio"
    requires: frozenset[str] = frozenset({"net_income", "cfo", "total_assets"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                (pl.col("net_income") - pl.col("cfo")) / (pl.col("total_assets") + 1e-8)
            ).alias(self.produces)
        )


@dataclass(slots=True)
class DebtToAssetRatioFactor(FundamentalFactor):
    name: str = "debt_to_asset_ratio"
    produces: str = "debt_to_asset_ratio"
    requires: frozenset[str] = frozenset({"total_liabilities", "total_assets"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.when(pl.col("total_assets").abs() > 1e-8)
            .then(
                pl.col("total_liabilities").cast(pl.Float64)
                / pl.col("total_assets").cast(pl.Float64)
            )
            .otherwise(None)
            .alias(self.produces)
        )


@dataclass(slots=True)
class ROEStabilityFactor(FundamentalFactor):
    window: int = 8
    name: str = "roe_stability"
    produces: str = "roe_stability"
    requires: frozenset[str] = frozenset({"symbol", "roe"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                1.0 / (pl.col("roe").rolling_std(self.window).over("symbol") + 1e-8)
            ).alias(self.produces)
        )
