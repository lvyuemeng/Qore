from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qore_factor.base import OHLCVFactor


@dataclass(slots=True)
class AverageAmountFactor(OHLCVFactor):
    window: int = 20
    name: str = field(init=False)
    produces: str = field(init=False)
    requires: frozenset[str] = frozenset({"symbol", "amount"})

    def __post_init__(self) -> None:
        self.name = f"average_amount_{self.window}d"
        self.produces = f"avg_amount_{self.window}d"

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("amount")
            .rolling_mean(self.window)
            .over("symbol")
            .alias(self.produces)
        )


@dataclass(slots=True)
class MinimumAmountFactor(OHLCVFactor):
    window: int = 20
    name: str = field(init=False)
    produces: str = field(init=False)
    requires: frozenset[str] = frozenset({"symbol", "amount"})

    def __post_init__(self) -> None:
        self.name = f"minimum_amount_{self.window}d"
        self.produces = f"min_amount_{self.window}d"

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("amount")
            .rolling_min(self.window)
            .over("symbol")
            .alias(self.produces)
        )


@dataclass(slots=True)
class PositionToLiquidityRatioFactor(OHLCVFactor):
    liquidity_column: str = "avg_amount_20d"
    position_column: str = "target_position_cny"
    name: str = field(init=False)
    produces: str = field(init=False)
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        suffix = self.liquidity_column.removeprefix("avg_").removeprefix("min_")
        self.name = f"position_to_{suffix}_ratio"
        self.produces = f"position_to_{suffix}_ratio"
        self.requires = frozenset({self.position_column, self.liquidity_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.when(pl.col(self.liquidity_column).abs() > 1e-8)
            .then(
                pl.col(self.position_column).cast(pl.Float64)
                / pl.col(self.liquidity_column).cast(pl.Float64)
            )
            .otherwise(None)
            .alias(self.produces)
        )


@dataclass(slots=True)
class CapacityPenaltyFactor(OHLCVFactor):
    ratio_column: str = "position_to_amount_20d_ratio"
    threshold: float = 0.10
    name: str = field(init=False)
    produces: str = field(init=False)
    requires: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"capacity_penalty_{self.ratio_column}"
        self.produces = f"capacity_penalty_{self.ratio_column}"
        self.requires = frozenset({self.ratio_column})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.when(pl.col(self.ratio_column).is_null())
            .then(None)
            .when(pl.col(self.ratio_column) <= self.threshold)
            .then(1.0)
            .otherwise(self.threshold / (pl.col(self.ratio_column) + 1e-8))
            .alias(self.produces)
        )
