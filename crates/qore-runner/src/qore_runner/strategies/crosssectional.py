from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl
from qore_data.universe import TradingSession

from qore_runner.strategy import (
    StrategyContext,
    align_signals_to_universe,
    tradeable_universe_frame,
)


@dataclass(slots=True)
class CrossSectionalScreener:
    factor_weights: dict[str, float]
    name: str = "cross_sectional_screener"
    compatible_sessions: frozenset[TradingSession] = frozenset({"nav", "auction"})
    signal_freq: Literal["event", "daily", "weekly", "monthly"] = "monthly"

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset(self.factor_weights)

    def generate(self, context: StrategyContext) -> pl.LazyFrame:
        universe_frame = tradeable_universe_frame(
            context.universe,
            context.date,
            context.calendar,
            self.compatible_sessions,
        )
        if not self.factor_weights:
            return align_signals_to_universe(
                pl.DataFrame(schema={"symbol": pl.String, "signal": pl.Float64}).lazy(),
                universe_frame,
            )
        scored = context.factor_lf.select(
            "symbol",
            pl.sum_horizontal(
                *(
                    pl.col(column) * weight
                    for column, weight in self.factor_weights.items()
                )
            ).alias("signal"),
        )
        return align_signals_to_universe(scored, universe_frame)
