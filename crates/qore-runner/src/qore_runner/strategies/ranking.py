from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import polars as pl
from qore_data.universe import TradingSession

from qore_runner.schedule import RebalanceSchedule
from qore_runner.strategy import (
    StrategyContext,
    align_signals_to_universe,
    tradeable_universe_frame,
)


class ScoreProvider(Protocol):
    required_columns: frozenset[str]

    def predict_scores(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame: ...


class ScoreCombiner(Protocol):
    def combine(
        self,
        scores: pl.LazyFrame,
        overlay_frame: pl.DataFrame | pl.LazyFrame | None,
    ) -> pl.LazyFrame: ...


@dataclass(slots=True)
class WeightedOverlayCombiner:
    alpha: float = 0.0

    def combine(
        self,
        scores: pl.LazyFrame,
        overlay_frame: pl.DataFrame | pl.LazyFrame | None,
    ) -> pl.LazyFrame:
        if self.alpha == 0.0 or overlay_frame is None:
            return scores.select(
                "symbol", pl.col("signal").cast(pl.Float64, strict=False)
            )
        overlay = (
            pl.DataFrame(overlay_frame.collect())
            if isinstance(overlay_frame, pl.LazyFrame)
            else overlay_frame
        )
        if overlay.is_empty() or "symbol" not in overlay.columns:
            return scores.select(
                "symbol", pl.col("signal").cast(pl.Float64, strict=False)
            )
        value_col = "overlay" if "overlay" in overlay.columns else "score"
        normalized = pl.DataFrame(
            overlay.lazy()
            .with_columns(
                pl.col("symbol").cast(pl.String, strict=False),
                pl.col(value_col).cast(pl.Float64, strict=False).alias("overlay"),
            )
            .filter(pl.col("symbol").is_not_null())
            .select("symbol", "overlay")
            .unique(subset=["symbol"], keep="last")
            .collect()
        )
        return (
            scores.join(normalized.lazy(), on="symbol", how="left")
            .with_columns(
                (
                    (1.0 - self.alpha) * pl.col("signal").cast(pl.Float64, strict=False)
                    + self.alpha * pl.col("overlay").fill_null(0.0)
                ).alias("signal")
            )
            .select("symbol", "signal")
        )


@dataclass(slots=True)
class RankingStrategy:
    score_provider: ScoreProvider
    combiner: ScoreCombiner | None = None
    name: str = "ranking"
    compatible_sessions: frozenset[TradingSession] = frozenset(
        {"auction", "nav", "continuous"}
    )
    rebalance_schedule: RebalanceSchedule = field(
        default_factory=lambda: RebalanceSchedule(frequency="weekly")
    )

    @property
    def signal_freq(self) -> Literal["event", "daily", "weekly", "monthly"]:
        return self.rebalance_schedule.frequency

    def strategy_rebalance_schedule(self) -> RebalanceSchedule:
        return self.rebalance_schedule

    @property
    def required_columns(self) -> frozenset[str]:
        return self.score_provider.required_columns

    def generate(self, context: StrategyContext) -> pl.LazyFrame:
        universe_frame = tradeable_universe_frame(
            context.universe,
            context.date,
            context.calendar,
            self.compatible_sessions,
        )
        scores = self.score_provider.predict_scores(context.factor_lf)
        combined = (
            self.combiner.combine(scores, context.providers.signal_overlay)
            if self.combiner
            else scores
        )
        return align_signals_to_universe(combined, universe_frame)
