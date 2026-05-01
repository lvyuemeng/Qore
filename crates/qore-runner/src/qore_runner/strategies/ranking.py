from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


class ScoreProvider(Protocol):
    required_columns: frozenset[str]

    def predict_scores(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame: ...


class ScoreCombiner(Protocol):
    def combine(
        self, scores: pl.LazyFrame, overlay_frame: pl.DataFrame | None
    ) -> pl.LazyFrame: ...


@dataclass(slots=True)
class WeightedOverlayCombiner:
    alpha: float = 0.0

    def combine(
        self,
        scores: pl.LazyFrame,
        overlay_frame: pl.DataFrame | None,
    ) -> pl.LazyFrame:
        if self.alpha == 0.0 or overlay_frame is None:
            return scores.select("symbol", pl.col("signal").cast(pl.Float64))
        if overlay_frame.is_empty() or "symbol" not in overlay_frame.columns:
            return scores.select("symbol", pl.col("signal").cast(pl.Float64))
        value_col = "overlay" if "overlay" in overlay_frame.columns else "score"
        normalized = pl.DataFrame(
            overlay_frame.lazy()
            .with_columns(
                pl.col("symbol").cast(pl.String),
                pl.col(value_col).cast(pl.Float64).alias("overlay"),
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
                    (1.0 - self.alpha) * pl.col("signal").cast(pl.Float64)
                    + self.alpha * pl.col("overlay").fill_null(0.0)
                ).alias("signal"),
            )
            .select("symbol", "signal")
        )


@dataclass(slots=True)
class RankingStrategy:
    score_provider: ScoreProvider
    combiner: ScoreCombiner | None = None
    name: str = "ranking"

    @property
    def required_columns(self) -> frozenset[str]:
        return self.score_provider.required_columns

    def generate(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        scores = self.score_provider.predict_scores(factor_lf)
        if self.combiner:
            return self.combiner.combine(scores, None)
        return scores
