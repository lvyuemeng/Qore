from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import polars as pl
from qore_core.instrument import TradingSession

from qore_runner.strategy import (
    StrategyContext,
    StrategyResult,
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
        overlays: Mapping[str, float],
    ) -> pl.LazyFrame: ...


@dataclass(slots=True)
class WeightedOverlayCombiner:
    alpha: float = 0.0

    def combine(
        self,
        scores: pl.LazyFrame,
        overlays: Mapping[str, float],
    ) -> pl.LazyFrame:
        if self.alpha == 0.0 or not overlays:
            return scores.select(
                "symbol", pl.col("signal").cast(pl.Float64, strict=False)
            )
        overlay_frame = pl.DataFrame(
            {
                "symbol": list(overlays),
                "overlay": list(overlays.values()),
            },
            schema={"symbol": pl.String, "overlay": pl.Float64},
        )
        return (
            scores.join(overlay_frame.lazy(), on="symbol", how="left")
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
    signal_freq: Literal["event", "daily", "weekly", "monthly"] = "weekly"
    overlay_key: str = "signal_overlays"

    @property
    def required_columns(self) -> frozenset[str]:
        return self.score_provider.required_columns

    def generate(self, context: StrategyContext) -> StrategyResult:
        overlays = _signal_overlays(context.inputs, self.overlay_key)
        universe_frame = tradeable_universe_frame(
            context.universe,
            context.date,
            context.calendar,
            self.compatible_sessions,
        )
        scores = self.score_provider.predict_scores(context.factor_lf)
        combined = self.combiner.combine(scores, overlays) if self.combiner else scores
        return StrategyResult(align_signals_to_universe(combined, universe_frame))


def _signal_overlays(
    inputs: Mapping[str, object],
    overlay_key: str,
) -> Mapping[str, float]:
    value = inputs.get(overlay_key, inputs.get("score_overlays"))
    if not isinstance(value, Mapping):
        return {}
    overlays: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        if not isinstance(raw, int | float | str):
            continue
        try:
            overlays[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return overlays
