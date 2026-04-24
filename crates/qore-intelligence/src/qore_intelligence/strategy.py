from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from qore_runner.strategies.ranking import RankingStrategy, WeightedOverlayCombiner

from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.pipeline import ModelPipeline
from qore_intelligence.model.registry import ModelRegistry


@dataclass(slots=True)
class ModelPipelineScoreProvider:
    pipeline: ModelPipeline
    required_columns: frozenset[str]

    @classmethod
    def from_settings(
        cls,
        settings: IntelligenceSettings,
        model_name: str = "stock_ranker",
    ) -> ModelPipelineScoreProvider:
        artifact = ModelRegistry.from_settings(settings).load(model_name)
        return cls(
            pipeline=ModelPipeline.from_trained_artifact(artifact),
            required_columns=frozenset(artifact.manifest.feature_schema.factor_columns),
        )

    def predict_scores(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        frame = pl.DataFrame(factor_lf.collect())
        if frame.is_empty():
            return pl.DataFrame(
                schema={"symbol": pl.String, "signal": pl.Float64}
            ).lazy()
        scores = self.pipeline.predict_score(frame.lazy()).cast(
            pl.Float64, strict=False
        )
        return pl.DataFrame(
            {
                "symbol": frame.get_column("symbol"),
                "signal": scores,
            }
        ).lazy()


def build_ranking_strategy(
    settings: IntelligenceSettings,
    *,
    model_name: str = "stock_ranker",
    overlay_alpha: float = 0.0,
) -> RankingStrategy:
    return RankingStrategy(
        score_provider=ModelPipelineScoreProvider.from_settings(
            settings,
            model_name=model_name,
        ),
        combiner=WeightedOverlayCombiner(alpha=overlay_alpha),
    )
