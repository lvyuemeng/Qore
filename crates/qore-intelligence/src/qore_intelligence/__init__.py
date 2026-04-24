# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntelligenceSettings:
    model_store_root: str = "models"
    news_llm_daily_budget: int = 50
    news_llm_model: str = "claude-sonnet-4-20250514"
    news_finbert_model: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    news_score_half_life_days: int = 5


from qore_intelligence.combine import SignalCombiner
from qore_intelligence.model.artifact import (
    FeatureSchema,
    ModelArtifactManifest,
    ModelPayload,
    RankerSpec,
    TrainedModelArtifact,
    TrainingMetadata,
)
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.normalizer import (
    CrossSectionalZScore,
    RankScaler,
    RobustScaler,
)
from qore_intelligence.model.pipeline import ModelPipeline
from qore_intelligence.model.registry import ModelRegistry
from qore_intelligence.model.workflow import (
    TrainingRun,
    fit_and_save_model,
    fit_and_save_model_from_store,
    training_frame_from_store,
)
from qore_intelligence.signal.llm import EventExtraction, LLMExtractor
from qore_intelligence.signal.score import NewsPipeline
from qore_intelligence.signal.sentiment import FinBERT
from qore_intelligence.signal.triage import Triage
from qore_intelligence.strategy import (
    ModelPipelineScoreProvider,
    build_ranking_strategy,
)

__all__ = [
    "CrossSectionalZScore",
    "EventExtraction",
    "FeatureSchema",
    "FinBERT",
    "IntelligenceSettings",
    "LLMExtractor",
    "ModelArtifactManifest",
    "ModelPayload",
    "ModelPipeline",
    "ModelPipelineScoreProvider",
    "ModelRegistry",
    "MultiHorizonRanker",
    "NewsPipeline",
    "RankScaler",
    "RankerSpec",
    "RobustScaler",
    "SignalCombiner",
    "TrainedModelArtifact",
    "TrainingMetadata",
    "TrainingRun",
    "Triage",
    "build_ranking_strategy",
    "fit_and_save_model",
    "fit_and_save_model_from_store",
    "training_frame_from_store",
]
