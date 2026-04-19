from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.normalizer import XNormalizer, YTransformer


@dataclass(slots=True)
class ModelPayload:
    x_normalizer: XNormalizer
    y_transformer: YTransformer
    model: MultiHorizonRanker


@dataclass(slots=True)
class FeatureSchema:
    factor_columns: list[str]
    target_columns: list[str]


@dataclass(slots=True)
class RankerSpec:
    model_family: str
    horizons: list[int]
    ensemble_weights: dict[str, float]


@dataclass(slots=True)
class TrainingMetadata:
    validation_metrics: dict[str, float] = field(default_factory=dict)
    training_window: dict[str, str] | None = None
    trained_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ModelArtifact:
    model_name: str
    feature_schema: FeatureSchema
    ranker_spec: RankerSpec
    training_metadata: TrainingMetadata = field(default_factory=TrainingMetadata)
    payload: ModelPayload | None = None
