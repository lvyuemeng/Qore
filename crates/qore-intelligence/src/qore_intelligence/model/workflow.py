from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from qore_data.store.duckdb import QoreStore

from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.artifact import TrainedModelArtifact
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.normalizer import (
    CrossSectionalZScore,
    RankScaler,
    XNormalizer,
    YTransformer,
)
from qore_intelligence.model.pipeline import ModelPipeline
from qore_intelligence.model.registry import ModelRegistry


@dataclass(slots=True)
class TrainingRun:
    artifact: TrainedModelArtifact
    artifact_path: Path


def fit_and_save_model(
    *,
    intelligence_settings: IntelligenceSettings,
    model_name: str,
    factor_lf: pl.LazyFrame,
    store: QoreStore,
    version: str | None = None,
    model: MultiHorizonRanker | None = None,
    x_normalizer: XNormalizer | None = None,
    y_transformer: YTransformer | None = None,
) -> TrainingRun:
    pipeline = ModelPipeline(
        x_normalizer=x_normalizer or RankScaler(),
        y_transformer=y_transformer or CrossSectionalZScore(),
        model=model or MultiHorizonRanker(),
    )
    artifact = pipeline.fit(
        factor_lf,
        store,
        model_name=model_name,
    )
    artifact_path = ModelRegistry.from_settings(intelligence_settings).save(
        artifact,
        version,
    )
    return TrainingRun(artifact=artifact, artifact_path=artifact_path)


def training_frame_from_store(
    *,
    store: QoreStore,
    factor_names: list[str],
    forward_returns: pl.LazyFrame,
    score_column: str = "z_score",
    start: date | None = None,
    end: date | None = None,
) -> pl.LazyFrame:
    factor_scores = store.read("factor_scores")
    if start is not None:
        factor_scores = factor_scores.filter(pl.col("date") >= start)
    if end is not None:
        factor_scores = factor_scores.filter(pl.col("date") <= end)
    factor_scores = factor_scores.filter(pl.col("factor_name").is_in(factor_names))
    pivoted = pl.DataFrame(
        factor_scores.select("date", "symbol", "factor_name", score_column).collect()
    )
    pivoted = pivoted.pivot(
        on="factor_name", index=["date", "symbol"], values=score_column
    ).lazy()
    return pivoted.join(forward_returns, on=["date", "symbol"], how="inner")


def fit_and_save_model_from_store(
    *,
    intelligence_settings: IntelligenceSettings,
    model_name: str,
    store: QoreStore,
    factor_names: list[str],
    forward_returns: pl.LazyFrame,
    version: str | None = None,
    model: MultiHorizonRanker | None = None,
    x_normalizer: XNormalizer | None = None,
    y_transformer: YTransformer | None = None,
    score_column: str = "z_score",
    start: date | None = None,
    end: date | None = None,
) -> TrainingRun:
    factor_lf = training_frame_from_store(
        store=store,
        factor_names=factor_names,
        forward_returns=forward_returns,
        score_column=score_column,
        start=start,
        end=end,
    )
    return fit_and_save_model(
        intelligence_settings=intelligence_settings,
        model_name=model_name,
        factor_lf=factor_lf,
        store=store,
        version=version,
        model=model,
        x_normalizer=x_normalizer,
        y_transformer=y_transformer,
    )
