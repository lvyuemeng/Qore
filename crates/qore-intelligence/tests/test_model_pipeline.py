from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from qore_core.config import QoreConfig
from qore_intelligence.model.normalizer import CrossSectionalZScore, RankScaler
from qore_intelligence.model.pipeline import ModelPipeline


def test_rank_scaler_outputs_percentiles() -> None:
    scaler = RankScaler()
    x = np.array([[3.0, 10.0], [1.0, 30.0], [2.0, 20.0]])
    scaler.fit(x)
    transformed = scaler.transform(x)
    assert transformed.shape == x.shape
    assert float(transformed.min()) >= 0.0
    assert float(transformed.max()) <= 1.0


def test_cross_sectional_zscore_normalizes_by_group() -> None:
    transformer = CrossSectionalZScore()
    y = np.array([1.0, 2.0, 10.0, 12.0])
    groups = np.array([2, 2])
    transformed = transformer.fit_transform(y, groups)
    assert np.isclose(transformed[:2].mean(), 0.0)
    assert np.isclose(transformed[2:].mean(), 0.0)


def test_model_pipeline_load_path_derived_from_config(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {"intelligence": {"model_store_root": str(tmp_path)}}
    )
    pipeline = ModelPipeline.from_config("stock_ranker", config)
    pipeline.trained_on = pipeline.trained_on or __import__("datetime").date.today()
    saved_path = pipeline.save("unit-test")
    loaded = ModelPipeline.load("stock_ranker", config, version="unit-test")
    assert (
        saved_path == Path(tmp_path) / "stock_ranker" / "unit-test" / "pipeline.joblib"
    )
    assert loaded.name == "stock_ranker"


def test_model_pipeline_predict_score_returns_series() -> None:
    config = QoreConfig()
    pipeline = ModelPipeline.from_config("stock_ranker", config)
    x = np.array([[1.0, 3.0], [2.0, 4.0]])
    pipeline.x_normalizer.fit(x)
    result = pipeline.predict_score(
        pl.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}).lazy()
    )
    assert result.name == "score"
    assert len(result) == 2
