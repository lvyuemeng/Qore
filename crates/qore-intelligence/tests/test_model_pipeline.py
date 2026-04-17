from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from qore_core.config import QoreConfig
from qore_data.store.duckdb import QoreStore
from qore_intelligence.model.normalizer import CrossSectionalZScore, RankScaler
from qore_intelligence.model.pipeline import ModelPipeline
from qore_intelligence.model.validation import PurgedKFold, PurgedTimeSplit


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


def test_purged_time_split_separates_train_and_test() -> None:
    frame = pl.DataFrame(
        {
            "date": pl.date_range(date(2026, 1, 1), date(2026, 1, 10), eager=True),
            "value": list(range(10)),
        }
    )
    train_df, test_df = PurgedTimeSplit(horizon_days=2, embargo_days=1).split(
        frame,
        split_date=date(2026, 1, 6),
    )
    assert train_df.get_column("date").max() < date(2026, 1, 4)
    assert test_df.get_column("date").min() >= date(2026, 1, 7)


def test_purged_kfold_yields_non_empty_splits() -> None:
    frame = pl.DataFrame(
        {
            "date": pl.date_range(date(2026, 1, 1), date(2026, 1, 15), eager=True),
            "value": list(range(15)),
        }
    )
    splits = list(PurgedKFold(n_splits=3, horizon_days=1, embargo_days=1).split(frame))
    assert len(splits) > 0
    assert all(
        not train_df.is_empty() and not test_df.is_empty()
        for train_df, test_df in splits
    )


def test_model_pipeline_fit_records_validation_ic(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            },
            "intelligence": {
                "horizons": [1, 2],
                "ensemble_weights": {"1d": 0.5, "2d": 0.5},
            },
        }
    )
    pipeline = ModelPipeline.from_config("stock_ranker", config)
    factor_lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 3),
                date(2026, 1, 4),
                date(2026, 1, 4),
                date(2026, 1, 5),
                date(2026, 1, 5),
                date(2026, 1, 6),
                date(2026, 1, 6),
            ],
            "symbol": ["AAA", "BBB"] * 6,
            "factor_a": [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.4, 0.2, 0.5, 0.2, 0.6, 0.3],
            "forward_return_1d": [
                0.01,
                0.02,
                0.02,
                0.01,
                0.03,
                0.01,
                0.04,
                0.02,
                0.05,
                0.02,
                0.06,
                0.03,
            ],
            "forward_return_2d": [
                0.02,
                0.03,
                0.03,
                0.02,
                0.04,
                0.02,
                0.05,
                0.03,
                0.06,
                0.03,
                0.07,
                0.04,
            ],
        }
    ).lazy()

    pipeline.fit(factor_lf, QoreStore.from_config(config))

    assert set(pipeline.validation_ic) == {"1d", "2d"}
