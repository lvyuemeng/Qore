from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.artifact import (
    FeatureSchema,
    ModelArtifactManifest,
    ModelPayload,
    RankerSpec,
    TrainedModelArtifact,
)
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.normalizer import CrossSectionalZScore, RankScaler
from qore_intelligence.model.pipeline import ModelPipeline
from qore_intelligence.model.registry import ModelRegistry
from qore_intelligence.model.validation import PurgedKFold, PurgedTimeSplit
from qore_intelligence.model.workflow import (
    fit_and_save_model_from_store,
    training_frame_from_store,
)


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


def test_model_registry_path_derived_from_settings(tmp_path: Path) -> None:
    settings = IntelligenceSettings(
        model_store_root=str(tmp_path),
    )
    registry = ModelRegistry.from_settings(settings)
    artifact = TrainedModelArtifact(
        manifest=ModelArtifactManifest(
            model_name="stock_ranker",
            feature_schema=FeatureSchema(
                factor_columns=["factor_a"],
                target_columns=[
                    "forward_return_20d",
                    "forward_return_60d",
                    "forward_return_252d",
                ],
            ),
            ranker_spec=RankerSpec(
                model_family="multi_horizon_ranker",
                horizons=[20, 60, 252],
                ensemble_weights={"20d": 1 / 3, "60d": 1 / 3, "252d": 1 / 3},
            ),
        ),
        payload=ModelPayload(
            x_normalizer=RankScaler(),
            y_transformer=CrossSectionalZScore(),
            model=MultiHorizonRanker(horizons=[20, 60, 252]),
        ),
    )
    saved_path = registry.save(artifact, "unit-test")
    loaded = registry.load("stock_ranker", version="unit-test")
    assert (
        saved_path == Path(tmp_path) / "stock_ranker" / "unit-test" / "artifact.joblib"
    )
    assert loaded.manifest.model_name == "stock_ranker"


def test_model_pipeline_predict_score_returns_series() -> None:
    pipeline = ModelPipeline.from_settings()
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
    train_max = train_df.get_column("date").max()
    test_min = test_df.get_column("date").min()
    assert isinstance(train_max, date)
    assert isinstance(test_min, date)
    assert train_max < date(2026, 1, 4)
    assert test_min >= date(2026, 1, 7)


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


def test_model_pipeline_fit_returns_artifact(tmp_path: Path) -> None:
    data_settings = DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )
    pipeline = ModelPipeline(
        x_normalizer=RankScaler(),
        y_transformer=CrossSectionalZScore(),
        model=MultiHorizonRanker(
            horizons=[1, 2],
            weights={"1d": 0.5, "2d": 0.5},
        ),
    )
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

    artifact = pipeline.fit(
        factor_lf,
        QoreStore.from_settings(data_settings),
        model_name="stock_ranker",
    )

    assert artifact.manifest.model_name == "stock_ranker"
    assert set(artifact.manifest.training_metadata.validation_metrics) == {"1d", "2d"}
    restored = ModelPipeline.from_trained_artifact(artifact)
    assert restored.model.horizons == [1, 2]


def test_training_frame_from_store_pivots_factor_scores(tmp_path: Path) -> None:
    data_settings = DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )
    store = QoreStore.from_settings(data_settings)
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [
                    date(2026, 1, 1),
                    date(2026, 1, 1),
                    date(2026, 1, 1),
                    date(2026, 1, 1),
                ],
                "symbol": ["AAA", "AAA", "BBB", "BBB"],
                "factor_name": ["factor_a", "factor_b", "factor_a", "factor_b"],
                "raw_value": [0.1, 0.3, 0.2, 0.4],
                "z_score": [0.1, 0.3, 0.2, 0.4],
                "rank_pct": [0.5, 0.8, 0.6, 0.9],
            }
        ),
    )
    frame = pl.DataFrame(
        training_frame_from_store(
            store=store,
            factor_names=["factor_a", "factor_b"],
            forward_returns=pl.DataFrame(
                {
                    "date": [date(2026, 1, 1), date(2026, 1, 1)],
                    "symbol": ["AAA", "BBB"],
                    "forward_return_1d": [0.01, 0.02],
                }
            ).lazy(),
        ).collect()
    )

    assert set(frame.columns) == {
        "date",
        "symbol",
        "factor_a",
        "factor_b",
        "forward_return_1d",
    }
    assert frame.height == 2


def test_fit_and_save_model_from_store_returns_saved_artifact(tmp_path: Path) -> None:
    data_settings = DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )
    intelligence_settings = IntelligenceSettings(
        model_store_root=str(tmp_path / "models"),
    )
    store = QoreStore.from_settings(data_settings)
    store.write(
        "factor_scores",
        pl.DataFrame(
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
                ],
                "symbol": ["AAA", "BBB"] * 4,
                "factor_name": ["factor_a"] * 8,
                "raw_value": [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.4, 0.2],
                "z_score": [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.4, 0.2],
                "rank_pct": [0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            }
        ),
    )
    run = fit_and_save_model_from_store(
        intelligence_settings=intelligence_settings,
        model_name="stock_ranker",
        store=store,
        factor_names=["factor_a"],
        forward_returns=pl.DataFrame(
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
                ],
                "symbol": ["AAA", "BBB"] * 4,
                "forward_return_1d": [0.01, 0.02, 0.02, 0.01, 0.03, 0.01, 0.04, 0.02],
            }
        ).lazy(),
        version="store-train",
        model=MultiHorizonRanker(horizons=[1], weights={"1d": 1.0}),
    )

    assert run.artifact.manifest.model_name == "stock_ranker"
    assert run.artifact.manifest.feature_schema.factor_columns == ["factor_a"]
    assert run.artifact.manifest.ranker_spec.horizons == [1]
    assert (
        run.artifact_path
        == tmp_path / "models" / "stock_ranker" / "store-train" / "artifact.joblib"
    )
