from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl
import polars.selectors as cs
from qore_core.config import QoreConfig
from qore_data.store.duckdb import QoreStore

from qore_intelligence.model.artifact import (
    FeatureSchema,
    ModelArtifact,
    ModelPayload,
    RankerSpec,
    TrainingMetadata,
)
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.normalizer import (
    CrossSectionalZScore,
    RankScaler,
    XNormalizer,
    YTransformer,
)
from qore_intelligence.model.validation import PurgedKFold

_RESERVED_COLUMNS = {"symbol", "date"}


@dataclass
class ModelPipeline:
    x_normalizer: XNormalizer
    y_transformer: YTransformer
    model: MultiHorizonRanker

    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        *,
        x_normalizer: XNormalizer | None = None,
        y_transformer: YTransformer | None = None,
    ) -> ModelPipeline:
        del config
        return cls(
            x_normalizer=x_normalizer or RankScaler(),
            y_transformer=y_transformer or CrossSectionalZScore(),
            model=MultiHorizonRanker(),
        )

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> ModelPipeline:
        payload = artifact.payload
        if payload is None:
            msg = "ModelArtifact payload is missing."
            raise ValueError(msg)
        return cls(
            x_normalizer=payload.x_normalizer,
            y_transformer=payload.y_transformer,
            model=payload.model,
        )

    def fit(
        self,
        factor_lf: pl.LazyFrame,
        store: QoreStore,
        *,
        model_name: str,
    ) -> ModelArtifact:
        del store
        frame = factor_lf.collect()
        if frame.is_empty():
            msg = "Cannot fit ModelPipeline on an empty factor frame."
            raise ValueError(msg)
        feature_columns = _feature_columns(frame)
        x = frame.select(feature_columns).to_numpy().astype(float, copy=False)
        self.x_normalizer.fit(x)
        transformed_x = self.x_normalizer.transform(x)
        group_labels = (
            frame.get_column("date").to_numpy()
            if "date" in frame.columns
            else np.zeros(len(frame))
        )
        targets = self._extract_targets(frame, group_labels)
        self.model.fit(transformed_x, targets, feature_columns)
        validation_metrics = self._compute_validation_ic(frame, feature_columns)
        training_window = _training_window(frame)
        return ModelArtifact(
            model_name=model_name,
            feature_schema=FeatureSchema(
                factor_columns=feature_columns,
                target_columns=[
                    f"forward_return_{horizon}d" for horizon in self.model.horizons
                ],
            ),
            ranker_spec=RankerSpec(
                model_family="multi_horizon_ranker",
                horizons=list(self.model.horizons),
                ensemble_weights=dict(self.model.weights),
            ),
            training_metadata=TrainingMetadata(
                validation_metrics=validation_metrics,
                training_window=training_window,
                trained_at=datetime.now(UTC),
            ),
            payload=ModelPayload(
                x_normalizer=self.x_normalizer,
                y_transformer=self.y_transformer,
                model=self.model,
            ),
        )

    def predict_score(self, factor_lf: pl.LazyFrame) -> pl.Series:
        frame = factor_lf.collect()
        if frame.is_empty():
            return pl.Series(name="score", values=[], dtype=pl.Float64)
        feature_columns = _feature_columns(frame)
        x = frame.select(feature_columns).to_numpy().astype(float, copy=False)
        transformed_x = self.x_normalizer.transform(x)
        transformed = pl.DataFrame(transformed_x, schema=feature_columns)
        return self.model.predict_score(transformed)

    def _extract_targets(
        self,
        frame: pl.DataFrame,
        group_labels: np.ndarray,
    ) -> dict[int, np.ndarray]:
        targets: dict[int, np.ndarray] = {}
        counts = _group_counts(group_labels)
        for horizon in self.model.horizons:
            column = f"forward_return_{horizon}d"
            if column not in frame.columns:
                msg = f"Missing target column {column!r} for model fitting."
                raise ValueError(msg)
            y = frame.get_column(column).to_numpy().astype(float, copy=False)
            targets[horizon] = self.y_transformer.fit_transform(y, counts)
        return targets

    def _compute_validation_ic(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, float]:
        unique_dates = (
            frame.get_column("date").n_unique() if "date" in frame.columns else 0
        )
        if unique_dates < 4:
            return {f"{horizon}d": 0.0 for horizon in self.model.horizons}

        splitter = PurgedKFold(
            n_splits=min(5, max(2, unique_dates // 2)),
            horizon_days=min(self.model.horizons),
            embargo_days=1,
        )
        fold_scores: dict[int, list[float]] = {
            horizon: [] for horizon in self.model.horizons
        }
        for train_df, test_df in splitter.split(frame):
            if train_df.is_empty() or test_df.is_empty():
                continue
            x_normalizer = _fresh_x_normalizer(self.x_normalizer)
            y_transformer = _fresh_y_transformer(self.y_transformer)
            train_x = (
                train_df.select(feature_columns).to_numpy().astype(float, copy=False)
            )
            x_normalizer.fit(train_x)
            train_x = x_normalizer.transform(train_x)
            group_counts = _group_counts(train_df.get_column("date").to_numpy())
            train_targets = {
                horizon: y_transformer.fit_transform(
                    train_df.get_column(f"forward_return_{horizon}d")
                    .to_numpy()
                    .astype(float, copy=False),
                    group_counts,
                )
                for horizon in self.model.horizons
            }
            model = MultiHorizonRanker(
                horizons=list(self.model.horizons),
                weights=dict(self.model.weights),
                model_params=dict(self.model.model_params),
            )
            model.fit(train_x, train_targets, feature_columns)

            test_x = (
                test_df.select(feature_columns).to_numpy().astype(float, copy=False)
            )
            predictions = model.predict_score(
                pl.DataFrame(x_normalizer.transform(test_x), schema=feature_columns)
            )
            for horizon in self.model.horizons:
                ic = _daily_ic_mean(
                    test_df.get_column("date"),
                    predictions,
                    test_df.get_column(f"forward_return_{horizon}d"),
                )
                if ic is not None:
                    fold_scores[horizon].append(ic)

        return {
            f"{horizon}d": float(np.mean(scores)) if scores else 0.0
            for horizon, scores in fold_scores.items()
        }


def _feature_columns(frame: pl.DataFrame) -> list[str]:
    numeric_columns = set(frame.select(cs.numeric()).columns)
    columns = [
        column
        for column in frame.columns
        if column in numeric_columns
        and column not in _RESERVED_COLUMNS
        and not column.startswith("forward_return_")
    ]
    if not columns:
        msg = "No numeric factor columns available for model pipeline."
        raise ValueError(msg)
    return columns


def _group_counts(group_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(group_labels)
    if labels.ndim != 1:
        msg = "group labels must be 1D"
        raise ValueError(msg)
    _, counts = np.unique(labels, return_counts=True)
    return counts.astype(int)


def _fresh_x_normalizer(x_normalizer: XNormalizer) -> XNormalizer:
    if isinstance(x_normalizer, RankScaler):
        return RankScaler()
    msg = f"Unsupported XNormalizer clone type: {type(x_normalizer).__name__}"
    raise TypeError(msg)


def _fresh_y_transformer(y_transformer: YTransformer) -> YTransformer:
    if isinstance(y_transformer, CrossSectionalZScore):
        return CrossSectionalZScore()
    msg = f"Unsupported YTransformer clone type: {type(y_transformer).__name__}"
    raise TypeError(msg)


def _daily_ic_mean(
    dates: pl.Series,
    predictions: pl.Series,
    forward_returns: pl.Series,
) -> float | None:
    frame = pl.DataFrame(
        {
            "date": dates,
            "prediction": predictions.cast(pl.Float64),
            "forward_return": forward_returns.cast(pl.Float64),
        }
    )
    daily = (
        frame.filter(
            pl.col("prediction").is_not_null() & pl.col("forward_return").is_not_null()
        )
        .group_by("date")
        .agg(pl.corr("prediction", "forward_return").alias("ic"))
        .filter(pl.col("ic").is_not_null())
    )
    if daily.is_empty():
        return None
    value = daily.get_column("ic").mean()
    return None if value is None else float(value)


def _training_window(frame: pl.DataFrame) -> dict[str, str] | None:
    if "date" not in frame.columns or frame.is_empty():
        return None
    dates = frame.get_column("date")
    start = dates.min()
    end = dates.max()
    if start is None or end is None:
        return None
    return {"start": start.isoformat(), "end": end.isoformat()}
