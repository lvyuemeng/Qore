from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass(slots=True)
class MultiHorizonRanker:
    horizons: list[int] = field(default_factory=lambda: [20, 60, 252])
    weights: dict[str, float] = field(default_factory=dict)
    model_params: dict[str, object] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    _coefs: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.horizons:
            msg = "MultiHorizonRanker requires at least one horizon."
            raise ValueError(msg)
        if not self.weights:
            equal_weight = 1.0 / len(self.horizons)
            self.weights = {f"{horizon}d": equal_weight for horizon in self.horizons}

    def fit(
        self,
        x: np.ndarray,
        targets: dict[int, np.ndarray],
        feature_columns: list[str],
    ) -> None:
        values = np.asarray(x, dtype=float)
        self.feature_columns = list(feature_columns)
        self._coefs = {}
        for horizon in self.horizons:
            y = np.asarray(targets.get(horizon), dtype=float).reshape(-1)
            if y.size != values.shape[0]:
                msg = f"Target length mismatch for horizon {horizon}"
                raise ValueError(msg)
            centered_x = values - values.mean(axis=0)
            centered_y = y - y.mean()
            denom = np.square(centered_x).sum(axis=0)
            denom = np.where(denom <= 1e-12, 1.0, denom)
            coef = (centered_x * centered_y[:, None]).sum(axis=0) / denom
            self._coefs[horizon] = coef.astype(float)

    def predict_score(self, x: pl.DataFrame) -> pl.Series:
        if x.is_empty():
            return pl.Series(name="score", values=[], dtype=pl.Float64)
        feature_frame = x.select(pl.all().exclude("symbol", "date"))
        values = feature_frame.to_numpy().astype(float, copy=False)
        if values.shape[1] == 0:
            return pl.Series(
                name="score", values=np.zeros(values.shape[0], dtype=float)
            )
        if not self._coefs:
            return pl.Series(name="score", values=values.mean(axis=1))

        score = np.zeros(values.shape[0], dtype=float)
        for horizon in self.horizons:
            coef = self._coefs.get(horizon)
            if coef is None:
                continue
            weight = float(self.weights.get(f"{horizon}d", 0.0))
            score += weight * (values @ coef)
        return pl.Series(name="score", values=score)
