from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class XNormalizer(Protocol):
    def fit(self, x: np.ndarray) -> None: ...

    def transform(self, x: np.ndarray) -> np.ndarray: ...


class YTransformer(Protocol):
    def fit_transform(self, y: np.ndarray, groups: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class RankScaler:
    _sorted_columns: list[np.ndarray] = field(default_factory=list)

    def fit(self, x: np.ndarray) -> None:
        values = _ensure_2d_float(x)
        self._sorted_columns = [
            np.sort(values[:, idx]) for idx in range(values.shape[1])
        ]

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = _ensure_2d_float(x)
        if not self._sorted_columns:
            msg = "RankScaler must be fit before transform()."
            raise ValueError(msg)
        transformed = np.zeros_like(values, dtype=float)
        for idx, sorted_values in enumerate(self._sorted_columns):
            right = np.searchsorted(sorted_values, values[:, idx], side="right")
            left = np.searchsorted(sorted_values, values[:, idx], side="left")
            transformed[:, idx] = (left + right) / 2.0 / len(sorted_values)
        return np.clip(transformed, 0.0, 1.0)


@dataclass(slots=True)
class RobustScaler:
    _median: np.ndarray | None = None
    _iqr: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        values = _ensure_2d_float(x)
        self._median = np.median(values, axis=0)
        q1 = np.percentile(values, 25, axis=0)
        q3 = np.percentile(values, 75, axis=0)
        self._iqr = np.where((q3 - q1) == 0.0, 1.0, q3 - q1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = _ensure_2d_float(x)
        if self._median is None or self._iqr is None:
            msg = "RobustScaler must be fit before transform()."
            raise ValueError(msg)
        return (values - self._median) / self._iqr


@dataclass(slots=True)
class CrossSectionalZScore:
    def fit_transform(self, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).reshape(-1)
        result = values.copy()
        for indexer in _group_indexers(groups, len(values)):
            group_values = values[indexer]
            mean = float(group_values.mean())
            std = float(group_values.std())
            scale = std if std > 1e-12 else 1.0
            result[indexer] = (group_values - mean) / scale
        return result


def _ensure_2d_float(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim != 2:
        msg = f"Expected 1D or 2D array, got shape {values.shape}"
        raise ValueError(msg)
    return values


def _group_indexers(groups: np.ndarray, size: int) -> list[np.ndarray]:
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        msg = "groups must be a 1D array"
        raise ValueError(msg)
    if group_array.size == size:
        unique = np.unique(group_array)
        return [np.where(group_array == value)[0] for value in unique]
    counts = np.asarray(group_array, dtype=int)
    if counts.sum() != size:
        msg = "group counts must sum to the target length"
        raise ValueError(msg)
    indexers: list[np.ndarray] = []
    start = 0
    for count in counts:
        stop = start + int(count)
        indexers.append(np.arange(start, stop))
        start = stop
    return indexers
