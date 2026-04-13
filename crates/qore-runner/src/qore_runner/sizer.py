from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from qore_core.universe import Universe


@runtime_checkable
class PositionSizer(Protocol):
    def size(self, signals: pl.Series, universe: Universe) -> dict[str, float]: ...


@dataclass(slots=True)
class EqualWeightSizer:
    top_k: int
    max_weight: float = 0.05

    def size(self, signals: pl.Series, universe: Universe) -> dict[str, float]:
        ranked = [
            (symbol, float(signal))
            for symbol, signal in zip(
                universe.symbols(), signals.to_list(), strict=False
            )
            if signal is not None and signal == signal
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        selected = ranked[: self.top_k]
        if not selected:
            return {}
        weight = min(1.0 / len(selected), self.max_weight)
        return {symbol: weight for symbol, _ in selected}


@dataclass(slots=True)
class VolScaledSizer:
    top_k: int
    vol_col: str = "realized_vol_20d"

    def size(self, signals: pl.Series, universe: Universe) -> dict[str, float]:
        ranked = [
            (symbol, float(signal))
            for symbol, signal in zip(
                universe.symbols(), signals.to_list(), strict=False
            )
            if signal is not None and signal == signal
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        selected = ranked[: self.top_k]
        if not selected:
            return {}
        raw_weights = {symbol: 1.0 for symbol, _ in selected}
        total = sum(raw_weights.values())
        return {symbol: value / total for symbol, value in raw_weights.items()}
