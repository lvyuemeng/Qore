from __future__ import annotations

from dataclasses import dataclass, field
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
    max_weight: float = 0.10
    volatility: dict[str, float] = field(default_factory=dict)

    def with_volatility(self, volatility: dict[str, float]) -> VolScaledSizer:
        return VolScaledSizer(
            top_k=self.top_k,
            vol_col=self.vol_col,
            max_weight=self.max_weight,
            volatility=dict(volatility),
        )

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

        inverse_vol = {
            symbol: self._inverse_volatility(symbol) for symbol, _ in selected
        }
        total = sum(inverse_vol.values())
        normalized = {
            symbol: value / total
            for symbol, value in inverse_vol.items()
            if total > 0.0
        }
        return _cap_and_renormalize(normalized, self.max_weight)

    def _inverse_volatility(self, symbol: str) -> float:
        volatility = float(self.volatility.get(symbol, 1.0))
        if volatility <= 0.0:
            return 1.0
        return 1.0 / volatility


def _cap_and_renormalize(
    weights: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    if not weights:
        return {}
    capped = dict(weights)
    for _ in range(len(capped) + 1):
        overweight = {
            symbol for symbol, weight in capped.items() if weight > max_weight
        }
        if not overweight:
            break
        fixed_weight = sum(max_weight for _ in overweight)
        residual_symbols = [symbol for symbol in capped if symbol not in overweight]
        residual_budget = max(1.0 - fixed_weight, 0.0)
        residual_total = sum(capped[symbol] for symbol in residual_symbols)
        for symbol in overweight:
            capped[symbol] = max_weight
        if not residual_symbols or residual_total <= 0.0:
            break
        for symbol in residual_symbols:
            capped[symbol] = capped[symbol] / residual_total * residual_budget
    total = sum(capped.values())
    if total <= 0.0:
        return {}
    return {symbol: weight / total for symbol, weight in capped.items()}
