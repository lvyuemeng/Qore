from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PositionSizer(Protocol):
    required_columns: frozenset[str]

    def size(self, signals: pl.DataFrame) -> dict[str, float]: ...


@dataclass(slots=True)
class EqualWeightSizer:
    top_k: int
    max_weight: float = 0.05
    required_columns: frozenset[str] = frozenset()

    def size(self, signals: pl.DataFrame) -> dict[str, float]:
        selected = _selected_signals(signals, self.top_k)
        if selected.is_empty():
            return {}
        weight = min(1.0 / selected.height, self.max_weight)
        return _weights_by_symbol(
            selected.with_columns(pl.lit(weight).alias("weight")),
            weight_col="weight",
        )


@dataclass(slots=True)
class VolScaledSizer:
    top_k: int
    vol_col: str = "realized_vol_20d"
    max_weight: float = 0.10
    volatility: dict[str, float] = field(default_factory=dict)

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset({self.vol_col})

    def with_volatility(self, volatility: dict[str, float]) -> VolScaledSizer:
        return VolScaledSizer(
            top_k=self.top_k,
            vol_col=self.vol_col,
            max_weight=self.max_weight,
            volatility=dict(volatility),
        )

    def size(self, signals: pl.DataFrame) -> dict[str, float]:
        selected = _selected_signals(signals, self.top_k)
        if selected.is_empty():
            return {}
        prepared = self._with_volatility(selected).with_columns(
            pl.when(pl.col(self.vol_col).cast(pl.Float64, strict=False) > 0.0)
            .then(1.0 / pl.col(self.vol_col).cast(pl.Float64, strict=False))
            .otherwise(1.0)
            .alias("inverse_vol")
        )
        total = prepared.select(pl.col("inverse_vol").sum().alias("total")).item()
        total_value = float(total) if isinstance(total, (int, float)) else 0.0
        if total_value <= 0.0:
            return {}
        normalized = _weights_by_symbol(
            prepared.with_columns(
                (pl.col("inverse_vol") / total_value).alias("weight")
            ),
            weight_col="weight",
        )
        return _cap_and_renormalize(normalized, self.max_weight)

    def _with_volatility(self, signals: pl.DataFrame) -> pl.DataFrame:
        if not self.volatility:
            return signals.with_columns(
                pl.col(self.vol_col).cast(pl.Float64, strict=False).fill_null(1.0)
            )
        overrides = pl.DataFrame(
            {
                "symbol": list(self.volatility),
                "override_volatility": list(self.volatility.values()),
            },
            schema={"symbol": pl.String, "override_volatility": pl.Float64},
        )
        return (
            signals.join(overrides, on="symbol", how="left")
            .with_columns(
                pl.coalesce(
                    "override_volatility",
                    pl.col(self.vol_col).cast(pl.Float64, strict=False),
                    pl.lit(1.0),
                ).alias(self.vol_col)
            )
            .drop("override_volatility")
        )


def _selected_signals(signals: pl.DataFrame, top_k: int) -> pl.DataFrame:
    if (
        signals.is_empty()
        or "symbol" not in signals.columns
        or "signal" not in signals.columns
    ):
        return pl.DataFrame(schema={"symbol": pl.String, "signal": pl.Float64})
    return pl.DataFrame(
        signals.lazy()
        .filter(pl.col("signal").is_not_null() & pl.col("signal").is_finite())
        .sort("signal", descending=True)
        .head(top_k)
        .collect()
    )


def _weights_by_symbol(frame: pl.DataFrame, weight_col: str) -> dict[str, float]:
    return {
        str(symbol): float(weight)
        for symbol, weight in frame.select("symbol", weight_col).iter_rows()
    }


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
