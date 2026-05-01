from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PositionSizer(Protocol):
    required_columns: frozenset[str]

    def size(self, signals: pl.DataFrame) -> pl.DataFrame:
        """Returns ``{symbol, weight}`` with weights capped to ``max_weight`` and summing to 1.0."""
        ...


@dataclass(slots=True)
class EqualWeightSizer:
    max_weight: float = 0.05
    required_columns: frozenset[str] = frozenset()

    def size(self, signals: pl.DataFrame) -> pl.DataFrame:
        if signals.is_empty() or "symbol" not in signals.columns:
            return _empty_weights_frame()
        n = max(signals.height, 1)
        weight = min(1.0 / n, self.max_weight)
        weights = pl.DataFrame(
            {"symbol": signals.get_column("symbol"), "weight": [weight] * n},
            schema={"symbol": pl.String, "weight": pl.Float64},
        )
        if 1.0 / n <= self.max_weight:
            return weights
        return _cap_and_renormalize(weights, self.max_weight)


@dataclass(slots=True)
class VolScaledSizer:
    vol_col: str
    max_weight: float = 0.10
    volatility: dict[str, float] = field(default_factory=dict)

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset({self.vol_col})

    def with_volatility(self, volatility: dict[str, float]) -> VolScaledSizer:
        return VolScaledSizer(
            vol_col=self.vol_col,
            max_weight=self.max_weight,
            volatility=dict(volatility),
        )

    def size(self, signals: pl.DataFrame) -> pl.DataFrame:
        if signals.is_empty() or "symbol" not in signals.columns:
            return _empty_weights_frame()
        vol = self._with_volatility(signals).get_column(self.vol_col)
        inv = pl.Series((1.0 / v) if v and v > 0.0 else 1.0 for v in vol.to_list())
        total = float(inv.sum())
        if total <= 0.0:
            return _empty_weights_frame()
        weights = pl.DataFrame(
            {"symbol": signals.get_column("symbol"), "weight": (inv / total).to_list()},
            schema={"symbol": pl.String, "weight": pl.Float64},
        )
        return _cap_and_renormalize(weights, self.max_weight)

    def _with_volatility(self, signals: pl.DataFrame) -> pl.DataFrame:
        if not self.volatility:
            return signals.with_columns(
                pl.col(self.vol_col).cast(pl.Float64, strict=False).fill_null(1.0)
            )
        overrides = pl.DataFrame(
            {
                "symbol": list(self.volatility),
                "override_vol": list(self.volatility.values()),
            },
            schema={"symbol": pl.String, "override_vol": pl.Float64},
        )
        return (
            signals.join(overrides, on="symbol", how="left")
            .with_columns(
                pl.coalesce(
                    pl.col("override_vol").cast(pl.Float64, strict=False),
                    pl.col(self.vol_col).cast(pl.Float64, strict=False),
                    pl.lit(1.0),
                ).alias(self.vol_col)
            )
            .drop("override_vol")
        )


def _cap_and_renormalize(weights: pl.DataFrame, max_weight: float) -> pl.DataFrame:
    if weights.is_empty():
        return _empty_weights_frame()
    keep = (
        weights.lazy()
        .select(
            pl.col("symbol").cast(pl.String, strict=False),
            pl.col("weight").cast(pl.Float64, strict=False).fill_null(0.0),
        )
        .filter(pl.col("weight") > 0.0)
        .collect()
    )
    if keep.is_empty():
        return _empty_weights_frame()
    total = float(keep.get_column("weight").sum())
    if total <= 0.0:
        return _empty_weights_frame()
    rows = [
        (s, w / total)
        for s, w in zip(
            keep.get_column("symbol").to_list(),
            keep.get_column("weight").to_list(),
            strict=False,
        )
    ]
    n = len(rows)
    if int(1.0 / max_weight) >= n:
        return pl.DataFrame(
            {"symbol": [s for s, _ in rows], "weight": [1.0 / n] * n},
            schema={"symbol": pl.String, "weight": pl.Float64},
        )
    rows.sort(key=lambda x: -x[1])
    for _ in range(n + 1):
        k = sum(1 for _, w in rows if w > max_weight)
        if k == 0:
            break
        residual_weight = sum(w for _, w in rows[k:])
        budget = max(1.0 - k * max_weight, 0.0)
        for i in range(k):
            rows[i] = (rows[i][0], max_weight)
        if residual_weight > 0.0 and budget >= 0.0:
            for i in range(k, n):
                rows[i] = (rows[i][0], rows[i][1] / residual_weight * budget)
    nt = sum(w for _, w in rows)
    if nt <= 0.0:
        return _empty_weights_frame()
    return pl.DataFrame(
        {"symbol": [s for s, _ in rows], "weight": [w / nt for _, w in rows]},
        schema={"symbol": pl.String, "weight": pl.Float64},
    )


def _empty_weights_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"symbol": pl.String, "weight": pl.Float64})
