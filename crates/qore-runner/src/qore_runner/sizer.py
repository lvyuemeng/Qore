from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PositionSizer(Protocol):
    required_columns: frozenset[str]

    def prepare(
        self, signals: pl.LazyFrame, factor_lf: pl.LazyFrame
    ) -> pl.DataFrame: ...

    def size(self, signals: pl.DataFrame) -> pl.DataFrame: ...

    def cap(self, weights: pl.DataFrame, max_single: float) -> pl.DataFrame: ...

    def pipe(
        self,
        signals: pl.LazyFrame,
        factor_lf: pl.LazyFrame,
        *,
        max_single: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]: ...


@dataclass(slots=True)
class EqualWeightSizer:
    top_k: int
    max_weight: float = 0.05
    required_columns: frozenset[str] = frozenset()

    def prepare(self, signals: pl.LazyFrame, factor_lf: pl.LazyFrame) -> pl.DataFrame:
        return _prepare_signals(
            signals, factor_lf, required_columns=self.required_columns
        )

    def cap(self, weights: pl.DataFrame, max_single: float) -> pl.DataFrame:
        return _cap_and_renormalize_frame(weights, max_weight=max_single)

    def pipe(
        self,
        signals: pl.LazyFrame,
        factor_lf: pl.LazyFrame,
        *,
        max_single: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        prepared = self.prepare(signals, factor_lf)
        raw = self.size(prepared)
        return prepared, self.cap(raw, max_single)

    def size(self, signals: pl.DataFrame) -> pl.DataFrame:
        selected = _selected_signals(signals, self.top_k)
        if selected.is_empty():
            return _empty_weights_frame()
        weight = min(1.0 / selected.height, self.max_weight)
        return pl.DataFrame(
            selected.with_columns(pl.lit(weight).alias("weight")),
            schema={"symbol": pl.String, "signal": pl.Float64, "weight": pl.Float64},
        ).select("symbol", "weight")


@dataclass(slots=True)
class VolScaledSizer:
    top_k: int
    vol_col: str = "realized_vol_20d"
    max_weight: float = 0.10
    volatility: dict[str, float] = field(default_factory=dict)

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset({self.vol_col})

    def prepare(self, signals: pl.LazyFrame, factor_lf: pl.LazyFrame) -> pl.DataFrame:
        return _prepare_signals(
            signals, factor_lf, required_columns=self.required_columns
        )

    def cap(self, weights: pl.DataFrame, max_single: float) -> pl.DataFrame:
        return _cap_and_renormalize_frame(weights, max_weight=max_single)

    def pipe(
        self,
        signals: pl.LazyFrame,
        factor_lf: pl.LazyFrame,
        *,
        max_single: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        prepared = self.prepare(signals, factor_lf)
        raw = self.size(prepared)
        return prepared, self.cap(raw, max_single)

    def with_volatility(self, volatility: dict[str, float]) -> VolScaledSizer:
        return VolScaledSizer(
            top_k=self.top_k,
            vol_col=self.vol_col,
            max_weight=self.max_weight,
            volatility=dict(volatility),
        )

    def size(self, signals: pl.DataFrame) -> pl.DataFrame:
        selected = _selected_signals(signals, self.top_k)
        if selected.is_empty():
            return _empty_weights_frame()
        prepared = self._with_volatility(selected).with_columns(
            pl.when(pl.col(self.vol_col).cast(pl.Float64, strict=False) > 0.0)
            .then(1.0 / pl.col(self.vol_col).cast(pl.Float64, strict=False))
            .otherwise(1.0)
            .alias("inverse_vol")
        )
        total = prepared.select(pl.col("inverse_vol").sum().alias("total")).item()
        total_value = float(total) if isinstance(total, (int, float)) else 0.0
        if total_value <= 0.0:
            return _empty_weights_frame()
        normalized = pl.DataFrame(
            prepared.with_columns((pl.col("inverse_vol") / total_value).alias("weight"))
        ).select("symbol", "weight")
        return _cap_and_renormalize_frame(normalized, self.max_weight)

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


def _prepare_signals(
    signals: pl.LazyFrame,
    factor_lf: pl.LazyFrame,
    *,
    required_columns: frozenset[str],
) -> pl.DataFrame:
    columns = sorted(required_columns)
    if not columns:
        return pl.DataFrame(signals.collect())
    extras = factor_lf.select("symbol", *columns)
    return pl.DataFrame(signals.join(extras, on="symbol", how="left").collect())


def _cap_and_renormalize_frame(
    weights: pl.DataFrame, max_weight: float
) -> pl.DataFrame:
    if weights.is_empty():
        return _empty_weights_frame()
    working = pl.DataFrame(
        weights.lazy()
        .select(
            pl.col("symbol").cast(pl.String, strict=False),
            pl.col("weight").cast(pl.Float64, strict=False).fill_null(0.0),
        )
        .filter(pl.col("symbol").is_not_null() & (pl.col("weight") > 0.0))
        .collect()
    )
    if working.is_empty():
        return _empty_weights_frame()
    total = working.select(pl.col("weight").sum()).item()
    total_value = float(total) if isinstance(total, (int, float)) else 0.0
    if total_value <= 0.0:
        return _empty_weights_frame()
    rows = [
        (str(symbol), float(weight) / total_value)
        for symbol, weight in working.iter_rows()
    ]
    for _ in range(len(rows) + 1):
        overweight_idx = [
            idx for idx, (_, weight) in enumerate(rows) if weight > max_weight
        ]
        if not overweight_idx:
            break
        residual_idx = [idx for idx in range(len(rows)) if idx not in overweight_idx]
        fixed_weight = max_weight * len(overweight_idx)
        residual_budget = max(1.0 - fixed_weight, 0.0)
        residual_total = sum(rows[idx][1] for idx in residual_idx)
        for idx in overweight_idx:
            symbol, _ = rows[idx]
            rows[idx] = (symbol, max_weight)
        if not residual_idx or residual_total <= 0.0:
            continue
        for idx in residual_idx:
            symbol, weight = rows[idx]
            rows[idx] = (symbol, weight / residual_total * residual_budget)
    normalized_total = sum(weight for _, weight in rows)
    if normalized_total <= 0.0:
        return _empty_weights_frame()
    return pl.DataFrame(
        {
            "symbol": [symbol for symbol, _ in rows],
            "weight": [weight / normalized_total for _, weight in rows],
        },
        schema={"symbol": pl.String, "weight": pl.Float64},
    )


def _empty_weights_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"symbol": pl.String, "weight": pl.Float64})
