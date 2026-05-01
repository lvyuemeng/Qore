from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from qore_factor.base import Factor

NormalizeMethod = Literal["zscore", "rank_pct"]
EVALUATION_SCHEMA: dict[str, pl.DataType] = {
    "signal_key": pl.String(),
    "horizon": pl.Int64(),
    "ic_mean": pl.Float64(),
    "ic_std": pl.Float64(),
    "icir": pl.Float64(),
    "observations": pl.Int64(),
}


@dataclass(slots=True)
class FactorPipeline:
    factors: list[Factor] = field(default_factory=list)
    normalize: NormalizeMethod | None = None
    normalize_group_by: list[str] = field(default_factory=lambda: ["date"])
    neutralize_by: list[str] = field(default_factory=list)

    def add(self, *factors: Factor) -> FactorPipeline:
        self.factors.extend(factors)
        return self

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        available = set(lf.collect_schema().names())
        computed: list[str] = []
        factors = self.factors

        for factor in factors:
            missing = sorted(factor.requires - available)
            if missing:
                msg = f"Factor '{factor.name}' missing required columns: {missing}"
                raise ValueError(msg)
            lf = factor.compute(lf)
            available.add(factor.produces)
            computed.append(factor.produces)

        if self.neutralize_by:
            lf = self._apply_neutralization(lf, computed)

        if self.normalize is not None:
            lf = self._apply_normalization(lf, computed)

        return lf

    def evaluate(
        self,
        factor_lf: pl.LazyFrame,
        forward_returns: pl.LazyFrame,
        horizons: list[int],
    ) -> pl.DataFrame:
        factor_columns = [factor.produces for factor in self.factors]
        if not factor_columns or not horizons:
            return pl.DataFrame(schema=EVALUATION_SCHEMA)

        required_return_columns = [f"forward_return_{horizon}d" for horizon in horizons]
        forward_schema = set(forward_returns.collect_schema().names())
        missing_returns = [
            column for column in required_return_columns if column not in forward_schema
        ]
        if missing_returns:
            msg = f"Missing forward return columns: {missing_returns}"
            raise ValueError(msg)

        factor_frame = self.run(factor_lf).select("date", "symbol", *factor_columns)
        joined = factor_frame.join(
            forward_returns,
            on=["date", "symbol"],
            how="inner",
        )
        metric_specs = [
            (signal_key, horizon, f"__ic__{signal_key}__{horizon}")
            for signal_key in factor_columns
            for horizon in horizons
        ]
        daily_ic = pl.DataFrame(
            joined.group_by("date")
            .agg(
                [
                    pl.corr(
                        pl.col(signal_key).cast(pl.Float64),
                        pl.col(f"forward_return_{horizon}d").cast(pl.Float64),
                    ).alias(alias)
                    for signal_key, horizon, alias in metric_specs
                ]
            )
            .collect()
        )
        if daily_ic.is_empty():
            return pl.DataFrame(schema=EVALUATION_SCHEMA)

        signal_keys: list[str] = []
        horizon_values: list[int] = []
        ic_means: list[float | None] = []
        ic_stds: list[float | None] = []
        icirs: list[float | None] = []
        observations: list[int] = []
        for signal_key, horizon, alias in metric_specs:
            series = daily_ic.get_column(alias).drop_nulls()
            ic_mean = series.mean()
            ic_std = series.std()
            ic_mean_value = float(ic_mean) if isinstance(ic_mean, int | float) else None
            ic_std_value = float(ic_std) if isinstance(ic_std, int | float) else None
            signal_keys.append(signal_key)
            horizon_values.append(horizon)
            ic_means.append(ic_mean_value)
            ic_stds.append(ic_std_value)
            icirs.append(
                None
                if ic_mean_value is None or ic_std_value in (None, 0.0)
                else ic_mean_value / ic_std_value
            )
            observations.append(series.len())

        return pl.DataFrame(
            {
                "signal_key": signal_keys,
                "horizon": horizon_values,
                "ic_mean": ic_means,
                "ic_std": ic_stds,
                "icir": icirs,
                "observations": observations,
            },
            schema=EVALUATION_SCHEMA,
        )

    def _apply_neutralization(
        self,
        lf: pl.LazyFrame,
        factor_columns: list[str],
    ) -> pl.LazyFrame:
        if not factor_columns:
            return lf

        expressions = [
            (pl.col(column) - pl.col(column).mean().over(self.neutralize_by)).alias(
                column
            )
            for column in factor_columns
        ]
        return lf.with_columns(expressions)

    def _apply_normalization(
        self,
        lf: pl.LazyFrame,
        factor_columns: list[str],
    ) -> pl.LazyFrame:
        if not factor_columns:
            return lf

        if self.normalize == "zscore":
            group_by = self.normalize_group_by
            expressions = [
                (
                    (pl.col(column) - pl.col(column).mean().over(group_by))
                    / (pl.col(column).std().over(group_by) + 1e-8)
                ).alias(f"{column}_z")
                for column in factor_columns
            ]
            return lf.with_columns(expressions)

        if self.normalize == "rank_pct":
            group_by = self.normalize_group_by
            expressions = [
                (
                    pl.col(column).rank(method="average").over(group_by)
                    / pl.len().over(group_by)
                ).alias(f"{column}_rank_pct")
                for column in factor_columns
            ]
            return lf.with_columns(expressions)

        msg = f"Unsupported normalization method: {self.normalize}"
        raise ValueError(msg)
