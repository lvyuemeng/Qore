from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl
from qore_data.store.duckdb import QoreStore

from qore_factor.base import Factor

NormalizeMethod = Literal["zscore", "rank_pct"]


@dataclass(slots=True)
class FactorPipeline:
    factors: list[Factor] = field(default_factory=list)
    _normalize_method: NormalizeMethod | None = None
    _normalize_group_by: list[str] = field(default_factory=lambda: ["date"])
    _neutralize_by: list[str] = field(default_factory=list)

    def add(self, *factors: Factor) -> FactorPipeline:
        self.factors.extend(factors)
        return self

    def normalize(
        self,
        method: NormalizeMethod = "zscore",
        group_by: list[str] | None = None,
    ) -> FactorPipeline:
        self._normalize_method = method
        self._normalize_group_by = group_by or ["date"]
        return self

    def neutralize(self, by: list[str]) -> FactorPipeline:
        self._neutralize_by = by
        return self

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        available = set(lf.collect_schema().names())
        computed: list[str] = []

        for factor in self.factors:
            missing = sorted(factor.requires - available)
            if missing:
                msg = f"Factor '{factor.name}' missing required columns: {missing}"
                raise ValueError(msg)
            lf = factor.compute(lf)
            available.add(factor.produces)
            computed.append(factor.produces)

        if self._neutralize_by:
            lf = self._apply_neutralization(lf, computed)

        if self._normalize_method is not None:
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
            return pl.DataFrame(
                schema={
                    "factor_name": pl.String,
                    "horizon": pl.Int64,
                    "ic_mean": pl.Float64,
                    "ic_std": pl.Float64,
                    "icir": pl.Float64,
                    "observations": pl.Int64,
                }
            )

        factor_frame = self.run(factor_lf).select("date", "symbol", *factor_columns)
        joined = factor_frame.join(
            forward_returns,
            on=["date", "symbol"],
            how="inner",
        )

        metric_frames: list[pl.DataFrame] = []
        for factor_name in factor_columns:
            for horizon in horizons:
                return_column = f"forward_return_{horizon}d"
                if return_column not in joined.collect_schema().names():
                    msg = f"Missing forward return column: {return_column}"
                    raise ValueError(msg)

                daily_ic = pl.DataFrame(
                    joined.select(
                        "date",
                        pl.col(factor_name).cast(pl.Float64).alias("factor_value"),
                        pl.col(return_column).cast(pl.Float64).alias("forward_return"),
                    )
                    .filter(
                        pl.col("factor_value").is_not_null()
                        & pl.col("forward_return").is_not_null()
                    )
                    .group_by("date")
                    .agg(pl.corr("factor_value", "forward_return").alias("ic"))
                    .filter(pl.col("ic").is_not_null())
                    .collect()
                )

                metric_frames.append(
                    pl.DataFrame(
                        [
                            {
                                "factor_name": factor_name,
                                "horizon": horizon,
                                "ic_mean": _series_stat(daily_ic, "mean"),
                                "ic_std": _series_stat(daily_ic, "std"),
                                "icir": _icir(daily_ic),
                                "observations": daily_ic.height,
                            }
                        ]
                    )
                )

        return pl.concat(metric_frames, how="vertical")

    def to_factor_scores(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        result = self.run(lf)
        factor_columns = [factor.produces for factor in self.factors]
        if not factor_columns:
            return pl.LazyFrame(
                schema={
                    "date": pl.Date,
                    "symbol": pl.String,
                    "factor_name": pl.String,
                    "raw_value": pl.Float64,
                    "z_score": pl.Float64,
                    "rank_pct": pl.Float64,
                }
            )

        schema_names = set(result.collect_schema().names())
        frames: list[pl.LazyFrame] = []
        for column in factor_columns:
            z_col = f"{column}_z"
            rank_col = f"{column}_rank_pct"
            frames.append(
                result.select(
                    pl.col("date"),
                    pl.col("symbol"),
                    pl.lit(column).alias("factor_name"),
                    pl.col(column).cast(pl.Float64).alias("raw_value"),
                    (
                        pl.col(z_col).cast(pl.Float64)
                        if z_col in schema_names
                        else pl.lit(None, dtype=pl.Float64)
                    ).alias("z_score"),
                    (
                        pl.col(rank_col).cast(pl.Float64)
                        if rank_col in schema_names
                        else pl.lit(None, dtype=pl.Float64)
                    ).alias("rank_pct"),
                )
            )

        return pl.concat(frames, how="vertical")

    def persist(self, lf: pl.LazyFrame, store: QoreStore) -> None:
        store.write("factor_scores", self.to_factor_scores(lf))

    def _apply_neutralization(
        self,
        lf: pl.LazyFrame,
        factor_columns: list[str],
    ) -> pl.LazyFrame:
        if not factor_columns:
            return lf

        expressions = [
            (pl.col(column) - pl.col(column).mean().over(self._neutralize_by)).alias(
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

        if self._normalize_method == "zscore":
            expressions = []
            for column in factor_columns:
                mean = pl.col(column).mean().over(self._normalize_group_by)
                std = pl.col(column).std().over(self._normalize_group_by)
                expressions.append(
                    ((pl.col(column) - mean) / (std + 1e-8)).alias(f"{column}_z")
                )
            return lf.with_columns(expressions)

        if self._normalize_method == "rank_pct":
            expressions = [
                (
                    pl.col(column).rank(method="average").over(self._normalize_group_by)
                    / pl.len().over(self._normalize_group_by)
                ).alias(f"{column}_rank_pct")
                for column in factor_columns
            ]
            return lf.with_columns(expressions)

        msg = f"Unsupported normalization method: {self._normalize_method}"
        raise ValueError(msg)


def _series_stat(frame: pl.DataFrame, stat: Literal["mean", "std"]) -> float | None:
    if frame.is_empty():
        return None
    series = frame.get_column("ic")
    value = series.mean() if stat == "mean" else series.std()
    return float(value) if isinstance(value, (int, float)) else None


def _icir(frame: pl.DataFrame) -> float | None:
    ic_mean = _series_stat(frame, "mean")
    ic_std = _series_stat(frame, "std")
    if ic_mean is None or ic_std in (None, 0.0):
        return None
    return ic_mean / ic_std
