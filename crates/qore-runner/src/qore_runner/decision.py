from __future__ import annotations

from datetime import date

import polars as pl

DECISION_SIGNAL_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.String,
    "signal": pl.String,
    "weight_target": pl.Float64,
    "weight_current": pl.Float64,
    "weight_delta": pl.Float64,
    "score_value": pl.Float64,
}


def rank_symbols(
    signal_frame: pl.DataFrame,
    *,
    top_k: int | None = None,
    score_column: str = "signal",
    descending: bool = True,
) -> list[str]:
    """Sort symbols by ``score_column``, apply ``top_k``, return ordered list."""
    if signal_frame.is_empty() or score_column not in signal_frame.columns:
        return []
    working = signal_frame.filter(
        pl.col(score_column).is_not_null() & pl.col(score_column).is_finite()
    )
    if working.is_empty():
        return []
    ranked = working.sort(score_column, descending=descending)
    if top_k is not None:
        ranked = ranked.head(max(top_k, 0))
    return [
        str(s)
        for s in ranked.get_column("symbol").unique(maintain_order=True).to_list()
    ]


def execution_signals(
    *,
    target_weights: pl.DataFrame,
    current_weights: pl.DataFrame,
    as_of: date,
    score_value_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Diff target vs current weights, produce buy/sell/hold signals."""
    epsilon = 1e-12
    target = target_weights.select(
        pl.col("symbol").cast(pl.String),
        pl.col("weight").cast(pl.Float64).alias("weight_target"),
    ).unique(subset=["symbol"], keep="last")
    current = current_weights.select(
        pl.col("symbol").cast(pl.String),
        pl.col("weight").cast(pl.Float64).alias("weight_current"),
    ).unique(subset=["symbol"], keep="last")
    score = (
        score_value_frame.select(
            pl.col("symbol").cast(pl.String),
            pl.col("signal").cast(pl.Float64).alias("score_value"),
        ).unique(subset=["symbol"], keep="last")
        if score_value_frame is not None
        else pl.DataFrame(schema={"symbol": pl.String, "score_value": pl.Float64})
    )
    result = (
        target.join(current, on="symbol", how="full", coalesce=True)
        .join(score, on="symbol", how="left")
        .with_columns(
            pl.col("weight_target").fill_null(0.0),
            pl.col("weight_current").fill_null(0.0),
        )
        .with_columns(
            (pl.col("weight_target") - pl.col("weight_current")).alias("weight_delta")
        )
        .with_columns(
            pl.when(pl.col("weight_delta") > epsilon)
            .then(pl.lit("buy"))
            .when(pl.col("weight_delta") < -epsilon)
            .then(pl.lit("sell"))
            .when(pl.col("weight_target") > epsilon)
            .then(pl.lit("hold"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("signal")
        )
        .filter(pl.col("signal").is_not_null())
        .with_columns(pl.lit(as_of).cast(pl.Date).alias("date"))
        .select(
            "date",
            "symbol",
            "signal",
            "weight_target",
            "weight_current",
            "weight_delta",
            "score_value",
        )
    )
    return result
