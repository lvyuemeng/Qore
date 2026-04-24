from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cache

import polars as pl

from qore_runner.strategy import StrategySelectionSpec


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    as_of: date
    frame: pl.DataFrame

    @property
    def selected_symbols(self) -> frozenset[str]:
        if self.frame.is_empty():
            return frozenset()
        return frozenset(
            str(symbol)
            for symbol in self.frame.filter(pl.col("selected"))
            .get_column("symbol")
            .to_list()
        )

    @property
    def force_exit_symbols(self) -> frozenset[str]:
        if self.frame.is_empty():
            return frozenset()
        return frozenset(
            str(symbol)
            for symbol in self.frame.filter(
                pl.col("exclude_reason")
                .cast(pl.String, strict=False)
                .str.starts_with("force_exit")
            )
            .get_column("symbol")
            .to_list()
        )


@dataclass(frozen=True, slots=True)
class DecisionPipeline:
    as_of: date
    selection: StrategySelectionSpec
    drawdown_stop: float

    def resolve(
        self,
        signal_frame: pl.DataFrame,
        overlay_frame: pl.DataFrame | pl.LazyFrame | None,
        nav: pl.Series,
    ) -> StrategyDecision:
        normalized_signal = _normalize_signal_frame(signal_frame)
        ranked_symbols = _ranked_symbol_frame(
            normalized_signal,
            selection=self.selection,
        )
        base = pl.DataFrame(
            normalized_signal.lazy()
            .join(
                ranked_symbols.lazy().with_columns(
                    pl.lit(True, dtype=pl.Boolean).alias("_rank_selected")
                ),
                on="symbol",
                how="left",
            )
            .with_columns(
                pl.col("_rank_selected").fill_null(False).alias("selected"),
                pl.when(
                    ~pl.col("signal").is_not_null().and_(pl.col("signal").is_finite())
                )
                .then(pl.lit("missing_signal"))
                .when(~pl.col("_rank_selected").fill_null(False))
                .then(pl.lit("rank_cutoff"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("exclude_reason"),
            )
            .drop("_rank_selected")
            .collect()
        )
        frame = self._apply_overlay(base, overlay_frame)
        frame = self._apply_drawdown(frame, nav)
        return StrategyDecision(as_of=self.as_of, frame=frame)

    def _apply_overlay(
        self,
        base_frame: pl.DataFrame,
        overlay_frame: pl.DataFrame | pl.LazyFrame | None,
    ) -> pl.DataFrame:
        overlay = _normalize_decision_overlay(overlay_frame)
        if overlay.is_empty():
            return base_frame
        return pl.DataFrame(
            base_frame.lazy()
            .join(overlay.lazy(), on="symbol", how="left")
            .with_columns(
                pl.when(pl.col("overlay_selected").is_null())
                .then(pl.col("selected"))
                .otherwise(pl.col("selected") & pl.col("overlay_selected"))
                .alias("selected"),
                pl.when(~pl.col("overlay_selected"))
                .then(
                    pl.col("overlay_exclude_reason")
                    .fill_null("overlay_blocked")
                    .cast(pl.String, strict=False)
                )
                .otherwise(pl.col("exclude_reason").cast(pl.String, strict=False))
                .alias("exclude_reason"),
            )
            .select("symbol", "signal", "selected", "exclude_reason")
            .collect()
        )

    def _apply_drawdown(self, frame: pl.DataFrame, nav: pl.Series) -> pl.DataFrame:
        if len(nav) <= 1:
            return frame
        peak_value = nav.max()
        latest_value = nav.tail(1).item()
        peak = float(peak_value) if isinstance(peak_value, (int, float)) else 0.0
        latest = float(latest_value) if isinstance(latest_value, (int, float)) else 0.0
        if peak <= 0.0 or (peak - latest) / peak < self.drawdown_stop:
            return frame
        return pl.DataFrame(
            frame.lazy()
            .with_columns(
                pl.when(pl.col("selected"))
                .then(pl.lit(False))
                .otherwise(pl.col("selected"))
                .alias("selected"),
                pl.when(pl.col("selected"))
                .then(pl.lit("risk_drawdown_stop"))
                .otherwise(pl.col("exclude_reason"))
                .alias("exclude_reason"),
            )
            .collect()
        )


def _normalize_signal_frame(signal_frame: pl.DataFrame) -> pl.DataFrame:
    if signal_frame.is_empty() or "symbol" not in signal_frame.columns:
        return pl.DataFrame(schema={"symbol": pl.String, "signal": pl.Float64})
    signal_col = "signal" if "signal" in signal_frame.columns else None
    if signal_col is None:
        return pl.DataFrame(schema={"symbol": pl.String, "signal": pl.Float64})
    return pl.DataFrame(
        signal_frame.lazy()
        .with_columns(
            pl.col("symbol").cast(pl.String, strict=False),
            pl.col(signal_col).cast(pl.Float64, strict=False).alias("signal"),
        )
        .filter(pl.col("symbol").is_not_null())
        .select("symbol", "signal")
        .unique(subset=["symbol"], keep="last")
        .collect()
    )


def _ranked_symbol_frame(
    signal_frame: pl.DataFrame,
    *,
    selection: StrategySelectionSpec,
) -> pl.DataFrame:
    if signal_frame.is_empty() or "symbol" not in signal_frame.columns:
        return pl.DataFrame(schema={"symbol": pl.String})
    ranked = signal_frame.filter(
        pl.col(selection.score_column).is_not_null()
        & pl.col(selection.score_column).is_finite()
    )
    if ranked.is_empty():
        return pl.DataFrame(schema={"symbol": pl.String})
    sort_columns: list[pl.Expr | str] = [
        pl.col(selection.score_column).is_null(),
        selection.score_column,
    ]
    descending = [False, selection.descending]
    for column in cached_tie_break_columns(selection.tie_break_columns):
        if column in ranked.columns and column != selection.score_column:
            sort_columns.append(column)
            descending.append(False)
    ranked = ranked.sort(sort_columns, descending=descending)
    if selection.top_k is not None:
        ranked = ranked.head(max(selection.top_k, 0))
    return pl.DataFrame(
        ranked.lazy()
        .select(pl.col("symbol").cast(pl.String, strict=False).alias("symbol"))
        .filter(pl.col("symbol").is_not_null())
        .collect()
    )


def _normalize_decision_overlay(
    overlay_frame: pl.DataFrame | pl.LazyFrame | None,
) -> pl.DataFrame:
    if overlay_frame is None:
        return _empty_decision_overlay()
    overlay = (
        pl.DataFrame(overlay_frame.collect())
        if isinstance(overlay_frame, pl.LazyFrame)
        else overlay_frame
    )
    if overlay.is_empty() or "symbol" not in overlay.columns:
        return _empty_decision_overlay()
    selected_expr = (
        pl.col("selected").cast(pl.Boolean, strict=False)
        if "selected" in overlay.columns
        else (
            pl.col("eligible").cast(pl.Boolean, strict=False)
            & ~pl.col("force_exit").cast(pl.Boolean, strict=False).fill_null(False)
            if "eligible" in overlay.columns and "force_exit" in overlay.columns
            else pl.lit(None, dtype=pl.Boolean)
        )
    )
    reason_expr = (
        pl.col("exclude_reason").cast(pl.String, strict=False)
        if "exclude_reason" in overlay.columns
        else (
            pl.col("not_selected_reason").cast(pl.String, strict=False)
            if "not_selected_reason" in overlay.columns
            else (
                pl.col("reason_code").cast(pl.String, strict=False)
                if "reason_code" in overlay.columns
                else pl.lit(None, dtype=pl.String)
            )
        )
    )
    return pl.DataFrame(
        overlay.lazy()
        .with_columns(
            pl.col("symbol").cast(pl.String, strict=False),
            selected_expr.alias("overlay_selected"),
            reason_expr.alias("overlay_exclude_reason"),
        )
        .filter(pl.col("symbol").is_not_null())
        .select("symbol", "overlay_selected", "overlay_exclude_reason")
        .unique(subset=["symbol"], keep="last")
        .collect()
    )


def _empty_decision_overlay() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "overlay_selected": pl.Boolean,
            "overlay_exclude_reason": pl.String,
        }
    )


@cache
def cached_tie_break_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    return columns
