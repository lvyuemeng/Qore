from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl

from qore_runner import RunnerSettings
from qore_runner.decision import DecisionPipeline, StrategyDecision
from qore_runner.sizer import PositionSizer
from qore_runner.strategy import Strategy, StrategyContext


@dataclass(frozen=True, slots=True)
class RunnerDiagnostics:
    candidate_count: int
    signal_count: int
    selected_count: int
    non_selected_count: int
    drawdown_blocked: bool


@dataclass(slots=True)
class TargetPortfolio:
    date: date
    weights_frame: pl.DataFrame
    signals: pl.DataFrame
    decision: StrategyDecision
    diagnostics: RunnerDiagnostics


@dataclass(slots=True)
class StrategyRunner:
    strategy: Strategy
    sizer: PositionSizer
    settings: RunnerSettings
    _cached_decision: StrategyDecision | None = None
    _cached_schedule_bucket: tuple[int, int, int] | tuple[int, int] | date | None = None

    @classmethod
    def from_settings(
        cls,
        settings: RunnerSettings,
        strategy: Strategy,
        sizer: PositionSizer,
    ) -> StrategyRunner:
        return cls(strategy=strategy, sizer=sizer, settings=settings)

    def step(
        self,
        context: StrategyContext,
        nav: pl.Series,
    ) -> TargetPortfolio:
        signal_lf = self.strategy.generate(context)
        prepared_signal_frame, sized_target_frame = self.sizer.pipe(
            signal_lf,
            context.factor_lf,
            max_single=self.settings.max_single,
        )
        decision = self._resolve_decision(context, prepared_signal_frame, nav)
        selected_signal_frame = pl.DataFrame(
            prepared_signal_frame.lazy()
            .join(
                decision.frame.lazy().select("symbol", "selected"),
                on="symbol",
                how="left",
            )
            .filter(pl.col("selected").fill_null(False))
            .drop("selected")
            .collect()
        )
        selected_weights_frame = _selected_weights_frame(
            sized_target_frame,
            decision.frame,
        )
        diagnostics = self._diagnostics(prepared_signal_frame, decision, nav)
        return TargetPortfolio(
            date=context.date,
            weights_frame=selected_weights_frame,
            signals=selected_signal_frame,
            decision=decision,
            diagnostics=diagnostics,
        )

    def _resolve_decision(
        self,
        context: StrategyContext,
        signal_frame: pl.DataFrame,
        nav: pl.Series,
    ) -> StrategyDecision:
        has_external_overlay = context.providers.decision_overlay is not None
        bucket = _schedule_bucket(self.strategy.signal_freq, context.date)
        if (
            not has_external_overlay
            and self._cached_decision is not None
            and self._cached_schedule_bucket == bucket
        ):
            return self._cached_decision
        decision = DecisionPipeline(
            as_of=context.date,
            selection=context.selection,
            drawdown_stop=self.settings.drawdown_stop,
        ).resolve(
            signal_frame=signal_frame,
            overlay_frame=context.providers.decision_overlay,
            nav=nav,
        )
        if not has_external_overlay:
            self._cached_decision = decision
            self._cached_schedule_bucket = bucket
        return decision

    def _diagnostics(
        self,
        signal_frame: pl.DataFrame,
        decision: StrategyDecision,
        nav: pl.Series,
    ) -> RunnerDiagnostics:
        signal_count = (
            int(
                signal_frame.select(
                    pl.col("signal")
                    .is_not_null()
                    .and_(pl.col("signal").is_finite())
                    .sum()
                ).item()
            )
            if not signal_frame.is_empty() and "signal" in signal_frame.columns
            else 0
        )
        selected_count = int(decision.frame.filter(pl.col("selected")).height)
        non_selected_count = int(decision.frame.height - selected_count)
        drawdown_blocked = False
        if len(nav) > 1:
            peak_value = nav.max()
            latest_value = nav.tail(1).item()
            peak = float(peak_value) if isinstance(peak_value, (int, float)) else 0.0
            latest = (
                float(latest_value) if isinstance(latest_value, (int, float)) else 0.0
            )
            drawdown_blocked = (
                peak > 0.0 and (peak - latest) / peak >= self.settings.drawdown_stop
            )
        return RunnerDiagnostics(
            candidate_count=int(signal_frame.height),
            signal_count=signal_count,
            selected_count=selected_count,
            non_selected_count=non_selected_count,
            drawdown_blocked=drawdown_blocked,
        )


def _schedule_bucket(
    freq: Literal["event", "daily", "weekly", "monthly"],
    as_of: date,
) -> tuple[int, int, int] | tuple[int, int] | date:
    if freq in {"event", "daily"}:
        return as_of
    if freq == "weekly":
        iso_year, iso_week, _ = as_of.isocalendar()
        return (iso_year, iso_week)
    return (as_of.year, as_of.month)


def _selected_weights_frame(
    target_frame: pl.DataFrame,
    decision_frame: pl.DataFrame,
) -> pl.DataFrame:
    if target_frame.is_empty() or decision_frame.is_empty():
        return pl.DataFrame(schema={"symbol": pl.String, "weight": pl.Float64})
    return pl.DataFrame(
        target_frame.lazy()
        .join(
            decision_frame.lazy().select("symbol", "selected"),
            on="symbol",
            how="left",
        )
        .filter(pl.col("selected").fill_null(False))
        .select("symbol", "weight")
        .collect()
    )
