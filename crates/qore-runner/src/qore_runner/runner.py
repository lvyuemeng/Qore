from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
    decision_signals: pl.DataFrame
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
        current_weights: pl.DataFrame | None = None,
    ) -> TargetPortfolio:
        self._validate_factor_contract(context)
        signal_lf = self.strategy.generate(context)
        prepared_signal_frame, _ = self.sizer.pipe(
            signal_lf,
            context.factor_lf,
            max_single=self.settings.max_single,
        )
        current_weights_frame = (
            current_weights
            if current_weights is not None
            else pl.DataFrame(schema={"symbol": pl.String, "weight": pl.Float64})
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
        selected_weights_frame = self.sizer.cap(
            self.sizer.size(selected_signal_frame),
            max_single=self.settings.max_single,
        )
        decision_signals = decision.execution_signals(
            target_weights=selected_weights_frame,
            current_weights=current_weights_frame,
        )
        diagnostics = self._diagnostics(
            prepared_signal_frame,
            selected_weights_frame,
            nav,
        )
        return TargetPortfolio(
            date=context.date,
            weights_frame=selected_weights_frame,
            signals=selected_signal_frame,
            decision_signals=decision_signals,
            decision=decision,
            diagnostics=diagnostics,
        )

    def _validate_factor_contract(self, context: StrategyContext) -> None:
        schema_names = set(context.factor_lf.collect_schema().names())
        if "symbol" not in schema_names:
            msg = "Strategy factor frame must contain 'symbol' column."
            raise ValueError(msg)
        missing = sorted(self.strategy.required_columns - schema_names)
        if missing:
            msg = f"Missing required factor columns for strategy: {missing}"
            raise ValueError(msg)

    def _resolve_decision(
        self,
        context: StrategyContext,
        signal_frame: pl.DataFrame,
        nav: pl.Series,
    ) -> StrategyDecision:
        has_external_overlay = context.providers.decision_overlay is not None
        bucket = self.strategy.strategy_rebalance_schedule().bucket(context.date)
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
        selected_weights_frame: pl.DataFrame,
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
        selected_count = int(selected_weights_frame.height)
        non_selected_count = int(max(signal_frame.height - selected_count, 0))
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
