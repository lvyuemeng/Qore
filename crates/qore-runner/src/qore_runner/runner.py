from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TypeVar

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.config import QoreConfig
from qore_core.instrument import SessionInstrument
from qore_core.universe import Universe

from qore_runner.risk import RiskManager
from qore_runner.sizer import PositionSizer
from qore_runner.strategy import Strategy, StrategyContext

TInstrument = TypeVar("TInstrument", bound=SessionInstrument)


@dataclass(frozen=True, slots=True)
class RunnerDiagnostics:
    eligible_count: int
    signal_count: int
    selected_count: int
    risk_blocked: bool


@dataclass(slots=True)
class TargetPortfolio:
    date: date
    weights: dict[str, float]
    signals: pl.DataFrame
    risk_triggered: bool
    diagnostics: RunnerDiagnostics


@dataclass(slots=True)
class StrategyRunner[TInstrument: SessionInstrument]:
    strategy: Strategy[TInstrument]
    sizer: PositionSizer
    risk_manager: RiskManager

    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        strategy: Strategy[TInstrument],
        sizer: PositionSizer,
    ) -> StrategyRunner[TInstrument]:
        return cls(
            strategy=strategy,
            sizer=sizer,
            risk_manager=RiskManager.from_config(config),
        )

    def step(
        self,
        factor_lf: pl.LazyFrame,
        inputs: Mapping[str, object] | None,
        universe: Universe[TInstrument],
        date: date,
        current_weights: dict[str, float],
        nav: pl.Series,
        calendar: TradingCalendar,
    ) -> TargetPortfolio:
        signal_lf = self.strategy.generate(
            StrategyContext(
                factor_lf=factor_lf,
                universe=universe,
                date=date,
                calendar=calendar,
                inputs=inputs or {},
            )
        ).signals
        signal_frame = self._signal_frame_for_sizer(signal_lf, factor_lf)
        target = self.sizer.size(signal_frame)
        adjusted = self.risk_manager.apply(target, current_weights, nav)
        risk_triggered = adjusted == {} and target != {}
        diagnostics = self._diagnostics(signal_frame, target, risk_triggered)
        return TargetPortfolio(
            date=date,
            weights=adjusted,
            signals=signal_frame,
            risk_triggered=risk_triggered,
            diagnostics=diagnostics,
        )

    def _signal_frame_for_sizer(
        self,
        signals: pl.LazyFrame,
        factor_lf: pl.LazyFrame,
    ) -> pl.DataFrame:
        required_columns = sorted(self.sizer.required_columns)
        if not required_columns:
            return pl.DataFrame(signals.collect())
        extras = factor_lf.select("symbol", *required_columns)
        return pl.DataFrame(signals.join(extras, on="symbol", how="left").collect())

    def _diagnostics(
        self,
        signal_frame: pl.DataFrame,
        target: dict[str, float],
        risk_triggered: bool,
    ) -> RunnerDiagnostics:
        stats = signal_frame.select(
            pl.len().alias("eligible_count"),
            pl.col("signal")
            .is_not_null()
            .and_(pl.col("signal").is_finite())
            .sum()
            .alias("signal_count"),
        ).row(0, named=True)
        eligible_count = int(stats["eligible_count"])
        signal_count = int(stats["signal_count"])
        return RunnerDiagnostics(
            eligible_count=eligible_count,
            signal_count=signal_count,
            selected_count=len(target),
            risk_blocked=risk_triggered,
        )
