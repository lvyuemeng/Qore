from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from qore_core.calendar import TradingCalendar
from qore_core.config import QoreConfig
from qore_core.universe import Universe
from qore_runner.risk import RiskManager
from qore_runner.sizer import PositionSizer
from qore_runner.strategy import Strategy


@dataclass(slots=True)
class TargetPortfolio:
    date: date
    weights: dict[str, float]
    signals: pl.Series
    risk_triggered: bool


@dataclass(slots=True)
class StrategyRunner:
    strategy: Strategy
    sizer: PositionSizer
    risk_manager: RiskManager

    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        strategy: Strategy,
        sizer: PositionSizer,
    ) -> "StrategyRunner":
        return cls(
            strategy=strategy,
            sizer=sizer,
            risk_manager=RiskManager.from_config(config),
        )

    def step(
        self,
        factor_lf: pl.LazyFrame,
        news_scores: dict[str, float] | None,
        universe: Universe,
        date: date,
        current_weights: dict[str, float],
        nav: pl.Series,
        calendar: TradingCalendar,
    ) -> TargetPortfolio:
        del news_scores
        signals = self.strategy.generate(factor_lf, universe, date, calendar)
        target = self.sizer.size(signals, universe)
        adjusted = self.risk_manager.apply(target, current_weights, nav)
        return TargetPortfolio(
            date=date,
            weights=adjusted,
            signals=signals,
            risk_triggered=adjusted == {} and target != {},
        )
