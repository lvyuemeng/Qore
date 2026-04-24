# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    max_single: float = 0.05
    drawdown_stop: float = 0.15


from qore_runner.calendar import TradingCalendar
from qore_runner.decision import DecisionPipeline
from qore_runner.runner import (
    RunnerDiagnostics,
    StrategyDecision,
    StrategyRunner,
    TargetPortfolio,
)
from qore_runner.sizer import EqualWeightSizer, PositionSizer, VolScaledSizer
from qore_runner.strategies.behavioral import BehavioralGatedStrategy
from qore_runner.strategies.crosssectional import CrossSectionalScreener
from qore_runner.strategies.ranking import RankingStrategy, WeightedOverlayCombiner
from qore_runner.strategy import (
    Strategy,
    StrategyContext,
    StrategyProviderFrames,
    StrategySelectionSpec,
)

__all__ = [
    "BehavioralGatedStrategy",
    "CrossSectionalScreener",
    "DecisionPipeline",
    "EqualWeightSizer",
    "PositionSizer",
    "RankingStrategy",
    "RunnerDiagnostics",
    "RunnerSettings",
    "Strategy",
    "StrategyContext",
    "StrategyDecision",
    "StrategyProviderFrames",
    "StrategyRunner",
    "StrategySelectionSpec",
    "TargetPortfolio",
    "TradingCalendar",
    "VolScaledSizer",
    "WeightedOverlayCombiner",
]
