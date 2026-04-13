from __future__ import annotations

from datetime import date

import polars as pl
from qore_core import QoreConfig, StockInstrument, TradingCalendar, Universe
from qore_runner.risk import RiskManager
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.behavioral import BehavioralGatedStrategy
from qore_runner.strategies.crosssectional import CrossSectionalScreener


class StubRegimeDetector:
    def __init__(self, scale_value: float) -> None:
        self.scale_value = scale_value

    def scale(self, lf: pl.LazyFrame, as_of: date) -> float:
        del lf, as_of
        return self.scale_value


def test_risk_manager_caps_weights() -> None:
    manager = RiskManager(max_single=0.4, drawdown_stop=0.2)
    result = manager.apply({"A": 0.8, "B": 0.6}, {}, pl.Series("nav", [1.0, 1.1]))
    assert max(result.values()) <= 0.5


def test_strategy_runner_produces_target_portfolio() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="AAA", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB", exchange="SZ", industry="tech"),
        ]
    )
    strategy = CrossSectionalScreener({"factor_a": 1.0})
    runner = StrategyRunner.from_config(
        QoreConfig(), strategy, EqualWeightSizer(top_k=1)
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.2, 0.8],
        }
    ).lazy()
    result = runner.step(
        factor_lf,
        None,
        universe,
        date(2026, 4, 13),
        {},
        pl.Series("nav", [1.0, 1.02]),
        TradingCalendar.from_config(QoreConfig()),
    )
    assert result.date == date(2026, 4, 13)
    assert len(result.weights) == 1


def test_behavioral_gated_strategy_scales_base_signal() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="AAA", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB", exchange="SZ", industry="tech"),
        ]
    )
    strategy = BehavioralGatedStrategy(
        base=CrossSectionalScreener({"factor_a": 1.0}),
        regime_detector=StubRegimeDetector(0.8),
        min_scale=0.3,
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.2, 0.8],
            "realized_vol_20d": [0.5, 0.5],
        }
    ).lazy()

    signals = strategy.generate(
        factor_lf,
        universe,
        date(2026, 4, 13),
        TradingCalendar.from_config(QoreConfig()),
    )

    assert signals.to_list() == [0.10666666666666667, 0.4266666666666667]
