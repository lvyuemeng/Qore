from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_core import QoreConfig, StockInstrument, TradingCalendar, Universe
from qore_intelligence.combine import SignalCombiner
from qore_runner.risk import RiskManager
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer, VolScaledSizer
from qore_runner.strategies.behavioral import BehavioralGatedStrategy
from qore_runner.strategies.crosssectional import CrossSectionalScreener
from qore_runner.strategies.ranking import RankingStrategy


class StubPipeline:
    def predict_score(self, factor_lf: pl.LazyFrame) -> pl.Series:
        df = factor_lf.collect()
        return pl.Series(name="score", values=df.get_column("model_score").to_list())


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
        None,
        universe,
        date(2026, 4, 13),
        TradingCalendar.from_config(QoreConfig()),
    )

    assert signals.to_list() == [0.10666666666666667, 0.4266666666666667]


def test_ranking_strategy_blends_news_scores() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="AAA", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB", exchange="SZ", industry="tech"),
        ]
    )
    strategy = RankingStrategy(
        pipeline=StubPipeline(),
        combiner=SignalCombiner(news_alpha=0.25),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "model_score": [0.4, 0.2],
        }
    ).lazy()

    signals = strategy.generate(
        factor_lf,
        {"AAA": 1.0, "BBB": -0.2},
        universe,
        date(2026, 4, 13),
        TradingCalendar.from_config(QoreConfig()),
    )

    assert signals.to_list() == pytest.approx([0.55, 0.1])


def test_vol_scaled_sizer_uses_inverse_volatility_weights() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="AAA", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB", exchange="SZ", industry="tech"),
            StockInstrument(symbol="CCC", exchange="SZ", industry="utility"),
        ]
    )
    sizer = VolScaledSizer(top_k=3, max_weight=0.6).with_volatility(
        {"AAA": 0.2, "BBB": 0.4, "CCC": 0.8}
    )

    weights = sizer.size(pl.Series("signal", [0.9, 0.8, 0.7]), universe)

    assert list(weights) == ["AAA", "BBB", "CCC"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["AAA"] > weights["BBB"] > weights["CCC"]


def test_vol_scaled_sizer_caps_single_name_weight() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="AAA", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB", exchange="SZ", industry="tech"),
        ]
    )
    sizer = VolScaledSizer(top_k=2, max_weight=0.55).with_volatility(
        {"AAA": 0.05, "BBB": 1.0}
    )

    weights = sizer.size(pl.Series("signal", [0.9, 0.8]), universe)

    assert weights["AAA"] == pytest.approx(0.55)
    assert weights["BBB"] == pytest.approx(0.45)


def test_strategy_runner_injects_volatility_into_vol_scaled_sizer() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="AAA", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB", exchange="SZ", industry="tech"),
        ]
    )
    runner = StrategyRunner.from_config(
        QoreConfig.model_validate({"stock": {"max_weight": 0.7}}),
        CrossSectionalScreener({"factor_a": 1.0}),
        VolScaledSizer(top_k=2, max_weight=0.7),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.9, 0.8],
            "realized_vol_20d": [0.2, 0.4],
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

    assert result.weights["AAA"] > result.weights["BBB"]
    assert sum(result.weights.values()) == pytest.approx(1.0)
