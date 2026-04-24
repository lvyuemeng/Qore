from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_data.universe import Universe
from qore_runner import RunnerSettings, TradingCalendar
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer, VolScaledSizer
from qore_runner.strategies.behavioral import BehavioralGatedStrategy
from qore_runner.strategies.crosssectional import CrossSectionalScreener
from qore_runner.strategies.ranking import RankingStrategy, WeightedOverlayCombiner
from qore_runner.strategy import (
    StrategyContext,
    StrategyProviderFrames,
    StrategySelectionSpec,
)


def _frame_universe(symbols: list[str]) -> Universe:
    return Universe.from_frame(
        pl.DataFrame({"symbol": symbols, "is_tradeable": [True for _ in symbols]}),
        symbol_col="symbol",
        tradeable_col="is_tradeable",
        suspended_col=None,
        session_marker="auction",
    )


def _weights_dict(frame: pl.DataFrame) -> dict[str, float]:
    if frame.is_empty():
        return {}
    return {str(symbol): float(weight) for symbol, weight in frame.iter_rows()}


class StubScoreProvider:
    required_columns = frozenset({"model_score"})

    def predict_scores(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        return factor_lf.select("symbol", pl.col("model_score").alias("signal"))


class StubRegimeDetector:
    def __init__(self, scale_value: float) -> None:
        self.scale_value = scale_value

    def scale(self, lf: pl.LazyFrame, as_of: date) -> float:
        del lf, as_of
        return self.scale_value


def test_strategy_runner_produces_target_portfolio() -> None:
    universe = _frame_universe(["AAA", "BBB"])
    strategy = CrossSectionalScreener({"factor_a": 1.0})
    runner = StrategyRunner.from_settings(
        RunnerSettings(), strategy, EqualWeightSizer(top_k=1)
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.2, 0.8],
        }
    ).lazy()
    result = runner.step(
        StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
        ),
        pl.Series("nav", [1.0, 1.02]),
    )
    assert result.date == date(2026, 4, 13)
    assert result.weights_frame.height == 1


def test_behavioral_gated_strategy_scales_base_signal() -> None:
    universe = _frame_universe(["AAA", "BBB"])
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

    signals = pl.DataFrame(
        strategy.generate(
            StrategyContext(
                factor_lf=factor_lf,
                universe=universe,
                date=date(2026, 4, 13),
                calendar=TradingCalendar(),
            )
        ).collect()
    )

    assert signals.get_column("signal").to_list() == [
        0.10666666666666667,
        0.4266666666666667,
    ]


def test_ranking_strategy_blends_overlay_scores() -> None:
    universe = _frame_universe(["AAA", "BBB"])
    strategy = RankingStrategy(
        score_provider=StubScoreProvider(),
        combiner=WeightedOverlayCombiner(alpha=0.25),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "model_score": [0.4, 0.2],
        }
    ).lazy()

    signals = pl.DataFrame(
        strategy.generate(
            StrategyContext(
                factor_lf=factor_lf,
                universe=universe,
                date=date(2026, 4, 13),
                calendar=TradingCalendar(),
                providers=StrategyProviderFrames(
                    signal_overlay=pl.DataFrame(
                        {
                            "symbol": ["AAA", "BBB"],
                            "overlay": [1.0, -0.2],
                        }
                    )
                ),
            )
        ).collect()
    )

    assert signals.get_column("signal").to_list() == pytest.approx([0.55, 0.1])


def test_vol_scaled_sizer_uses_inverse_volatility_weights() -> None:
    sizer = VolScaledSizer(top_k=3, max_weight=0.6).with_volatility(
        {"AAA": 0.2, "BBB": 0.4, "CCC": 0.8}
    )

    weights_frame = sizer.size(
        pl.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "signal": [0.9, 0.8, 0.7],
                "realized_vol_20d": [0.2, 0.4, 0.8],
            }
        )
    )

    weights = _weights_dict(weights_frame)
    assert list(weights) == ["AAA", "BBB", "CCC"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["AAA"] > weights["BBB"] > weights["CCC"]


def test_vol_scaled_sizer_caps_single_name_weight() -> None:
    sizer = VolScaledSizer(top_k=2, max_weight=0.55).with_volatility(
        {"AAA": 0.05, "BBB": 1.0}
    )

    weights_frame = sizer.size(
        pl.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "signal": [0.9, 0.8],
                "realized_vol_20d": [0.05, 1.0],
            }
        )
    )

    weights = _weights_dict(weights_frame)
    assert weights["AAA"] == pytest.approx(0.55)
    assert weights["BBB"] == pytest.approx(0.45)


def test_strategy_runner_joins_volatility_for_vol_scaled_sizer() -> None:
    universe = _frame_universe(["AAA", "BBB"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(max_single=0.7),
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
        StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
        ),
        pl.Series("nav", [1.0, 1.02]),
    )
    weights = _weights_dict(result.weights_frame)
    assert weights["AAA"] > weights["BBB"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_strategy_runner_applies_decision_overlay_and_force_exit() -> None:
    universe = _frame_universe(["AAA", "BBB", "CCC"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=2),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "factor_a": [0.8, 0.7, 0.6],
        }
    ).lazy()

    result = runner.step(
        StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
            providers=StrategyProviderFrames(
                decision_overlay=pl.DataFrame(
                    {
                        "symbol": ["AAA", "BBB", "CCC"],
                        "selected": [True, False, True],
                        "exclude_reason": [None, "force_exit:audit", None],
                    }
                )
            ),
        ),
        pl.Series("nav", [1.0]),
    )
    assert set(result.weights_frame.get_column("symbol").to_list()) == {"AAA", "CCC"}
    assert result.decision.force_exit_symbols == frozenset({"BBB"})
    bbb_reason = (
        result.decision.frame.filter(pl.col("symbol") == "BBB")
        .get_column("exclude_reason")
        .item()
    )
    assert bbb_reason == "force_exit:audit"
    assert result.diagnostics.non_selected_count == 1


def test_strategy_runner_selection_spec_limits_ranked_symbols() -> None:
    universe = _frame_universe(["AAA", "BBB", "CCC"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=2),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "factor_a": [0.9, 0.8, 0.1],
        }
    ).lazy()

    result = runner.step(
        StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
            selection=StrategySelectionSpec(top_k=1),
        ),
        pl.Series("nav", [1.0]),
    )
    assert set(result.weights_frame.get_column("symbol").to_list()) == {"AAA"}
    assert result.diagnostics.candidate_count == 3


def test_strategy_runner_reuses_monthly_decision_within_same_month() -> None:
    universe = _frame_universe(["AAA", "BBB"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    first_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.9, 0.1],
        }
    ).lazy()
    second_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.1, 0.9],
        }
    ).lazy()

    first = runner.step(
        StrategyContext(
            factor_lf=first_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
            selection=StrategySelectionSpec(top_k=1),
        ),
        pl.Series("nav", [1.0]),
    )
    second = runner.step(
        StrategyContext(
            factor_lf=second_lf,
            universe=universe,
            date=date(2026, 4, 20),
            calendar=TradingCalendar(),
            selection=StrategySelectionSpec(top_k=1),
        ),
        pl.Series("nav", [1.0]),
    )
    assert set(first.weights_frame.get_column("symbol").to_list()) == {"AAA"}
    assert set(second.weights_frame.get_column("symbol").to_list()) == {"AAA"}


def test_strategy_runner_supports_frame_wrapped_universe_without_objects() -> None:
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.9, 0.1],
        }
    ).lazy()
    frame_universe = _frame_universe(["AAA", "BBB"])

    result = runner.step(
        StrategyContext(
            factor_lf=factor_lf,
            universe=frame_universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
            selection=StrategySelectionSpec(top_k=1),
        ),
        pl.Series("nav", [1.0]),
    )
    assert set(result.weights_frame.get_column("symbol").to_list()) == {"AAA"}


def test_strategy_runner_drawdown_sets_exclude_reason() -> None:
    universe = _frame_universe(["AAA", "BBB"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(drawdown_stop=0.1),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    factor_lf = pl.DataFrame({"symbol": ["AAA", "BBB"], "factor_a": [0.9, 0.8]}).lazy()

    result = runner.step(
        StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
            selection=StrategySelectionSpec(top_k=1),
        ),
        pl.Series("nav", [1.0, 0.85]),
    )
    assert result.weights_frame.is_empty()
    assert result.diagnostics.drawdown_blocked is True
    assert (
        result.decision.frame.filter(pl.col("symbol") == "AAA")
        .get_column("exclude_reason")
        .item()
        == "risk_drawdown_stop"
    )
