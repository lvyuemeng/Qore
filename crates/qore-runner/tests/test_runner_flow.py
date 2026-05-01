from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_runner import (
    DECISION_SIGNAL_SCHEMA,
    EqualWeightSizer,
    VolScaledSizer,
    execution_signals,
    rank_symbols,
)
from qore_runner.strategies.behavioral import BehavioralGatedStrategy
from qore_runner.strategies.crosssectional import CrossSectionalScreener
from qore_runner.strategies.ranking import RankingStrategy

D = date(2026, 4, 13)


def _signals(strategy, factor_data):
    lf = pl.DataFrame(factor_data).lazy()
    return pl.DataFrame(strategy.generate(lf).collect())


# -- rank_symbols ---------------------------------------------------------------


def test_rank_symbols_returns_top_k() -> None:
    signals = pl.DataFrame(
        {"symbol": ["AAA", "BBB", "CCC", "DDD"], "signal": [0.9, 0.8, 0.7, 0.6]}
    )
    result = rank_symbols(signals, top_k=3)
    assert result == ["AAA", "BBB", "CCC"]


def test_rank_symbols_filters_non_finite() -> None:
    signals = pl.DataFrame(
        {"symbol": ["AAA", "BBB", "CCC"], "signal": [0.9, float("nan"), 0.7]}
    )
    result = rank_symbols(signals, top_k=10)
    assert result == ["AAA", "CCC"]


def test_rank_symbols_custom_score_column() -> None:
    signals = pl.DataFrame({"symbol": ["AAA", "BBB"], "score": [0.2, 0.8]})
    result = rank_symbols(signals, score_column="score", descending=True)
    assert result == ["BBB", "AAA"]


def test_rank_symbols_ascending() -> None:
    signals = pl.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "signal": [0.9, 0.8, 0.7]})
    result = rank_symbols(signals, descending=False)
    assert result == ["CCC", "BBB", "AAA"]


def test_rank_symbols_empty_when_no_signals() -> None:
    assert rank_symbols(pl.DataFrame()) == []


# -- EqualWeightSizer ----------------------------------------------------------


def test_equal_weight_sizer_distributes_equally() -> None:
    signals = pl.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "signal": [0.9, 0.8, 0.7]})
    sizer = EqualWeightSizer(max_weight=0.5)
    weights = sizer.size(signals)
    # size() assigns min(1/n, max_weight) = min(0.333, 0.5) = 0.333
    assert weights.get_column("weight").to_list() == pytest.approx(
        [1.0 / 3, 1.0 / 3, 1.0 / 3]
    )


# -- VolScaledSizer ------------------------------------------------------------


def test_vol_scaled_sizer_uses_inverse_volatility_weights() -> None:
    sizer = VolScaledSizer(max_weight=0.6, vol_col="realized_vol_20d").with_volatility(
        {"AAA": 0.2, "BBB": 0.4, "CCC": 0.8}
    )
    wf = sizer.size(
        pl.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "signal": [0.9, 0.8, 0.7],
                "realized_vol_20d": [0.2, 0.4, 0.8],
            }
        )
    )
    w = dict(
        zip(
            wf.get_column("symbol").to_list(),
            wf.get_column("weight").to_list(),
            strict=False,
        )
    )
    assert list(w) == ["AAA", "BBB", "CCC"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["AAA"] > w["BBB"] > w["CCC"]


def test_vol_scaled_sizer_caps_single_name_weight() -> None:
    sizer = VolScaledSizer(max_weight=0.55, vol_col="realized_vol_20d").with_volatility(
        {"AAA": 0.05, "BBB": 1.0}
    )
    wf = sizer.size(
        pl.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "signal": [0.9, 0.8],
                "realized_vol_20d": [0.05, 1.0],
            }
        )
    )
    w = dict(
        zip(
            wf.get_column("symbol").to_list(),
            wf.get_column("weight").to_list(),
            strict=False,
        )
    )
    assert len(w) == 2
    assert all(ww <= 0.55 for ww in w.values())


# -- execution_signals ---------------------------------------------------------


def test_execution_signals_buy_on_new_targets() -> None:
    target = pl.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]})
    current = pl.DataFrame(schema={"symbol": pl.String, "weight": pl.Float64})
    result = execution_signals(target_weights=target, current_weights=current, as_of=D)
    assert list(result.columns) == list(DECISION_SIGNAL_SCHEMA)
    assert result.get_column("signal").to_list() == ["buy", "buy"]


def test_execution_signals_sell_when_weight_drops() -> None:
    target = pl.DataFrame({"symbol": ["AAA"], "weight": [0.0]})
    current = pl.DataFrame({"symbol": ["AAA"], "weight": [0.5]})
    result = execution_signals(target_weights=target, current_weights=current, as_of=D)
    assert result.get_column("signal").to_list() == ["sell"]


def test_execution_signals_hold_unchanged() -> None:
    target = pl.DataFrame({"symbol": ["AAA"], "weight": [0.3]})
    current = pl.DataFrame({"symbol": ["AAA"], "weight": [0.3]})
    result = execution_signals(target_weights=target, current_weights=current, as_of=D)
    assert result.get_column("signal").to_list() == ["hold"]


# -- behavioral strategy -------------------------------------------------------


class StubRegimeDetector:
    def scale(self, lf) -> float:
        return 0.8


def test_behavioral_gated_strategy_scales_base_signal() -> None:
    strategy = BehavioralGatedStrategy(
        base=CrossSectionalScreener({"factor_a": 1.0}),
        regime_detector=StubRegimeDetector(),
        min_scale=0.3,
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.2, 0.8],
            "realized_vol_20d": [0.5, 0.5],
        }
    ).lazy()
    signals = pl.DataFrame(strategy.generate(factor_lf).collect())
    assert signals.get_column("signal").to_list() == pytest.approx([0.16, 0.64])


# -- ranking strategy ----------------------------------------------------------


class StubScoreProvider:
    required_columns = frozenset({"model_score"})

    def predict_scores(self, lf):
        return lf.select("symbol", pl.col("model_score").alias("signal"))


def test_ranking_strategy_passes_through_scores() -> None:
    strategy = RankingStrategy(score_provider=StubScoreProvider())
    factor_lf = pl.DataFrame(
        {"symbol": ["AAA", "BBB"], "model_score": [0.4, 0.2]}
    ).lazy()
    signals = pl.DataFrame(strategy.generate(factor_lf).collect())
    assert signals.get_column("signal").to_list() == pytest.approx([0.4, 0.2])
