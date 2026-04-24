from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from qore_backtest import BacktestSettings, TradingCalendar
from qore_backtest.engine import BacktestEngine
from qore_backtest.metrics import compute_metrics
from qore_backtest.simulate import fill_order
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_data.universe import Universe
from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.workflow import fit_and_save_model
from qore_intelligence.strategy import build_ranking_strategy
from qore_runner import RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener
from qore_runner.strategies.ranking import RankingStrategy, WeightedOverlayCombiner
from qore_runner.strategy import StrategyContext


class StubScoreProvider:
    required_columns = frozenset({"factor_a"})

    def predict_scores(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        return factor_lf.select("symbol", pl.col("factor_a").alias("signal"))


def _stock_universe(symbols: list[str]) -> Universe:
    exchange = [
        "SH"
        if symbol.endswith(".SH")
        else "SZ"
        if symbol.endswith(".SZ")
        else "BJ"
        if symbol.endswith(".BJ")
        else "SH"
        for symbol in symbols
    ]
    return Universe.from_frame(
        pl.DataFrame(
            {
                "symbol": symbols,
                "exchange": exchange,
                "industry": ["test" for _ in symbols],
                "price_limit_pct": [0.10 for _ in symbols],
                "session": ["auction" for _ in symbols],
            }
        ),
        session_col="session",
    )


def _data_settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )


def _intelligence_settings(tmp_path: Path) -> IntelligenceSettings:
    return IntelligenceSettings(
        model_store_root=str(tmp_path / "models"),
        news_llm_daily_budget=50,
        news_llm_model="claude-sonnet-4-20250514",
        news_finbert_model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
        news_score_half_life_days=5,
    )


def test_fill_order_stock_fills_next_day() -> None:
    price_data = pl.DataFrame(
        {
            "date": [date(2026, 4, 13)],
            "open": [10.0],
            "is_suspended": [False],
            "limit_up": [False],
            "limit_down": [False],
        }
    )
    fill = fill_order(
        "600519.SH",
        "auction",
        date(2026, 4, 10),
        "buy",
        100.0,
        price_data,
        BacktestSettings(),
        TradingCalendar(),
    )
    assert fill.status == "filled"
    assert fill.fill_date == date(2026, 4, 13)


def test_backtest_engine_runs_end_to_end(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "factor_name": ["factor_a", "factor_a"],
                "raw_value": [0.1, 0.2],
                "z_score": [0.1, 0.9],
                "rank_pct": [0.5, 1.0],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "open": [10.0, 10.0],
                "high": [11.0, 12.0],
                "low": [9.0, 9.5],
                "close": [10.1, 11.0],
                "volume": [100, 120],
                "amount": [1000.0, 1200.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, False],
                "limit_up": [False, False],
                "limit_down": [False, False],
            }
        ),
    )
    universe = _stock_universe(["AAA.SH", "BBB.SZ"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_settings(
        BacktestSettings(),
        runner,
        store,
        TradingCalendar(),
    )
    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))
    metrics = compute_metrics(result)
    assert result.nav.height == 1
    assert result.fills.height == 1
    assert result.diagnostics.height == 1
    assert "annualized_return" in metrics
    assert "win_rate" in metrics


def test_backtest_engine_passes_signal_overlays_to_runner(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "factor_name": ["factor_a", "factor_a"],
                "raw_value": [0.1, 0.2],
                "z_score": [0.1, 0.2],
                "rank_pct": [0.5, 1.0],
            }
        ),
    )
    store.write(
        "news_scores",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "score": [1.0, -1.0],
                "event_type": ["earnings", "regulatory"],
                "source_layer": ["llm", "llm"],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "open": [10.0, 10.0],
                "high": [11.0, 12.0],
                "low": [9.0, 9.5],
                "close": [10.1, 11.0],
                "volume": [100, 120],
                "amount": [1000.0, 1200.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, False],
                "limit_up": [False, False],
                "limit_down": [False, False],
            }
        ),
    )
    universe = _stock_universe(["AAA.SH", "BBB.SZ"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        RankingStrategy(
            score_provider=StubScoreProvider(),
            combiner=WeightedOverlayCombiner(alpha=0.8),
        ),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_settings(
        BacktestSettings(),
        runner,
        store,
        TradingCalendar(),
    )

    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))

    assert result.positions.filter(pl.col("date") == date(2026, 4, 13)).get_column(
        "symbol"
    ).to_list() == ["AAA.SH"]
    assert result.diagnostics.get_column("fill_request_count").item() == 1
    assert result.diagnostics.get_column("rejected_count").item() == 1


def test_backtest_engine_runs_with_saved_model_registry_artifact(
    tmp_path: Path,
) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    fit_and_save_model(
        intelligence_settings=_intelligence_settings(tmp_path),
        model_name="stock_ranker",
        factor_lf=pl.DataFrame(
            {
                "date": [
                    date(2026, 1, 1),
                    date(2026, 1, 1),
                    date(2026, 1, 2),
                    date(2026, 1, 2),
                    date(2026, 1, 3),
                    date(2026, 1, 3),
                    date(2026, 1, 4),
                    date(2026, 1, 4),
                    date(2026, 1, 5),
                    date(2026, 1, 5),
                    date(2026, 1, 6),
                    date(2026, 1, 6),
                ],
                "symbol": ["AAA.SH", "BBB.SZ"] * 6,
                "factor_a": [
                    0.1,
                    0.2,
                    0.2,
                    0.1,
                    0.3,
                    0.1,
                    0.4,
                    0.2,
                    0.5,
                    0.2,
                    0.6,
                    0.3,
                ],
                "forward_return_1d": [
                    0.01,
                    0.02,
                    0.02,
                    0.01,
                    0.03,
                    0.01,
                    0.04,
                    0.02,
                    0.05,
                    0.02,
                    0.06,
                    0.03,
                ],
            }
        ).lazy(),
        store=store,
        version="integration",
        model=MultiHorizonRanker(horizons=[1], weights={"1d": 1.0}),
    )
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "factor_name": ["factor_a", "factor_a"],
                "raw_value": [0.1, 0.9],
                "z_score": [0.1, 0.9],
                "rank_pct": [0.5, 1.0],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "open": [10.0, 10.0],
                "high": [10.5, 11.0],
                "low": [9.8, 9.9],
                "close": [10.1, 11.0],
                "volume": [100, 120],
                "amount": [1000.0, 1200.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, False],
                "limit_up": [False, False],
                "limit_down": [False, False],
            }
        ),
    )
    universe = _stock_universe(["AAA.SH", "BBB.SZ"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        build_ranking_strategy(_intelligence_settings(tmp_path)),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_settings(
        BacktestSettings(),
        runner,
        store,
        TradingCalendar(),
    )

    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))

    assert result.positions.filter(pl.col("date") == date(2026, 4, 13)).get_column(
        "symbol"
    ).to_list() == ["BBB.SZ"]


def test_backtest_engine_enforces_force_exit_overlay_liquidation(
    tmp_path: Path,
) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [
                    date(2026, 4, 13),
                    date(2026, 4, 13),
                    date(2026, 4, 14),
                    date(2026, 4, 14),
                ],
                "symbol": ["AAA.SH", "BBB.SZ", "AAA.SH", "BBB.SZ"],
                "factor_name": ["factor_a", "factor_a", "factor_a", "factor_a"],
                "raw_value": [0.9, 0.1, 0.9, 0.1],
                "z_score": [0.9, 0.1, 0.9, 0.1],
                "rank_pct": [1.0, 0.5, 1.0, 0.5],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [
                    date(2026, 4, 13),
                    date(2026, 4, 13),
                    date(2026, 4, 14),
                    date(2026, 4, 14),
                    date(2026, 4, 15),
                    date(2026, 4, 15),
                ],
                "symbol": [
                    "AAA.SH",
                    "BBB.SZ",
                    "AAA.SH",
                    "BBB.SZ",
                    "AAA.SH",
                    "BBB.SZ",
                ],
                "open": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "high": [10.5, 10.2, 10.5, 10.2, 10.5, 10.2],
                "low": [9.8, 9.8, 9.8, 9.8, 9.8, 9.8],
                "close": [10.1, 10.0, 10.1, 10.0, 10.1, 10.0],
                "volume": [100, 100, 100, 100, 100, 100],
                "amount": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
                "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "is_suspended": [False, False, False, False, False, False],
                "limit_up": [False, False, False, False, False, False],
                "limit_down": [False, False, False, False, False, False],
            }
        ),
    )
    universe = _stock_universe(["AAA.SH", "BBB.SZ"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_settings(
        BacktestSettings(),
        runner,
        store,
        TradingCalendar(),
        decision_overlays_by_day={
            date(2026, 4, 14): pl.DataFrame(
                {
                    "symbol": ["AAA.SH", "BBB.SZ"],
                    "selected": [False, True],
                    "exclude_reason": ["force_exit:audit", None],
                }
            )
        },
    )

    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 14))

    assert result.diagnostics.get_column("force_exit_count").to_list() == [0, 1]
    assert result.diagnostics.get_column("decision_non_selected_count").to_list() == [
        1,
        1,
    ]
    assert result.diagnostics.get_column("forced_liquidation_symbols").to_list() == [
        "",
        "AAA.SH",
    ]
    assert result.diagnostics.get_column("decision_selected_count").to_list() == [1, 1]
    assert result.diagnostics.get_column("decision_new_symbols").to_list() == [
        "",
        "",
    ]
    assert result.diagnostics.get_column("decision_dropped_symbols").to_list() == [
        "",
        "",
    ]
    assert (
        result.positions.filter(
            (pl.col("date") == date(2026, 4, 14)) & (pl.col("symbol") == "AAA.SH")
        )
        .get_column("weight")
        .item()
        == 0.0
    )


def test_strategy_runner_exposes_diagnostics() -> None:
    universe = _stock_universe(["AAA", "BBB"])
    runner = StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    factor_lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "factor_a": [0.1, 0.9],
        }
    ).lazy()

    result = runner.step(
        StrategyContext(
            factor_lf=factor_lf,
            universe=universe,
            date=date(2026, 4, 13),
            calendar=TradingCalendar(),
        ),
        pl.Series("nav", [1.0]),
    )

    assert result.diagnostics.candidate_count == 2
    assert result.diagnostics.signal_count == 2
    assert result.diagnostics.selected_count == 1
    assert result.diagnostics.non_selected_count == 1
    assert result.diagnostics.drawdown_blocked is False
