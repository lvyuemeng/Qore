from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from qore_backtest.engine import BacktestEngine
from qore_backtest.metrics import compute_metrics
from qore_backtest.simulate import fill_order
from qore_core import QoreConfig, StockInstrument, TradingCalendar, Universe
from qore_data.store.duckdb import QoreStore
from qore_intelligence.combine import SignalCombiner
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.workflow import fit_and_save_model
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener
from qore_runner.strategies.ranking import RankingStrategy


class StubPipeline:
    def predict_score(self, factor_lf: pl.LazyFrame) -> pl.Series:
        df = factor_lf.collect()
        return pl.Series(name="score", values=df.get_column("factor_a").to_list())


def test_fill_order_stock_fills_next_day() -> None:
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
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
        inst,
        date(2026, 4, 10),
        "buy",
        100.0,
        price_data,
        QoreConfig().backtest,
        TradingCalendar.from_config(QoreConfig()),
    )
    assert fill.status == "filled"
    assert fill.fill_date == date(2026, 4, 13)


def test_backtest_engine_runs_end_to_end(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
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
    universe = Universe(
        [
            StockInstrument(symbol="AAA.SH", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB.SZ", exchange="SZ", industry="tech"),
        ]
    )
    runner = StrategyRunner.from_config(
        config,
        CrossSectionalScreener({"factor_a": 1.0}),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_config(
        config,
        runner,
        store,
        TradingCalendar.from_config(config),
    )
    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))
    metrics = compute_metrics(result)
    assert result.nav.height == 1
    assert len(result.fills) == 1
    assert "annualized_return" in metrics
    assert "win_rate" in metrics


def test_backtest_engine_passes_news_scores_to_runner(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
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
    universe = Universe(
        [
            StockInstrument(symbol="AAA.SH", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB.SZ", exchange="SZ", industry="tech"),
        ]
    )
    runner = StrategyRunner.from_config(
        config,
        RankingStrategy(
            pipeline=StubPipeline(), combiner=SignalCombiner(news_alpha=0.8)
        ),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_config(
        config,
        runner,
        store,
        TradingCalendar.from_config(config),
    )

    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))

    assert list(result.positions[0]) == ["AAA.SH"]


def test_backtest_engine_runs_with_saved_model_registry_artifact(
    tmp_path: Path,
) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            },
            "intelligence": {"model_store_root": str(tmp_path / "models")},
        }
    )
    store = QoreStore.from_config(config)
    fit_and_save_model(
        config=config,
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
    universe = Universe(
        [
            StockInstrument(symbol="AAA.SH", exchange="SH", industry="bank"),
            StockInstrument(symbol="BBB.SZ", exchange="SZ", industry="tech"),
        ]
    )
    runner = StrategyRunner.from_config(
        config,
        RankingStrategy.from_config(config),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_config(
        config,
        runner,
        store,
        TradingCalendar.from_config(config),
    )

    result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))

    assert list(result.positions[0]) == ["BBB.SZ"]
