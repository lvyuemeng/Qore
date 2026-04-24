from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_backtest import BacktestSettings, TradingCalendar
from qore_backtest.engine import BacktestEngine
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_data.universe import Universe
from qore_runner import RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener


def _data_settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )


def _stock_universe(symbols: list[str]) -> Universe:
    exchange = ["SH" if symbol.endswith(".SH") else "SZ" for symbol in symbols]
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


def _seed_factor_and_market_data(store: QoreStore) -> None:
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
                "symbol": ["AAA.SH", "BBB.SZ", "AAA.SH", "BBB.SZ", "AAA.SH", "BBB.SZ"],
                "open": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "nav": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
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


def test_backtest_result_consistency_on_force_exit_transition(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    _seed_factor_and_market_data(store)
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

    result = engine.run(
        _stock_universe(["AAA.SH", "BBB.SZ"]),
        date(2026, 4, 13),
        date(2026, 4, 14),
    )

    diagnostics = result.diagnostics.sort("date")
    assert result.nav.height == diagnostics.height == 2
    assert result.turnover.height == diagnostics.height
    position_dates = (
        result.positions.select("date")
        .unique()
        .sort("date")
        .get_column("date")
        .to_list()
    )
    diagnostic_dates = diagnostics.get_column("date").to_list()
    assert all(position_day in diagnostic_dates for position_day in position_dates)

    for (
        request_count,
        filled_count,
        pending_count,
        rejected_count,
    ) in diagnostics.select(
        "fill_request_count", "filled_count", "pending_count", "rejected_count"
    ).iter_rows():
        assert request_count == filled_count + pending_count + rejected_count

    assert diagnostics.get_column("forced_liquidation_symbols").to_list() == [
        "",
        "AAA.SH",
    ]
    assert diagnostics.get_column("force_exit_count").to_list() == [0, 1]
    assert set(result.fills.get_column("status").to_list()) == {"filled"}

    filled_quantity_by_day = (
        result.fills.group_by("date")
        .agg(pl.col("quantity").sum().alias("filled_quantity"))
        .sort("date")
    )
    turnover_by_day = result.turnover.sort("date").select("date", "turnover")
    comparison = turnover_by_day.join(
        filled_quantity_by_day,
        on="date",
        how="left",
    ).with_columns(pl.col("filled_quantity").fill_null(0.0))
    for turnover, filled_quantity in comparison.select(
        "turnover", "filled_quantity"
    ).iter_rows():
        assert isinstance(turnover, float)
        assert isinstance(filled_quantity, float)
        assert turnover == pytest.approx(filled_quantity)

    for turnover, commission, risk_flag in result.turnover.select(
        "turnover", "commission", "risk_flag"
    ).iter_rows(named=False):
        assert isinstance(turnover, float)
        assert isinstance(commission, float)
        assert isinstance(risk_flag, bool)
