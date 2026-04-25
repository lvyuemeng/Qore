from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_backtest import (
    BacktestSettings,
    NullSignalOverlaySource,
    StoreFactorSource,
    StoreMarketDataSource,
    TradingCalendar,
)
from qore_backtest.engine import BacktestEngine
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_data.universe import Universe
from qore_runner import RebalanceSchedule, RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener


def _data_settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )


def _engine(tmp_path: Path) -> BacktestEngine:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    return BacktestEngine.from_settings(
        BacktestSettings(),
        StrategyRunner.from_settings(
            RunnerSettings(),
            CrossSectionalScreener({"factor_a": 1.0}),
            EqualWeightSizer(top_k=1),
        ),
        TradingCalendar(),
        factor_source=StoreFactorSource(store=store),
        market_data_source=StoreMarketDataSource(store=store),
        signal_overlay_source=NullSignalOverlaySource(),
    )


def test_execution_plan_preserves_daily_delay_parity(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    universe = Universe.from_frame(
        pl.DataFrame(
            {
                "symbol": ["AAA.SH", "FUND.NAV", "FUT.CONT"],
                "session": ["auction", "nav", "continuous"],
                "subscription_delay": [None, 3, None],
                "redemption_delay": [None, 4, None],
            }
        ),
        session_col="session",
    )
    requests = pl.DataFrame(
        {
            "symbol": ["AAA.SH", "AAA.SH", "FUND.NAV", "FUND.NAV", "FUT.CONT"],
            "direction": ["buy", "sell", "buy", "sell", "buy"],
            "quantity": [0.1, 0.1, 0.2, 0.2, 0.3],
        }
    )

    plan = engine._execution_plan(
        requests,
        date(2026, 4, 13),
        universe.execution_metadata(),
    )
    fill_dates = {
        (symbol, direction): fill_date
        for symbol, direction, fill_date in plan.select(
            "symbol", "direction", "fill_date"
        ).iter_rows()
    }

    assert fill_dates[("AAA.SH", "buy")] == date(2026, 4, 14)
    assert fill_dates[("AAA.SH", "sell")] == date(2026, 4, 14)
    assert fill_dates[("FUND.NAV", "buy")] == date(2026, 4, 16)
    assert fill_dates[("FUND.NAV", "sell")] == date(2026, 4, 17)
    assert fill_dates[("FUT.CONT", "buy")] == date(2026, 4, 13)


def test_execution_plan_raises_for_missing_execution_metadata(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    requests = pl.DataFrame(
        {
            "symbol": ["AAA.SH", "MISSING.SZ"],
            "direction": ["buy", "sell"],
            "quantity": [0.5, 0.5],
        }
    )
    execution_metadata = pl.DataFrame(
        {
            "symbol": ["AAA.SH"],
            "session": ["auction"],
            "dataset": ["stock_ohlcv"],
            "buy_delay": [1],
            "sell_delay": [1],
        }
    )

    with pytest.raises(ValueError, match="MISSING\\.SZ"):
        engine._execution_plan(requests, date(2026, 4, 13), execution_metadata)


def test_execution_plan_uses_strategy_schedule_delay_defaults(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    engine = BacktestEngine.from_settings(
        BacktestSettings(),
        StrategyRunner.from_settings(
            RunnerSettings(),
            CrossSectionalScreener(
                {"factor_a": 1.0},
                rebalance_schedule=RebalanceSchedule(
                    frequency="daily",
                    buy_delay=2,
                    sell_delay=3,
                ),
            ),
            EqualWeightSizer(top_k=1),
        ),
        TradingCalendar(),
        factor_source=StoreFactorSource(store=store),
        market_data_source=StoreMarketDataSource(store=store),
        signal_overlay_source=NullSignalOverlaySource(),
    )
    requests = pl.DataFrame(
        {
            "symbol": ["AAA.SH", "AAA.SH"],
            "direction": ["buy", "sell"],
            "quantity": [0.1, 0.1],
        }
    )
    execution_metadata = pl.DataFrame(
        {
            "symbol": ["AAA.SH"],
            "session": ["auction"],
            "dataset": ["stock_ohlcv"],
            "buy_delay": [None],
            "sell_delay": [None],
        }
    )

    plan = engine._execution_plan(requests, date(2026, 4, 13), execution_metadata)
    fill_dates = {
        (symbol, direction): fill_date
        for symbol, direction, fill_date in plan.select(
            "symbol", "direction", "fill_date"
        ).iter_rows()
    }

    assert fill_dates[("AAA.SH", "buy")] == date(2026, 4, 15)
    assert fill_dates[("AAA.SH", "sell")] == date(2026, 4, 16)
