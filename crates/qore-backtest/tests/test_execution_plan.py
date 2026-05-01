from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from qore_backtest import (
    BacktestSettings,
    TradingCalendar,
)
from qore_backtest.calendar import TradingCalendar
from qore_backtest.engine import BacktestEngine
from qore_data import DataSettings
from qore_data.fetch import MarketSource as StoreMarketDataSource
from qore_data.store.duckdb import QoreStore
from qore_runner import RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer


def _data_settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "qore.duckdb"), parquet_root=str(tmp_path / "raw")
    )


def _engine(tmp_path: Path) -> BacktestEngine:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    return BacktestEngine.from_settings(
        BacktestSettings(buy_delay=0, sell_delay=0),
        StrategyRunner.from_settings(RunnerSettings(), EqualWeightSizer(top_k=1)),
        TradingCalendar(),
        signal_source=_StoreFactorSource(store=store),
        market_data_source=StoreMarketDataSource(store=store),
    )


class _StoreFactorSource:
    def __init__(self, store: QoreStore, dataset: str = "strategy_factors") -> None:
        self._store = store
        self._dataset = dataset

    def frame_for_day(self, trading_day: date) -> pl.DataFrame | None:
        lf = self._store.read(
            self._dataset, filters={"date": trading_day}, backend="duckdb"
        )
        frame = pl.DataFrame(lf.collect())
        return frame if not frame.is_empty() else None


def test_execution_plan_uses_default_delays(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    requests = pl.DataFrame(
        {
            "symbol": ["AAA.SH", "AAA.SH"],
            "direction": ["buy", "sell"],
            "quantity": [0.1, 0.1],
        }
    )
    plan = TradingCalendar().fill_plan(requests, date(2026, 4, 13), 1, 2)
    dates = {
        (r, d): fd
        for r, d, fd in plan.select("symbol", "direction", "fill_date").iter_rows()
    }
    assert dates[("AAA.SH", "buy")] == date(2026, 4, 14)
    assert dates[("AAA.SH", "sell")] == date(2026, 4, 15)


def test_execution_plan_uses_strategy_schedule_delays(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    engine = BacktestEngine.from_settings(
        BacktestSettings(buy_delay=2, sell_delay=3),
        StrategyRunner.from_settings(RunnerSettings(), EqualWeightSizer(top_k=1)),
        TradingCalendar(),
        signal_source=_StoreFactorSource(store=store),
        market_data_source=StoreMarketDataSource(store=store),
    )
    requests = pl.DataFrame(
        {
            "symbol": ["AAA.SH", "AAA.SH"],
            "direction": ["buy", "sell"],
            "quantity": [0.1, 0.1],
        }
    )
    TradingCalendar().fill_plan(requests, date(2026, 4, 13), 2, 3)
