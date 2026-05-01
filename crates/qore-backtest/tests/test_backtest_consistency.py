from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from qore_backtest import BacktestSettings, MappingDayFrameSource, TradingCalendar
from qore_backtest.engine import BacktestEngine
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_runner import RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer


@dataclass(frozen=True, slots=True)
class StoreMarketDataSource:
    store: QoreStore

    def frame_for_day(self, trading_day: date) -> pl.DataFrame | None:
        frame = pl.DataFrame(
            self.store.read(
                "stock_ohlcv", filters={"date": trading_day}, backend="duckdb"
            ).collect()
        )
        return frame if not frame.is_empty() else None


def _data_settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "qore.duckdb"), parquet_root=str(tmp_path / "raw")
    )


def _seed_market_data(store: QoreStore) -> None:
    d13, d14, d15 = date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15)
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [d13, d13, d14, d14, d15, d15],
                "symbol": ["AAA.SH", "BBB.SZ", "AAA.SH", "BBB.SZ", "AAA.SH", "BBB.SZ"],
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


def test_backtest_result_consistency_on_force_exit_transition(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_data_settings(tmp_path))
    _seed_market_data(store)
    d13, d14 = date(2026, 4, 13), date(2026, 4, 14)
    factor_by_day = {
        d13: pl.DataFrame({"symbol": ["AAA.SH", "BBB.SZ"], "factor_a": [0.9, 0.1]}),
        d14: pl.DataFrame({"symbol": ["AAA.SH", "BBB.SZ"], "factor_a": [0.9, 0.1]}),
    }
    runner = StrategyRunner.from_settings(RunnerSettings(), EqualWeightSizer(top_k=1))
    engine = BacktestEngine.from_settings(
        BacktestSettings(
            buy_delay=0, sell_delay=0, start=date(2026, 4, 13), end=date(2026, 4, 14)
        ),
        runner,
        TradingCalendar(),
        signal_source=MappingDayFrameSource(factor_by_day),
        market_data_source=StoreMarketDataSource(store=store),
        decision_source=MappingDayFrameSource(
            {
                d14: pl.DataFrame(
                    {
                        "symbol": ["AAA.SH", "BBB.SZ"],
                        "selected": [False, True],
                        "exclude_reason": ["force_exit:audit", None],
                    }
                )
            }
        ),
    )
    result = engine.run()
    diagnostics = result.diagnostics.sort("date")
    assert result.nav.height == diagnostics.height == 2
