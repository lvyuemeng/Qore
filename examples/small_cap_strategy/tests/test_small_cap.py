from __future__ import annotations

from datetime import date

import pytest
from qore_backtest import TradingCalendar
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from small_cap_strategy.workflow import (
    _monthly_rebalance_days,
    _strategy_spec,
    run_small_cap_workflow,
)


def _has_required_small_cap_data(store: QoreStore, strategy_end: object) -> bool:
    checks = (
        store.read(
            "index_constituents",
            filters={"index_symbol": "8841431.WI", "as_of": strategy_end},
            columns=["symbol"],
            backend="duckdb",
        )
        .limit(1)
        .collect(),
        store.read(
            "stock_profiles",
            filters={"as_of": strategy_end},
            columns=["symbol"],
            backend="duckdb",
        )
        .limit(1)
        .collect(),
        store.read(
            "stock_ohlcv",
            columns=["date", "symbol"],
            backend="duckdb",
        )
        .limit(1)
        .collect(),
    )
    return all(not frame.is_empty() for frame in checks)


def test_strategy_spec_defaults() -> None:
    spec = _strategy_spec()
    assert spec.benchmark == "8841431.WI"
    assert spec.top_n == 20
    assert spec.primary_factor == "total_market_cap"
    assert spec.primary_ascending is True


def test_monthly_rebalance_days_returns_first_trading_day_per_month() -> None:
    days = _monthly_rebalance_days(
        TradingCalendar(),
        start=date(2026, 1, 1),
        end=date(2026, 3, 31),
    )
    assert len(days) == 3
    assert {(d.year, d.month) for d in days} == {(2026, 1), (2026, 2), (2026, 3)}


def test_run_small_cap_workflow_real_data_contract() -> None:
    spec = _strategy_spec()
    store = QoreStore.from_settings(
        DataSettings(db_path="data/qore.duckdb", parquet_root="data/raw")
    )
    strategy_end = spec.end
    if not _has_required_small_cap_data(store, strategy_end):
        pytest.skip(
            "Small-cap real-data contract test skipped: required datasets missing"
        )

    result = run_small_cap_workflow()

    assert not result.nav.is_empty()
    assert "date" in result.nav.columns
    assert "nav" in result.nav.columns
