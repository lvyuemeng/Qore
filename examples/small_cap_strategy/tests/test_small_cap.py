from __future__ import annotations

from datetime import date

import polars as pl
import pytest
import small_cap_strategy.workflow as workflow_module
from qore_backtest import TradingCalendar
from qore_data import DataSettings, StockPipeline
from qore_data.store.duckdb import QoreStore
from qore_runner import RebalanceSchedule
from small_cap_strategy.workflow import (
    _strategy_spec,
    prepare_small_cap_data,
    run_small_cap_workflow,
)


def _has_required_small_cap_data(store: QoreStore, strategy_end: date) -> bool:
    checks = (
        pl.DataFrame(
            store.read(
                "index_constituents",
                filters={"index_symbol": "000852.SH", "as_of": strategy_end},
                columns=["symbol"],
                backend="duckdb",
            )
            .limit(1)
            .collect()
        ),
        pl.DataFrame(
            store.read(
                "stock_info",
                columns=["symbol"],
                backend="duckdb",
            )
            .limit(1)
            .collect()
        ),
        pl.DataFrame(
            store.read("stock_ohlcv", columns=["date", "symbol"], backend="duckdb")
            .limit(1)
            .collect()
        ),
        pl.DataFrame(
            store.read("fundamentals", columns=["symbol", "roe"], backend="duckdb")
            .filter(pl.col("roe").is_not_null())
            .limit(1)
            .collect()
        ),
    )
    return all(not frame.is_empty() for frame in checks)


def test_strategy_spec_defaults() -> None:
    spec = _strategy_spec()
    assert spec.benchmark == "000852.SH"
    assert spec.top_n == 20
    assert spec.primary_factor == "total_market_cap"
    assert spec.primary_ascending is True


def test_rebalance_schedule_returns_first_trading_day_per_month() -> None:
    calendar = TradingCalendar()
    days = (
        RebalanceSchedule(frequency="monthly", buy_delay=1, sell_delay=2)
        .schedule(
            trading_days=pl.DataFrame(
                {
                    "date": calendar.trading_days_between(
                        date(2026, 1, 1), date(2026, 3, 31)
                    )
                },
                schema={"date": pl.Date},
            )
        )
        .get_column("date")
        .to_list()
    )
    assert len(days) == 3
    assert {(d.year, d.month) for d in days} == {(2026, 1), (2026, 2), (2026, 3)}


def test_run_small_cap_workflow_fails_clearly_when_no_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def _empty_selection(*args, **kwargs):
        del args, kwargs
        float_cols = [
            "total_market_cap",
            "total_assets",
            "total_liabilities",
            "roe",
            "operating_cashflow",
            "pe_ttm",
            "pb",
            "debt_to_asset_ratio",
            "position_to_amount_20d_ratio",
            "min_amount_20d",
        ]
        bool_cols = ["is_st", "is_suspended", "limit_up", "limit_down"]
        schema = dict.fromkeys(float_cols, pl.Float64)
        schema.update(dict.fromkeys(bool_cols, pl.Boolean))
        schema.update(
            {
                "listing_days": pl.Int64,
                "symbol": pl.String,
                "date": pl.Date,
            }
        )
        return pl.DataFrame(schema=schema).lazy()

    class _FakePipe:
        store = None

        def selection(self, query):
            del query
            return _empty_selection()

        async def close(self):
            return None

    monkeypatch.setattr(
        workflow_module.StockPipeline,
        "from_settings",
        lambda *a, **kw: _FakePipe(),
    )
    with pytest.raises(ValueError, match="produced no rebalance selection snapshots"):
        run_small_cap_workflow(
            data_settings=DataSettings(
                db_path=str(tmp_path / "qore.duckdb"),
                parquet_root=str(tmp_path / "raw"),
            )
        )


@pytest.mark.asyncio
async def test_prepare_small_cap_data_fails_clearly_when_no_constituents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _empty_resolve(self, index_symbol, as_of):
        del self, index_symbol, as_of
        return pl.Series("symbol", [], dtype=pl.String)

    class _Fake:
        async def close(self) -> None:
            return

    _fake = _Fake()
    monkeypatch.setattr(workflow_module.StockPipeline, "resolve", _empty_resolve)
    monkeypatch.setattr(
        workflow_module.StockPipeline,
        "from_settings",
        lambda *a, **kw: StockPipeline(
            store=_fake,
            quote=_fake,
            financial=_fake,
            analyst=_fake,
            announcement=_fake,
            index=_fake,
        ),
    )

    with pytest.raises(ValueError, match="resolved no constituents"):
        await prepare_small_cap_data()


def test_run_small_cap_workflow_real_data_contract() -> None:
    spec = _strategy_spec()
    store = None
    try:
        store = QoreStore.from_settings(
            DataSettings(db_path="data/qore.duckdb", parquet_root="data/raw")
        )
    except Exception as exc:
        pytest.skip(
            f"Small-cap real-data contract test skipped: store unavailable ({exc})"
        )
    assert store is not None
    strategy_end = spec.end
    if not _has_required_small_cap_data(store, strategy_end):
        pytest.skip(
            "Small-cap real-data contract test skipped: required datasets missing"
        )

    result = run_small_cap_workflow()

    assert not result.nav.is_empty()
    assert "date" in result.nav.columns
    assert "nav" in result.nav.columns
