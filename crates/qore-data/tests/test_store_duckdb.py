from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore


def _settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )


def test_store_write_and_read_stock_ohlcv(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))
    df = pl.DataFrame(
        {
            "date": [date(2026, 4, 10)],
            "symbol": ["600519.SH"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [1000],
            "amount": [2000.0],
            "adj_factor": [1.0],
            "is_suspended": [False],
            "limit_up": [False],
            "limit_down": [False],
        }
    )
    store.write("stock_ohlcv", df)
    result = pl.DataFrame(store.read("stock_ohlcv").collect())
    assert result.get_column("symbol").to_list() == ["600519.SH"]


def test_register_all_views_works_without_files(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))
    store.register_all_views()
    result = pl.DataFrame(store.sql("select * from stock_ohlcv").collect())
    assert result.is_empty()


def test_register_all_views_reads_written_dataset(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))
    df = pl.DataFrame(
        {
            "date": [date(2026, 4, 10)],
            "symbol": ["600519.SH"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [1000],
            "amount": [2000.0],
            "adj_factor": [1.0],
            "is_suspended": [False],
            "limit_up": [False],
            "limit_down": [False],
        }
    )
    store.write("stock_ohlcv", df)
    store.register_all_views()
    result = pl.DataFrame(store.sql("select symbol from stock_ohlcv").collect())
    assert result.get_column("symbol").to_list() == ["600519.SH"]


def test_store_read_semantics_match_between_parquet_and_duckdb(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))
    df = pl.DataFrame(
        {
            "date": [date(2026, 4, 10), date(2026, 4, 10)],
            "symbol": ["600519.SH", "000001.SZ"],
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [1000, 2000],
            "amount": [2000.0, 3000.0],
            "adj_factor": [1.0, 1.0],
            "is_suspended": [False, False],
            "limit_up": [False, False],
            "limit_down": [False, False],
        }
    )
    store.write("stock_ohlcv", df)

    parquet_result = pl.DataFrame(
        store.read_parquet(
            "stock_ohlcv",
            filters={"date": date(2026, 4, 10)},
            columns=["symbol", "close"],
        ).collect()
    ).sort("symbol")
    duckdb_result = pl.DataFrame(
        store.read_duckdb(
            "stock_ohlcv",
            filters={"date": date(2026, 4, 10)},
            columns=["symbol", "close"],
        ).collect()
    ).sort("symbol")

    assert parquet_result.equals(duckdb_result)


def test_store_rejects_missing_columns(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))
    df = pl.DataFrame({"date": [date(2026, 4, 10)], "symbol": ["600519.SH"]})
    try:
        store.write("stock_ohlcv", df)
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("store.write should reject incomplete schema")


def test_store_deduplicates_existing_rows_across_writes(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))
    df = pl.DataFrame(
        {
            "date": [date(2026, 4, 10)],
            "symbol": ["600519.SH"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [1000],
            "amount": [2000.0],
            "adj_factor": [1.0],
            "is_suspended": [False],
            "limit_up": [False],
            "limit_down": [False],
        }
    )
    store.write("stock_ohlcv", df)
    store.write("stock_ohlcv", df)

    result = pl.DataFrame(store.read("stock_ohlcv").collect())
    assert result.height == 1


def test_store_rejects_unknown_filter_columns(tmp_path: Path) -> None:
    store = QoreStore.from_settings(_settings(tmp_path))

    try:
        store.read("stock_ohlcv", filters={"missing": "value"})
    except KeyError as exc:
        assert "Unknown filter column" in str(exc)
    else:
        raise AssertionError("store.read should reject unknown filter columns")
