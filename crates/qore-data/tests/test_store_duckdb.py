from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from qore_core import QoreConfig
from qore_data.store.duckdb import QoreStore


def test_store_write_and_read_stock_ohlcv(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
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
    result = store.read("stock_ohlcv").collect()
    assert result.get_column("symbol").to_list() == ["600519.SH"]


def test_register_all_views_works_without_files(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
    store.register_all_views()
    result = store.sql("select * from stock_ohlcv").collect()
    assert result.is_empty()


def test_register_all_views_reads_written_dataset(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
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
    result = store.sql("select symbol from stock_ohlcv").collect()
    assert result.get_column("symbol").to_list() == ["600519.SH"]


def test_store_rejects_missing_columns(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
    df = pl.DataFrame({"date": [date(2026, 4, 10)], "symbol": ["600519.SH"]})
    try:
        store.write("stock_ohlcv", df)
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("store.write should reject incomplete schema")


def test_store_deduplicates_existing_rows_across_writes(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
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

    result = store.read("stock_ohlcv").collect()
    assert result.height == 1


def test_store_rejects_unknown_filter_columns(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)

    try:
        store.read("stock_ohlcv", filters={"missing": "value"})
    except KeyError as exc:
        assert "Unknown filter column" in str(exc)
    else:
        raise AssertionError("store.read should reject unknown filter columns")
