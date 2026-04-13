from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_core import QoreConfig
from qore_data.store.duckdb import QoreStore
from qore_factor.fundamental.value import BookToPriceFactor
from qore_factor.ohlcv.momentum import MomentumFactor
from qore_factor.pipeline import FactorPipeline


def test_pipeline_adds_factor_columns() -> None:
    lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 1),
                date(2026, 1, 2),
            ],
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "close": [10.0, 11.0, 20.0, 21.0],
            "pb": [2.0, 2.2, 1.5, 1.6],
        }
    ).lazy()

    pipeline = FactorPipeline().add(
        MomentumFactor(lookback=1, skip=0), BookToPriceFactor()
    )
    result = pipeline.run(lf).collect()

    assert "mom_1d_skip0" in result.columns
    assert "bp" in result.columns


def test_pipeline_normalizes_cross_sectionally() -> None:
    lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
            ],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "pb": [2.0, 1.0, 4.0, 2.0],
        }
    ).lazy()

    pipeline = FactorPipeline().add(BookToPriceFactor()).normalize(method="zscore")
    result = pipeline.run(lf).collect()

    assert "bp_z" in result.columns


def test_pipeline_evaluates_ic_metrics() -> None:
    factor_lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
            ],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "pb": [2.0, 1.0, 4.0, 2.0],
        }
    ).lazy()
    forward_returns = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
            ],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "forward_return_1d": [0.5, 1.0, 0.25, 0.5],
        }
    ).lazy()

    metrics = (
        FactorPipeline()
        .add(BookToPriceFactor())
        .evaluate(
            factor_lf,
            forward_returns,
            horizons=[1],
        )
    )

    assert metrics.height == 1
    assert metrics.get_column("factor_name").to_list() == ["bp"]
    assert metrics.get_column("horizon").to_list() == [1]
    assert metrics.get_column("ic_mean").to_list() == [1.0]
    assert metrics.get_column("observations").to_list() == [2]


def test_pipeline_persists_factor_scores(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    store = QoreStore.from_config(config)
    lf = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "symbol": ["AAA", "BBB"],
            "pb": [2.0, 1.0],
        }
    ).lazy()

    FactorPipeline().add(BookToPriceFactor()).normalize(method="zscore").persist(
        lf, store
    )

    result = (
        store.read("factor_scores").collect().sort(["date", "symbol", "factor_name"])
    )
    assert result.height == 2
    assert result.get_column("factor_name").to_list() == ["bp", "bp"]
    assert result.get_column("raw_value").to_list() == pytest.approx([0.5, 1.0])
    assert result.get_column("z_score").null_count() == 0
