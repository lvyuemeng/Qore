from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_core import QoreConfig
from qore_data.store.duckdb import QoreStore
from qore_factor.fundamental.growth import (
    NetProfitGrowthFactor,
    ProfitGrowthPremiumFactor,
    RevenueGrowthFactor,
)
from qore_factor.fundamental.quality import (
    AccrualRatioFactor,
    AssetTurnoverFactor,
    CFOYieldFactor,
    GrossMarginFactor,
)
from qore_factor.fundamental.value import BookToPriceFactor
from qore_factor.ohlcv.momentum import MomentumFactor
from qore_factor.ohlcv.volatility import RealizedVolatilityFactor
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


def test_pipeline_adds_realized_volatility_factor() -> None:
    lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 4),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 4),
            ],
            "symbol": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
            "close": [10.0, 10.2, 10.1, 10.4, 20.0, 19.9, 20.2, 20.5],
        }
    ).lazy()

    result = FactorPipeline().add(RealizedVolatilityFactor(window=2)).run(lf).collect()

    assert "realized_vol_2d" in result.columns
    assert result.get_column("realized_vol_2d").null_count() > 0
    assert result.get_column("realized_vol_2d").drop_nulls().len() > 0


def test_pipeline_adds_fundamental_quality_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB"],
            "revenue": [100.0, 200.0],
            "gross_margin": [0.35, 0.20],
            "total_assets": [400.0, 500.0],
        }
    ).lazy()

    result = (
        FactorPipeline()
        .add(GrossMarginFactor(), AssetTurnoverFactor())
        .run(lf)
        .collect()
        .sort("symbol")
    )

    assert result.get_column("gross_margin").to_list() == [0.35, 0.20]
    assert result.get_column("asset_turnover").to_list() == pytest.approx([0.25, 0.4])


def test_pipeline_adds_fundamental_cashflow_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB"],
            "cfo": [80.0, 60.0],
            "net_income": [100.0, 50.0],
            "total_assets": [400.0, 200.0],
        }
    ).lazy()

    result = (
        FactorPipeline()
        .add(CFOYieldFactor(), AccrualRatioFactor())
        .run(lf)
        .collect()
        .sort("symbol")
    )

    assert result.get_column("cfo_yield").to_list() == pytest.approx([0.2, 0.3])
    assert result.get_column("accrual_ratio").to_list() == pytest.approx([0.05, -0.05])


def test_pipeline_adds_fundamental_growth_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB"],
            "revenue_growth_yoy": [0.10, 0.05],
            "net_profit_growth_yoy": [0.18, 0.02],
        }
    ).lazy()

    result = (
        FactorPipeline()
        .add(
            RevenueGrowthFactor(),
            NetProfitGrowthFactor(),
            ProfitGrowthPremiumFactor(),
        )
        .run(lf)
        .collect()
        .sort("symbol")
    )

    assert result.get_column("revenue_growth_yoy").to_list() == pytest.approx(
        [0.10, 0.05]
    )
    assert result.get_column("net_profit_growth_yoy").to_list() == pytest.approx(
        [0.18, 0.02]
    )
    assert result.get_column("profit_growth_premium").to_list() == pytest.approx(
        [0.08, -0.03]
    )


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
