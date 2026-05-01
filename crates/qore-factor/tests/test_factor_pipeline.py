from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_factor.event.alert import AlertCondition, AlertRule, build_alert_frame
from qore_factor.fundamental.growth import ProfitGrowthPremiumFactor
from qore_factor.fundamental.quality import (
    AccrualRatioFactor,
    AssetTurnoverFactor,
    CFOYieldFactor,
    DebtToAssetRatioFactor,
    ROEStabilityFactor,
)
from qore_factor.fundamental.value import BookToPriceFactor
from qore_factor.ohlcv.liquidity import (
    AverageAmountFactor,
    CapacityPenaltyFactor,
    MinimumAmountFactor,
    PositionToLiquidityRatioFactor,
    liquidity_capacity_factors,
)
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
    result = _collect_dataframe(pipeline.run(lf))

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

    pipeline = FactorPipeline(normalize="zscore").add(BookToPriceFactor())
    result = _collect_dataframe(pipeline.run(lf))

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

    result = _collect_dataframe(
        FactorPipeline().add(RealizedVolatilityFactor(window=2)).run(lf)
    )

    assert "realized_vol_2d" in result.columns
    assert result.get_column("realized_vol_2d").null_count() > 0
    assert result.get_column("realized_vol_2d").drop_nulls().len() > 0


def test_pipeline_adds_liquidity_capacity_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
            ],
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "amount": [
                10_000_000.0,
                12_000_000.0,
                8_000_000.0,
                5_000_000.0,
                4_000_000.0,
                6_000_000.0,
            ],
            "target_position_cny": [
                800_000.0,
                800_000.0,
                800_000.0,
                900_000.0,
                900_000.0,
                900_000.0,
            ],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline()
        .add(
            AverageAmountFactor(window=2),
            MinimumAmountFactor(window=2),
            PositionToLiquidityRatioFactor(liquidity_column="avg_amount_2d"),
            CapacityPenaltyFactor(
                ratio_column="position_to_amount_2d_ratio", threshold=0.10
            ),
        )
        .run(lf)
    ).sort(["symbol", "date"])

    avg_amount = result.get_column("avg_amount_2d").to_list()
    assert avg_amount[0] is None
    assert avg_amount[1] == pytest.approx(11_000_000.0)
    assert avg_amount[2] == pytest.approx(10_000_000.0)
    assert avg_amount[3] is None
    assert avg_amount[4] == pytest.approx(4_500_000.0)
    assert avg_amount[5] == pytest.approx(5_000_000.0)

    min_amount = result.get_column("min_amount_2d").to_list()
    assert min_amount[0] is None
    assert min_amount[1] == pytest.approx(10_000_000.0)
    assert min_amount[2] == pytest.approx(8_000_000.0)
    assert min_amount[3] is None
    assert min_amount[4] == pytest.approx(4_000_000.0)
    assert min_amount[5] == pytest.approx(4_000_000.0)

    liquidity_ratio = result.get_column("position_to_amount_2d_ratio").to_list()
    assert liquidity_ratio[0] is None
    assert liquidity_ratio[1] == pytest.approx(800_000.0 / 11_000_000.0)
    assert liquidity_ratio[2] == pytest.approx(0.08)
    assert liquidity_ratio[3] is None
    assert liquidity_ratio[4] == pytest.approx(0.2)
    assert liquidity_ratio[5] == pytest.approx(0.18)
    penalties = result.get_column(
        "capacity_penalty_position_to_amount_2d_ratio"
    ).to_list()
    assert penalties[0] is None
    assert penalties[1] == pytest.approx(1.0)
    assert penalties[2] == pytest.approx(1.0)
    assert penalties[3] is None
    assert penalties[4] == pytest.approx(0.5)
    assert penalties[5] == pytest.approx(0.5555555556)


def test_liquidity_capacity_factor_factory_names() -> None:
    factors = liquidity_capacity_factors(
        window=5, position_column="target_position_cny"
    )
    names = [f.produces for f in factors]
    assert names == ["avg_amount_5d", "min_amount_5d", "position_to_amount_5d_ratio"]


def test_pipeline_adds_fundamental_quality_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB"],
            "revenue": [100.0, 200.0],
            "total_assets": [400.0, 500.0],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline().add(AssetTurnoverFactor()).run(lf)
    ).sort("symbol")

    assert result.get_column("asset_turnover").to_list() == pytest.approx([0.25, 0.4])


def test_pipeline_adds_fundamental_cashflow_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB"],
            "operating_cashflow": [80.0, 60.0],
            "net_income": [100.0, 50.0],
            "total_assets": [400.0, 200.0],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline().add(CFOYieldFactor(), AccrualRatioFactor()).run(lf)
    ).sort("symbol")

    assert result.get_column("cfo_yield").to_list() == pytest.approx([0.2, 0.3])
    assert result.get_column("accrual_ratio").to_list() == pytest.approx([0.05, -0.05])


def test_pipeline_adds_debt_to_asset_ratio_factor() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB", "CCC"],
            "total_liabilities": [120.0, 150.0, 10.0],
            "total_assets": [400.0, 300.0, 0.0],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline().add(DebtToAssetRatioFactor()).run(lf)
    ).sort("symbol")

    assert result.get_column("debt_to_asset_ratio").to_list()[:2] == pytest.approx(
        [0.3, 0.5]
    )
    assert result.row(2, named=True)["debt_to_asset_ratio"] is None


def test_debt_to_asset_ratio_factor_supports_alias_configuration() -> None:
    lf = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "liabilities": [30.0, 40.0],
            "assets": [100.0, 80.0],
        }
    ).lazy()

    factor = DebtToAssetRatioFactor(
        liabilities_column="liabilities",
        assets_column="assets",
        produces="leverage_ratio",
        name="leverage_ratio",
    )
    result = _collect_dataframe(FactorPipeline().add(factor).run(lf)).sort("symbol")

    assert factor.requires == frozenset({"liabilities", "assets"})
    assert result.get_column("leverage_ratio").to_list() == pytest.approx([0.3, 0.5])


def test_build_alert_frame_from_generic_conditions() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 4, 18), date(2026, 4, 18), date(2026, 4, 18)],
            "symbol": ["AAA", "BBB", "CCC"],
            "pct_change": [-0.08, -0.03, -0.09],
            "turnover_cny": [8_000_000.0, 8_000_000.0, 3_000_000.0],
            "has_adverse_audit_opinion": [False, True, True],
        }
    ).lazy()

    alerts = pl.DataFrame(
        build_alert_frame(
            lf,
            rules=(
                AlertRule(
                    name="single_day_drop",
                    conditions=(
                        AlertCondition("pct_change", "le", -0.07),
                        AlertCondition("turnover_cny", "gt", 5_000_000.0),
                    ),
                ),
                AlertRule(
                    name="adverse_audit_context",
                    conditions=(
                        AlertCondition("has_adverse_audit_opinion", "eq", True),
                    ),
                    action="record_alert",
                ),
            ),
        ).collect()
    )
    alerts = alerts.sort(["alert_name", "symbol"])

    assert alerts.height == 3
    assert alerts.get_column("alert_name").to_list() == [
        "adverse_audit_context",
        "adverse_audit_context",
        "single_day_drop",
    ]
    assert alerts.get_column("symbol").to_list() == ["BBB", "CCC", "AAA"]
    assert alerts.get_column("alert_action").to_list() == [
        "record_alert",
        "record_alert",
        "emit_alert",
    ]


def test_pipeline_adds_fundamental_growth_factors() -> None:
    lf = pl.DataFrame(
        {
            "date": [date(2026, 3, 31), date(2026, 3, 31)],
            "symbol": ["AAA", "BBB"],
            "revenue_growth_yoy": [0.10, 0.05],
            "net_profit_growth_yoy": [0.18, 0.02],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline()
        .add(
            ProfitGrowthPremiumFactor(
                net_profit_growth_column="net_profit_growth_yoy",
                revenue_growth_column="revenue_growth_yoy",
            ),
        )
        .run(lf)
    ).sort("symbol")

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
    assert metrics.get_column("signal_key").to_list() == ["bp"]
    assert metrics.get_column("horizon").to_list() == [1]
    assert metrics.get_column("ic_mean").to_list() == [1.0]
    assert metrics.get_column("observations").to_list() == [2]


def test_roe_stability_factor() -> None:
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
            "roe": [0.15, 0.12, 0.13, 0.14, 0.10, 0.08, 0.09, 0.11],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline().add(ROEStabilityFactor(window=3)).run(lf)
    ).sort(["symbol", "date"])

    assert "roe_stability" in result.columns
    stable_aaa = result.filter(pl.col("symbol") == "AAA").get_column("roe_stability")
    assert stable_aaa[0] is None
    assert stable_aaa[3] > 0
    stable_bbb = result.filter(pl.col("symbol") == "BBB").get_column("roe_stability")
    assert stable_bbb[3] > 0


def test_pipeline_neutralizes_cross_sectionally() -> None:
    lf = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
            ],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "industry": ["Bank", "Tech", "Bank", "Tech"],
            "pb": [2.0, 1.0, 4.0, 2.0],
        }
    ).lazy()

    result = _collect_dataframe(
        FactorPipeline(neutralize_by=["date", "industry"])
        .add(BookToPriceFactor())
        .run(lf)
    )

    assert "bp" in result.columns
    assert result.get_column("bp").to_list() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0], abs=1e-8
    )


def _collect_dataframe(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    collected = df.collect() if isinstance(df, pl.LazyFrame) else df
    if not isinstance(collected, pl.DataFrame):
        msg = "Expected DataFrame during factor test materialization."
        raise TypeError(msg)
    return collected
