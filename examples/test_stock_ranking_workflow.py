from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from qore_data.store.duckdb import QoreStore

from examples.stock_ranking_workflow import (
    _seed_backtest_inputs,
    build_stock_strategy_assembly,
)


def test_build_stock_strategy_assembly_composes_audit_exclusion_capacity_and_alerts(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    _seed_backtest_inputs(store)

    assembly = build_stock_strategy_assembly(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
        suggested_max_aum_cny=50_000_000.0,
        top_n=20,
        audit_exclusion_days=365,
        liquidity_lookback_days=2,
    )

    assert set(assembly.selection_frame.get_column("symbol").to_list()) == {
        "AAA.SH",
        "BBB.SZ",
        "CCC.SZ",
    }
    selected_symbols = set(assembly.selected_frame.get_column("symbol").to_list())
    assert "BBB.SZ" not in selected_symbols
    assert assembly.decision_frame.height == 3
    overlay = assembly.decision_frame.sort("symbol")
    bbb = overlay.filter(pl.col("symbol") == "BBB.SZ").row(0, named=True)
    assert bool(bbb["selected"]) is False
    assert bbb["exclude_reason"] in {"audit", "audit|capacity"}
    assert overlay.filter(~pl.col("selected")).height >= 1
    assert set(assembly.alert_frame.get_column("alert_name").to_list()) == {
        "adverse_audit_context",
        "single_day_drop",
    }
    statuses = set(
        pl.DataFrame(
            assembly.decision_frame.with_columns(
                pl.when(pl.col("selected"))
                .then(pl.lit("selected"))
                .when(pl.col("exclude_reason").is_not_null())
                .then(pl.lit("excluded"))
                .otherwise(pl.lit("eligible_not_selected"))
                .alias("pool_status")
            )
        )
        .get_column("pool_status")
        .to_list()
    )
    assert "excluded" in statuses
    assert statuses.issubset({"selected", "eligible_not_selected", "excluded"})


def test_build_stock_strategy_overlay_is_deterministic_for_same_as_of(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    _seed_backtest_inputs(store)

    historical = build_stock_strategy_assembly(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
        suggested_max_aum_cny=50_000_000.0,
        top_n=20,
        audit_exclusion_days=365,
        liquidity_lookback_days=2,
    )
    fresh = build_stock_strategy_assembly(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
        suggested_max_aum_cny=50_000_000.0,
        top_n=20,
        audit_exclusion_days=365,
        liquidity_lookback_days=2,
    )

    assert (
        historical.decision_frame.sort("symbol").to_dicts()
        == fresh.decision_frame.sort("symbol").to_dicts()
    )
