from __future__ import annotations

from pathlib import Path

from qore_core import QoreConfig

from examples.stock_ranking_workflow import (
    build_stock_category_report,
    main,
    run_stock_ranking_workflow,
)


def test_run_stock_ranking_workflow_returns_backtest_result(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            },
            "intelligence": {"model_store_root": str(tmp_path / "models")},
        }
    )

    result = run_stock_ranking_workflow(config)

    assert result.nav.height == 1
    assert list(result.positions[0]) == ["BBB.SZ"]


def test_build_stock_category_report_returns_industry_summary(tmp_path: Path) -> None:
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            },
            "intelligence": {"model_store_root": str(tmp_path / "models")},
        }
    )

    report = build_stock_category_report(config)

    assert set(report.columns) == {
        "industry",
        "board",
        "symbol_count",
        "avg_total_market_cap",
        "avg_report_count",
        "announcement_count",
    }
    assert report.height == 2
    assert report.get_column("announcement_count").to_list() == [1, 2]


def test_stock_workflow_main_accepts_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "qore.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  db_path: {tmp_path / 'qore.duckdb'}",
                f"  parquet_root: {tmp_path / 'raw'}",
                "intelligence:",
                f"  model_store_root: {tmp_path / 'models'}",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
