from __future__ import annotations

from pathlib import Path

import pytest
from qore_data import DataSettings
from small_cap_strategy.workflow import (
    _strategy_spec,
    run_small_cap_workflow,
)


def test_strategy_spec_is_static_example_contract() -> None:
    spec = _strategy_spec()
    assert spec.benchmark == "8841431.WI"
    assert spec.primary_factor == "total_market_cap"


def test_run_small_cap_workflow_requires_universe_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No universe data found"):
        run_small_cap_workflow(
            data_settings=DataSettings(
                db_path=str(tmp_path / "qore.duckdb"),
                parquet_root=str(tmp_path / "raw"),
            ),
        )
