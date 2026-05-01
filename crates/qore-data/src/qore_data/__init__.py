# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataSettings:
    db_path: str = "data/qore.duckdb"
    parquet_root: str = "data/raw"
    concurrency: int = 20
    delay_min: float = 0.05
    delay_max: float = 0.15
    timeout: float = 15.0
    max_retries: int = 3
    retry_budget: int = 50
    cooldown_min: float = 0.5
    cooldown_max: float = 2.0
    retry_backoff_min: float = 0.2
    retry_backoff_max: float = 1.0


from qore_data.fetch import StockPipeline

__all__ = [
    "DataSettings",
    "StockPipeline",
]
