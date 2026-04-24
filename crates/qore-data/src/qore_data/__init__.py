# ruff: noqa: E402
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataSettings:
    db_path: str = "data/qore.duckdb"
    parquet_root: str = "data/raw"
    eastmoney_concurrency: int = 8
    eastmoney_delay_min: float = 0.2
    eastmoney_delay_max: float = 0.5
    eastmoney_timeout: float = 15.0
    eastmoney_max_retries: int = 3
    eastmoney_retry_budget: int = 20
    eastmoney_cooldown_min: float = 1.5
    eastmoney_cooldown_max: float = 8.0
    eastmoney_retry_backoff_min: float = 0.5
    eastmoney_retry_backoff_max: float = 2.0


from qore_data.fetch import (
    fetch_analyst_forecast,
    fetch_announcements,
    fetch_audit_opinions,
    fetch_daily,
    fetch_fundamentals,
    fetch_minute,
    fetch_profile,
    fetch_tick,
)
from qore_data.instrument import (
    DerivativeInstrument,
    FundInstrument,
    Instrument,
    SessionInstrument,
    StockInstrument,
    TradingSession,
)
from qore_data.universe import (
    CandidateFilter,
    CandidateSort,
    StockCandidateSpec,
    StockSelectionPipeline,
    StockSelectionScope,
    StockUniverseQuery,
    Universe,
    build_stock_universe_frame_from_index,
    build_stock_universe_from_index,
    snapshot_index_constituents,
    snapshot_stock_analyst_forecasts,
    snapshot_stock_announcements,
    snapshot_stock_audit_opinions,
    snapshot_stock_profiles,
    snapshot_stock_statuses,
)

__all__ = [
    "CandidateFilter",
    "CandidateSort",
    "DataSettings",
    "DerivativeInstrument",
    "FundInstrument",
    "Instrument",
    "SessionInstrument",
    "StockCandidateSpec",
    "StockInstrument",
    "StockSelectionPipeline",
    "StockSelectionScope",
    "StockUniverseQuery",
    "TradingSession",
    "Universe",
    "build_stock_universe_frame_from_index",
    "build_stock_universe_from_index",
    "fetch_analyst_forecast",
    "fetch_announcements",
    "fetch_audit_opinions",
    "fetch_daily",
    "fetch_fundamentals",
    "fetch_minute",
    "fetch_profile",
    "fetch_tick",
    "snapshot_index_constituents",
    "snapshot_stock_analyst_forecasts",
    "snapshot_stock_announcements",
    "snapshot_stock_audit_opinions",
    "snapshot_stock_profiles",
    "snapshot_stock_statuses",
]
