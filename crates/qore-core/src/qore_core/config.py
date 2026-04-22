from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
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


class IntelligenceConfig(BaseModel):
    model_store_root: str = "models"
    news_llm_daily_budget: int = 50
    news_llm_model: str = "claude-sonnet-4-20250514"
    news_finbert_model: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    news_score_half_life_days: int = 5


class StockRunnerConfig(BaseModel):
    rebalance_freq: Literal["D", "W", "M"] = "W"
    top_k: int = 50
    max_weight: float = 0.05
    benchmark: str = "000300.SH"


class FundRunnerConfig(BaseModel):
    rebalance_freq: Literal["M"] = "M"
    top_k: int = 20
    max_weight: float = 0.10


class DerivativeRunnerConfig(BaseModel):
    rebalance_freq: Literal["D", "W"] = "D"
    roll_days_before_expiry: int = 5


class BacktestConfig(BaseModel):
    initial_capital: float = 10_000_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    drawdown_stop: float = 0.15


class QoreConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    stock: StockRunnerConfig = Field(default_factory=StockRunnerConfig)
    fund: FundRunnerConfig = Field(default_factory=FundRunnerConfig)
    derivative: DerivativeRunnerConfig = Field(default_factory=DerivativeRunnerConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> QoreConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})
