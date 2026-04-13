from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    db_path: str = "data/qore.duckdb"
    parquet_root: str = "data/raw"
    eastmoney_concurrency: int = 8
    eastmoney_delay_min: float = 0.2
    eastmoney_delay_max: float = 0.5


class LGBMParams(BaseModel):
    objective: str = "lambdarank"
    metric: str = "ndcg"
    ndcg_eval_at: list[int] = [5, 10, 20]
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    verbose: int = -1
    seed: int = 42


class IntelligenceConfig(BaseModel):
    model_store_root: str = "models"
    horizons: list[int] = [20, 60, 252]
    ensemble_weights: dict[str, float] = {"20d": 0.3, "60d": 0.4, "252d": 0.3}
    lgbm: LGBMParams = Field(default_factory=LGBMParams)
    news_llm_daily_budget: int = 50
    news_llm_model: str = "claude-sonnet-4-20250514"
    news_finbert_model: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    news_score_half_life_days: int = 5

    @model_validator(mode="after")
    def validate_weights(self) -> IntelligenceConfig:
        weight_sum = sum(self.ensemble_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            msg = f"ensemble_weights must sum to 1.0, got {weight_sum:.4f}"
            raise ValueError(msg)
        return self


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
