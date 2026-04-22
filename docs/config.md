# Qore Configuration

## Purpose

`QoreConfig` is the single entrypoint for infrastructure and runtime behavior.
New crate code should read paths, source behavior, retry/cooldown behavior,
runner defaults, and backtest defaults from config instead of ad hoc constructor
arguments.

It is not the long-term home for trained model outputs.
Model metadata, feature schema, learned weights, training summaries, and trained
payload state belong to model artifacts, not static repository config.

Source of truth:

- `crates/qore-core/src/qore_core/config.py`
- `docs/design.md`

## Current Config Tree

`QoreConfig` currently contains:

- `data`
  - `db_path`
  - `parquet_root`
  - `eastmoney_concurrency`
  - `eastmoney_delay_min`
  - `eastmoney_delay_max`
  - `eastmoney_timeout`
  - `eastmoney_max_retries`
  - `eastmoney_retry_budget`
  - `eastmoney_cooldown_min`
  - `eastmoney_cooldown_max`
  - `eastmoney_retry_backoff_min`
  - `eastmoney_retry_backoff_max`
- `intelligence`
  - `model_store_root`
  - `news_llm_daily_budget`
  - `news_llm_model`
  - `news_finbert_model`
  - `news_score_half_life_days`
- `stock`
  - `rebalance_freq`
  - `top_k`
  - `max_weight`
  - `benchmark`
- `fund`
  - `rebalance_freq`
  - `top_k`
  - `max_weight`
- `derivative`
  - `rebalance_freq`
  - `roll_days_before_expiry`
- `backtest`
  - `initial_capital`
  - `commission`
  - `slippage`
  - `drawdown_stop`

## What config should contain

Keep these in config:

- filesystem roots and storage locations
- source/fetch behavior such as concurrency, timeout, retry, cooldown
- runtime budgets such as LLM call budgets
- runner/backtest runtime defaults

Do not keep these in config:

- trained model payload state
- selected factor schema used by a saved model
- learned ensemble weights
- training metadata and validation summaries
- model-family-specific learned settings produced by training

## Model Artifact Boundary

The current intelligence boundary is:

- `ModelArtifactManifest`: persisted metadata only
- `ModelPayload`: trained runtime objects only
- `TrainedModelArtifact`: manifest + payload envelope for persistence
- `ModelPipeline`: fit/predict runtime behavior only
- `ModelRegistry`: load/save behavior only

Practical rule:

- `QoreConfig` tells the system where to store and how to operate
- model artifacts tell the system what was trained and what trained state to load

## Usage Pattern

Load config from YAML:

```python
from qore_core.config import QoreConfig

config = QoreConfig.from_yaml("config/qore.yaml")
```

Create config-driven components:

```python
from qore_core import TradingCalendar
from qore_data.fetcher import EastMoneyFetcher
from qore_data.store.duckdb import QoreStore
from qore_intelligence.model.registry import ModelRegistry

calendar = TradingCalendar.from_config(config)
fetcher = EastMoneyFetcher.from_config(config)
store = QoreStore.from_config(config)
registry = ModelRegistry.from_config(config)
```

## Example YAML

```yaml
data:
  db_path: data/qore.duckdb
  parquet_root: data/raw
  eastmoney_concurrency: 8
  eastmoney_delay_min: 0.2
  eastmoney_delay_max: 0.5
  eastmoney_timeout: 15.0
  eastmoney_max_retries: 3
  eastmoney_retry_budget: 20
  eastmoney_cooldown_min: 1.5
  eastmoney_cooldown_max: 8.0
  eastmoney_retry_backoff_min: 0.5
  eastmoney_retry_backoff_max: 2.0

intelligence:
  model_store_root: models
  news_llm_daily_budget: 50
  news_llm_model: claude-sonnet-4-20250514
  news_finbert_model: IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
  news_score_half_life_days: 5

stock:
  rebalance_freq: W
  top_k: 50
  max_weight: 0.05
  benchmark: 000300.SH

fund:
  rebalance_freq: M
  top_k: 20
  max_weight: 0.10

derivative:
  rebalance_freq: D
  roll_days_before_expiry: 5

backtest:
  initial_capital: 10000000
  commission: 0.0003
  slippage: 0.0005
  drawdown_stop: 0.15
```

## Current Constraints

- config is still library-first; there is no final CLI/operator config contract yet
- operator workflow documentation is still being built
- `data` contains EastMoney runtime hardening controls because the current source scope is EastMoney-first
- broader source expansion may later require more source-specific subsections or a cleaner source registry shape

## Rules For New Code

- put runtime paths and operator/runtime behavior under `QoreConfig`
- prefer `from_config()` over raw path injection
- do not add crate runtime config that bypasses `QoreConfig`
- do not put trained model metadata or training outputs into static config
- keep config aligned with `docs/design.md` and current crate behavior, not legacy `src/quant_trade`
