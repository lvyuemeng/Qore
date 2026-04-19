# Qore Configuration

## Purpose

`QoreConfig` is the single configuration entrypoint for infrastructure and runtime
behavior. New crate code should read filesystem paths, source behavior, and operator
runtime settings from config rather than ad hoc constructor arguments.

It should not be used as the long-term home for trained model tuning outputs.
Model hyperparameters, selected factor sets, learned weights, and training metadata
belong to exported model artifacts, not static repository config.

Source of truth:

- `crates/qore-core/src/qore_core/config.py`
- `docs/design.md`

## Current Config Tree

`QoreConfig` currently contains these sections:

- `data`
  - `db_path`
  - `parquet_root`
  - `eastmoney_concurrency`
  - `eastmoney_delay_min`
  - `eastmoney_delay_max`
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

## Usage Pattern

Load config from YAML:

```python
from qore_core.config import QoreConfig

config = QoreConfig.from_yaml("config/qore.yaml")
```

Classes that depend on config should expose `from_config()`:

```python
fetcher = EastMoneyFetcher.from_config(config)
store = QoreStore.from_config(config)
pipeline = ModelPipeline.from_config(config)
```

## Example YAML

```yaml
data:
  db_path: data/qore.duckdb
  parquet_root: data/raw
  eastmoney_concurrency: 8
  eastmoney_delay_min: 0.2
  eastmoney_delay_max: 0.5

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
  max_weight: 0.1

derivative:
  rebalance_freq: D
  roll_days_before_expiry: 5

backtest:
  initial_capital: 10000000
  commission: 0.0003
  slippage: 0.0005
  drawdown_stop: 0.15
```

## Model Artifact Boundary

The intended boundary is:

- `QoreConfig`: model store location, runtime budgets, source behavior, backtest behavior
- model export/import artifact: tuned hyperparameters, factor schema, selected horizons, ensemble weights, training summary, validation outputs

Examples of values that should live with the saved model artifact rather than static config:

- Optuna-selected hyperparameters
- factor list or feature schema used at fit time
- horizon set used by the trained model
- ensemble weights learned or selected during training
- training/validation summaries

## Current Constraints

- Paths are still library-first; no official CLI config contract exists yet
- Current code still keeps some model-shape settings under `intelligence`; that is now a known design mismatch to be removed
- End-to-end operator examples are still being documented

## Configuration Rules For New Code

- Put runtime paths and operator/runtime behavior under `QoreConfig`
- Prefer `from_config()` over raw path injection
- Do not add crate runtime config that bypasses `QoreConfig`
- Do not put trained model tuning outputs or training metadata into static config
- Keep config aligned with `docs/design.md`, not legacy `src/quant_trade`
