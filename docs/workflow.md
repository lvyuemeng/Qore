# Qore Workflow

## Purpose

This document describes the current usable workflow in the rewrite. It is a
library-first flow, not a finalized CLI contract.

## Current Workflow

The current manual sequence is:

1. Load `QoreConfig`
2. Fetch raw datasets with `qore-data`
3. Persist raw datasets through `QoreStore`
4. Read store-backed lazy frames
5. Compute and normalize factors with `qore-factor`
6. Persist factor outputs into `factor_scores`
7. Fit or load ranking models in `qore-intelligence`
8. Optionally compute and persist `news_scores`
9. Generate target portfolios with `qore-runner`
10. Evaluate results in `qore-backtest`

## Crate Responsibilities

- `qore-core`: config, calendar, instrument, universe
- `qore-data`: source adapters, fetch dispatch, DuckDB + Parquet storage
- `qore-factor`: lazy factor computation, normalization, evaluation, persistence
- `qore-intelligence`: ranking model pipeline, validation, signal scoring
- `qore-runner`: strategy logic, gating, sizing, target portfolio generation
- `qore-backtest`: fill simulation, accounting skeleton, performance metrics

## Minimal Building Blocks

```python
from qore_core.config import QoreConfig
from qore_data.fetcher.eastmoney import EastMoneyFetcher
from qore_data.store.duckdb import QoreStore
from qore_factor.pipeline import FactorPipeline
from qore_intelligence.model.pipeline import ModelPipeline
from qore_runner.runner import StrategyRunner
from qore_backtest.engine import BacktestEngine

config = QoreConfig.from_yaml("config/qore.yaml")
fetcher = EastMoneyFetcher.from_config(config)
store = QoreStore.from_config(config)
model_pipeline = ModelPipeline.from_config(config)
backtest = BacktestEngine.from_config(config)
```

## Custom Factor Workflow

To add a factor:

1. Create the factor class in the matching module family under `crates/qore-factor/src/qore_factor/`
2. Inherit from the correct factor protocol family
3. Define `name`, `requires`, and `produces`
4. Implement `compute(self, lf: pl.LazyFrame) -> pl.LazyFrame`
5. Keep the factor lazy; do not call `.collect()` inside the factor
6. Add focused tests in `crates/qore-factor/tests/test_factor_pipeline.py`

## Current Reality

- The architecture boundary is visible and usable
- A partial end-to-end stock ranking flow exists
- Operators still need to wire steps manually from Python
- There is no supported one-command daily workflow yet
- Production hardening and reporting remain incomplete

## Near-Term Workflow Target

The next workflow milestone is one reproducible A-share stock ranking path with:

- explicit config file
- one supported example script or CLI entrypoint
- documented inputs and outputs
- fetch -> factor -> model -> runner -> backtest reproducibility
