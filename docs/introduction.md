# Qore Introduction and Usage

## What Qore is

Qore is a library-first quantitative research stack built as workspace crates.
Crates expose reusable APIs; they do not embed product CLI entrypoints.

Current runtime crates:

- `qore-data`: fetch/store/snapshot/universe assembly
- `qore-factor`: factor computation and normalization pipelines
- `qore-intelligence`: model pipeline, registry, and optional news/signal layers
- `qore-runner`: strategy decision and sizing
- `qore-backtest`: execution simulation, diagnostics, metrics, and result views

Current runnable end-to-end reference:

- workspace package: `examples/small_cap_strategy`

## Configuration model (merged reference)

Use crate-local typed settings in runtime code:

- `DataSettings`
- `IntelligenceSettings`
- `RunnerSettings`
- `BacktestSettings`

`QoreConfig` can still be used in composition adapters, but crate internals should
not require global cross-crate config objects.

Keep in config:

- storage paths and roots
- source runtime knobs (concurrency/timeout/retry/cooldown)
- runtime defaults and budgets

Do not keep in config:

- trained model payloads
- learned schema/weights from model fitting
- training summaries as static repository config

## Basic workflow

Typical flow is:

1. fetch or prepare raw datasets into `QoreStore`
2. build a stock universe via `StockSelectionPipeline`
3. compute factors and persist `factor_scores`
4. train/load model and build strategy
5. run strategy through runner and backtest
6. analyze `BacktestResult` / `BacktestView`

## Minimal usage walkthrough

```python
from datetime import date

from qore_backtest import BacktestSettings, TradingCalendar
from qore_backtest.engine import BacktestEngine
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_data.universe import StockSelectionPipeline, StockCandidateSpec
from qore_factor.pipeline import FactorPipeline
from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.strategy import build_ranking_strategy
from qore_runner import RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer

# 1) Data/store settings
data_settings = DataSettings(db_path="data/qore.duckdb", parquet_root="data/raw")
store = QoreStore.from_settings(data_settings)

# 2) Build universe directly (convenience API)
pipeline = StockSelectionPipeline.from_index(
    store,
    index_symbol="000300.SH",
    as_of=date(2026, 4, 13),
)
candidate_spec = StockCandidateSpec(top_n=50)
universe = pipeline.universe(candidate_spec)

# 3) Factor pipeline (example shape)
factor_pipeline = FactorPipeline()
# factor_pipeline = factor_pipeline.add(...).normalize(method="zscore")
# factor_scores = factor_pipeline.run(selection_lf).collect()
# store.write("factor_scores", factor_scores)

# 4) Intelligence strategy (example shape)
intelligence_settings = IntelligenceSettings(model_store_root="models")
ranker = MultiHorizonRanker()
strategy = build_ranking_strategy(ranker)

# 5) Runner + backtest
runner = StrategyRunner.from_settings(
    RunnerSettings(),
    strategy,
    EqualWeightSizer(top_k=50),
)
engine = BacktestEngine.from_settings(
    BacktestSettings(),
    runner,
    store,
    TradingCalendar(),
)
result = engine.run(universe, date(2026, 4, 13), date(2026, 4, 30))

# 6) Analyze
metrics = result.nav
view = result.view().with_drawdown()
```

## Quick API notes

- Universe convenience: prefer `pipeline.universe(candidate_spec)`.
- Frame output is still available when needed: `pipeline.universe_frame(candidate_spec)`.
- Backtest output is frame-native (`nav`, `positions`, `turnover`, `fills`, `diagnostics`).
- Visualization is method-owned from result view (`result.view().plot()...`).

## Current constraints

- crates are libraries; no product CLI contract in crate runtime
- official CLI/operator path is a future layer
- examples remain the current executable workflow interface

## Read next

- `docs/workflow.md` (contributor extension guide)
- `docs/roadmap.md` (active checklist)
- `docs/design.md` (current architecture snapshot)
