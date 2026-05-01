# Qore Introduction and Usage

## What Qore is

Qore is a library-first quantitative research stack built as workspace crates.
Crates expose reusable APIs; they do not embed product CLI entrypoints.

Current runtime crates:

- `qore-data`: fetch/store/selection/candidate filtering
- `qore-factor`: factor computation and normalization pipelines
- `qore-intelligence`: model pipeline, registry, and optional news/signal layers
- `qore-runner`: strategy decision and sizing
- `qore-backtest`: execution simulation, diagnostics, metrics, and result views

Current runnable end-to-end reference:

- workspace package: `examples/small_cap_strategy`

## Configuration model

Use crate-local typed settings in runtime code:

| Setting | Key fields |
|---|---|
| `DataSettings` | `db_path`, `parquet_root`, `concurrency`, `delay_min`, `delay_max`, `timeout`, `max_retries`, `retry_budget`, `cooldown_min`, `cooldown_max`, `retry_backoff_min`, `retry_backoff_max` |
| `RunnerSettings` | `max_single`, `drawdown_stop` |
| `BacktestSettings` | `initial_capital`, `commission`, `slippage`, `drawdown_stop`, `cadence` |
| `IntelligenceSettings` | model store paths |

`QoreConfig` can still be used in composition adapters, but crate internals should
not require global cross-crate config objects.

## Basic workflow

Typical flow is:

1. `StockPipeline` fetch → `QoreStore` (DuckDB + Parquet)
2. `SelectionSource.batch(dates)` → `pl.LazyFrame` (one SQL query across all datasets)
3. `FactorPipeline` → factor columns
4. `StockCandidateSpec.apply()` + `CandidateEligibilityPolicy.capacity_filter()` → selection snapshots
5. `StrategyRunner` → target weights
6. `BacktestEngine` → fills/nav/diagnostics/view

## Minimal usage walkthrough

```python
from datetime import date

from qore_backtest import (
    BacktestSettings,
    DateColumnDayFrameSource,
    NullSignalOverlaySource,
    TradingCalendar,
)
from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_data import (
    CandidateEligibilityPolicy,
    CandidateFilter,
    DataSettings,
    StockCandidateSpec,
    StockPipeline,
    Universe,
)
from qore_data.sources import MarketSource, SelectionSource
from qore_data.store.duckdb import QoreStore
from qore_factor.fundamental.quality import DebtToAssetRatioFactor
from qore_factor.pipeline import FactorPipeline
from qore_runner import RebalanceSchedule, RunnerSettings
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener

spec = ...  # see examples/small_cap_strategy for full StrategySpec

# 1) Prepare data (async)
pipe = StockPipeline.from_settings(data_settings)
symbols = await pipe.resolve(spec.benchmark, spec.end)
await pipe.stock_profiles(symbols, spec.end)
await pipe.stock_daily(symbols, spec.start, spec.end)
await pipe.fundamentals(symbols, spec.end)
await pipe.announcements(symbols, spec.start, spec.end)
await pipe.audit_opinions(symbols, spec.start, spec.end)
await pipe.analyst_forecasts(symbols, spec.end)

# 2) Build selection via SQL view
source = SelectionSource(store, spec.benchmark)
selection_lf = source.batch(rebalance_dates)

# 3) Factor pipeline
factor_frame = pl.DataFrame(
    FactorPipeline()
        .add(spec.debt_to_asset_factor)
        .run(selection_lf)
        .select("date", "symbol", spec.primary_factor)
        .collect()
)

# 4) Candidate filtering
candidate_spec = StockCandidateSpec(filters=spec.filters, min_listing_days=60)
policy = CandidateEligibilityPolicy(capacity_ratio_limit=0.10, min_daily_amount_cny=10_000_000)
snapshots = pl.DataFrame(
    candidate_spec.apply(selection_lf)
        .with_columns(policy.capacity_filter().alias("_cp"))
        .filter(pl.col("_cp"))
        .sort(spec.primary_factor, descending=False)
        .head(spec.top_n)
        .collect()
)

# 5) Run backtest
engine = BacktestEngine.from_settings(
    BacktestSettings(),
    StrategyRunner.from_settings(
        RunnerSettings(),
        CrossSectionalScreener({spec.primary_factor: 1.0}, rebalance_schedule=spec.rebalance_schedule),
        EqualWeightSizer(top_k=spec.top_n),
    ),
    TradingCalendar(),
    factor_source=DateColumnDayFrameSource(frame=factor_frame),
    market_data_source=MarketSource(store=store),
)
result = engine.run(universe, spec.start, spec.end)

# 6) Analyze
print(result.nav)
print(result.diagnostics)
result.view().with_drawdown().plot().overview()
```

## Quick API notes

- `StockPipeline`: resolve constituents, fetch 6 datasets, build statuses + factors.
- `SelectionSource`: DuckDB SQL view joining all datasets; `batch(dates)` returns `pl.LazyFrame`.
- `CandidateSpec`/`StockCandidateSpec`: chainable filters (`gt`, `lt`, `between`, `in`, etc.).
- `CandidateEligibilityPolicy`: liquidity and capacity guards via `capacity_filter()`.
- `BacktestEngine`: protocol-driven — inject `MarketSource`/`DateColumnDayFrameSource`.
- `BacktestResult`: `.nav`, `.positions`, `.turnover`, `.fills`, `.diagnostics`.
- `result.view().with_drawdown().plot().overview()`.

## Running the reference workflow

```bash
uv run --package small-cap-strategy small-cap-strategy
```

With custom data paths:

```bash
uv run --package small-cap-strategy small-cap-strategy --db-path data/custom.duckdb --parquet-root data/custom
```

Prepare data only:

```bash
uv run --package small-cap-strategy small-cap-strategy --prepare-data
```

## Current constraints

- Crates are libraries; no product CLI contract in crate runtime.
- Official CLI/operator path is a future layer.
- Examples remain the current executable workflow interface.
- All EastMoney API calls go through `HardenedJsonFetcher` with rate limiting and anti-crawl rotation.

## Read next

- `docs/workflow.md` (contributor extension guide)
- `docs/roadmap.md` (active checklist)
- `docs/design.md` (current architecture snapshot)
- `docs/config.md` (settings reference)
