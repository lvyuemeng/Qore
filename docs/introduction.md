# Qore Introduction

## What Qore is now

`Qore` is a crate-first quantitative research and backtest workspace.
The active stack is:

- `qore-core`: config, instruments, calendar, universe
- `qore-data`: fetchers, store, stock-selection pipelines
- `qore-factor`: lazy factor pipelines
- `qore-intelligence`: model training, artifacts, optional signal layers
- `qore-runner`: strategy-to-portfolio construction
- `qore-backtest`: execution simulation, accounting, metrics

The repository is still library-first. The current direction is to make data and
universe assembly read like a pipeline instead of a pile of eager helper calls.

## Store-first, pipeline-second mental model

`QoreStore` remains the persisted retrieval surface, but most stock workflow code
should now be written in two layers:

1. use `QoreStore` to persist and query datasets lazily
2. use `StockSelectionPipeline` to compose only the stock inputs needed for the
   current task

This replaces the older helper-heavy style where one function eagerly joined
profiles, statuses, fundamentals, forecasts, daily market data, and
announcement counts even when the caller only needed a subset.

## Storage semantics

Source of truth:

- `crates/qore-data/src/qore_data/store/schema.py`
- `crates/qore-data/src/qore_data/store/duckdb.py`

Core behavior:

- persisted layer: Parquet files under `data/raw` or the configured parquet root
- query layer: DuckDB views registered over those Parquet datasets
- write path: validate -> cast -> partition -> deduplicate -> write Parquet -> refresh views
- read path: one dataset/filter/column interface for both Parquet and DuckDB-backed reads

`QoreStore.read()` supports:

- `backend="parquet"`: Parquet-native lazy scans
- `backend="duckdb"`: DuckDB-backed filtered/projection queries returned as `LazyFrame`
- `backend="auto"`: currently resolves to Parquet priority

Practical rule:

- use Parquet-native reads for broad analytical scans
- use DuckDB reads for narrower filtered retrieval and lookup-like access

## Current named datasets

The store currently supports these persisted datasets:

- market data: `stock_ohlcv`, `fund_nav`, `derivative_ohlcv`
- fundamentals and metadata: `fundamentals`, `index_constituents`, `stock_profiles`, `stock_statuses`, `analyst_forecasts`, `announcements`
- factor and signal data: `factor_scores`, `news_scores`

## Stock selection pipeline

Source of truth:

- `crates/qore-data/src/qore_data/universe.py`

The stock workflow now starts from `StockSelectionPipeline.from_index(...)`:

```python
from datetime import date

from qore_data.universe import StockSelectionPipeline

pipeline = StockSelectionPipeline.from_index(
    store,
    index_symbol="8841431.WI",
    as_of=date(2026, 4, 19),
    announcement_start=date(2026, 4, 1),
    announcement_end=date(2026, 4, 30),
)
```

The pipeline stays lazy and stage-owned until you collect.

### Stage model

The available stages are:

- `profiles`: names, board, listing date, market cap, share counts
- `statuses`: ST state, suspension state, price-limit pct, tradeability
- `fundamentals`: latest available fundamental snapshot as of `as_of`
- `forecasts`: analyst forecast snapshot as of `as_of`
- `daily_market`: `amount`, `limit_up`, `limit_down` for the day
- `announcements`: announcement-window counts for the requested window

You can request exactly what you need:

```python
frame = pipeline.with_stages("profiles", "fundamentals").collect()
report = pipeline.with_category_inputs().category_report()
```

### Typical usage patterns

Build a full joined research frame:

```python
selection_frame = pipeline.with_default_selection_inputs().collect()
```

Build only candidate inputs required by a filter spec:

```python
from qore_data.universe import CandidateFilter, CandidateSort, StockCandidateSpec

candidate_spec = StockCandidateSpec(
    filters=(
        CandidateFilter("roe", "gt", 0.0, fill_null=float("-inf")),
        CandidateFilter("pe_ttm", "between", (0.0, 50.0)),
    ),
    sort_by=(CandidateSort("total_market_cap"),),
    top_n=20,
    min_listing_days=60,
    exclude_limit_up=True,
)

candidates = pipeline.candidates(candidate_spec)
universe = pipeline.to_universe(candidate_spec)
```

Build an industry or board report without unnecessary joins:

```python
category_report = pipeline.with_category_inputs().category_report()
```

### Why this is better than the old helper style

- category reports do not need fundamentals or daily-market joins
- candidate filtering only pulls the stages implied by the active filters and sorts
- announcement-window aggregation stays separate from reusable snapshots
- fundamentals stay a distinct latest-as-of snapshot instead of hidden post-processing
- callers can reuse one base pipeline and branch into multiple outputs cheaply

### What should be cached early vs resolved late

- cacheable snapshots: `index_constituents`, `stock_profiles`, `stock_statuses`, `analyst_forecasts`
- reusable as-of snapshot logic: latest available fundamentals as of the selection date
- request-time aggregations: announcement-window counts, category summaries, candidate ranking and top-n selection
- factor-related joins: fundamentals, forecasts, and daily market-state columns used by the active strategy filters or ranking rules
- execution-related state: suspension and limit-state fields, which are usually day-specific rather than long-lived metadata

## Current limitations

- store filters are still equality-based, not a full query language
- batch symbol retrieval usually still means reading a slice and refining in Polars
- `backend="auto"` currently prefers Parquet reads
- richer A-share metadata is still being expanded
- strategy YAML parsing and operator-facing config assembly still belong to the
  future CLI or workflow layer, not the data crate surface

## Where to start

- read `docs/design.md` for the target architecture
- read `docs/roadmap.md` for current rewrite priorities and missing pieces
- inspect `crates/qore-data/src/qore_data/universe.py` for the pipeline API
- use `examples/stock_ranking_workflow.py` as the current end-to-end reference path
