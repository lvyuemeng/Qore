# Qore Contributor Workflow

## Purpose

This guide is for contributors extending crate features.

Hard rule: crates remain libraries. Do not add product CLI entrypoints in
crate packages; put runnable examples under `examples/`.

## Contributor extension map

- `qore-data`: fetcher sectors, store schemas, selection views, candidate filtering.
- `qore-factor`: factor classes and `FactorPipeline` behavior.
- `qore-intelligence`: model and signal pipeline capabilities.
- `qore-runner`: strategy decision/sizing contracts.
- `qore-backtest`: execution simulation, diagnostics, metrics, and view/plot APIs.

## Architecture rules

- Keep hot paths frame-native (`pl.DataFrame` / `pl.LazyFrame`).
- Prefer joins/select/aggregations over Python list/dict reconstruction.
- Keep crate internals on crate-local typed settings.
- Keep `QoreConfig` adapters in composition code only.
- Maintain deterministic strategy/backtest behavior.
- Greedy storage: fetch maximally, filter at read time via SQL views.
- No `hasattr`/`isinstance`/`getattr` dynamic dispatch in hot paths; use protocols and data structures.

## Concrete extension recipes

### 1) Add a new factor

1. Add implementation under `crates/qore-factor/src/qore_factor/...`.
2. Keep it lazy; do not call `.collect()` inside factor compute.
3. Register/use it in `FactorPipeline` composition.
4. Add tests in `crates/qore-factor/tests/`.

```python
from __future__ import annotations
from dataclasses import dataclass
import polars as pl

@dataclass(frozen=True, slots=True)
class AmountMomentumFactor:
    produces: str = "amount_mom_20"

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("amount").cast(pl.Float64)
            .pct_change(20).over("symbol")
            .alias(self.produces)
        )
```

### 2) Add a new data fetcher sector

1. Create `fetcher/your_sector.py` extending `BaseJsonFetcher` from `fetcher/_base.py`.
2. Use `self._json_fetcher.fetch_json(RequestSpec(...))` for HTTP calls.
3. Use `_fetch_paginated` (inherited) for multi-page endpoints.
4. For batch endpoints that need chunking, follow `_gather_chunked` pattern from `financial.py`.
5. Wire into `StockPipeline` in `fetch.py`.
6. Add store schema in `store/schema.py`.
7. Export from `fetcher/__init__.py`.

```python
from qore_data.fetcher._base import BaseJsonFetcher, build_json_fetcher
from qore_data.fetcher.http import RequestSpec

class YourFetcher(BaseJsonFetcher):
    async def batch_your_data(
        self, instruments: list[StockInstrument], as_of: date
    ) -> pl.DataFrame:
        ...
```

### 3) Add/adjust selection logic

Primary files:

- `crates/qore-data/src/qore_data/fetch.py` — `CandidateSpec`, `StockCandidateSpec`, `CandidateEligibilityPolicy`, `CandidateFilter`.
- `crates/qore-data/src/qore_data/sources/__init__.py` — `SelectionSource` SQL view.

Rules:

- Filters are `pl.Expr` expressions composed in `StockCandidateSpec.apply()`.
- `SelectionSource._SELECTION_VIEW_SQL` defines the DuckDB view joining all datasets.
- Capacity/liquidity filters go in `CandidateEligibilityPolicy.capacity_filter()`.

### 4) Extend runner strategy behavior

Primary files:

- `crates/qore-runner/src/qore_runner/strategies/`
- `crates/qore-runner/src/qore_runner/sizer.py`

Rules:

- Output canonical decision frame.
- Keep sizing contracts DataFrame-first.
- Avoid ad-hoc symbol loops for frame-computable operations.

### 5) Extend backtest behavior

Primary file: `crates/qore-backtest/src/qore_backtest/engine.py`.

Protocol contracts:

- `MarketDataSource` — provides OHLCV data for fills.
- `DayFrameSource` — factor/signal/decision overlay data.
- `NullSignalOverlaySource` / `NullDecisionOverlaySource` — pass-through implementations.

Engine does not import DuckDB, Parquet, or `QoreStore`. Data sources are injected.

### 6) Extend view/visualization

Primary file: `crates/qore-backtest/src/qore_backtest/view.py`.

Rules:

- Extend `BacktestView` methods as pure-return transforms.
- Keep plotting behind method-owned facade (`result.view().plot().equity()`).
- Plotting dependency in uv dependency group `viz`.

### 7) Add a new runnable workflow

Place under `examples/`. Follow the pattern from `examples/small_cap_strategy/`:

```text
examples/your_strategy/
├── pyproject.toml
├── README.md
├── src/your_strategy/
│   └── workflow.py     # StrategySpec, prepare_*_data, run_*_workflow, main, cli
└── tests/
    └── test_your_strategy.py
```

Key imports:

- `qore_data.DataSettings`, `qore_data.StockPipeline`, `qore_data.Universe`
- `qore_data.sources.SelectionSource`, `qore_data.sources.MarketSource`
- `qore_factor.FactorPipeline`, `qore_factor.fundamental.quality.DebtToAssetRatioFactor`
- `qore_runner.RebalanceSchedule`, `qore_runner.runner.StrategyRunner`, `qore_runner.sizer.EqualWeightSizer`
- `qore_runner.strategies.crosssectional.CrossSectionalScreener`
- `qore_backtest.BacktestSettings`, `qore_backtest.TradingCalendar`, `qore_backtest.engine.BacktestEngine`
- `qore_backtest.DateColumnDayFrameSource`, `qore_backtest.NullSignalOverlaySource`

## Tests to add with each feature

- One behavior-focused unit test for local logic.
- One integration-style test for cross-artifact consistency when contracts change.
- Live IO tests gated behind `_require_live_io()` (pattern from `test_fetcher.py`).

## Validation checklist

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

If unrelated workspace blockers prevent full-suite pass, run targeted crate checks
and document blockers clearly in change notes.

## Final contributor checklist

- [ ] Feature implemented as library API in the correct crate.
- [ ] No crate-level product CLI added.
- [ ] Tests cover behavior and integration consistency.
- [ ] Docs updated (`docs/design.md`, `docs/roadmap.md`, `docs/workflow.md`).
- [ ] Fetcher additions follow `BaseJsonFetcher` pattern and include telemetry.
