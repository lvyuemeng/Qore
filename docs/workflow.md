# Qore Contributor Workflow

## Purpose

This guide is for contributors extending crate features.

Hard rule: crates remain libraries. Do not add product CLI entrypoints in
crate packages; put runnable examples under `examples/`.

## Contributor extension map

- `qore-data`: dataset/snapshot/universe and pipeline stage logic
- `qore-factor`: factor implementations and factor pipeline behavior
- `qore-intelligence`: model and signal pipeline capabilities
- `qore-runner`: strategy decision/sizing contracts
- `qore-backtest`: execution realism, diagnostics, metrics, and view/plot APIs

## Architecture rules

- Keep hot paths frame-native (`pl.DataFrame` / `pl.LazyFrame`).
- Prefer joins/select/aggregations over Python list/dict reconstruction.
- Keep crate internals on crate-local typed settings.
- Keep `QoreConfig` adapters in composition code only.
- Maintain deterministic strategy/backtest behavior.
- Avoid helper APIs that coalesce multiple pipeline stages into a single opaque call;
  compose explicit steps at workflow boundary.

## Concrete extension recipes

### 1) Add a new factor

1. Add implementation under `crates/qore-factor/src/qore_factor/...`.
2. Keep it lazy; do not call `.collect()` inside factor compute.
3. Register/use it in `FactorPipeline` composition.
4. Add tests in `crates/qore-factor/tests/test_factor_pipeline.py`.

Example factor:

```python
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class AmountMomentumFactor:
    name: str = "amount_mom_20"

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (
                pl.col("amount")
                .cast(pl.Float64, strict=False)
                .pct_change(20)
                .over("symbol")
            ).alias(self.name)
        )
```

### 2) Add/adjust stock selection stage

Primary file: `crates/qore-data/src/qore_data/universe.py`.

Rules:

- stage method should return new pipeline via `_replace(...)`
- joins remain lazy
- collect only at explicit output boundaries (`collect`, `candidates`, `universe_frame`, `universe`)

Example shape:

```python
def with_custom_stage(self) -> StockSelectionPipeline:
    custom = self.store.read_duckdb("custom_dataset", filters={"as_of": self.scope.as_of})
    frame = self.frame.join(custom, on="symbol", how="left")
    return self._replace(frame=frame, stage="profiles")
```

### 3) Extend runner strategy behavior

Primary files:

- `crates/qore-runner/src/qore_runner/strategy.py`
- `crates/qore-runner/src/qore_runner/strategies/`

Rules:

- output canonical decision frame
- keep sizing contracts DataFrame-first
- avoid ad-hoc symbol loops for frame-computable operations

### 4) Extend backtest behavior

Primary file: `crates/qore-backtest/src/qore_backtest/engine.py`.

Rules:

- execution planning from `Universe.execution_metadata()`
- diagnostics/result assembly remains frame-native
- preserve deterministic ordering even if parallelization is used in bounded blocks

### 5) Extend view/visualization

Primary file: `crates/qore-backtest/src/qore_backtest/view.py`.

Rules:

- extend `BacktestView` methods as pure-return transforms
- keep plotting behind method-owned facade (`result.view().plot().equity()`)
- keep plotting dependency in uv dependency group `viz`

### 6) Extend intelligence model workflow

Primary files:

- `crates/qore-intelligence/src/qore_intelligence/model/pipeline.py`
- `crates/qore-intelligence/src/qore_intelligence/model/registry.py`

Rules:

- keep training/inference inputs as wide frames (`date`, `symbol`, factor columns)
- do not add pipe-coalesced helpers such as `fit_and_save_*`; keep fit/save as explicit composition
- keep store ingestion helper focused on frame assembly only (no model fit/registry side effects)

## Tests to add with each feature

- one behavior-focused unit test for local logic
- one integration-style test for cross-artifact consistency when contracts change

Good integration candidates:

- runner decision frame + sizer output consistency
- backtest fill requests + fills + diagnostics parity
- view transforms (`window`, `with_drawdown`) with deterministic assertions

## Validation checklist

```bash
uv run ruff format .
uv run ruff check .
uv run ty check .
uv run pytest
```

If unrelated workspace blockers prevent full-suite pass, run targeted crate checks
and document blockers clearly in change notes.

## Final contributor checklist

- [ ] feature implemented as library API in the correct crate
- [ ] no crate-level product CLI added
- [ ] tests cover behavior and integration consistency
- [ ] docs updated (`docs/introduction.md`, `docs/workflow.md`, `docs/roadmap.md`, `docs/design.md`)
- [ ] roadmap progress/checklist updated for touched stream
