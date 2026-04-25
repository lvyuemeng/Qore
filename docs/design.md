# Qore Design (Current State)

## Status snapshot

Qore is a uv-workspace, crate-first quantitative research stack. Runtime crates
are library APIs; product CLI/operator contracts are deferred to a future layer.

Active runtime crates:

- `qore-data`
- `qore-factor`
- `qore-intelligence`
- `qore-runner`
- `qore-backtest`

`qore-core` is removed from active runtime/dependency graph.

## Core architecture principles

1. Frame-native runtime contracts (`pl.DataFrame`/`pl.LazyFrame`) in hot paths.
2. Crate-local typed settings for crate internals.
3. Workflow/composition layer can adapt external config to crate settings.
4. Deterministic behavior for runner and backtest execution.
5. Library-first crates with clean composition boundaries.

## Crate responsibilities

### qore-data

- source fetch/snapshot functions
- DuckDB + Parquet store (`QoreStore`)
- universe and stock selection pipeline (`StockSelectionPipeline`, `Universe`)

Current convenience direction:

- prefer direct universe output from pipeline: `pipeline.universe(...)`
- keep frame output available when needed: `pipeline.universe_frame(...)`

### qore-factor

- factor transforms and pipeline composition
- normalization/evaluation on factor outputs
- frame-native operations and lazy-first compute semantics

### qore-intelligence

- model pipeline/training/predict orchestration
- artifact/registry boundaries
- optional signal overlays and scoring components

### qore-runner

- strategy decision frame generation
- sizing and portfolio target construction
- typed, frame-native runner step contract

### qore-backtest

- execution planning/fill simulation
- accounting and diagnostics assembly
- frame-native result surfaces (`BacktestResult`)
- view/plot composition (`BacktestView`)

## Data and execution flow

Canonical workflow path:

1. data snapshot/fetch -> `QoreStore`
2. selection pipeline -> `Universe`
3. factor compute -> `factor_scores`
4. intelligence model/signal -> strategy signal
5. runner -> target weights
6. backtest -> fills/nav/diagnostics/view

## Settings boundary

Crate-local settings are authoritative for internals:

- `DataSettings`
- `IntelligenceSettings`
- `RunnerSettings`
- `BacktestSettings`

Global `QoreConfig` (if used) belongs to composition boundaries only.

## Backtest/result contract state

Backtest outputs are frame-native:

- `BacktestResult.nav`
- `BacktestResult.positions`
- `BacktestResult.turnover`
- `BacktestResult.fills`
- `BacktestResult.diagnostics`

Visualization/view API is method-owned:

- `result.view()` -> `BacktestView`
- `with_drawdown`, `with_benchmark`, `window`
- `plot().equity()/overview()/tearsheet()`

Plot backend dependency is managed through uv dependency group `viz`.

## Constraints and anti-patterns

- avoid object/list/dict reconstruction where frame expressions can represent logic
- avoid hidden coupling to global config in crate internals
- avoid embedding product CLI logic in crate packages
- avoid broad eager joins when stage-owned lazy composition can express intent

## Reference entrypoints

- user workflow intro: `docs/introduction.md`
- contributor workflow: `docs/workflow.md`
- active checklist and priorities: `docs/roadmap.md`
