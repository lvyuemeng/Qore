# Qore Rewrite Roadmap

## Objective

Ship a library-first, architecture-clean Qore stack (`qore-*` crates) with:

- typed, frame-native runner/backtest hot paths,
- crate-local settings (no global config coupling in crate internals),
- simplified package boundaries before CLI unification,
- one reproducible A-share workflow from fetch -> factor -> model -> runner -> backtest.

Legacy compatibility is allowed only as thin transitional shims.

## Current Direction (Hard Constraints)

1. Architecture-first changes are allowed to be compatibility-breaking.
2. Keep execution/dataflow Polars-native where practical.
3. Avoid opaque parsing in crate internals (`_safe_*`, permissive row coercion).
4. Keep typed semantics in hot paths (dataclasses/typed frame contracts).
5. Keep settings crate-local and exported at package root.
6. Keep workflow/CLI layer as the only future config-unification boundary.

## Boundary Decisions

### Legacy migration inventory (merged)

`docs/migration-inventory.md` is merged into this roadmap and treated as closed reference material.

- Legacy `src/quant_trade` modules are no longer planning anchors for active architecture work.
- Current development priority is direct `qore-*` crate evolution, not one-to-one legacy mapping.
- Practical carry-over policy:
  - `src/quant_trade/client/eastmoney.py` -> keep endpoint reverse-engineering only; runtime stays in `qore_data.fetcher.eastmoney`.
  - `src/quant_trade/feature/*` -> carry formulas only when needed by active factor workflow.
  - `src/quant_trade/model/*` -> carry concepts only; implementation authority stays in `qore_intelligence`.
  - Non-essential legacy scripts/transforms/config adapters stay retired unless required by the new workflow boundary.

Legacy build order notes are superseded by active workstreams and priority checklists in this document.

### Calendar ownership

- `qore_runner.calendar.TradingCalendar`: single runtime calendar owner.
- `qore_backtest`: consumes runner calendar directly; no crate-local duplicate.
- `qore_core.calendar`: removed as part of `qore-core` runtime-surface merge-down.

Policy: retain exactly one calendar module (`qore_runner.calendar`) during this phase.

### Settings ownership

- `BacktestSettings`, `RunnerSettings`, `DataSettings`, `IntelligenceSettings` now live in each package `__init__.py`.
- Legacy `settings.py` modules are removed.
- Internal and user imports should use package-root settings symbols.

## What Is Done

- Backtest typed refactor baseline landed:
  - typed execution structures (`SymbolExecutionSpec`, typed fill/market rows),
  - reduced dict/list row plumbing in active execution path,
  - session/dataset routing pre-bound in execution specs,
  - typed cadence scaffold (`daily`/`intraday`) on `BacktestSettings`.
- Runner selection/sizing path tightened toward frame-native joins.
- Runner/backtest efficiency slice landed:
  - `TargetPortfolio` now carries `weights_frame` as canonical runner output (dict weights removed),
  - backtest fill-request planning now joins target/current weight frames directly (no per-step dict->frame rebuild),
  - fill execution no longer materializes typed row dataclass lists before simulation,
  - execution metadata planning moved to frame-native expression pipeline instead of row-wise Python spec reconstruction.
- Runner sizing contracts are now frame-native end to end:
  - `PositionSizer.size` and `PositionSizer.cap` operate on `pl.DataFrame` weights,
  - `StrategyRunner.step` no longer accepts dict-based current weights.
- `Universe` ownership migration started:
  - `Universe` contract is now merged into `qore_data.universe` (no split `universe_contract` module),
  - `qore-data` package root exports `Universe`, reducing direct runtime dependence on `qore_core.universe`,
  - runner strategy session typing now imports `TradingSession` from `qore_data.universe` instead of `qore_core.instrument`.
- `qore-core` universe surface removed:
  - `qore_core.universe` module deleted,
  - `Universe` usage is now owned by `qore-data`.
- `Instrument` ownership migration started:
  - `qore_data.instrument` now owns `StockInstrument`/`FundInstrument`/`DerivativeInstrument` + `TradingSession`,
  - `qore-data` fetch/source/eastmoney runtime imports no longer depend on `qore_core.instrument`.
- `qore-core` dependency edges reduced:
  - `qore-data`, `qore-factor`, `qore-runner`, `qore-intelligence`, and `qore-backtest` no longer declare `qore-core` package dependency.
- `qore-core` removal completed in workspace code:
  - `crates/qore-core` source/tests/package metadata removed,
  - workspace source mapping for `qore-core` removed.
- Backtest run-state initialization is now method-owned (`BacktestRunState.initialize(...)`) instead of inline empty-frame construction in the engine loop.
- Session/dataset restrictions for execution planning are now owned by `Universe.execution_metadata()`; backtest engine consumes precomputed metadata.
- Backtest run-state moved further toward columnar storage:
  - day fills are accumulated as frame rows (`fills_frame`) and returned as DataFrame in `BacktestResult`.
- Universe API cleanup landed:
  - removed legacy `to_universe` / `to_universe_frame` compatibility methods,
  - canonical universe output is frame-first (`universe_frame(...)` + explicit `Universe.from_frame(...)` where needed).
- Snapshot argument model normalized:
  - `StockSnapshotSpec` now dispatches through a single typed `SnapshotQuery` input.
- Fundamentals retrieval extended:
  - selection scope now supports fundamentals date windows (`fundamentals_start`, `fundamentals_end`) and no longer hard-codes latest-only helper wrappers.
- Universe helper naming and API simplification completed:
  - removed `*_lazy` helper naming in universe internals,
  - removed duplicate `candidate_universe_frame(...)` surface and kept canonical `universe_frame(...)`.
- Backtest execution-plan guardrails + parity coverage landed:
  - execution planning now rejects symbols with missing/invalid execution session metadata,
  - daily fill-delay parity tests cover auction/nav/continuous session behavior.
- Backtest integration consistency coverage improved:
  - force-exit transition flow now validates cross-artifact consistency (`diagnostics`, `fills`, `turnovers`, `positions`, `nav`) in one compact integration test.
- Backtest diagnostics/result assembly is now frame-native end-to-end:
  - removed per-day `BacktestDiagnosticsRow` object materialization in engine assembly,
  - `BacktestResult` now returns positions/turnover as DataFrames (no dict/list reconstruction).
- Fill execution row assembly reduced further:
  - `_fills_from_frame(...)` no longer builds intermediate dict-record frames before final schema assembly,
  - fill output now materializes directly into final typed DataFrame columns.
- Crate-local settings migration completed for data/intelligence/runner/backtest package roots.
- Backtest visualization kickoff landed:
  - `BacktestResult.view()` + frame-native `BacktestView` landed,
  - `window(...)`, `with_drawdown(...)`, and `with_benchmark(...)` fluent view methods landed,
  - `plot()` facade landed with backend managed via uv dependency group (`viz`).
- Documentation split by audience landed:
  - `docs/introduction.md` now merges user intro + config + basic workflow usage,
  - `docs/workflow.md` is contributor-focused with crate-level extension guidance.
- Universe convenience API improved:
  - `StockSelectionPipeline.universe(...)` and `StockUniverseQuery.universe()` now return `Universe` directly,
  - users no longer need to manually wrap `universe_frame(...)` for common flows.

## Active Workstreams

### 1) Runner and backtest typed contracts

- Remove residual object-typed row parsing from hot loops.
- Keep strategy/backtest contracts centered on semantic frames.
- Add architecture tests for typed contracts and session/dataset consistency.
- Continue eliminating remaining Python list/set-heavy branching where frame-native joins/aggregations can express the same logic.

### 2) Config and package-boundary simplification

- Keep crate APIs free of `QoreConfig` requirements.
- Restrict any `QoreConfig` adapters to workflow/example composition boundaries.
- Continue shrinking `qore-core` to compatibility-only exports.
- Move remaining `qore_core` runtime contracts (`instrument`, `universe`) behind crate-local owners and keep only temporary shims.
- `qore-core` is no longer an active crate; config ownership is moved to workflow-local typed config structures.

### 3) Backtest realism and diagnostics

- Improve pending fill/retry behavior and execution diagnostics.
- Keep deterministic ordering with optional parallel batch fill boundaries.

### 4) Optional visualization support

- API kickoff landed per proposal (`BacktestView` + `BacktestResult.view()` + guarded `plot()` facade).
- Keep base install dependency-free; keep plotting runtime dependency behind uv dependency group (`viz`).
- Initial proposal artifact remains at `docs/backtest-visualization-proposal.md`.

Proposed pipe-style API (no implementation yet):

- Add a typed view model container, e.g. `BacktestView`:
  - `nav: pl.DataFrame`
  - `drawdown: pl.DataFrame | None`
  - `benchmarks: dict[str, pl.DataFrame]`
  - `trades: pl.DataFrame | None`
  - `diagnostics: pl.DataFrame | None`
- Expose fluent, method-owned pipeline style from `BacktestResult`:
  - `result.view().with_drawdown().with_benchmark("CSI300", bench_df).plot().equity()`
  - `result.view().window(start=..., end=...).plot().overview()`
- Keep plotting entrypoints behind objects (`.plot().equity()`, `.plot().tearsheet()`), not free helper functions.
- Keep base install dependency-free; activate plotting via uv dependency group only.

## Priority Checklist

### P0 - Stability after boundary cleanup

- [x] Validate no runtime imports still require removed `settings.py` modules.
- [x] Run lint/compile checks on touched crates.
- [x] Keep package root exports stable for user imports.

### P1 - Backtest engine completion

- [x] Finish typed/columnar result assembly paths.
- [x] Add parity tests for daily cadence before/after planner refinements.
- [x] Add guardrails against session inference from runtime market rows.

### P2 - Runner contract simplification

- [x] Keep one canonical decision frame (`selected`, `exclude_reason`, timing fields).
- [x] Reduce remaining helper indirection in decision/sizing flows.

### P3 - Workflow and operator path

- [x] Document one supported reproducible A-share runbook.
- [ ] Define CLI/entrypoint contract after crate boundaries stabilize.

## Acceptance Gates

The roadmap slice is complete when:

- active crate APIs compose with crate-local settings only,
- runner/backtest hot paths are typed and frame-native (no opaque row parsing),
- only one runtime calendar module remains (`qore_runner.calendar`),
- visualization scope is approved in roadmap before any plotting/runtime dependency changes,
- `qore-core` is removed from active workspace runtime and dependency graph,
- one end-to-end A-share workflow remains reproducible on supported entrypoints,
- crate deliverables remain library-first (no product CLI entrypoint embedded in crates).
