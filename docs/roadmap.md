# Qore Refactor Roadmap

## Objective

Move Qore to a signal-first architecture:

- factor computation stays in `qore-factor` as reusable transforms,
- runner and backtest consume only strategy decisions/signals,
- workflow is composition-only and does not materialize library internals.

## Non-negotiable Rules

1. No workflow-owned factor persistence.
2. No `factor_scores` dataset in execution path.
3. No `factor_name`-driven long-table contracts between runner/backtest.
4. Runner/backtest interfaces must be signal-oriented (`buy`/`hold`/`sell` + sizing metadata).
5. Strategy/workflow code must never rebuild ranking/storage semantics that belong in crates.

## Target Architecture

### 1) `qore-factor`: pure composition primitives

- Keep `FactorPipeline` as factor-compute/normalize/neutralize composition.
- Remove convenience methods that pre-bake domain composition (for example liquidity-capacity builder shortcuts).
- Keep outputs frame-native and wide by symbol/date.

### 2) `qore-runner`: signal contract as primary output

- Standardize runner output to typed decision frames:
  - `date`, `symbol`, `signal`, `weight_target`, optional `reason`.
- Ranking/cutoff ownership stays in runner strategy internals.
- No public contract requiring concrete factor labels.

### 2.1) `qore-runner.runner` concrete refactor scope

- Keep runner contracts factor-agnostic:
  - no `factor_name` field, no factor label dispatch, no long-table factor assumptions.
- Keep runner hot path frame-native:
  - no Python list/materialized object loops in selection/weight/decision delta flow.
- Keep source and strategy behavior protocol-dispatched:
  - runner consumes `Strategy.generate(...)` and provider frames via protocol, not hardcoded dataset logic.
- Unify runner outputs into one canonical decision-signal DataFrame contract:
  - required columns: `date`, `symbol`, `signal`, `weight_target`, `weight_current`, `weight_delta`.
  - optional columns: `reason`, `score_value`, strategy metadata.
- Avoid double representations in runner API:
  - no parallel object + frame semantics for the same decision state.

### 3) `qore-backtest`: execute decisions, not factors

- Backtest engine accepts runner decisions directly.
- Remove reliance on long-form factor score tables.
- Keep provider/store access focused on market/execution data, not factor-name dispatch.

### 3.1) `qore-backtest.engine` hard cleanup requirements

- No pre-baked internal dataset reads in engine runtime paths (for example market/news/factor table assumptions).
- All runtime data sources must be constructor-injected contracts.
- `BacktestEngine` must not own `QoreStore`; storage access belongs to externally provided sources.
- Source protocols must dispatch by behavior/capability, not by hardcoded source names or dataset labels.
- Remove all `factor_name`-style materialization/pivot compatibility from engine.
- Remove helper-style transformation layers that convert object -> frame repeatedly in hot paths.
- Keep run loop frame-native end-to-end (request/plan/fill/diagnostics as DataFrame operations).
- Keep `_empty_*` shapes owned by `BacktestRunState.initialize(...)` only.
- Keep diagnostics schema ownership co-located in `BacktestRunState.initialize(...)`.
- Replace summary object-to-frame adapter patterns with direct DataFrame summary records.

### 3.2) `qore-backtest.engine` concrete refactor scope

- Engine must not own implicit IO policy:
  - market/factor/overlay data acquisition must come from injected protocols only.
  - engine may cache frames, but must not decide dataset names/columns internally.
  - engine must not depend on `QoreStore` in type signature or construction path.
- Engine must reject factor-label style payloads:
  - any `factor_name` long-form contract is invalid in execution path.
  - accepted factor input is wide frame keyed by `symbol` (+ optional `date`).
- Engine execution should stay in DataFrame pipeline form:
  - request -> execution plan -> fills -> turnover -> diagnostics all expressed as frame transforms.
  - `list[Fill]`/tuple-row adaptation loops are forbidden in core path.
- Engine summary should be frame-first:
  - daily summary emitted directly as one-row DataFrame contract.
  - diagnostics schema co-located with summary contract.
- Source abstraction should be unified:
  - avoid repeated protocol variants that differ only by transport (`db` vs `dataframe`) while doing same behavior.
  - define one behavioral protocol per responsibility (factor, market, overlay) with shared frame contract.

### 4) `qore-data`: remove factor-score schema coupling

- Deprecate and remove `factor_scores` from canonical store schema.
- Keep universe/profile/fundamental/market datasets as selection inputs.
- If derived snapshots are needed, store strategy decision artifacts instead of factor-name rows.

### 4.1) `qore-intelligence` alignment

- Remove `factor_name` + long-table training/serving assumptions from intelligence workflows.
- Training/inference inputs must consume wide symbol/date signal frames.
- Model workflow should not require `factor_scores` pivots as canonical path.

### 5) `small_cap_strategy`: composition-only example

- No `WorkflowConfig` wrapper dataclass; accept crate `*Settings` objects directly.
- No writing/reading `factor_scores`.
- Use crate APIs only (selection -> runner -> backtest -> view).

## Implementation Order

1. Remove factor-score APIs from `qore-factor` and workflow usage.
2. Introduce/complete signal decision contract in runner.
3. Refactor `qore-runner.runner` to a single frame-native decision-signal contract and remove parallel object/frame decision semantics.
4. Refactor `qore-backtest.engine` to protocol-injected sources and fully frame-native execution/fill/summary path.
5. Remove `factor_name`/`factor_scores` coupling from `qore-intelligence` model workflow.
6. Remove remaining `factor_scores` schema dependencies in data paths.
7. Final cleanup of small-cap example to only wire library methods.

## Current Findings (No-Code Proposal Snapshot)

- `qore-backtest.engine` still contains helper-heavy flow and implicit source assumptions; this conflicts with source-injection and composition-only goals.
- `qore-backtest.engine` still has object-frame dual representations in fill/summary paths; this should be collapsed to one DataFrame-native contract.
- `qore-intelligence` model workflow still required store-level long-frame compatibility; this should be replaced with wide-frame dataset ingestion.
- source contracts in `engine.py` are still repetitive (`db`/`provider` wrappers with same runtime behavior) and not yet unified by capability.

## Progress Snapshot

- `qore-backtest.engine` now consumes injected market + signal-overlay sources (no hardcoded `news_scores` read path).
- `qore-backtest.engine` fill resolution path is now frame-native and removed row-wise fill object materialization loops in core execution.
- factor long-form guardrail is enforced in backtest execution (`factor_name` long-frame rejected).
- `small_cap_strategy` now injects factor source explicitly into backtest engine.
- `BacktestDaySummary` object-first daily assembly has been removed from run-loop usage; daily diagnostics are emitted directly as one-row DataFrame records.
- backtest-side factor column patching compatibility has been removed; runner now validates required factor columns and fails fast on invalid factor frames.
- `BacktestEngine` no longer owns `QoreStore`; factor/market/overlay IO dependencies are now supplied via injected source providers.
- store-backed sources now encapsulate storage dependency internally (`StoreFactorSource`, `StoreMarketDataSource`, `StoreSignalOverlaySource`) while engine stays storage-agnostic.
- `_factor_frame_for_day` no longer performs explicit schema-guard checks for required columns; contract validation stays in runner.
- repeated source-frame materialization behavior in `*_for_day` methods has been consolidated via a shared materialization helper.
- `qore-intelligence` no longer exposes store-coalescing workflow helpers; wide training frames are composed explicitly in workflow code (`store.read(...).select(...).join(...)`) then passed to `ModelPipeline.fit` and `ModelRegistry.save`.
- Remaining major slices:
  - unify remaining repetitive source protocol surface into slimmer capability-oriented adapters,
  - remove remaining `factor_scores` schema coupling in data/store paths.

## State Update (WP-B / WP-A)

- `WP-B` status: **partial**.
  - Done: source injection direction and frame-native fill path in core.
  - Done: summary emit is frame-first in run loop.
  - Done: engine storage dependency removed (`QoreStore` no longer in engine constructor/state).
  - Not done: final engine helper minimization pass and source-surface consolidation.
- `WP-A` status: **partial**.
  - Done: runner exposes decision signal frame.
  - Done: factor-frame required-column contract validation moved to runner.
  - Not done: final contract documentation + intelligence-side alignment.
- `WP-C` status: **complete**.
  - Done: removed canonical `factor_scores`/`factor_name` pivot dependency from intelligence model workflow store ingestion path.
  - Done: moved workflow inputs to wide-frame contract (`factor_columns`, `factor_dataset`).
  - Done: removed coalesced fit/save workflow helpers in favor of explicit composition through model pipeline + registry.

## Proposed Refactor Plan (No Code Yet)

0. boundary reset (first action):
   - declare hard ownership: strategy/runner own factor contract validity; backtest only executes provided frames.
   - remove all backtest-side compatibility behavior for missing factor columns and shape patching.
1. `runner` contract freeze:
   - define canonical decision-signal frame schema and document required/optional columns.
   - ensure selection and sizing metadata are emitted only as frame columns.
2. `runner` execution simplification:
   - remove any duplicated decision state that is not required by downstream contracts.
   - preserve diagnostics as frame-derived metrics, not object-derived counters.
3. `engine` source injection finalization:
   - define one behavioral protocol per responsibility (factor/market/overlay) with shared frame-return contract.
   - remove transport-specific duplication (`db` vs dataframe wrappers with same behavior).
   - remove internal pre-baked source read paths from engine runtime.
   - remove `QoreStore` from `BacktestEngine`; sources receive dependencies externally at composition time.
4. `engine` fill path vectorization:
   - replace row-wise fill conversion helpers with frame-native fill preparation and status derivation.
   - keep delay/session routing as frame transformations only.
5. `engine` summary/diagnostics frame-first rewrite:
   - remove `BacktestDaySummary` object-first usage in run loop.
   - write daily summary directly as DataFrame row(s) in run path.
   - append diagnostics from same summary frame contract without object->frame adapters.
6. `engine` helper minimization:
   - remove `engine` helpers that perform strategy-role compatibility (`normalize factor source`, `ensure factor exists` style behavior).
   - keep only execution-role helpers (execution plan, forced liquidation, return aggregation) and protocol boundary helpers.
   - keep empty schemas only in `BacktestRunState.initialize(...)`.
   - remove source-name-oriented helper/wrapper code; retain only capability-oriented protocol adapters.
7. `qore-intelligence` wide-frame migration:
   - replace factor-label dependent model IO with wide frame contracts.
8. parity + regression sweep:
   - compare nav/positions/turnover/diagnostics parity versus baseline scenarios.
   - add explicit tests asserting no `factor_name`/`factor_scores` dependency in runner/backtest intelligence paths.

## Engine/Runner Work Packages (Concrete)

### WP-A: Runner Contract Package (`qore-runner/src/qore_runner/runner.py`)

- Goal: one canonical frame contract from strategy decision to execution.
- Deliverables:
  - documented `decision_signals` schema and invariants.
  - no factor-label coupling in runner API.
  - diagnostics derived from frame state only.
- Verification:
  - runner flow tests pass with selection, overlay, rebalance cache, drawdown guard.
  - no new list/object-based hot path transformations.

### WP-B: Engine Source/Execution Package (`qore-backtest/src/qore_backtest/engine.py`)

- Goal: protocol-injected sources and frame-only execution path.
- Deliverables:
  - zero pre-baked source assumptions in runtime.
  - no `QoreStore` dependency in `BacktestEngine`.
  - unified source protocols by behavior (factor/market/overlay), no transport-duplicated interfaces.
  - no factor materialization compatibility (`factor_name` long form unsupported).
  - frame-native fill and summary pipelines.
  - no strategy-role compatibility logic in engine (`normalize_factor_source_frame`/`ensure_required_factor_columns` style behavior removed).
- Verification:
  - backtest flow parity on nav/positions/turnover/diagnostics.
  - tests cover missing/invalid injected source data and protocol guardrails.
  - tests assert backtest fails fast on invalid factor frames instead of patching/filling missing factor columns.
  - tests assert engine construction/execution does not require store handle.

### WP-B1: Immediate Next Slice (Plan First)

- Remove `BacktestDaySummary` object-centric row assembly from run path and switch to direct one-row DataFrame summary emit.
- Remove backtest-side factor fallback/compatibility helpers entirely; delegate contract strictness to strategy/runner.
- Keep engine strict: if factor input does not satisfy strategy-required columns, fail fast at boundary (no padding/default fill in engine).
- Remove `QoreStore` from `BacktestEngine` constructor and `from_settings` path.
- Collapse repetitive source wrappers into capability-based protocol adapters.
- Verify with focused parity tests (`test_backtest_flow.py`, `test_backtest_consistency.py`, `test_runner_flow.py`) and add one explicit boundary-failure test.

### WP-C: Intelligence Contract Package (`crates/qore-intelligence`)

- Goal: remove factor-label dependency from training/inference interfaces.
- Deliverables:
  - wide signal frame training/inference inputs.
  - no canonical dependence on `factor_scores`/pivot logic.
- Verification:
  - model workflow tests pass with wide-frame fixtures only.
