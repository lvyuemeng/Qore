# Qore Rewrite Roadmap

## Goal

Rewrite the repository into the `Qore` architecture defined in `docs/design.md`.
This is a staged replacement of legacy `src/quant_trade`, not an incremental refactor.

## Document Map

- `docs/design.md`: canonical architecture and design rules
- `docs/roadmap.md`: rewrite status, milestones, phases, and next priorities
- `docs/config.md`: current configuration structure and rules
- `docs/workflow.md`: current library-first workflow and near-term workflow target
- `docs/migration-inventory.md`: legacy-to-new mapping details

## Rewrite Principles

1. New architecture first, compatibility second.
2. No new product evolution in legacy `src/quant_trade` except migration support.
3. Every new class that depends on paths or runtime parameters should expose `from_config()`.
4. All new storage targets DuckDB + Parquet, never ArcticDB.
5. Instrument-specific behavior uses `singledispatch`, not `isinstance` chains.
6. Analytical pipelines stay lazy until an explicit collection point.
7. `.ai/refs/akshare/` and legacy provider code are references, not runtime dependencies.
8. Preserve formulas and endpoint knowledge, not legacy abstractions.
9. Separate behavior from persisted data structures, especially in model export/import design.

## Current State

Rewrite stage: late `Milestone B` and early `Milestone C`.

Current macro status:

- Foundation and workspace migration are complete enough for active crate-first work
- Data, factor, intelligence, runner, and backtest layers all exist and can interact
- One partial stock-ranking flow exists across the new crates
- The system is still library-first and not yet an operator-ready product workflow

What is working now:

- `qore-core`: typed config, calendar, instrument, and universe primitives are in place
- `qore-data`: EastMoney-backed stock and fund data paths work through DuckDB + Parquet storage
- `qore-factor`: lazy factor computation, normalization, evaluation, and persistence are present
- `qore-intelligence`: baseline ranking model, validation IC recording, and news-score persistence exist
- `qore-runner`: ranking flow, news blending, and volatility-aware sizing are partially integrated
- `qore-backtest`: dispatched fills, engine skeleton, and metrics exist for a basic simulation loop

What is not done yet:

- No official user-facing CLI or stable crate entrypoint contract
- No single documented daily workflow that a user can run without code assembly
- No production-grade EastMoney resilience evidence under sustained load
- No completed benchmark-quality stock-ranking backtest on real historical datasets
- No broader source expansion beyond the current EastMoney-first scope
- Current intelligence design still mixes model behavior and persisted model state more than desired
- Some current model-shape settings still live in config even though they should move into trained artifacts
- Stock-universe-specific metadata is still thin for serious A-share workflows

## Next Step

Immediate next step:

- deliver one reproducible A-share stock workflow that wires fetch -> factor -> model -> runner -> backtest from documented config and commands
- separate model artifact data, model registry behavior, and fit/predict pipeline behavior in the intelligence layer design

After that:

- harden EastMoney operational behavior
- tighten runner/backtest realism
- decide whether source expansion remains deferred or becomes active scope

## Macro Plan

Milestones are macro delivery checkpoints. Phases are implementation slices inside those milestones.

### Milestone A - Foundation

Goal: stop legacy-first development and establish the workspace, core types, and migration rules.

Contains phases:

- Phase 0: Freeze and prepare
- Phase 1: Workspace bootstrap
- Phase 2: Core domain rewrite

Status: largely complete.

### Milestone B - Data to Signal

Goal: make the new stack capable of producing usable inputs, factors, and ranking signals.

Contains phases:

- Phase 3: Data layer rewrite
- Phase 4: Factor engine rewrite
- Phase 5: Intelligence rewrite

Status: mostly in place, but still missing production hardening and cleaner workflow packaging.

### Milestone C - Portfolio to Backtest

Goal: turn model outputs into target portfolios and evaluate them through the new backtest layer.

Contains phases:

- Phase 6: Runner rewrite
- Phase 7: Backtest rewrite

Status: started, with runnable skeletons but incomplete accounting realism and operator workflow.

### Milestone D - Cutover

Goal: move active usage fully onto the new crates and retire legacy runtime paths.

Contains phases:

- Phase 8: Migration cutover and legacy removal

Status: not started.

## Phase Status Snapshot

Status key:

- `[ ]` not started
- `[-]` in progress
- `[x]` completed

| Phase | Scope | Status | Current summary |
| --- | --- | --- | --- |
| Phase 0 | Freeze and prepare | `[x]` | Design, migration stance, and `.ai/` reference workflow are established |
| Phase 1 | Workspace bootstrap | `[x]` | uv workspace and crate layout exist; legacy package is no longer primary |
| Phase 2 | Core domain rewrite | `[x]` | Typed config, instrument, calendar, and universe foundations are in place |
| Phase 3 | Data layer rewrite | `[-]` | Dispatch and store exist; EastMoney and store hardening still remain |
| Phase 4 | Factor engine rewrite | `[-]` | Lazy factor pipeline exists; factor breadth and tests still need expansion |
| Phase 5 | Intelligence rewrite | `[-]` | Baseline ranking pipeline exists; broader validation and signal maturity remain |
| Phase 6 | Runner rewrite | `[-]` | Ranking, news blending, and volatility-aware sizing exist; portfolio construction remains shallow |
| Phase 7 | Backtest rewrite | `[-]` | Engine skeleton and metrics exist; deeper accounting realism remains |
| Phase 8 | Cutover and legacy removal | `[ ]` | New crate flow is not yet the sole supported runtime path |

## Phase Details

### Phase 0 - Freeze and Prepare

Macro goal: stop deepening the legacy architecture and prepare for parallel rewrite.

Micro deliverables:

- keep `docs/design.md` as the canonical target spec
- keep `.ai/` as local reference material only
- maintain migration inventory and legacy marking

Status: complete.

### Phase 1 - Workspace Bootstrap

Macro goal: replace the single-package layout with a uv workspace monorepo.

Micro deliverables:

- workspace root uses `uv`
- crate skeletons exist under `crates/`
- shared lint, type, and test tooling is centralized

Status: complete.

### Phase 2 - Core Domain Rewrite (`qore-core`)

Macro goal: establish immutable, typed platform foundations.

Micro deliverables:

- sealed instrument model
- centralized `QoreConfig`
- session-aware trading calendar
- homogeneous universe model

Status: complete enough for dependent crate work.

### Phase 3 - Data Layer Rewrite (`qore-data`)

Macro goal: rebuild ingestion and storage around typed protocols and Parquet-lake storage.

Micro deliverables:

- `singledispatch` fetch APIs by instrument type
- EastMoney fetcher reimplemented without runtime `akshare`
- DuckDB + Parquet store and dataset schema registry
- validated read/write and named dataset views

Current state:

- filter validation, repeated-write deduplication, and several EastMoney datasets are already implemented
- broader endpoint coverage and operational hardening remain unfinished
- richer stock-universe metadata still remains unfinished, especially historical constituent, industry, and status views

### Phase 4 - Factor Engine Rewrite (`qore-factor`)

Macro goal: move feature engineering to composable lazy factor pipelines.

Micro deliverables:

- `Factor` protocol with `requires` and `produces`
- lazy `FactorPipeline`
- normalization, neutralization, evaluation, and persistence
- initial useful factor families for ranking workflows

Current state:

- lazy computation, normalization, neutralization, evaluation, and `factor_scores` persistence exist
- realized-volatility plus multiple quality, cashflow, and growth factors are already present
- more factor breadth and stronger reconstruction/testing coverage remain

### Phase 5 - Intelligence Rewrite (`qore-intelligence`)

Macro goal: unify model ranking and news signals into one intelligence layer.

Micro deliverables:

- model normalizers and ranking model
- fit/predict pipeline separated from artifact data and registry behavior
- purged and walk-forward-style validation
- optional signal modules and signal combination

Current state:

- baseline multi-horizon ranking pipeline exists and records validation IC during fitting
- article-derived news scoring exists
- current implementation still conflates some persisted model metadata with runtime pipeline behavior
- current config still carries some model-shape settings that should migrate into exported artifacts
- signal-stack maturity and broader validation coverage still remain

### Phase 6 - Runner Rewrite (`qore-runner`)

Macro goal: define strategy generation and portfolio construction independently from execution simulation.

Micro deliverables:

- strategy protocol and ranking-based strategy
- screener and behavioral gating flows
- sizers, risk manager, and `StrategyRunner`

Current state:

- runner now threads `news_scores` and factor-derived volatility into portfolio generation
- broader strategy depth, risk behavior, and cleaner portfolio construction still remain

### Phase 7 - Backtest Rewrite (`qore-backtest`)

Macro goal: implement session-aware execution simulation and portfolio accounting.

Micro deliverables:

- dispatched `fill_order()` by instrument type
- `BacktestEngine.from_config()`
- portfolio/accounting result model and metrics

Current state:

- engine skeleton, dispatched fills, and metrics exist
- deeper accounting realism and more complete integration still remain

### Phase 8 - Migration Cutover and Legacy Removal

Macro goal: move active workflows fully to `qore-*` crates and retire obsolete runtime paths.

Micro deliverables:

- create supported entrypoints on top of the new crates
- remove ArcticDB runtime dependence
- retire `src/quant_trade`, legacy scripts, and stale local artifacts
- update docs to describe only the new architecture

Status: not started.

## Active Implementation Priorities

### Now

- build one supported end-to-end A-share example workflow
- redesign intelligence persistence boundary so model artifact data is separate from model loading/saving behavior
- document exact config, datasets, and command sequence
- verify the current crate chain on one benchmark universe
- add the stock-universe information needed for credible pool definition and category-aware evaluation

### Next

- add EastMoney resilience testing, retry budgeting, and crawl telemetry
- improve factor coverage only where it directly strengthens the current ranking flow
- tighten runner and backtest realism around sizing, accounting, and diagnostics
- extend stock-universe metadata with useful A-share-specific information from EastMoney-reimplemented endpoints

### Later

- decide explicit source expansion scope for Yahoo and crypto markets
- perform cutover and legacy removal after the new path is genuinely usable

## Legacy Mapping

| Legacy area | Status | Target home |
| --- | --- | --- |
| `src/quant_trade/transform.py` | retire | none |
| `src/quant_trade/client/eastmoney.py` | reverse engineer and rewrite | `qore-data/fetcher/eastmoney.py` |
| `src/quant_trade/provider/akshare.py` | do not port directly | reference only via `.ai/refs/akshare/` |
| `src/quant_trade/provider/baostock.py` | optional future reference | possible future source adapter |
| `src/quant_trade/config/arctic.py` | retire | `qore-data/store/duckdb.py` |
| `src/quant_trade/feature/process.py` | formula extraction only | `qore-factor/*` |
| `src/quant_trade/feature/store.py` | retire | split across `qore-data` and `qore-core` |
| `src/quant_trade/model/process.py` | selective reuse | `qore-intelligence/model/*` |
| `src/quant_trade/model/lgb.py` | selective reuse | `qore-intelligence/model/*` |
| `src/quant_trade/model/store.py` | rewrite | `qore-intelligence/model/pipeline.py` |
| `scripts/smoke_train.py` | retire | crate CLI / examples |

## Risks

- directly migrating legacy classes would drag old abstractions into the new design
- ArcticDB compatibility pressure could weaken the rewrite boundary
- runtime reuse of AkShare would violate the target contract
- eager legacy assumptions may be hidden inside old factor pipelines and need lazy re-expression
- early skeleton success can mask the remaining operator and production gaps

## Definition of Done

The rewrite is complete when:

- repository structure matches the monorepo design
- all active code paths run through `qore-*` crates rather than `quant_trade`
- storage is DuckDB + Parquet rather than ArcticDB
- instrument-specific behavior is expressed via `singledispatch`
- model training and loading are config-derived and versioned
- one full stock-ranking workflow runs end to end through the new stack
- legacy `src/quant_trade` is removed or archived out of the active path
