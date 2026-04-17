# Qore Rewrite Roadmap

## Goal

Rewrite the repository into the `Qore` architecture defined in `docs/design.md`.
This is not an incremental refactor of `quant_trade`; it is a staged replacement of
the current single-package project with a uv-workspace monorepo built around typed,
config-driven crates and a ranking-first workflow.

## Current Repository Assessment

The current codebase is useful as a source of domain knowledge and endpoint reverse
engineering, but it does not match the target architecture.

### What exists today

- Single Python package: `src/quant_trade`
- Direct provider integrations: `akshare`, `baostock`, EastMoney client wrappers
- ArcticDB/LMDB-centered storage in `src/quant_trade/config/arctic.py`
- Eager `polars.DataFrame` feature engineering in `src/quant_trade/feature/*`
- LightGBM training stack in `src/quant_trade/model/*`
- Ad hoc scripts and local model artifacts in `scripts/`, `model/`, `db/`, `logs/`

### Main gaps vs `docs/design.md`

- No uv workspace or crate layout under `crates/`
- Package identity still `quant-trade`, not `qore`
- Direct `akshare` imports are widespread, while target design bans them in crates
- Storage backend is ArcticDB, while target design requires DuckDB + Parquet lake
- Architecture is eager/DataFrame-oriented, while target design is LazyFrame-first
- No sealed instrument union, homogeneous universe model, or session-typed routing
- No `singledispatch` public data/backtest API by instrument type
- Config is partial and file-path-driven, not centralized as `QoreConfig`
- Intelligence/model/signal/runner/backtest boundaries do not exist yet
- Existing scripts couple training, storage, feature joining, and prediction tightly

### Migration stance

- Preserve the current project only as reference code during the rewrite
- Preserve `.ai/` as local reference material only; keep it gitignored and out of runtime dependencies
- Do not try to adapt ArcticDB storage or `quant_trade` package into the final shape
- Reuse endpoint knowledge, normalization logic, and factor ideas where valuable
- Build new crates in parallel, then retire legacy modules once parity is reached

## Rewrite Principles

1. New architecture first, compatibility second.
2. No new features in legacy `src/quant_trade` unless needed for migration support.
3. Every new class that depends on paths or tuning params must use `from_config()`.
4. All new data-path logic targets DuckDB + Parquet, never ArcticDB.
5. All instrument-specific behavior uses `singledispatch`, never `isinstance` chains.
6. All new analytical pipelines stay lazy until explicitly allowed collection points.
7. Treat `.ai/refs/akshare/` and current provider code as references, not dependencies.
8. Keep `.ai/` ignored by git; it is a local reverse-engineering workspace, not product source.

## Target End State

```text
qore/
|- .ai/
|- crates/
|  |- qore-core/
|  |- qore-data/
|  |- qore-factor/
|  |- qore-intelligence/
|  |- qore-runner/
|  \- qore-backtest/
|- data/
|- models/
|- docs/
|- justfile
\- pyproject.toml
```

.ai/ remains local-only reference material and should stay gitignored.

Legacy directories such as `src/quant_trade`, `db/`, `model/`, and ad hoc scripts
should become migration-only and then be removed.

## Progress Checklist

Status key:

- `[ ]` not started
- `[-]` in progress
- `[x]` completed

### Phase 0 checklist

- `[x]` `docs/design.md` exists as the canonical target spec
- `[x]` `docs/roadmap.md` exists and frames the rewrite as replacement, not refactor
- `[x]` `.ai/` is treated as reference-only material
- `[x]` `.ai/` is ignored by git
- `[ ]` Add or finalize `.ai/AGENTS.md` rules aligned with the design doc
- `[x]` Add or finalize `just ai-refs` for local reference sync
- `[x]` Mark legacy areas in `README.md`
- `[x]` Create explicit migration inventory per legacy module

### Phase 1 checklist

- `[x]` Rewrite root `pyproject.toml` to uv workspace format
- `[x]` Create `crates/` directory structure
- `[x]` Create `crates/qore-core`
- `[x]` Create `crates/qore-data`
- `[x]` Create `crates/qore-factor`
- `[x]` Create `crates/qore-intelligence`
- `[x]` Create `crates/qore-runner`
- `[x]` Create `crates/qore-backtest`
- `[x]` Move shared lint/type/test config to workspace root
- `[x]` Stop treating `src/quant_trade` as the primary package

### Phase 2 checklist

- `[x]` Implement `qore_core/instrument.py`
- `[x]` Implement `qore_core/config.py`
- `[x]` Implement `qore_core/calendar.py`
- `[x]` Implement `qore_core/universe.py`
- `[x]` Add config-derived constructors where required
- `[x]` Verify homogeneous-universe constraint

### Phase 3 checklist

- `[x]` Implement typed source protocols in `qore-data`
- `[x]` Implement `fetch_daily()` via `singledispatch`
- `[x]` Implement `fetch_minute()` via `singledispatch`
- `[x]` Implement `fetch_tick()` via `singledispatch`
- `[x]` Implement `fetch_fundamentals()` via `singledispatch`
- `[-]` Rebuild EastMoney fetcher without runtime `akshare` dependency
- `[x]` Implement dataset schema registry
- `[-]` Implement DuckDB + Parquet store
- `[x]` Register named dataset views

Current note:

- `qore-data` now validates read filters, deduplicates repeated parquet writes, and covers fund holdings, analyst forecast data, and stock announcements alongside daily/nav/fundamental EastMoney paths; store ergonomics and broader endpoint coverage remain in progress.

### Phase 4 checklist

- `[-]` Define `Factor` protocol
- `[-]` Implement core factor families
- `[-]` Implement lazy `FactorPipeline`
- `[-]` Implement normalization and neutralization flow
- `[x]` Persist factor outputs into `factor_scores`

Current note:

- `qore-factor` now computes factors lazily, supports normalization and neutralization, evaluates cross-sectional IC/ICIR, persists standardized outputs into `factor_scores`, and includes realized-volatility plus legacy-inspired fundamental quality, cashflow, and growth formulas extracted as design-aligned factors rather than direct class ports.

### Phase 5 checklist

- `[x]` Implement model normalizers
- `[x]` Implement `MultiHorizonRanker`
- `[x]` Implement config-driven `ModelPipeline`
- `[-]` Port purged / walk-forward validation concepts into the new layer
- `[-]` Implement signal modules (`triage`, `sentiment`, `llm`, `score`)
- `[x]` Implement `SignalCombiner`

Current note:

- `qore-intelligence` now has config-derived normalizers, a baseline multi-horizon ranking model, a persisted `ModelPipeline`, a news pipeline that scores and writes article-derived `news_scores`, and purged validation primitives that now feed recorded validation ICs during fitting.

### Phase 6 checklist

- `[-]` Define `Strategy` protocol
- `[-]` Implement ranking-based strategy
- `[-]` Implement screener strategy
- `[-]` Implement behavioral gating wrapper
- `[-]` Implement sizers and risk manager
- `[-]` Implement `StrategyRunner`

Current note:

- `qore-runner` now threads `news_scores` through strategy generation, blends them in the ranking path, and includes an inverse-volatility sizer with capped renormalization; broader end-to-end portfolio construction still remains in progress.

### Phase 7 checklist

- `[-]` Implement `fill_order()` dispatch by instrument type
- `[-]` Implement `BacktestEngine`
- `[-]` Implement portfolio/accounting result model
- `[-]` Implement metrics module

Current note:

- `qore-backtest` now includes dispatched fills, a runnable engine skeleton, and richer result metrics, but still needs tighter integration with completed data/model layers.

### Phase 8 checklist

- `[ ]` Move active workflows to new crate entrypoints
- `[ ]` Remove dependency on ArcticDB runtime paths
- `[ ]` Remove dependency on `src/quant_trade`
- `[ ]` Retire or archive `scripts/smoke_train.py`
- `[ ]` Retire or archive legacy local artifacts under `db/`, `model/`, and `logs/`
- `[ ]` Update docs to describe only the new architecture

## Phased Roadmap

## Phase 0 - Freeze and Prepare

Objective: stop deepening the legacy architecture and prepare the repository for a
parallel rewrite.

Deliverables:

- Keep `docs/design.md` as the canonical target spec
- Add `.ai/AGENTS.md` with the rules from the design doc
- Add `just ai-refs` and clone `.ai/refs/akshare` locally
- Create a migration inventory document mapping legacy modules to target crates
- Mark `src/quant_trade` as legacy in `README.md`

Exit criteria:

- Team works against the Qore design, not the current package shape
- Reference repo for AkShare endpoint reading is available locally and ignored by git

## Phase 1 - Workspace Bootstrap

Objective: replace the single-package layout with a uv workspace monorepo.

Deliverables:

- Rewrite root `pyproject.toml` to use `[tool.uv.workspace]`
- Create crate skeletons under `crates/`
- Rename package identity from `quant-trade` to `qore`
- Standardize tool config: ruff, mypy strict, pytest, optional extras per crate
- Add crate-local tests and minimal import smoke checks

Suggested order:

1. `qore-core`
2. `qore-data`
3. `qore-factor`
4. `qore-intelligence`
5. `qore-runner`
6. `qore-backtest`

Exit criteria:

- `uv sync` installs a workspace with independent crates
- Legacy package is no longer the primary entrypoint

## Phase 2 - Core Domain Rewrite (`qore-core`)

Objective: establish the immutable, typed foundation that every other crate uses.

Scope:

- `instrument.py`: `StockInstrument`, `FundInstrument`, `DerivativeInstrument`
- `config.py`: full `QoreConfig` tree with `from_yaml()`
- `calendar.py`: `TradingCalendar` and session-aware `fill_date()`
- `universe.py`: homogeneous `Universe`

Migration notes:

- Do not carry over current helper, storage, or provider abstractions from `src/quant_trade`

Exit criteria:

- Core types exist exactly in the shape expected by `docs/design.md`
- Unit checks cover homogeneous universe and session fill semantics

## Phase 3 - Data Layer Rewrite (`qore-data`)

Objective: rebuild ingestion and storage around typed protocols and Parquet lakehouse
storage.

Scope:

- Source protocols: `StockSource`, `FundSource`, `DerivativeSource`
- Public fetch API: `fetch_daily`, `fetch_minute`, `fetch_tick`, `fetch_fundamentals`
- EastMoney fetcher implemented with `httpx`, using `.ai/refs/akshare/` only as reference
- `QoreStore` with DuckDB + Parquet datasets and named schemas
- Registration of dataset views and validated read/write paths

Legacy reuse candidates:

- EastMoney endpoint knowledge from `src/quant_trade/client/eastmoney.py`
- Column mapping logic from current builders/parsers
- Universe/index constituent mapping ideas from provider code

Legacy code to retire, not port directly:

- `src/quant_trade/provider/akshare.py`
- `src/quant_trade/provider/baostock.py`
- `src/quant_trade/config/arctic.py`
- `src/quant_trade/feature/store.py`

Exit criteria:

- No crate imports `akshare`
- Daily/fundamental dispatch works by instrument type
- Store reads return `pl.LazyFrame`
- Storage writes target `data/raw` and DuckDB views

## Phase 4 - Factor Engine Rewrite (`qore-factor`)

Objective: move feature engineering from eager ad hoc transforms to composable lazy
factor pipelines.

Scope:

- `Factor` protocol with `requires` and `produces`
- `FactorPipeline.add()`, `.normalize()`, `.neutralize()`, `.run()`, `.evaluate()`
- Initial factor set: momentum, book-to-price, ROE stability, SUE, carry
- Standardized persistence into `factor_scores`

Legacy reuse candidates:

- Factor formulas from `src/quant_trade/feature/process.py`
- Cross-sectional normalization concepts from `SectorGroup` / `CrossSectionFlow`

Important rewrite decisions:

- Preserve formulas, not classes
- Replace eager `.collect()` patterns with LazyFrame expressions
- Separate raw factor computation from normalization storage

Exit criteria:

- Factor pipeline can compute and normalize lazily from store-backed inputs
- `factor_scores` dataset stores both `raw_value` and `z_score`

## Phase 5 - Intelligence Rewrite (`qore-intelligence`)

Objective: unify model ranking and news signal generation behind one intelligence
layer.

Scope:

- `model/normalizer.py`
- `model/lgbm_rank.py`
- `model/pipeline.py`
- `signal/triage.py`, `sentiment.py`, `llm.py`, `score.py`
- `combine.py`

Legacy reuse candidates:

- Purged CV ideas from `src/quant_trade/model/process.py`
- LightGBM tuning/training logic from `src/quant_trade/model/lgb.py`
- Model metadata persistence concepts from `src/quant_trade/model/store.py`

Important rewrite decisions:

- Rebuild training around `ModelPipeline.from_config()` and versioned model roots
- Replace current script-driven train/predict flow with reusable pipeline objects
- Make ranking multi-horizon by default
- Make news optional behind extras, default off until stable

Exit criteria:

- `ModelPipeline.load("stock_ranker", config)` resolves fully from config
- Walk-forward ranking training works end to end against store/factor outputs

## Phase 6 - Runner Rewrite (`qore-runner`)

Objective: define strategy generation and portfolio construction independently from
backtest execution.

Scope:

- `Strategy` protocol with session compatibility
- `RankingStrategy`, `CrossSectionalScreener`, `BehavioralGatedStrategy`
- `PositionSizer`, `RiskManager`, `StrategyRunner`

Legacy reuse candidates:

- Ranking output interpretation from `scripts/smoke_train.py`
- Existing factor weighting ideas

Exit criteria:

- Runner produces typed target portfolios from factor/model/news inputs
- Stock, fund, and derivative configuration paths are separated cleanly

## Phase 7 - Backtest Rewrite (`qore-backtest`)

Objective: implement session-aware execution simulation and portfolio accounting.

Scope:

- `fill_order()` via `singledispatch`
- `BacktestEngine.from_config()`
- metrics module for return/risk/IC/turnover analytics

Important rewrite decisions:

- Execution semantics are driven by instrument/session model, not symbol naming
- Stocks, funds, and derivatives each get explicit fill logic

Exit criteria:

- End-to-end backtest runs from universe + runner + store
- Metrics include both portfolio and ranking diagnostics

## Phase 8 - Migration Cutover and Legacy Removal

Objective: switch all active workflows to the new architecture and remove obsolete
code.

Tasks:

- Move user-facing commands to new crate-based entrypoints
- Migrate any needed local config into `QoreConfig`
- Deprecate and then remove `src/quant_trade`
- Remove ArcticDB-specific docs and scripts
- Archive or delete stale local artifacts under `db/`, `model/`, `logs/` as appropriate

Exit criteria:

- Mainline development happens only in `crates/qore-*`
- Legacy package and storage path are no longer required

## Module Mapping

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

- Directly migrating legacy classes will drag old abstractions into the new design
- ArcticDB data already stored locally may tempt partial compatibility layers
- Reusing AkShare runtime calls would violate the target contract and slow cleanup
- Large eager feature pipelines may hide assumptions that must be re-expressed lazily
- Legacy tests encode old architecture expectations and should not define the rewrite

## Recommended Execution Plan

### Milestone A - Foundation

- Finish Phase 0 to Phase 2 first
- Do not start factor/model rewrite before `qore-core` and `qore-data` APIs settle

### Milestone B - Data to Signal

- Complete Phase 3 to Phase 5
- Reach one working stock ranking flow on daily data before adding funds/news/derivatives

### Milestone C - Portfolio to Backtest

- Complete Phase 6 and Phase 7
- Validate one benchmark strategy end to end on A-shares

### Milestone D - Cutover

- Remove legacy package and obsolete assets
- Update docs and commands to reflect only the new architecture

## Definition of Done

The rewrite is complete when:

- Repository structure matches the monorepo design
- All new code depends on `qore-*` crates instead of `quant_trade`
- Data storage is DuckDB + Parquet, not ArcticDB
- Instrument-specific behavior is expressed via `singledispatch`
- Model training and loading are config-derived and versioned
- A full stock ranking backtest runs through the new stack end to end
- Legacy `src/quant_trade` is removed or archived outside the active codepath
