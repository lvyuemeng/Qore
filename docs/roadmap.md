# Qore Rewrite Roadmap

## Goal

Rewrite the repository into the `Qore` architecture in `docs/design.md` and the
library-first workflow in `docs/workflow.md`.

This remains a staged replacement of legacy `src/quant_trade`, not an
incremental refactor.

## Ground Rules

1. New architecture first, compatibility second.
2. New active work lands in `crates/qore-*`, not legacy runtime paths.
3. Analytical dataflow stays lazy until explicit execution boundaries.
4. Prefer Polars and backend-native operators before Python loops.
5. Use `singledispatch` or session-driven dispatch, not union-heavy branching.
6. Config is runtime infrastructure only; trained state belongs in artifacts.
7. Example strategy YAML is a reference input for future workflow or CLI layers,
   not a data-crate-owned runtime surface.

## Current State

- `qore-core`: usable typed config, instrument, calendar, homogeneous universe
- `qore-data`: usable DuckDB + Parquet store, EastMoney fetcher, staged stock-selection pipeline
- `qore-factor`: usable lazy factor pipeline and persistence baseline
- `qore-intelligence`: usable ranking pipeline and cleaner artifact boundary
- `qore-runner`: usable strategy/sizer/risk flow with generic overlays
- `qore-backtest`: usable execution/accounting skeleton with metrics baseline
- workflow reality: library-first composition works, but there is still no supported CLI or one-command operator path

## Highest-Priority Missing Features

These are the highest-priority gaps because they block the reference A-share
workflow and the example small-cap strategy in `examples/strategy/small_cap_quality_enhanced_monthly.yaml`.

### From the small-cap strategy example

- `debt_to_asset_ratio` belongs in `qore-factor`; the missing piece is raw balance-sheet coverage such as `total_liabilities`, not a data-layer precomputed ratio
- blocked-trade logic is incomplete; current data does not expose `ask1_volume` / `bid1_volume` or equivalent order-book state
- audit-opinion exclusion is still incomplete; raw opinion history and reusable event-state composition exist now, but downstream workflow enforcement is not finished
- single-name capacity checking is still missing; the strategy needs reusable liquidity and capital checks such as target-position-to-amount and minimum traded-value guards
- `st_warning` and delisting-risk handling are incomplete beyond the current coarse ST/suspension flags
- alert routing is still missing, but it should be a general workflow alert surface rather than intelligence-specific `ai_record_alert`
- intraday execution windows are low priority; end-of-day or next-day close-based daily execution is the preferred baseline
- strategy YAML parsing and operator workflow assembly are missing from the future CLI/workflow layer

### From design and workflow

- no official CLI or stable user-facing entrypoint contract
- no single documented daily workflow runnable without manual Python assembly
- no benchmark-quality end-to-end A-share validation on real historical data
- no production-grade EastMoney sustained-load proof or mature telemetry reporting
- no complete operator-facing workflow that goes fetch -> factor -> model -> runner -> backtest from one supported entry

## Milestones

### Milestone A - Foundation

- Goal: stable workspace, immutable core types, and shared development rules
- Status: mostly complete
- Remaining:
  - keep tightening touched-module typing
  - keep making `qore-core` APIs more method-owned and consistent

### Milestone B - Data to Signal

- Goal: produce reliable datasets, factors, and ranking signals
- Status: active
- Blocking gaps:
  - richer A-share metadata and event surfaces
  - stronger EastMoney hardening and telemetry
  - more factor coverage only where it supports the target stock workflow
  - cleaner packaging from stored data to model-ready signals

### Milestone C - Portfolio to Backtest

- Goal: turn signals into portfolios and evaluate them realistically
- Status: active
- Blocking gaps:
  - richer strategy families and operator-facing strategy assembly
  - better risk behavior and diagnostics
  - more realistic execution, pending fills, and accounting
  - support for execution constraints required by the stock example workflow

### Milestone D - Workflow and Cutover

- Goal: move from library-first internals to a supported user-facing workflow and retire legacy paths
- Status: not started
- Blocking gaps:
  - CLI or stable entrypoint contract
  - supported config/workflow assembly layer
  - cutover proof on real workflows
  - legacy runtime retirement

## Phases

### Phase 0 - Freeze and prepare

- Status: done
- Scope: design target, migration boundary, AI reference rules
- Remaining: maintenance only

### Phase 1 - Workspace bootstrap

- Status: done
- Scope: uv workspace, crate layout, shared lint/type/test tooling
- Remaining: maintenance only

### Phase 2 - Core domain rewrite (`qore-core`)

- Status: mostly done
- Done:
  - typed config
  - session-aware calendar
  - homogeneous universe
  - generic typing improvements on touched APIs
- Remaining:
  - keep improving pipe-style universe ergonomics
  - keep simplifying generic/session-driven dispatch surfaces

### Phase 3 - Data layer rewrite (`qore-data`)

- Status: active
- Done:
  - EastMoney stock and fund fetch baseline
  - DuckDB + Parquet store
  - request hardening helpers
  - staged stock-selection pipeline owned by `StockSelectionPipeline`
  - fundamentals schema and EastMoney parsing now cover raw `total_liabilities` for leverage-factor construction
  - raw `stock_audit_opinions` dataset and EastMoney announcement-derived audit-opinion ingestion baseline
  - lazy as-of audit-opinion state join on `StockSelectionPipeline`
- High-priority remaining:
  - raw balance-sheet coverage needed by factor-layer leverage fields such as `total_liabilities`
  - richer status/event metadata for ST warning, delisting risk, and audit opinion
  - order-book or equivalent blocked-trade data for limit-up and limit-down execution filters
  - reusable daily-liquidity and capital-capacity inputs for per-stock capacity checks
  - better announcement and event coverage for strategy filters and alerting
  - clearer store semantics for analytical scans vs filtered retrieval
  - sustained-load EastMoney validation and usable telemetry

### Phase 4 - Factor engine rewrite (`qore-factor`)

- Status: active
- Done:
  - lazy factor pipeline
  - normalization and evaluation baseline
  - persistence into `factor_scores`
  - event-aware audit-opinion factor composition baseline, kept independent of runner/backtest policy
  - reusable capacity-metric factor composition from daily liquidity inputs
  - generic alert-condition frame builder from workflow inputs, independent of intelligence subscribers
- High-priority remaining:
  - add only workflow-relevant factors and derived fields
  - support leverage and event-aware factors required by the stock example, including `debt_to_asset_ratio` from raw liabilities and assets
  - support reusable capacity metrics and penalties derived from daily liquidity inputs, independent of runner policy
  - improve reconstruction and test coverage

### Phase 5 - Intelligence rewrite (`qore-intelligence`)

- Status: active
- Done:
  - baseline ranking pipeline
  - artifact manifest/payload/runtime cleanup
  - news-score persistence baseline
- High-priority remaining:
  - lighter metadata inspection separate from payload loading
  - broader validation and signal-stack maturity
  - generic workflow alert sinks and event escalation surfaces that intelligence may consume but does not exclusively own

### Phase 6 - Runner rewrite (`qore-runner`)

- Status: active
- Done:
  - generic strategy boundary
  - shared sizing path
  - generic overlay inputs
  - baseline diagnostics
- High-priority remaining:
  - richer stock strategy assembly beyond raw ranking inputs
  - stronger rule-based exits and exclusion handling once event and capacity overlays are composed upstream
  - operator-facing strategy config integration at the workflow or CLI layer

### Phase 7 - Backtest rewrite (`qore-backtest`)

- Status: active
- Done:
  - session-dispatched fills baseline
  - cached daily reads
  - accounting loop and metrics baseline
- High-priority remaining:
  - pending-fill and retry realism
  - execution windows for stock-session workflows after day-level execution support is solid
  - better diagnostics around blocked trades, retries, and exits

### Phase 8 - Workflow and cutover

- Status: not started
- Scope:
  - supported example entrypoints
  - config parsing above crate internals
  - CLI or workflow package boundary
  - legacy retirement
- Remaining: all deliverables

## Active Priorities

### Now

- finish the stock data surfaces required by the small-cap monthly strategy
- finish the factor surfaces required by the small-cap monthly strategy, with derived leverage metrics computed in `qore-factor` instead of stored as raw datasets
- add workflow-composable capacity checks and generic alert surfaces before deepening runner-specific policy
- keep the stock-selection API method-owned and remove helper wrappers
- document the workflow boundary clearly: crates provide primitives, examples show composition, future CLI owns strategy/config parsing
- validate one reproducible A-share workflow on real historical data
- harden EastMoney with operational telemetry and sustained-load evidence

## Priority Checklists

### Checklist 1 - `debt_to_asset_ratio`

- `qore-data`: add raw `total_liabilities` coverage to point-in-time fundamentals [done]
- `qore-factor`: compute `debt_to_asset_ratio = total_liabilities / total_assets`
- `qore-factor`: keep the ratio lazy and reconstructible from raw fields
- `qore-runner` or workflow layer: consume the factor as a normal selection input instead of expecting a precomputed store field

### Checklist 2 - audit-opinion exclusion

Concrete implementation plan:

1. `qore-data`
   - add a persisted `stock_audit_opinions` dataset with at least `symbol`, `report_date`, `announce_date`, `opinion`, `opinion_code`, and source metadata [done]
   - implement EastMoney endpoint coverage or equivalent announcement-derived parsing for audit opinions [done: announcement-derived baseline]
   - keep raw opinion history point-in-time, not prefiltered into strategy-specific flags
2. `qore-data` universe/event surface
   - add a lazy as-of resolver that derives the latest known adverse opinion state per symbol as of the selection date [done]
   - expose a reusable event/status join for audit-opinion-driven exclusions without hardcoding one strategy [done: selection-pipeline join baseline]
3. `qore-factor` / workflow layer
   - if needed, expose event-aware boolean or age-based features such as `has_adverse_audit_opinion` and `adverse_audit_opinion_age_days` [done: factor composition baseline]
   - keep these derived fields outside raw dataset storage [done]
4. workflow layer
   - compose audit-opinion exclusion windows, capacity checks, and alert conditions before runner/backtest consumption
   - keep event semantics, alert semantics, and stock-specific thresholds outside runner/backtest core contracts
5. `qore-runner`
   - do not own event semantics; only consume already-composed generic overlays when workflow layers choose to provide them
   - if needed later, support rule-based universe exclusion or forced exit inputs from generic event/status overlays without embedding strategy-specific event details into runner APIs
6. `qore-backtest`
   - enforce next-open liquidation and exclusion persistence in execution/accounting flow
   - add diagnostics so adverse-opinion exits and re-entry blocks are visible in results

### Checklist 3 - stock capacity and general alerts

Concrete implementation plan:

1. `qore-data`
   - expose reusable daily liquidity inputs such as `amount` history and possibly turnover-derived summaries needed for stock-level capital checks
   - keep these as generic market-state inputs, not strategy-specific capacity flags
2. `qore-factor` / workflow layer
   - derive capacity metrics such as `avg_amount_20d`, `target_position_to_daily_amount`, and capacity penalties from workflow inputs [done: factor composition baseline]
   - derive general alert condition frames from price/turnover/event inputs without coupling alerts to intelligence consumers [done: generic alert frame baseline]
3. workflow layer
   - compose index-universe selection, factor filters, event exclusions, capacity checks, and alert rules into one reusable daily strategy assembly path
   - define generic alert actions such as `emit_alert` or `record_alert`, leaving downstream subscribers optional
4. `qore-intelligence`
   - optionally subscribe to generic alerts for context enrichment, but do not own the alert contract
5. `qore-runner` / `qore-backtest`
   - consume capacity-checked universes and generic overlays after workflow composition, without embedding stock-capacity semantics in their core APIs

### Next

- expand only the factor and event surfaces needed by the target workflow
- strengthen runner risk behavior and backtest realism
- move workflow/config assembly into a dedicated operator-facing layer

### Later

- decide explicit source expansion scope
- cut over active workflows fully to supported new entrypoints

## Definition of Done

The rewrite is complete when:

- active workflows run through `qore-*` crates only
- one supported A-share workflow runs end to end from a stable entrypoint
- stock example strategy requirements are either implemented or explicitly rejected by the supported workflow contract
- EastMoney is operationally hardened with evidence, not just local tests
- legacy `src/quant_trade` is removed or archived out of the active path
