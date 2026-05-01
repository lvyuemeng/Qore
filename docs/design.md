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

## Core architecture principles

1. Frame-native runtime contracts (`pl.DataFrame`/`pl.LazyFrame`) in hot paths.
2. Crate-local typed settings for crate internals.
3. Workflow/composition layer can adapt external config to crate settings.
4. Deterministic behavior for runner and backtest execution.
5. Library-first crates with clean composition boundaries.
6. Greedy-storage: fetch maximal data sets; filter at read time through SQL views.
7. Signal-first: runner/backtest consume strategy decisions; factors live in `qore-factor`.

## Crate responsibilities

### qore-data

- **Fetchers** (`fetcher/`): sector-specific HTTP fetchers wrapping EastMoney APIs.
  Six fetchers inherit `BaseJsonFetcher` from `_base.py`:
  `QuoteFetcher`, `FinancialFetcher`, `AnalystFetcher`, `AnnouncementFetcher`,
  `ConstituentFetcher`, `FundFetcher`. Plus `CSIndexFetcher` for index constituents via xls.
- **Fetch orchestration** (`fetch.py`): `StockPipeline` owns the full fetch → store
  lifecycle. Mode-aware (`backfill`/`refresh`/`local`). `_gather()` for chunked+semaphored
  per-symbol OHLCV; batch methods for profiles, fundamentals, analyst, announcements.
- **Selection** (`sources/`): `SelectionSource` owns a DuckDB SQL view joining
  `index_constituents`, `stock_profiles`, `stock_ohlcv`, `fundamentals`,
  `analyst_forecasts`, `strategy_factors`. `batch(dates)` returns `pl.LazyFrame`.
- **Store** (`store/`): `QoreStore` — pure DuckDB + Parquet reader-writer.
  No selection logic. Schema definitions in `store/schema.py`.
- **HTTP layer** (`fetcher/http.py`): `HardenedJsonFetcher` with retry, anti-crawl
  rotation, `RequestTelemetry` accumulation, and `RequestOutcome` structured diagnostics.
- **Candidate types** (in `fetch.py`): `CandidateSpec`, `StockCandidateSpec`,
  `CandidateFilter`, `CandidateSort`, `CandidateEligibilityPolicy`.

### qore-factor

- Factor classes: `OHLCVFactor`, `FundamentalFactor`, `CrossSectionalFactor`, `EventFactor`.
- `FactorPipeline` for composition and lazy execution.
- `build_liquidity_capacity_frame` for volume/liquidity filter construction used by `CandidateEligibilityPolicy`.

### qore-intelligence

- Model pipeline, registries, and normalizers (`RobustScaler`, `CrossSectionalZScore`, `RankScaler`).
- Model runners: `MultiHorizonRanker` (LGBM).
- Signal sources: `NewsPipeline`, `FinBERT`, `LLMExtractor`, `Triage`.
- Strategy adapter: `build_ranking_strategy`.

### qore-runner

- `StrategyRunner`: orchestrates decision → sizing → target portfolio.
- Strategies: `CrossSectionalScreener`, `RankingStrategy`, `BehavioralGatedStrategy`.
- Sizers: `EqualWeightSizer`, `VolScaledSizer`.
- Schedules: `RebalanceSchedule` with `frequency` and `buy_delay`/`sell_delay`.

### qore-backtest

- `BacktestEngine`: protocol-driven engine consuming `MarketDataSource`, `FactorSource`,
  `SignalOverlaySource`, `DecisionOverlaySource`. No data extraction imports (no DuckDB/Parquet/QoreStore).
- `BacktestResult`: frame-native outputs (`nav`, `positions`, `turnover`, `fills`, `diagnostics`).
- `BacktestView`: `with_drawdown()`, `with_benchmark()`, `window()`, `plot().overview()`.
- `fill_order`, `compute_metrics`.

## Data and execution flow

Canonical workflow path:

1. `StockPipeline` fetch → `QoreStore` (Parquet + DuckDB)
2. `SelectionSource.batch(dates)` → `pl.LazyFrame` (one SQL query)
3. `FactorPipeline` → factor columns
4. `CandidateSpec.apply()` + `CandidateEligibilityPolicy.capacity_filter()` → selection snapshots
5. `StrategyRunner` → target weights
6. `BacktestEngine` → fills/nav/diagnostics/view

## Fetcher architecture

```text
fetcher/
├── __init__.py          # exports all fetchers + http types
├── _base.py             # BaseJsonFetcher, ResponseGuard, HEADER_PROFILES,
│                        #   PREFIX, build_json_fetcher, _fetch_paginated,
│                        #   _frame_from_records, _to_stock, _symbol_digits, etc.
├── http.py              # HardenedJsonFetcher, RequestTelemetry, RequestOutcome,
│                        #   EndpointStats, RequestHardening, HeaderProfile
├── quote.py             # QuoteFetcher(BaseJsonFetcher): stock_daily, stock_profile,
│                        #   capital_flow, batch_stock_profiles (clist)
├── financial.py         # FinancialFetcher(BaseJsonFetcher): fundamentals,
│                        #   batch_fundamentals, _gather_chunked
├── analyst.py           # AnalystFetcher(BaseJsonFetcher): analyst_forecast,
│                        #   batch_analyst_forecasts (chunked at 100)
├── announcement.py      # AnnouncementFetcher(BaseJsonFetcher): greedy market-wide
│                        #   fetch + client-side symbol filtering
├── constituent.py       # ConstituentFetcher(BaseJsonFetcher): index_constituents
├── csindex.py           # CSIndexFetcher: index_constituents via xls workbook
└── fund.py              # FundFetcher(BaseJsonFetcher): fund_nav, fund_holdings
```

Key patterns:

- All EastMoney JSON fetchers extend `BaseJsonFetcher` (provides `__init__`,
  `from_settings`, `close`, `telemetry_snapshot`, `_fetch_paginated`).
- `_fetch_paginated` is unified in `_base.py` — shared by `FundFetcher` and `AnnouncementFetcher`.
- `announcement.py` uses greedy-storage: one market-wide paginated fetch, then
  `_unnest_items` to extract per-symbol rows from the `codes` array.
- `financial.py` and `analyst.py` chunk `SECURITY_CODE IN (...)` filters at 100 codes
  to avoid HTTP 414 (URL too long).
- `PREFIX = {"SH": "1", "SZ": "0", "BJ": "0"}` defined once in `_base.py`.

## Settings boundary

Crate-local settings are authoritative for internals:

| Setting | Fields |
|---|---|
| `DataSettings` | `db_path`, `parquet_root`, `concurrency`, `delay_min`, `delay_max`, `timeout`, `max_retries`, `retry_budget`, `cooldown_min`, `cooldown_max`, `retry_backoff_min`, `retry_backoff_max` |
| `IntelligenceSettings` | model store paths |
| `RunnerSettings` | `max_single`, `drawdown_stop` |
| `BacktestSettings` | `initial_capital`, `commission`, `slippage`, `drawdown_stop`, `cadence` |

Global `QoreConfig` (if used) belongs to composition boundaries only.

## Backtest/result contract state

Backtest outputs are frame-native:

- `BacktestResult.nav`
- `BacktestResult.positions`
- `BacktestResult.turnover`
- `BacktestResult.fills`
- `BacktestResult.diagnostics`

Visualization/view API is method-owned:

- `result.view()` → `BacktestView`
- `with_drawdown`, `with_benchmark`, `window`
- `plot().equity()/overview()/tearsheet()`

Plot backend dependency is managed through uv dependency group `viz`.

## Logging

Three loggers with structured key=value format:

| Logger | Location | Events |
|---|---|---|
| `qore.data.fetch` | `fetch.py` | `fetch_chunk` (dataset, chunk, symbols, rows, elapsed), `fetch_done`, `http_telemetry` (per-endpoint requests/successes/failures/retries/4xx/5xx/avg_latency) |
| `qore.data.http` | — removed | Replaced by `RequestOutcome` dataclass in `EndpointStats.recent_outcomes` ring buffer |
| `small_cap_strategy` | workflow.py | `workflow_start`, `selection_batch`, `factor_pipeline`, `candidate_filter`, `universe_build`, `backtest_done` |

HTTP telemetry surfaces via `_log_telemetry()` at the end of each `_gather` batch.

## Constraints and anti-patterns

- Avoid object/list/dict reconstruction where frame expressions can represent logic.
- Avoid hidden coupling to global config in crate internals.
- Avoid embedding product CLI logic in crate packages.
- Avoid broad eager joins when stage-owned lazy composition can express intent.
- Greedy storage: filter at read time, not at fetch time.
- No `hasattr`/`isinstance`/`getattr` dynamic dispatch in hot paths.
- Use protocols for type contracts; data structures for dispatch.

## Reference entrypoints

- User workflow intro: `docs/introduction.md`
- Contributor workflow: `docs/workflow.md`
- Active checklist and priorities: `docs/roadmap.md`
