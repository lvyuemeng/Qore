# Qore Refactor Roadmap

## Objective

Signal-first architecture: factors in `qore-factor`, runner/backtest consume decisions, workflow is pure crate-API composition.

## Done

(Streams A–NN2 completed — see git history)

### NN-2a/b/c — BaoStock session centralized (superseded)
Superseded by NN-2h (individual per-call login/logout). BaoStock sessions are no longer shared — each sync worker manages its own lifecycle.

### NN-2d — CSIndexFetcher merged into ConstituentFetcher
`csindex.py` removed. CSI XLS parsing logic consolidated to `_parse_csi_xls` in `constituent.py`. `ConstituentFetcher` now has 3-priority dispatch: CSI XLS → EastMoney fundztapi → BaoStock.

### NN-2e — QuoteFetcher source protocol refactor
`QuoteFetcher` refactored from `BaseJsonFetcher` subclass to standalone class with injectable `QuoteDaySource` / `QuoteProfileSource` / `CapitalFlowSource` protocols. Three sources:
- `_BaoStockQuoteSource` — baostock TCP via `asyncio.to_thread`, top priority
- `_NeteaseQuoteSource` — standalone `httpx.AsyncClient` for GBK CSV, mid priority
- `_EastMoneyQuoteSource` — shared `HardenedJsonFetcher`, lowest priority

Sources configured per-method dimension, not hardcoded in try/except ladders.

### NN-2f — `_bs_sync_profiles` stale type filter corrected
Bug fix: `br[3] != "1"` → `br[3] != ""` (outDate field was filtering all active stocks).

### NN-2g — BaoStock concurrent executor for `stock_daily`
`StockPipeline.stock_daily()` now uses `batch_fetch(BatchConfig.process(), kline_symbol, symbols)`. Each symbol runs in its own ProcessPoolExecutor worker.

### NN-2h — BaoStock sync consolidation + `baostock.py`
All BaoStock sync functions moved from `_base.py` to dedicated `baostock.py`:
- `kline()`, `profiles()`, `fundamentals()`, `constituents()` — module-level functions pickleable by dotted-path for ProcessPoolExecutor
- Each function handles `bs.login()`/`bs.logout()` internally
- `fundamentals_symbol()` — per-symbol DataFrame worker for `batch_fetch`
- Uses shared helpers (`_exchange_from_stock_code`, `_symbol_digits`, `_to_float`) from `_base.py`

### NN-2i — FinancialFetcher source protocol refactor
`FinancialFetcher` refactored from `BaseJsonFetcher` subclass to standalone class with `FinancialSource` protocol:
- `_BaoStockFinancialSource` — baostock TCP (priority)
- `_EastMoneyFinancialSource` — via `HardenedJsonFetcher` (fallback)
- `StockPipeline.fundamentals()` uses `batch_fetch` per symbol via `fundamentals_symbol()`

### NN-2j — Telemetry removed from `http.py`
Full telemetry stack removed:
- `RequestOutcome`, `EndpointStats`, `RequestTelemetry` — deleted
- `TelemetryReadable` protocol — deleted
- `HardenedJsonFetcher` no longer implements telemetry interfaces
- `RequestHardening.telemetry` field removed
- `BaseJsonFetcher.telemetry_snapshot()` removed
- `FinancialFetcher.telemetry_snapshot/telemetry_frame` removed
- `QuoteFetcher.telemetry_snapshot/telemetry_frame` removed
- `StockPipeline._log_telemetry()` removed
- `__init__.py` exports cleaned up

External API only uses semantic logging; individual HTTP request telemetry was unnecessary.

## Remaining

### Stream NN-3: Expanded fundamentals + stock_info split (done)

1. **NN-3a: Expand EMPTY_SCHEMA fundamentals** — 50-column schema with all BaoStock quarterly metrics + time-varying profile fields
2. **NN-3b: Multi-API BaoStock fundamentals** — Query all 6 quarterly APIs per symbol
3. **NN-3c: Split `stock_profiles` → `stock_info`** — Lean static-identity table
4. **NN-3d: Update schemas and view** — `schema.py`, `duckdb._build_selection_view`, `fetch.py`
5. **NN-3e: Clean up `financial.py`** — EastMoney financial source removed
6. **NN-3f: Update tests**

### Stream NN-3g: BaoStock dissected into sector modules + baostock.py deleted

BaoStock sync workers moved into each sector module. `baostock.py` deleted. `fetch.py` routes all BaoStock access through fetcher classes.

### Stream NN-3h: Xueqiu source protocol — cross-fetcher session + fallback

New `xueqiu.py` module with centralized `_XueqiuSession` (guest token acquisition, auto-refresh).

Source chains updated:
- **QuoteFetcher**: `BaoStock → Xueqiu → NetEase → EastMoney`
- **FinancialFetcher**: `BaoStock → Xueqiu` (Xueqiu fills PE/PB/PS that BaoStock leaves null)
- **AnalystFetcher**: `Xueqiu (EPS) → EastMoney (full rating)`
- **FundFetcher**: `Xueqiu (NAV+holdings) → EastMoney api.fund + datacenter-web`
- **ConstituentFetcher**: `CSI → EastMoney → Xueqiu → BaoStock`

### Stream NN-4: Source protocol refactor for remaining sectors

#### AnnouncementFetcher
- `_CNInfoSource` — standalone `httpx.AsyncClient` for POST-based CNInfo API
- `_EastMoneyAnnounceSource` — via `HardenedJsonFetcher`

### Operational sequence

1. NN-3a through NN-3h (done)
2. NN-4a: AnnouncementFetcher source protocol
