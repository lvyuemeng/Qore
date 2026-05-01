# qore-data

Data ingress, storage, and retrieval for Qore.

## Architecture

```text
src/qore_data/
├── __init__.py          # DataSettings, StockPipeline exports
├── fetch.py             # StockPipeline — fetch pipeline + corpus read API
├── store/
│   ├── duckdb.py        # QoreStore — DuckDB + Parquet reader-writer
│   └── schema.py        # Dataset schemas (index_constituents, stock_info, etc.)
└── fetcher/
    ├── __init__.py      # Exports all fetchers + http types
    ├── _base.py         # BaoStock session, helpers, URL/config constants
    ├── http.py          # HardenedJsonFetcher, RequestSpec, RequestPolicy
    ├── quote.py         # QuoteFetcher — BaoStock → Xueqiu → NetEase → EastMoney
    ├── financial.py     # FinancialFetcher — BaoStock → Xueqiu
    ├── analyst.py       # AnalystFetcher — Xueqiu → EastMoney
    ├── announcement.py  # AnnouncementFetcher — CNInfo → EastMoney
    ├── constituent.py   # ConstituentFetcher — CSI → EastMoney → Xueqiu → BaoStock
    ├── fund.py          # FundFetcher — Xueqiu → EastMoney
    └── xueqiu.py        # _XueqiuSession — guest token lifecycle for Xueqiu API
```

## Key exports

| Name | Kind | Description |
|---|---|---|
| `DataSettings` | `@dataclass` | DB path, concurrency, retry, cooldown settings |
| `StockPipeline` | `@dataclass` | Fetch pipeline + corpus read API |
| `QoreStore` | class | DuckDB + Parquet persistent store |

## Usage

### 1. Fetch: pull data from sources, auto-store

```python
from qore_data import DataSettings, StockPipeline

pipe = StockPipeline.from_settings(DataSettings())

# Resolve index constituents → writes to "index_constituents"
symbols = await pipe.resolve("000300.SH", as_of=date.today())

# Fetch OHLCV → auto-stores to "stock_ohlcv"
await pipe.stock_daily(symbols, start=date(2024, 1, 1))

# Fetch fundamentals → auto-stores to "fundamentals"
await pipe.fundamentals(symbols)

# Fetch analyst consensus → auto-stores to "analyst_forecasts"
await pipe.analyst_forecasts(symbols)

# Fetch static identity → auto-stores to "stock_info"
await pipe.stock_profiles(symbols)
```

### 2. Read: composable lazy retrieval

All reads return ``pl.LazyFrame`` — compose with standard Polars lazy API.

```python
# Single dataset by symbol and/or date
pipe.read("stock_ohlcv", symbols=["600519.SH"], dates=(date(2025,1,1), date(2025,3,31)))
pipe.read("fundamentals", symbols=["600519.SH"])

# Corpus: market data + fundamentals in one join
pipe.market_corpus(
    symbols=["600519.SH", "000001.SZ"],
    start=date(2025,1,1),
    end=date(2025,3,31),
    include_fundamentals=True,   # joins latest fundamentals per symbol
)

# Corpus: fundamentals + analyst forecasts
pipe.fundamental_corpus(
    symbols=["600519.SH"],
    as_of=date(2025,3,31),       # only fundamentals announced by this date
)

# Materialise with collect() or pass to FactorPipeline
df = pipe.read("stock_ohlcv", symbols=["600519.SH"]).collect()
```

### 3. Raw SQL (escape hatch)

```python
pipe.read_sql("SELECT symbol, roe FROM fundamentals WHERE roe > 0.2")
```

## Datasets

| Dataset | Partition | Dedup key | Description |
|---|---|---|---|
| `stock_ohlcv` | `year`, `symbol` | `date`, `symbol` | Daily OHLCV with limit-up/down flags |
| `fund_nav` | `year`, `symbol` | `date`, `symbol` | Fund unit NAV, accrued NAV, daily return |
| `fund_holdings` | `symbol` | `report_date`, `symbol`, `stock_symbol` | Fund position-level holdings |
| `fundamentals` | `year`, `symbol` | `announce_date`, `symbol`, `report_date` | 50-column quarterly financials + valuation |
| `index_constituents` | `index_symbol` | `as_of`, `index_symbol`, `symbol` | Index member snapshot by date |
| `stock_info` | (none) | `symbol` | Static identity: name, exchange, industry, board |
| `analyst_forecasts` | `symbol` | `as_of`, `symbol` | EPS consensus + rating breakdown |
| `announcements` | `symbol` | `symbol`, `art_code` | Company announcements |
| `stock_audit_opinions` | `symbol` | `symbol`, `art_code` | Audit opinion records |

## Source chains

Each fetcher tries sources in priority order, falling through on failure:

| Fetcher | Chain |
|---|---|
| QuoteFetcher | BaoStock → Xueqiu → NetEase → EastMoney |
| FinancialFetcher | BaoStock → Xueqiu |
| AnalystFetcher | Xueqiu → EastMoney |
| FundFetcher | Xueqiu → EastMoney |
| ConstituentFetcher | CSI → EastMoney → Xueqiu → BaoStock |
| AnnouncementFetcher | CNInfo → EastMoney |

BaoStock is the primary for OHLCV and quarterly financials. Xueqiu fills valuation
multiples (PE/PB) that BaoStock leaves null. EastMoney serves as fallback and is
used by the capital flow endpoint (EastMoney-only).
