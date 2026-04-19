# Qore

> Canonical AI prompt for the Qore quantitative platform.
> Python 3.13 · uv workspace · DuckDB + Parquet · Polars · Ranking-first

---

## 0. Project Identity

**Qore** is a production-grade quantitative trading platform for Chinese
markets. uv-workspace monorepo of independent crates. Every design decision
is intentional, typed, and verifiable.

Stack: Python 3.13, uv, Polars (LazyFrame-first), DuckDB (query engine over
Parquet lake), Pydantic v2, mypy strict, ruff, pytest.

---

## 1. Repository Layout & AI References

### 1.1 Monorepo structure

```text
qore/
├── .ai/
│   ├── refs/              ← cloned reference repos (gitignored)
│       └── akshare/       ← source reference for reverse engineering
├── crates/
│   ├── qore-core/
│   ├── qore-data/
│   ├── qore-factor/
│   ├── qore-intelligence/ ← merged qore-model + qore-signal
│   ├── qore-runner/       ← merged strategy + portfolio
│   └── qore-backtest/
├── data/
│   ├── raw/
│   ├── factor/
│   ├── signal/
│   └── qore.duckdb
├── examples/
│   └── stock_ranking_workflow.py
├── models/
├── Justfile
└── pyproject.toml         ← [tool.uv.workspace] members = ["crates/*"]
```

### 1.2 `.ai/` — reference material and AI rules

```markdown
## Coding rules for Qore

### Data layer

- Never import akshare directly in any crate source file.
- To understand an eastmoney endpoint, READ .ai/refs/akshare/ first.
  Key paths:
  akshare/stock/stock_zh_a_hist.py ← A-share OHLCV (kline endpoint)
  akshare/fund/fund_em_*.py ← Fund NAV endpoints
  akshare/stock/stock_individual_info_em.py ← Stock fundamentals
  akshare/index/index_stock_cons_*.py ← Index constituent APIs
  Pattern: find the URL, params, and response field mapping, then
  reimplement in qore_data/fetcher/eastmoney.py using httpx.

### Dispatch

- Use singledispatch for all instrument-type-specific operations.
- Never write: if isinstance(inst, StockInstrument): ...
  Write: the dispatch function itself.
- Unsupported type combinations (e.g. minute data for stocks) must
  have NO registered implementation — let the default raise TypeError.

### Config

- Every class that needs a filesystem path or runtime parameter must
  expose a @classmethod from_config(cls, config: QoreConfig) constructor.
- Never accept root: Path or similar as a plain argument in __init__.
```

```gitignore
# .gitignore additions
.ai/
```

```just
# Justfile — fetch AI reference material
ai-refs:
    #!/usr/bin/env bash
    mkdir -p .ai/refs
    if [ -d .ai/refs/akshare/.git ]; then
        echo "Updating akshare reference..."
        git -C .ai/refs/akshare pull --ff-only
    else
        echo "Cloning akshare reference..."
        git clone --depth=1 https://github.com/akfamily/akshare .ai/refs/akshare
    fi
    echo "Done. Reference at .ai/refs/akshare/"
```

Run once before starting `qore-data` implementation:

```bash
just ai-refs
# Then read .ai/refs/akshare/ before writing any fetcher
```

### 1.3 Examples are outside crates

Examples, demos, and reference workflows belong under `examples/`, not inside crate
packages. Crates should expose reusable building blocks; examples should compose those
building blocks into runnable workflows.

Example responsibilities:

- pre-given stock selection strategy examples
- end-to-end backtest examples using crate APIs
- category or basket evaluation examples that acquire more information before ranking
- operator-reference scripts that show the intended flow without becoming crate runtime API

---

## 2. Core Type System — `qore-core`

**Deps**: `pydantic>=2.0`

### 2.1 Instrument: sealed union with no shared optional fields

```python
# qore_core/instrument.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

TradingSession = Literal["auction", "nav", "continuous"]
# auction    — A-shares: T+1, 涨跌停, call-auction open/close
# nav        — Funds: NAV-based, T+N subscription/redemption lag
# continuous — Futures + crypto perps: T+0, margined, MTM, 24 h

@dataclass(frozen=True)
class StockInstrument:
    symbol: str                             # "600519.SH", "000858.SZ"
    exchange: Literal["SH", "SZ", "BJ"]
    industry: str                           # SW 申万 level-1 code
    price_limit_pct: float = 0.10           # 5% for ST stocks
    session: TradingSession = "auction"

@dataclass(frozen=True)
class FundInstrument:
    symbol: str                             # "110022", "159915"
    fund_type: Literal["active", "passive_etf", "bond", "mixed", "qdii"]
    subscription_delay: int = 1             # T+N to receive units
    redemption_delay: int = 2               # T+N to receive cash
    session: TradingSession = "nav"

@dataclass(frozen=True)
class DerivativeInstrument:
    """Commodity futures, financial futures, crypto spot/perp/futures.
    All share continuous-session semantics regardless of underlying."""
    symbol: str                             # "rb2501", "BTCUSDT-PERP", "IF2503"
    exchange: str                           # "SHFE", "DCE", "BINANCE"
    underlying: str
    derivative_type: Literal["futures", "perpetual", "option"]
    contract_size: float
    margin_rate: float
    quote_currency: str = "CNY"
    expiry: date | None = None              # None for perpetuals
    session: TradingSession = "continuous"

Instrument = StockInstrument | FundInstrument | DerivativeInstrument
```

**Rule**: every function that accepts `Instrument` must use `match` over all
three concrete arms. `getattr` fallbacks and `isinstance` chains are banned.

---

### 2.2 Config: infrastructure and runtime source of truth

```python
# qore_core/config.py
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

class DataConfig(BaseModel):
    db_path: str = "data/qore.duckdb"
    parquet_root: str = "data/raw"
    eastmoney_concurrency: int = 8
    eastmoney_delay_min: float = 0.2
    eastmoney_delay_max: float = 0.5

class IntelligenceConfig(BaseModel):
    """Infrastructure config for intelligence runtime, not trained model state."""
    model_store_root: str = "models"
    # News signal
    news_llm_daily_budget: int = 50
    news_llm_model: str = "claude-sonnet-4-20250514"
    news_finbert_model: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    news_score_half_life_days: int = 5

class StockRunnerConfig(BaseModel):
    rebalance_freq: Literal["D", "W", "M"] = "W"
    top_k: int = 50
    max_weight: float = 0.05
    benchmark: str = "000300.SH"

class FundRunnerConfig(BaseModel):
    rebalance_freq: Literal["M"] = "M"
    top_k: int = 20
    max_weight: float = 0.10

class DerivativeRunnerConfig(BaseModel):
    rebalance_freq: Literal["D", "W"] = "D"
    roll_days_before_expiry: int = 5

class BacktestConfig(BaseModel):
    initial_capital: float = 10_000_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    drawdown_stop: float = 0.15

class QoreConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    stock: StockRunnerConfig = Field(default_factory=StockRunnerConfig)
    fund: FundRunnerConfig = Field(default_factory=FundRunnerConfig)
    derivative: DerivativeRunnerConfig = Field(default_factory=DerivativeRunnerConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QoreConfig":
        import yaml
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
```

**Rule**: every class that needs a path or runtime parameter must expose
`@classmethod from_config(cls, config: QoreConfig)`. Never accept bare
`Path` arguments in `__init__`.

**Boundary**: config is for infrastructure and runtime behavior. Tuned model
hyperparameters, selected factor schema, learned ensemble weights, and training
metadata belong to exported model artifacts, not `QoreConfig`.

---

### 2.3 Calendar

```python
# qore_core/calendar.py
class TradingCalendar:
    def is_trading_day(self, d: date) -> bool: ...
    def next_trading_day(self, d: date, n: int = 1) -> date: ...
    def prev_trading_day(self, d: date, n: int = 1) -> date: ...
    def trading_days_between(self, start: date, end: date) -> list[date]: ...

    def fill_date(self, signal_date: date, inst: Instrument) -> date:
        """Earliest execution date for a signal generated on signal_date."""
        match inst:
            case StockInstrument():
                return self.next_trading_day(signal_date, 1)
            case FundInstrument(subscription_delay=n):
                return self.next_trading_day(signal_date, n)
            case DerivativeInstrument():
                return signal_date   # T+0

    @classmethod
    def from_config(cls, config: QoreConfig) -> "TradingCalendar":
        return cls()  # loads bundled holiday file; no config params needed yet
```

---

### 2.4 Universe

```python
# qore_core/universe.py
class Universe:
    """Homogeneous, typed container of Instruments."""
    def __init__(self, instruments: list[Instrument]) -> None:
        types = {type(i) for i in instruments}
        if len(types) > 1:
            raise TypeError(f"Universe must be homogeneous; got {types}")
        self._map = {i.symbol: i for i in instruments}
        self._suspended: dict[tuple[str, date], bool] = {}

    def symbols(self) -> list[str]: ...
    def get(self, symbol: str) -> Instrument: ...
    def __iter__(self): ...
    def __len__(self): ...

    def is_suspended(self, symbol: str, d: date) -> bool: ...
    def tradeable_on(self, d: date) -> "Universe":
        """Sub-universe excluding suspended symbols on d."""
        ...
```

---

## 3. Data Layer — `qore-data`

**Deps**: `httpx[http2]>=0.28`, `tenacity>=9`, `duckdb>=1.0`,
`polars>=1.0`, `pyarrow>=17`

### 3.1 Source protocols: split by instrument type, no mega-interface

Instead of one `DataSource` protocol with a `freq` parameter that most
instrument types cannot honour, define three narrow protocols.
Each fetcher implements only the protocol(s) it actually supports.

```python
# qore_data/source.py

class StockSource(Protocol):
    """Provides A-share data only."""
    async def stock_daily(
        self, inst: StockInstrument, start: date, end: date
    ) -> pl.DataFrame:
        """
        Columns: date, symbol, open, high, low, close, volume, amount,
                 adj_factor, is_suspended, limit_up, limit_down
        """
    async def fundamentals(
        self, inst: StockInstrument, fields: list[str], as_of: date
    ) -> pl.DataFrame:
        """Point-in-time fundamental snapshot."""

    async def index_constituents(
        self, index_symbol: str, as_of: date
    ) -> list[StockInstrument]: ...

    async def stock_profile(
        self, inst: StockInstrument, as_of: date
    ) -> pl.DataFrame:
        """Universe metadata snapshot for a single stock."""

class FundSource(Protocol):
    """Provides fund NAV and holdings data only."""
    async def fund_nav(
        self, inst: FundInstrument, start: date, end: date
    ) -> pl.DataFrame:
        """Columns: date, symbol, nav, acc_nav, daily_return"""

    async def fund_holdings(
        self, inst: FundInstrument, report_date: date
    ) -> pl.DataFrame: ...

class DerivativeSource(Protocol):
    """Provides derivative data at multiple frequencies."""
    async def derivative_daily(
        self, inst: DerivativeInstrument, start: date, end: date
    ) -> pl.DataFrame:
        """Columns: date, symbol, open, high, low, close,
                    volume, amount, open_interest, settle_price"""

    async def derivative_minute(
        self, inst: DerivativeInstrument, start: date, end: date,
        freq_minutes: int = 1,
    ) -> pl.DataFrame:
        """Columns: datetime, symbol, open, high, low, close, volume"""

    async def derivative_tick(
        self, inst: DerivativeInstrument, trading_date: date
    ) -> pl.DataFrame:
        """Full tick data for a single session."""
```

---

### 3.2 Dispatch fetch functions — the public API

`singledispatch` is the routing mechanism. Unsupported combinations (e.g.
minute data for stocks) have **no registered implementation** — the default
arm raises `TypeError`. This is intentional: there is no runtime guard, no
`assert_freq_supported` — unsupported calls are errors at dispatch time.

```python
# qore_data/fetch.py
from functools import singledispatch
from qore_core.instrument import Instrument, StockInstrument, FundInstrument, DerivativeInstrument

# ── Daily OHLCV ─────────────────────────────────────────────────────────────

@singledispatch
async def fetch_daily(
    inst: Instrument,
    start: date,
    end: date,
    source,            # typed by registered arm
) -> pl.DataFrame:
    raise TypeError(f"No daily fetch registered for {type(inst).__name__}")

@fetch_daily.register(StockInstrument)
async def _(inst: StockInstrument, start: date, end: date, source: StockSource) -> pl.DataFrame:
    return await source.stock_daily(inst, start, end)

@fetch_daily.register(FundInstrument)
async def _(inst: FundInstrument, start: date, end: date, source: FundSource) -> pl.DataFrame:
    return await source.fund_nav(inst, start, end)

@fetch_daily.register(DerivativeInstrument)
async def _(inst: DerivativeInstrument, start: date, end: date, source: DerivativeSource) -> pl.DataFrame:
    return await source.derivative_daily(inst, start, end)

# ── Minute OHLCV — DerivativeInstrument only ────────────────────────────────
# NO registration for StockInstrument or FundInstrument.
# Calling fetch_minute(StockInstrument(...), ...) hits the default → TypeError.

@singledispatch
async def fetch_minute(
    inst: Instrument,
    start: date,
    end: date,
    source,
    freq_minutes: int = 1,
) -> pl.DataFrame:
    raise TypeError(
        f"{type(inst).__name__} does not support sub-daily data. "
        f"Only DerivativeInstrument supports fetch_minute()."
    )

@fetch_minute.register(DerivativeInstrument)
async def _(
    inst: DerivativeInstrument,
    start: date, end: date,
    source: DerivativeSource,
    freq_minutes: int = 1,
) -> pl.DataFrame:
    return await source.derivative_minute(inst, start, end, freq_minutes)

# ── Tick data — DerivativeInstrument only ───────────────────────────────────

@singledispatch
async def fetch_tick(inst: Instrument, trading_date: date, source) -> pl.DataFrame:
    raise TypeError(f"{type(inst).__name__} does not support tick data.")

@fetch_tick.register(DerivativeInstrument)
async def _(inst: DerivativeInstrument, trading_date: date, source: DerivativeSource) -> pl.DataFrame:
    return await source.derivative_tick(inst, trading_date)

# ── Fundamentals — StockInstrument only ─────────────────────────────────────

@singledispatch
async def fetch_fundamentals(inst: Instrument, fields: list[str], as_of: date, source) -> pl.DataFrame:
    raise TypeError(f"Fundamentals not available for {type(inst).__name__}.")

@fetch_fundamentals.register(StockInstrument)
async def _(inst: StockInstrument, fields: list[str], as_of: date, source: StockSource) -> pl.DataFrame:
    return await source.fundamentals(inst, fields, as_of)
```

**Rule**: whenever a new fetch operation is needed, define a new
`@singledispatch` function and register only the instrument types that
genuinely support it. Never add a `freq` parameter to a single function
and branch internally. The dispatch function IS the type check.

---

### 3.3 EastMoney Fetcher — config-driven

```python
# qore_data/fetcher/eastmoney.py
class EastMoneyFetcher:
    """
    Implements StockSource and FundSource.
    Does NOT implement DerivativeSource.
    Before writing any endpoint: READ .ai/refs/akshare/ for URL patterns.
    """

    # EastMoney secid prefix by exchange
    _PREFIX = {"SH": "1", "SZ": "0", "BJ": "0"}
    _KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    _FUND_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
    _CONSTITUENT_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    _FINANCIAL_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    _ANALYST_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    _ANNOUNCE_URL = "https://np-anotice.eastmoney.com/anlist/gglist.aspx"

    def __init__(self, concurrency: int, delay_min: float, delay_max: float) -> None:
        self._sem = asyncio.Semaphore(concurrency)
        self._delay = (delay_min, delay_max)
        self._client = httpx.AsyncClient(
            http2=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                "Referer": "https://finance.eastmoney.com/",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
            timeout=15.0,
        )

    @classmethod
    def from_config(cls, config: QoreConfig) -> "EastMoneyFetcher":
        return cls(
            concurrency=config.data.eastmoney_concurrency,
            delay_min=config.data.eastmoney_delay_min,
            delay_max=config.data.eastmoney_delay_max,
        )
```

---

### 3.4 Store: typed named datasets, config-driven

```python
# qore_data/store/schema.py
import pyarrow as pa
from dataclasses import dataclass

@dataclass(frozen=True)
class Dataset:
    name: str
    schema: pa.Schema
    partition_cols: list[str]
    dedup_keys: list[str]       # columns that uniquely identify a row

DATASETS: dict[str, Dataset] = {
    "stock_ohlcv": Dataset(
        name="stock_ohlcv",
        schema=pa.schema([
            ("date", pa.date32()), ("symbol", pa.string()),
            ("open", pa.float64()), ("high", pa.float64()),
            ("low", pa.float64()),  ("close", pa.float64()),
            ("volume", pa.int64()),  ("amount", pa.float64()),
            ("adj_factor", pa.float64()),
            ("is_suspended", pa.bool_()),
            ("limit_up", pa.bool_()), ("limit_down", pa.bool_()),
        ]),
        partition_cols=["year", "symbol"],
        dedup_keys=["date", "symbol"],
    ),
    "fund_nav": Dataset(
        name="fund_nav",
        schema=pa.schema([
            ("date", pa.date32()), ("symbol", pa.string()),
            ("nav", pa.float64()), ("acc_nav", pa.float64()),
            ("daily_return", pa.float64()),
        ]),
        partition_cols=["year", "symbol"],
        dedup_keys=["date", "symbol"],
    ),
    "derivative_ohlcv": Dataset(
        name="derivative_ohlcv",
        schema=pa.schema([
            ("date", pa.date32()), ("symbol", pa.string()),
            ("open", pa.float64()), ("high", pa.float64()),
            ("low", pa.float64()),  ("close", pa.float64()),
            ("volume", pa.int64()),  ("amount", pa.float64()),
            ("open_interest", pa.int64()), ("settle_price", pa.float64()),
        ]),
        partition_cols=["year", "symbol"],
        dedup_keys=["date", "symbol"],
    ),
    "fundamentals": Dataset(
        name="fundamentals",
        schema=pa.schema([
            ("report_date", pa.date32()),    # period end date
            ("announce_date", pa.date32()),  # public announcement date (PIT)
            ("symbol", pa.string()),
            ("pe_ttm", pa.float64()), ("pb", pa.float64()),
            ("ps_ttm", pa.float64()), ("ev_ebitda", pa.float64()),
            ("roe", pa.float64()),    ("roa", pa.float64()),
            ("gross_margin", pa.float64()),
            ("revenue", pa.float64()), ("net_income", pa.float64()),
            ("total_assets", pa.float64()),
            ("operating_cashflow", pa.float64()),
        ]),
        partition_cols=["year", "symbol"],
        dedup_keys=["announce_date", "symbol", "report_date"],
    ),
    "stock_profiles": Dataset(
        name="stock_profiles",
        schema=pa.schema([
            ("as_of", pa.date32()), ("symbol", pa.string()),
            ("short_name", pa.string()), ("exchange", pa.string()),
            ("industry", pa.string()), ("board", pa.string()),
            ("listing_date", pa.date32()),
            ("total_market_cap", pa.float64()),
            ("float_market_cap", pa.float64()),
            ("total_shares", pa.float64()),
            ("float_shares", pa.float64()),
            ("is_st", pa.bool_()),
        ]),
        partition_cols=["symbol"],
        dedup_keys=["as_of", "symbol"],
    ),
    "factor_scores": Dataset(
        name="factor_scores",
        schema=pa.schema([
            ("date", pa.date32()), ("symbol", pa.string()),
            ("factor_name", pa.string()),
            ("raw_value", pa.float64()),
            ("z_score", pa.float64()),
            ("rank_pct", pa.float64()),
        ]),
        partition_cols=["date_month", "factor_name"],
        dedup_keys=["date", "symbol", "factor_name"],
    ),
    "news_scores": Dataset(
        name="news_scores",
        schema=pa.schema([
            ("date", pa.date32()), ("symbol", pa.string()),
            ("score", pa.float64()),
            ("event_type", pa.string()),
            ("source_layer", pa.string()),  # "triage"|"finbert"|"llm"
        ]),
        partition_cols=["date_month"],
        dedup_keys=["date", "symbol"],
    ),
}
```

```python
# qore_data/store/duckdb.py
class QoreStore:
    def __init__(self, db_path: str, parquet_root: str) -> None: ...

    @classmethod
    def from_config(cls, config: QoreConfig) -> "QoreStore":
        return cls(db_path=config.data.db_path, parquet_root=config.data.parquet_root)

    def register_all_views(self) -> None:
        """Called once at startup. Registers each DATASET as a DuckDB VIEW."""

    def read(
        self,
        dataset: str,
        filters: dict[str, object] | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Validates dataset name, returns LazyFrame (zero-copy Arrow)."""

    def write(self, dataset: str, df: pl.DataFrame) -> None:
        """Validate schema, partition, deduplicate, write Parquet."""

    def sql(self, query: str) -> pl.LazyFrame:
        """Escape hatch for analytical SQL (factor window computation, etc.)"""
```

### 3.5 Stock-universe-specific information still needed

For credible A-share workflows, data coverage is not just OHLCV plus point-in-time
fundamentals. The stock universe layer still needs richer metadata and membership views.
Reference `.ai/refs/akshare/` first when adding them.

Priority stock-universe information to support:

- historical index constituents for benchmark and pool definitions
- stable industry classification mapping for neutralization and group evaluation
- ST, suspension, delisting-risk, and price-limit-relevant status flags
- board / listing-segment tags such as main board, ChiNext, STAR, Beijing
- fund holdings and analyst forecast coverage linked cleanly to stock universes
- announcement and event coverage that can feed category evaluation or news triage

These are useful both for pre-given stock-selection strategies and for category-level
evaluation before ranking individual names.

---

## 4. Factor Engine — `qore-factor`

**Deps**: `polars>=1.0`, `scipy>=1.13`, `numpy>=2.0`

### 4.1 Factor Protocol: requires / produces

```python
# qore_factor/base.py
class Factor(Protocol):
    name: str
    produces: str              # output column name
    requires: frozenset[str]   # input columns that must exist

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Append self.produces column (Float64). Never call .collect()."""
```

Factor families by input domain:

```python
class OHLCVFactor:
    """requires ⊆ {date, symbol, open, high, low, close, volume, amount}"""

class FundamentalFactor:
    """requires ⊆ fundamental dataset columns (pb, roe, etc.)"""

class CrossSectionalFactor:
    """requires = {source_factor_column, date}.
    Produces a normalized or ranked version of an upstream factor."""
```

### 4.2 Pipeline

```python
# qore_factor/pipeline.py
class FactorPipeline:
    """Composable, lazy pipeline. No .collect() until .run() returns."""

    def add(self, *factors: Factor) -> "FactorPipeline":
        """Register factors. Validates requires against available columns."""

    def normalize(
        self,
        method: Literal["zscore", "rank_pct"] = "zscore",
        group_by: list[str] | None = None,   # default: ["date"]
    ) -> "FactorPipeline": ...

    def neutralize(
        self,
        by: list[str],          # e.g. ["industry"] or ["industry", "market_cap_tier"]
    ) -> "FactorPipeline": ...

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute: compute all factors → neutralize → normalize.
        Returns augmented LazyFrame. No .collect()."""

    def evaluate(
        self,
        factor_lf: pl.LazyFrame,
        forward_returns: pl.LazyFrame,
        horizons: list[int],
    ) -> pl.DataFrame:
        """IC and ICIR per factor per horizon."""
```

### 4.3 Key factor implementations

```python
# ohlcv/momentum.py
@dataclass
class MomentumFactor:
    lookback: int = 252
    skip: int = 21
    produces: str = field(init=False)
    requires: frozenset[str] = frozenset({"date", "symbol", "close"})

    def __post_init__(self) -> None:
        self.produces = f"mom_{self.lookback}d_skip{self.skip}"

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("close") / pl.col("close")
               .shift(self.lookback + self.skip).over("symbol") - 1
            ).alias(self.produces)
        )
    # For stocks: caller must add .shift(1).over("symbol") after pipeline
    # to enforce T+1 lag. This factor itself is session-agnostic.

# fundamental/value.py
@dataclass
class BookToPriceFactor:
    name: str = "bp"; produces: str = "bp"
    requires: frozenset[str] = frozenset({"pb"})
    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns((1.0 / pl.col("pb")).alias("bp"))

# fundamental/quality.py
@dataclass
class ROEStabilityFactor:
    window: int = 8
    name: str = "roe_stability"; produces: str = "roe_stability"
    requires: frozenset[str] = frozenset({"symbol", "roe"})
    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (1.0 / (pl.col("roe").rolling_std(self.window).over("symbol") + 1e-8))
            .alias("roe_stability")
        )

# fundamental/info.py
@dataclass
class SUEFactor:
    """Standardized Unexpected Earnings."""
    name: str = "sue"; produces: str = "sue"
    requires: frozenset[str] = frozenset({"actual_eps", "consensus_eps", "eps_std"})
    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            ((pl.col("actual_eps") - pl.col("consensus_eps"))
             / (pl.col("eps_std") + 1e-8)).alias("sue")
        )

# futures/carry.py
@dataclass
class CarryFactor:
    name: str = "carry"; produces: str = "carry"
    requires: frozenset[str] = frozenset({"near_price", "far_price", "days_to_roll"})
    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            ((pl.col("near_price") - pl.col("far_price"))
             / pl.col("near_price") / pl.col("days_to_roll")
            ).alias("carry")
        )
```

---

## 5. Intelligence Layer — `qore-intelligence`

**This is the merge of `qore-model` and `qore-signal`.**

Both crates produce signals that are eventually combined before entering the
runner. Keeping them together in `qore-intelligence` makes the combination
point (`SignalCombiner`) natural and avoids a dependency cycle.

**Deps**:

```toml
[project]
dependencies = [
    "qore-core", "qore-factor",
    "lightgbm>=4.6", "optuna>=4.7",
    "numpy>=2.0", "joblib>=1.4",
    "jieba>=0.42",
]
[project.optional-dependencies]
nlp = ["transformers>=4.40", "torch>=2.2"]
llm = ["litellm>=1.40"]
regime = ["hmmlearn>=0.3"]
```

**Internal layout**:

```text
qore-intelligence/src/qore_intelligence/
├── model/
│   ├── normalizer.py      ← XNormalizer, YTransformer protocols + impls
│   ├── lgbm_rank.py       ← ranking model core
│   ├── artifact.py        ← model export/import data structures
│   ├── registry.py        ← model loading/saving behavior
│   ├── ensemble.py        ← Optuna tuning and weight search
│   ├── regime.py          ← HMM MarketRegimeDetector (Phase 3)
│   └── pipeline.py        ← fit/predict orchestration only
├── signal/
│   ├── triage.py          ← Layer 1: regex + jieba
│   ├── sentiment.py       ← Layer 2: FinBERT-Chinese
│   ├── llm.py             ← Layer 3: litellm structured extraction
│   └── score.py           ← Layer 4: decay + DuckDB write
└── combine.py             ← SignalCombiner: merge model + news scores
```

### 5.1 Normalization

```python
# qore_intelligence/model/normalizer.py
class XNormalizer(Protocol):
    def fit(self, X: np.ndarray) -> None: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class RankScaler:
    """
    Cross-sectional rank → percentile [0, 1].
    Preferred for GBDT models: removes outliers, preserves order.
    fit() records per-feature rank boundaries from training data.
    transform() applies rank interpolation (no data leakage).
    """

class RobustScaler:
    """Median + IQR scaling. Alternative to RankScaler."""

class YTransformer(Protocol):
    def fit_transform(self, y: np.ndarray, groups: np.ndarray) -> np.ndarray: ...

class CrossSectionalZScore:
    """
    Z-score forward returns within each date group.
    Required for LambdaRank: raw returns are non-stationary;
    within-period z-scores are comparable across time.
    fit_transform() modifies in-place per group, no cross-group leakage.
    """
```

### 5.2 ModelPipeline: fit/predict orchestration only

```python
# qore_intelligence/model/pipeline.py
class ModelPipeline:
    """
    Complete ML pipeline: XNormalizer → SignalModel.
    It orchestrates fit/predict behavior only.
    It is not the persisted artifact and it is not the registry.
    """

    def __init__(
        self,
        x_normalizer: XNormalizer,
        y_transformer: YTransformer,
        model: "SignalModel",
    ) -> None:
        ...

    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        *,
        x_normalizer: XNormalizer | None = None,
        y_transformer: YTransformer | None = None,
    ) -> "ModelPipeline":
        """
        Factory for runtime defaults only.
        Does not resolve persisted model versions.
        """
        return cls(
            x_normalizer=x_normalizer or RankScaler(),
            y_transformer=y_transformer or CrossSectionalZScore(),
            model=MultiHorizonRanker(),
        )

    def fit(
        self,
        factor_lf: pl.LazyFrame,
        store: "QoreStore",
    ) -> None:
        """
        1. Materialize factor_lf (only place .collect() is allowed in this module)
        2. Build forward return targets per horizon from store
        3. x_normalizer.fit(X_train)
        4. y_transformer.fit_transform(y_train, date_groups)
        5. Walk-forward GroupKFold training
        """

    def predict_score(self, factor_lf: pl.LazyFrame) -> pl.Series:
        """x_normalizer.transform → model.predict → pl.Series(name='score')"""
```

### 5.3 Model artifact and registry separation

```python
# qore_intelligence/model/artifact.py
class ModelArtifact(BaseModel):
    """Pure persisted data. No loading/saving behavior."""

    model_name: str
    feature_columns: list[str]
    horizons: list[int]
    ensemble_weights: dict[str, float]
    model_params: dict[str, object]
    validation_metrics: dict[str, float]
    training_window: dict[str, str] | None = None
    trained_at: datetime
    payload: bytes | str

# qore_intelligence/model/registry.py
class ModelRegistry:
    """Loading/saving behavior. Paths come from config."""

    @classmethod
    def from_config(cls, config: QoreConfig) -> "ModelRegistry": ...

    def save(self, artifact: ModelArtifact, version: str | None = None) -> Path: ...
    def load(self, model_name: str, version: str = "latest") -> ModelArtifact: ...
```

Rules:

- `ModelPipeline` owns fit/predict behavior
- `ModelArtifact` owns persisted data only
- `ModelRegistry` owns export/import behavior only
- fields such as `trained_on`, `validation_ic`, and tuned hyperparameters belong in `ModelArtifact`, not `ModelPipeline`

### 5.4 Ranking model core

```python
# qore_intelligence/model/lgbm_rank.py
class MultiHorizonRanker:
    """Ranking model core. Trained settings come from fit or artifact load."""

    def predict_score(self, X: pl.DataFrame) -> pl.Series:
        """Weighted sum across horizon models."""
```

Optuna-generated hyperparameters and selected factor/model settings are packed into
the saved artifact. They are not declared as static config defaults.

### 5.5 News pipeline — litellm, layered

```python
# qore_intelligence/signal/llm.py
import litellm
from pydantic import BaseModel, Field

class EventExtraction(BaseModel):
    """Structured output — the only form of LLM output used by trading logic."""
    event_type: Literal["earnings", "guidance", "regulatory", "ma", "other"]
    direction: Literal["positive", "negative", "neutral"]
    magnitude: Literal["high", "medium", "low"]
    certainty: float = Field(ge=0.0, le=1.0)
    trading_relevant: bool

class LLMExtractor:
    @classmethod
    def from_config(cls, config: QoreConfig) -> "LLMExtractor":
        return cls(
            model=config.intelligence.news_llm_model,
            daily_budget=config.intelligence.news_llm_daily_budget,
        )

    async def extract(self, text: str) -> EventExtraction | None:
        """Returns None if daily budget exhausted."""
        if not self._budget.can_call():
            return None
        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        self._budget.record(response.usage)
        return EventExtraction.model_validate_json(
            response.choices[0].message.content
        )
```

**Why litellm**: single `acompletion()` interface over Claude, GPT-4, Gemini,
and local Ollama. Swap `news_llm_model` in config to switch providers without
touching code. `config.intelligence.news_llm_model = "ollama/qwen2.5:7b"` for
a fully local, zero-cost pipeline.

```python
# qore_intelligence/signal/score.py
class NewsPipeline:
    """Orchestrates all four layers and writes to store."""

    @classmethod
    def from_config(cls, config: QoreConfig, store: "QoreStore") -> "NewsPipeline":
        return cls(
            triage=Triage(),
            sentiment=FinBERT.from_config(config),
            llm=LLMExtractor.from_config(config),
            store=store,
            half_life=config.intelligence.news_score_half_life_days,
        )

    async def run(self, trading_date: date) -> None:
        """Crawl → triage → sentiment → llm → decay → write news_scores."""
```

### 5.6 SignalCombiner

```python
# qore_intelligence/combine.py
class SignalCombiner:
    """Merges factor-model scores with news scores into final signal."""

    def __init__(self, news_alpha: float = 0.0) -> None:
        # news_alpha=0 disables news entirely (default until Phase 4)
        self.news_alpha = news_alpha

    def combine(
        self,
        model_scores: pl.Series,          # index=symbol, from ModelPipeline
        news_scores: dict[str, float],    # from NewsPipeline
    ) -> pl.Series:
        if self.news_alpha == 0.0:
            return model_scores
        news = pl.Series(
            name="news",
            values=[news_scores.get(s, 0.0) for s in model_scores.name],
        )
        return (1 - self.news_alpha) * model_scores + self.news_alpha * news
```

---

## 6. Runner (Strategy + Portfolio) — `qore-runner`

**Deps**: `polars>=1.0`, `numpy>=2.0`, `cvxpy>=1.4` (optional)

### 6.1 Strategy Protocol — session-typed, not asset-locked

```python
# qore_runner/strategy.py
class Strategy(Protocol):
    """
    Session-typed signal generator.
    Compatible sessions declare execution mechanics, not asset names.
    A MomentumStrategy may run on auction-session stocks (weekly) or
    continuous-session futures (daily) — same logic, different universe.
    """
    name: str
    compatible_sessions: frozenset[TradingSession]
    signal_freq: Literal["event", "daily", "weekly", "monthly"]
    required_columns: frozenset[str]   # factor columns needed in input lf

    def generate(
        self,
        lf: pl.LazyFrame,
        universe: Universe,
        date: date,
        calendar: TradingCalendar,
    ) -> pl.Series:
        """
        Returns Series(name="signal", dtype=Float64).
        Index = symbol strings. NaN = no view.
        Must filter out symbols not tradeable on calendar.fill_date(date, inst).
        """
```

Concrete strategies:

```python
# strategies/ranking.py
class RankingStrategy:
    """Signal from ModelPipeline. Universal — any session."""
    compatible_sessions = frozenset({"auction", "nav", "continuous"})

    @classmethod
    def from_config(cls, config: QoreConfig) -> "RankingStrategy":
        registry = ModelRegistry.from_config(config)
        artifact = registry.load("stock_ranker")
        pipeline = ModelPipeline.from_artifact(artifact)
        combiner = SignalCombiner(news_alpha=0.0)
        return cls(pipeline=pipeline, combiner=combiner)

# strategies/crosssectional.py
class CrossSectionalScreener:
    """Weighted factor score without ML. Suitable for nav-session (funds)."""
    compatible_sessions = frozenset({"nav", "auction"})
    signal_freq = "monthly"

    def __init__(self, factor_weights: dict[str, float]) -> None: ...

# strategies/behavioral.py
class BehavioralGatedStrategy:
    """
    Wraps any base strategy with a regime + vol scaling gate.
    scale = clip(regime_scale * vol_scale, min_scale, 1.0)
    Inherits compatible_sessions from base strategy.
    """
    def __init__(
        self,
        base: Strategy,
        regime_detector: "MarketRegimeDetector | None" = None,
        vol_lookback: int = 20,
        min_scale: float = 0.5,
    ) -> None: ...
```

### 6.1.1 Example workflow shape

The first reference stock example under `examples/` should be built from these parts:

1. define a pre-given stock selection universe or basket
2. acquire more category-level information before name ranking
3. generate signals from factors and optional model or news layers
4. construct entries and target holdings through runner components
5. evaluate the resulting strategy with backtest metrics and diagnostics

This example is a composition layer, not a crate-owned runtime entrypoint.

### 6.2 PositionSizer, RiskManager, StrategyRunner

```python
# sizer.py
class PositionSizer(Protocol):
    def size(self, signals: pl.Series, universe: Universe) -> dict[str, float]: ...

class EqualWeightSizer:
    def __init__(self, top_k: int, max_weight: float = 0.05) -> None: ...

class VolScaledSizer:
    """Weight ∝ 1/realized_vol. No covariance matrix needed."""
    def __init__(self, top_k: int, vol_col: str = "realized_vol_20d") -> None: ...

class MaxDiversificationSizer:
    """cvxpy. Use only for universes <30 with stable correlations."""
    def __init__(self, top_k: int, max_weight: float = 0.10) -> None: ...

# risk.py
class RiskManager:
    @classmethod
    def from_config(cls, config: QoreConfig) -> "RiskManager":
        return cls(
            max_single=config.stock.max_weight,
            drawdown_stop=config.backtest.drawdown_stop,
        )

    def apply(
        self,
        target: dict[str, float],
        current: dict[str, float],
        nav: pl.Series,
    ) -> dict[str, float]:
        """Returns {} (all-cash) if drawdown stop triggered."""

# runner.py
@dataclass
class TargetPortfolio:
    date: date
    weights: dict[str, float]
    signals: pl.Series
    risk_triggered: bool

class StrategyRunner:
    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        strategy: Strategy,
        sizer: PositionSizer,
    ) -> "StrategyRunner":
        return cls(
            strategy=strategy,
            sizer=sizer,
            risk_manager=RiskManager.from_config(config),
        )

    def step(
        self,
        factor_lf: pl.LazyFrame,
        news_scores: dict[str, float] | None,
        universe: Universe,
        date: date,
        current_weights: dict[str, float],
        nav: pl.Series,
        calendar: TradingCalendar,
    ) -> TargetPortfolio: ...
```

---

## 7. Backtest — `qore-backtest`

**Deps**: `polars>=1.0`, `quantstats>=0.0.62`

### 7.1 Session-dispatched simulators

Same `singledispatch` pattern as the data layer — route by instrument
session type, no `if isinstance` chains.

```python
# qore_backtest/simulate.py
from functools import singledispatch

@singledispatch
def fill_order(
    inst: Instrument,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    price_data: pl.DataFrame,
    config: BacktestConfig,
    calendar: TradingCalendar,
) -> "Fill":
    raise TypeError(f"No fill logic for {type(inst).__name__}")

@fill_order.register(StockInstrument)
def _(inst: StockInstrument, order_date, direction, quantity, price_data, config, calendar) -> "Fill":
    """
    fill_date = next_trading_day(order_date)
    is_suspended → rejected
    limit_up + buy → pending, retry fill_date+1
    limit_down + sell → pending, retry fill_date+1
    fill_price = open[fill_date] * (1 ± slippage)
    """

@fill_order.register(FundInstrument)
def _(inst: FundInstrument, ...) -> "Fill":
    """fill_date = next_trading_day(order_date, inst.subscription_delay)
    fill_price = nav[fill_date]; apply fee"""

@fill_order.register(DerivativeInstrument)
def _(inst: DerivativeInstrument, ...) -> "Fill":
    """T+0. fill_price = next open. MTM daily. Margin tracking."""
```

### 7.2 Engine

```python
# qore_backtest/engine.py
class BacktestEngine:
    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        runner: StrategyRunner,
        store: QoreStore,
        calendar: TradingCalendar,
    ) -> "BacktestEngine":
        return cls(
            runner=runner, store=store,
            config=config.backtest, calendar=calendar,
        )

    def run(self, universe: Universe, start: date, end: date) -> "BacktestResult": ...
```

### 7.3 Metrics

```python
# qore_backtest/metrics.py
def compute_metrics(
    result: "BacktestResult",
    benchmark_nav: pl.Series | None = None,
) -> dict[str, float]:
    """
    annualized_return, sharpe_ratio, calmar_ratio, max_drawdown,
    sortino_ratio, information_ratio (if benchmark),
    ic_mean, ic_std, icir,
    avg_turnover, total_commission_cost, win_rate, profit_factor
    """
```

---

## 8. Dependency Summary

```toml
# qore-core:          pydantic>=2.0
# qore-data:          qore-core, httpx[http2]>=0.28, tenacity>=9,
#                     duckdb>=1.0, polars>=1.0, pyarrow>=17
# qore-factor:        qore-core, qore-data, polars>=1.0, scipy>=1.13, numpy>=2.0
# qore-intelligence:  qore-core, qore-factor,
#                     lightgbm>=4.6, optuna>=4.7, numpy>=2.0, joblib>=1.4,
#                     jieba>=0.42
#   [nlp]:            transformers>=4.40, torch>=2.2
#   [llm]:            litellm>=1.40
#   [regime]:         hmmlearn>=0.3
# qore-runner:        qore-core, qore-factor, qore-intelligence,
#                     polars>=1.0, numpy>=2.0
#   [opt]:            cvxpy>=1.4
# qore-backtest:      qore-core, qore-data, qore-runner,
#                     polars>=1.0, quantstats>=0.0.62
```

---

## 9. Hard Constraints

- `singledispatch` is the only permitted instrument-type branching mechanism.
  `isinstance` chains and `if asset_type ==` conditions are banned everywhere.
- Unsupported operations (minute data for stocks, fundamentals for funds) have
  **no registered implementation**. Let the dispatch default raise `TypeError`.
- Every class with a path or runtime dependency exposes
  `@classmethod from_config(cls, config: QoreConfig)`. No bare `Path` args.
- Tuned model hyperparameters, factor schema, learned weights, and training
  summaries are artifact data, not config.
- Never import `akshare` in any crate. Read `.ai/refs/akshare/` as reference only.
- `.collect()` is only permitted in fit/evaluation or execution boundaries such as model training and backtest execution.
  `BacktestEngine.run()`. Everywhere else: LazyFrame.
- `EventExtraction` (Pydantic model) is the only LLM output type that crosses
  into trading logic. Raw text is never used for decisions.
- `Universe` must be homogeneous — all instruments the same concrete type.
- `factor_scores` dataset always stores both `raw_value` and `z_score`.
  Normalization state must be reconstructible without re-running the pipeline.

---

## 10. Where to Start

```text
just ai-refs                    # clone .ai/refs/akshare/ first

crates/qore-core/src/qore_core/instrument.py   # sealed union + TradingSession
crates/qore-core/src/qore_core/config.py       # QoreConfig with from_yaml()
crates/qore-core/src/qore_core/calendar.py     # fill_date dispatch
crates/qore-core/src/qore_core/universe.py     # homogeneous container
crates/qore-data/src/qore_data/fetch.py        # singledispatch fetch functions
```

Verify before proceeding:

- `fetch_minute(StockInstrument(...), ...)` raises `TypeError`
- `fetch_daily(FundInstrument(...), ...)` calls `source.fund_nav()` correctly
- `ModelRegistry.from_config(config)` derives artifact paths entirely from config
- `QoreStore.from_config(config)` needs no other arguments
- A `Universe([StockInstrument(...), FundInstrument(...)])` raises `TypeError`

---

*Qore v4 — dispatch typing · config-driven init · .ai/ refs · merged intelligence*
