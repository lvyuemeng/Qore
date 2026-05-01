# Qore Configuration (Reference)

Primary user-facing config/workflow introduction: `docs/introduction.md`.

This file is kept as a short boundary reference.

## Configuration boundary

- Use crate-local typed settings in crate runtime APIs.
- Keep `QoreConfig` (if used) at workflow composition boundaries only.
- Do not couple crate internals to cross-crate global config objects.

## Settings reference

### DataSettings

```python
@dataclass(frozen=True, slots=True)
class DataSettings:
    db_path: str = "data/qore.duckdb"       # DuckDB database path
    parquet_root: str = "data/raw"          # Parquet file root directory
    concurrency: int = 20                   # Max concurrent HTTP requests
    delay_min: float = 0.05                 # Min random delay between requests (seconds)
    delay_max: float = 0.15                 # Max random delay between requests (seconds)
    timeout: float = 15.0                   # HTTP request timeout (seconds)
    max_retries: int = 3                    # Max retry attempts per request
    retry_budget: int = 50                  # Max total retries per endpoint
    cooldown_min: float = 0.5               # Min cooldown after anti-crawl hit (seconds)
    cooldown_max: float = 2.0               # Max cooldown after anti-crawl hit (seconds)
    retry_backoff_min: float = 0.2          # Min exponential backoff multiplier
    retry_backoff_max: float = 1.0          # Max exponential backoff multiplier
```

`DataSettings` feeds `build_json_fetcher()` → `HardenedJsonFetcher`, which wraps all
EastMoney HTTP calls with header rotation, rate limiting, anti-crawl detection, and retry.

### RunnerSettings

```python
@dataclass(frozen=True, slots=True)
class RunnerSettings:
    max_single: float = 0.05                # Max single position weight
    drawdown_stop: float = 0.15             # Drawdown stop threshold
```

### BacktestSettings

```python
@dataclass(frozen=True, slots=True)
class BacktestSettings:
    initial_capital: float = 10_000_000.0   # Starting capital
    commission: float = 0.0003              # Per-trade commission rate
    slippage: float = 0.0005                # Per-trade slippage rate
    drawdown_stop: float = 0.15             # Drawdown stop threshold
    cadence: Literal["daily", "intraday"] = "daily"
```

## What belongs in config

- Filesystem/storage locations.
- Source runtime knobs (timeout, retries, concurrency, cooldown).
- Runtime budgets and default behavior.

## What does not belong in config

- Trained model payload state.
- Learned weights/schema and training summaries.
- Model-family learned internals produced by fitting.

## Rule for new code

- Map boundary config to crate-local settings in workflow/example code.
- Keep crate internals typed and config-decoupled.
- Keep crates library-first; do not introduce product CLI entrypoints in crates.
