# Small-Cap Strategy Example

End-to-end small-cap workflow built on Qore crates.
Runs data preparation → selection → runner decisions → backtest execution.

## Prerequisites

- Python 3.13 with `uv` workspace
- Local DuckDB + Parquet store populated with required datasets

Prepare data first:

```bash
uv run --package small-cap-strategy small-cap-strategy --prepare-data
```

This fetches index constituents, profiles, daily OHLCV, fundamentals, audit opinions,
analyst forecasts, and announcements for the benchmark index constituents.

## Run

```bash
uv run --package small-cap-strategy small-cap-strategy
```

Optional args:

```bash
uv run --package small-cap-strategy small-cap-strategy \
  --db-path data/qore.duckdb \
  --parquet-root data/raw \
  --initial-capital 10000000 \
  --commission 0.0003 \
  --slippage 0.0005 \
  --drawdown-stop 0.15
```

## Required Datasets

| Dataset | Used for |
|---|---|
| `index_constituents` | Benchmark membership |
| `stock_profiles` | Market cap, board, listing date, ST flag |
| `stock_ohlcv` | Trading status, execution prices |
| `fundamentals` | roe, pe_ttm, pb, operating_cashflow, total_assets, total_liabilities |
| `strategy_factors` | avg_amount_20d, min_amount_20d, position_to_amount_20d_ratio |
| `stock_audit_opinions` | Audit-based exclusion |
| `analyst_forecasts` | Category report |
| `announcements` | Category report |

## Output

- Stock category report (industry × board with counts, market cap, forecasts, announcements)
- NAV frame
- Diagnostics frame
- Backtest overview plot (NAV + drawdown)

## Troubleshooting

**"No rebalance selection snapshots"** — data is prepared but no candidates pass the selection filters. Verify filters still admit candidates for the configured date range.

**"No universe data found"** — prepare data first with `--prepare-data`.

**Test skips as "required datasets missing"** — the real-data contract test requires a populated store. Run `--prepare-data` or set `QORE_RUN_LIVE_IO=1` for live IO tests.
