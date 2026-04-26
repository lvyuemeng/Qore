# Small-Cap Strategy Example

This example is the current end-to-end small-cap workflow built on Qore crates.
It runs selection -> runner decisions -> backtest execution without reading or
writing `factor_scores`.

## What the current workflow does

The workflow is implemented in `examples/small_cap_strategy/src/small_cap_strategy/workflow.py`.

1. Builds a fixed strategy spec in code (`_strategy_spec`) with:
   - benchmark index: `8841431.WI`
   - rebalance: monthly (`buy_delay=1`, `sell_delay=2`)
   - selection filters: `roe`, debt-to-asset ratio, operating cashflow, `pe_ttm`, `pb`
   - ranking key: `total_market_cap` ascending (smaller cap preferred)
2. Computes monthly rebalance dates with `RebalanceSchedule.schedule(...)` using
   `TradingCalendar` trading days.
3. For each rebalance date, builds selection inputs from `qore-data`:
   - `build_selection_inputs_for_as_of(...)`
   - `build_candidate_selection_snapshot(...)`
4. Adds factor transform in pipeline composition:
   - `FactorPipeline().add(DebtToAssetRatioFactor(...)).run(...)`
5. Produces two date-keyed frames for execution:
   - `factor_frame`: `date`, `symbol`, `total_market_cap`
   - `overlay_frame`: `date`, `symbol`, `selected`, `exclude_reason`
6. Runs `StrategyRunner` + `BacktestEngine`:
   - runner: `CrossSectionalScreener` + `EqualWeightSizer`
   - backtest inputs: `DateColumnDayFrameSource` for factor/decision overlay,
     `StoreMarketDataSource` for market prices.

## Can it run directly right now?

Yes, the code path is runnable directly as a CLI, but only if the required local
datasets are present in your DuckDB/Parquet store.

Run command:

```bash
uv run --package small-cap-strategy small-cap-strategy
```

Optional runtime args:

```bash
uv run --package small-cap-strategy small-cap-strategy \
  --db-path data/qore.duckdb \
  --parquet-root data/raw \
  --initial-capital 10000000 \
  --commission 0.0003 \
  --slippage 0.0005 \
  --drawdown-stop 0.15
```

## Data retrieval consistency and required datasets

Current retrieval is consistent with workflow logic and crate APIs. The workflow
constructs all selection inputs from store reads in `qore-data` and executes
trades from `qore-backtest` market reads.

At minimum, you need:

- `index_constituents` (for benchmark membership at each rebalance `as_of`)
- `stock_profiles` (market cap, board, listing date, ST flag)
- `stock_ohlcv` (status flags and execution prices)
- `fundamentals` (for `roe`, `pe_ttm`, `pb`, `operating_cashflow`, liabilities/assets)
- `strategy_factors` (liquidity/capacity columns such as
  `avg_amount_20d`, `min_amount_20d`, `position_to_amount_20d_ratio`)
- `stock_audit_opinions` (for audit-based exclusion state)

The helper `build_stock_category_report(...)` also uses:

- `analyst_forecasts`
- `announcements`

## Practical status of this example

- The workflow is executable and internally consistent for current contracts.
- It is not a zero-data demo; it depends on pre-populated local datasets.
- The real-data contract test (`examples/small_cap_strategy/tests/test_small_cap.py`)
  explicitly skips when required datasets are missing, and runs assertions only
  when local data is available.

## Output

Running the CLI prints:

- stock category report frame
- NAV frame
- diagnostics frame

Then it renders backtest overview plots via `result.view().with_drawdown().plot().overview()`.
