# qore-factor

Lazy factor transforms and pipeline composition for Qore.

## Key exports

### Pipeline

| Name | Description |
|---|---|
| `FactorPipeline` | Lazy composition of factor transforms. Constructor accepts `normalize`, `normalize_group_by`, `neutralize_by`. Call `.add(factors...)`, then `.run(lf)`. |

### Factor base types (markers)

| Name | Description |
|---|---|
| `Factor` | Protocol — any object with `.name`, `.produces`, `.requires`, `.compute(lf)` |
| `OHLCVFactor` | Marker for price/volume derived factors |
| `FundamentalFactor` | Marker for financial statement derived factors |
| `CrossSectionalFactor` | Marker for cross-sectional normalization/ranking |
| `EventFactor` | Marker for event/status overlay factors |

### OHLCV factors (`qore_factor.ohlcv`)

| Factor | Produces | Configurable inputs |
|---|---|---|
| `MomentumFactor(lookback, skip)` | `mom_{lookback}d_skip{skip}` | `close_column` (default: `close`) |
| `RealizedVolatilityFactor(window)` | `realized_vol_{window}d` | `close_column` (default: `close`) |
| `AverageAmountFactor(window)` | `avg_amount_{window}d` | `amount_column` (default: `amount`) |
| `MinimumAmountFactor(window)` | `min_amount_{window}d` | `amount_column` (default: `amount`) |
| `PositionToLiquidityRatioFactor(liquidity_column, position_column)` | `position_to_{suffix}_ratio` | — |
| `CapacityPenaltyFactor(ratio_column, threshold)` | `capacity_penalty_{ratio_column}` | — |

Helper: `liquidity_capacity_factors(window, position_column)` returns `(AverageAmount, MinimumAmount, PositionToLiquidityRatio)` with auto-linked column names.

### Fundamental factors (`qore_factor.fundamental`)

| Factor | Derivation | Configurable inputs |
|---|---|---|
| `AssetTurnoverFactor` | `revenue / total_assets` | `revenue_column`, `assets_column` |
| `CFOYieldFactor` | `operating_cashflow / total_assets` | `cfo_column`, `assets_column` |
| `AccrualRatioFactor` | `(net_income - cfo) / total_assets` | `net_income_column`, `cfo_column`, `assets_column` |
| `DebtToAssetRatioFactor` | `liabilities / assets` (zero-asset guard) | `liabilities_column`, `assets_column` |
| `ROEStabilityFactor` | `1 / roe.rolling_std(window).over(symbol)` | `roe_column`, `window` |
| `ProfitGrowthPremiumFactor` | `net_profit_yoy - revenue_growth_yoy` | `net_profit_growth_column`, `revenue_growth_column` |
| `BookToPriceFactor` | `1 / pb` | `pb_column` |
| `SUEFactor` | `(actual_eps - consensus_eps) / eps_std` | (fixed: `actual_eps`, `consensus_eps`, `eps_std`) |

### Event utilities (`qore_factor.event`)

| Name | Description |
|---|---|
| `AlertCondition(field, operator, value)` | One filter condition |
| `AlertRule(name, conditions, action)` | AND-combined conditions |
| `build_alert_frame(lf, rules)` | Apply rules, return alert table |

## Usage

```python
from datetime import date
import polars as pl
from qore_factor.pipeline import FactorPipeline
from qore_factor.fundamental.quality import DebtToAssetRatioFactor, AssetTurnoverFactor
from qore_factor.fundamental.value import BookToPriceFactor
from qore_factor.ohlcv.momentum import MomentumFactor

lf = pl.DataFrame({
    "date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 1), date(2026, 1, 2)],
    "symbol": ["AAA", "AAA", "BBB", "BBB"],
    "close": [10.0, 11.0, 20.0, 21.0],
    "pb": [2.0, 2.2, 1.5, 1.6],
    "total_liabilities": [120.0, 120.0, 150.0, 150.0],
    "total_assets": [400.0, 400.0, 300.0, 300.0],
    "revenue": [100.0, 100.0, 200.0, 200.0],
}).lazy()

result = (
    FactorPipeline(normalize="zscore")
    .add(MomentumFactor(lookback=1, skip=0),
         BookToPriceFactor(),
         DebtToAssetRatioFactor(),
         AssetTurnoverFactor())
    .run(lf)
).collect()
```

## Principles

- All factors operate on `pl.LazyFrame` — never call `.collect()` inside `compute()`.
- `requires` is auto-derived from configurable input column names — no stale frozensets.
- Every factor has configurable input/output column names except where the derivation is tightly coupled to a specific schema.
- `read()` from `qore-data` produces `pl.LazyFrame`, pass it directly to `FactorPipeline.run()`.
