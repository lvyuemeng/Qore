# qore-runner

Signal ranking, position sizing, and execution signal generation for Qore.

## Key exports

### Signal generation (`strategies/`)

| Strategy | What it does | `generate(factor_lf)` output |
|---|---|---|
| `CrossSectionalScreener(factor_weights)` | Weighted sum of factor columns as a single composite signal. | `{symbol, signal = Σ(col × weight)}` |
| `RankingStrategy(score_provider, combiner?)` | Delegates scoring to a `ScoreProvider`, optionally blends with a `ScoreCombiner` overlay. | `{symbol, signal}` from provider |
| `BehavioralGatedStrategy(base, regime_detector?, vol_column?, min_scale=0.5)` | Wraps any base `Strategy`, scales its signal by `max(min_scale, regime_scale × vol_scale)`. | `{symbol, signal = base_signal × scale}` |

**Supporting protocols:**

| Protocol | Required method | Description |
|---|---|---|
| `Strategy` | `generate(factor_lf) → pl.LazyFrame` | Accepts any object with `name`, `required_columns`, `generate()`. |
| `ScoreProvider` | `predict_scores(factor_lf) → pl.LazyFrame` | Returns `{symbol, signal}` from a model. |
| `ScoreCombiner` | `combine(scores, overlay) → pl.LazyFrame` | Blends provider scores with an overlay signal. |
| `WeightedOverlayCombiner(alpha)` | implements `combine()` | `signal = (1-α) × score + α × overlay`. Default `α=0` = passthrough. |

### Ranking & execution (`decision.py`)

```python
rank_symbols(signal_frame, *, top_k=None, score_column="signal", descending=True) → list[str]
```

Sort symbols by score, filter non-finite, apply `top_k`. Returns ordered symbol list — ready to feed into the sizer.

```python
execution_signals(*, target_weights, current_weights, as_of, score_value_frame=None) → pl.DataFrame
```

Diff target vs current weights → buy/sell/hold signals. Columns: `date`, `symbol`, `signal`, `weight_target`, `weight_current`, `weight_delta`, `score_value`.

### Position sizing (`sizer.py`)

| Sizer | Constructor | `size(signals)` returns |
|---|---|---|
| `EqualWeightSizer` | `max_weight=0.05` | Equal weights; skips capping pass if `n × max_weight ≤ 1.0`. |
| `VolScaledSizer` | `vol_col, max_weight=0.10, volatility={}` | Inverse-vol weights, proportionally capped & renormalized. |

`size()` is the only method in the `PositionSizer` protocol — weights are always returned fully capped and summing to 1.0. The `_cap_and_renormalize` algorithm sorts by weight descending, caps top-k at `max_weight`, and distributes residual proportionally. O(N log N + K) — no iterative loop in the common case.

`VolScaledSizer.with_volatility({"AAA": 0.2})` creates a clone with manual volatility overrides for specific symbols.

### Utilities

| Name | Description |
|---|---|
| `TradingCalendar` | Chinese A‑share calendar: `is_trading_day(d)`, `trading_days_between(s, e)` |
| `DECISION_SIGNAL_SCHEMA` | dict of `{column: pl.DataType}` for `execution_signals()` output |

## Usage

### Small-cap style workflow

```python
from datetime import date
import polars as pl
from qore_data import DataSettings, StockPipeline
from qore_factor.pipeline import FactorPipeline
from qore_factor.fundamental.quality import AssetTurnoverFactor
from qore_runner import rank_symbols, execution_signals, EqualWeightSizer
from qore_runner.strategies.crosssectional import CrossSectionalScreener

pipe = StockPipeline.from_settings(DataSettings())

# 1. Fetch data
symbols = await pipe.resolve("000852.SH", date.today())
await pipe.stock_daily(symbols, start=date(2024, 1, 1))
await pipe.fundamentals(symbols)
await pipe.stock_profiles(symbols)

# 2. Build factor pipeline
factor_lf = pipe.market_corpus(symbols, date(2024, 1, 1), date.today())
screener = CrossSectionalScreener({"total_market_cap": -1.0, "asset_turnover": 1.0})
signals = pl.DataFrame(screener.generate(factor_lf).collect())

# 3. Rank and select
selected = rank_symbols(signals, top_k=50)
selected_signals = signals.filter(pl.col("symbol").is_in(selected))

# 4. Size positions
sizer = EqualWeightSizer(max_weight=0.05)
weights = sizer.size(selected_signals)

# 5. Execution signals (diff vs current holdings)
execution = execution_signals(
    target_weights=weights,
    current_weights=current,  # from portfolio system
    as_of=date.today(),
)
```

### Volatility-scaled sizing

```python
sizer = VolScaledSizer(vol_col="realized_vol_20d", max_weight=0.10)\
    .with_volatility({"600519.SH": 0.15})  # manual vol override
weights = sizer.size(selected_signals)
```

### Behavioral gating

```python
from qore_runner.strategies.behavioral import BehavioralGatedStrategy

class MyRegime:
    def scale(self, lf):
        return 0.8  # pull back 20% in this regime

strategy = BehavioralGatedStrategy(
    base=CrossSectionalScreener({"roe": 1.0}),
    regime_detector=MyRegime(),
    vol_column="realized_vol_20d",
    min_scale=0.3,
)
```
