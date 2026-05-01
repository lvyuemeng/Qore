# qore-backtest

Per-day backtest simulation. Accepts signals, market data, and a sizer. Returns NAV, positions, fills, and performance metrics.

## Key exports

| Name | Description |
|---|---|
| `BacktestEngine(config, calendar, signals, market_data, sizer, top_k)` | Constructor — no `from_settings`. Run with `.run()`. |
| `BacktestResult` | `.nav`, `.positions`, `.turnover`, `.fills`, `.diagnostics`. Methods: `.view()` → `BacktestView`, `.metrics(benchmark_nav)` → `dict[str, float]`. |
| `BacktestSettings` | `initial_capital`, `commission`, `slippage`, `buy_delay`, `sell_delay`, `start`, `end`. |
| `BacktestView` | `.with_drawdown()`, `.with_benchmark(name, nav)`, `.window(start, end)`, `.plot()` → `BacktestPlotter`. |
| `BacktestPlotter` | `.equity()`, `.overview()`, `.timeseries(series, ...)` — returns matplotlib `Figure`. |
| `TradingCalendar` | Chinese A‑share calendar with fill scheduling: `fill_plan(requests, day, buy_delay, sell_delay)`. |

## Usage

```python
from qore_backtest import BacktestEngine, BacktestSettings, TradingCalendar
from qore_runner.sizer import EqualWeightSizer

engine = BacktestEngine(
    config=BacktestSettings(initial_capital=10_000_000, commission=0.0003),
    calendar=TradingCalendar(),
    signals=signals.lazy(),            # {date, symbol, signal}
    market_data=market.lazy(),         # {date, symbol, open, close, is_suspended, limit_up, limit_down}
    sizer=EqualWeightSizer(max_weight=0.05),
    top_k=50,
)
result = engine.run()

print(result.nav)
print(result.metrics(benchmark_nav=index_nav))
result.view().with_drawdown().plot().overview()
```

## Architecture

The engine loops over trading days. Per day:

1. **Filter** signals and market data to the current date
2. **Rank** symbols by score (`rank_symbols` from `qore-runner`)
3. **Size** positions (sizer returns fully-capped weights summing to 1.0)
4. **Diff** target weights vs current → buy/sell requests
5. **Fill** requests with delay, slippage, limit-up/down, suspension checks
6. **Return** = weighted return of filled positions × close/open
7. **Accumulate** NAV, positions, turnover, fills, diagnostics

## Metrics

`.metrics(benchmark_nav)` returns 10 metrics: `annualized_return`, `sharpe_ratio`, `calmar_ratio`, `max_drawdown`, `sortino_ratio`, `information_ratio`, `win_rate`, `profit_factor`, `avg_turnover`, `total_commission_cost`.

## Visualization

Plotting requires the `viz` dependency group:
```
uv sync --group viz
```
