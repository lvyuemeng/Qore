# small-cap-strategy

Small-cap strategy workflow project initialized via `uv init` and hosted as a
workspace member.

## Run

```bash
uv run --package small-cap-strategy small-cap-strategy
```

## Library Usage Notes

- `qore_data.StockSelectionPipeline` is used as the universe and feature join
  backbone (`with_profiles`, `with_statuses`, `with_fundamentals`,
  `with_daily_market`, `with_audit_opinion_state`).
- `qore_runner.CrossSectionalScreener + EqualWeightSizer` is used for
  signal-to-weight translation with monthly rebalance overlays.
- `qore_backtest.BacktestEngine` consumes `factor_scores` and
  `decision_overlays_by_day` to execute deterministic backtests.
- The example uses a fixed in-code strategy contract (`_strategy_spec`) instead of
  YAML parsing.
