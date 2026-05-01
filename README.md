# Qore

Qore is a library-first quantitative research and backtest workspace for Chinese
markets.

## Status

- Current phase: **architecture stabilization stage**
- Active architecture snapshot: `docs/design.md`
- Active execution checklist: `docs/roadmap.md`
- User workflow and usage: `docs/introduction.md`
- Contributor extension guide: `docs/workflow.md`
- Configuration reference: `docs/config.md`
- `src/quant_trade` is legacy migration reference, not target runtime architecture

## Repository Direction

The repository is a uv workspace monorepo of crate libraries:

| Crate | Purpose |
|---|---|
| `qore-data` | Data fetch, store (DuckDB + Parquet), selection pipeline, candidate filtering |
| `qore-factor` | Factor transforms, `FactorPipeline`, liquidity/capacity frames |
| `qore-intelligence` | Model pipeline, registry, signal overlays, normalizers |
| `qore-runner` | Strategy decision, sizing, rebalance scheduling |
| `qore-backtest` | Execution simulation, fills, diagnostics, metrics, result views |

Key architectural patterns:

- Frame-native contracts (`pl.DataFrame`/`pl.LazyFrame`) in hot paths.
- Crate-local typed settings (`DataSettings`, `RunnerSettings`, `BacktestSettings`, `IntelligenceSettings`).
- Greedy storage: maximal data ingest, filter at read time via SQL views.
- Signal-first: runner/backtest consume strategy decisions; factors live in `qore-factor`.
- Protocol-driven engine: backtest has no data import coupling.
- `result.view().with_drawdown().plot().overview()` for visualization.

## Quick Start

```bash
uv sync
```

Run the reference workflow:

```bash
uv run --package small-cap-strategy small-cap-strategy
```

Prepare data only:

```bash
uv run --package small-cap-strategy small-cap-strategy --prepare-data
```

## Code Quality

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Legacy Note

The old `quant_trade` code remains temporarily for migration reference. New platform
work should go into `crates/qore-*` and keep crates library-only.
