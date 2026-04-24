# Qore

Qore is a library-first quantitative research and backtest workspace for Chinese
markets.

## Status

- Current phase: **architecture stabilization stage**
- Active architecture snapshot: `docs/design.md`
- Active execution checklist: `docs/roadmap.md`
- User workflow and usage: `docs/introduction.md`
- Contributor extension guide: `docs/workflow.md`
- `src/quant_trade` is legacy migration reference, not target runtime architecture

## Repository Direction

The repository is now a uv workspace monorepo of crate libraries:

- `crates/qore-data`
- `crates/qore-factor`
- `crates/qore-intelligence`
- `crates/qore-runner`
- `crates/qore-backtest`

Key current direction:

- frame-native runner/backtest contracts
- crate-local typed settings (`DataSettings`, `IntelligenceSettings`, `RunnerSettings`, `BacktestSettings`)
- convenience universe API (`pipeline.universe(...)`) for normal usage
- result-view API (`result.view()`) with plotting support through dependency group `viz`

## Quick Start

```bash
uv sync
```

Run the reference workflow:

```bash
uv run python examples/stock_ranking_workflow.py
```

## Legacy Note

The old `quant_trade` code remains temporarily for migration reference. New platform
work should go into `crates/qore-*` and keep crates library-only.
