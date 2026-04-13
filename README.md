# Qore

Qore is a rewrite-in-progress quantitative research and trading platform for Chinese
markets.

## Status

- Active architecture target: `docs/design.md`
- Execution plan and checklist: `docs/roadmap.md`
- Current `src/quant_trade` package is legacy migration material, not the target design
- `.ai/` is local reference material only and stays gitignored

## Repository Direction

The repository is being rebuilt into a uv workspace monorepo:

- `crates/qore-core`
- `crates/qore-data`
- `crates/qore-factor`
- `crates/qore-intelligence`
- `crates/qore-runner`
- `crates/qore-backtest`

## Quick Start

```bash
uv sync
```

## Legacy Note

The old `quant_trade` code remains temporarily for migration reference. New platform
work should go into `crates/qore-*`.
