# Qore Agent Introduction

This repository is the `Qore` rewrite workspace.

- Treat `docs/design.md` as the canonical architecture specification.
- Treat `docs/roadmap.md` as the execution order and status tracker.
- Implement new work only under `crates/qore-*` unless migration support is required.
- Keep the stack aligned with the design: Python 3.13, uv workspace, Polars lazy pipelines, DuckDB + Parquet, strict typing.
- Never import `akshare` in crate runtime code; use `.ai/refs/akshare/` only for endpoint reverse engineering.
- Use `singledispatch` for instrument-specific behavior; do not add `isinstance` routing.
- Every class needing paths or tuning parameters must expose `from_config(config: QoreConfig)`.

## Workspace Basics

Install and sync the workspace:

```bash
uv sync --dev
```

Run commands from the repository root so workspace packages resolve correctly.

## Basic uv Commands

Run all tests:

```bash
uv run pytest
```

Run one test file:

```bash
uv run pytest crates/qore-core/tests/test_core_smoke.py
```

Run lint checks:

```bash
uv run ruff check .
```

Auto-fix lint issues when safe:

```bash
uv run ruff check --fix .
```

Format the codebase:

```bash
uv run ruff format .
```

Type check the codebase:

```bash
uv run ty check .
```

If `ty` is not installed in the environment yet, add it to dev dependencies first or run it as a tool.

## Existing Just Recipes

The root `Justfile` already exposes the common entrypoints:

```bash
just test
just lint
just format
just type
```

These recipes currently map to:

- `just test` -> `uv run pytest`
- `just lint` -> `uv run ruff check .`
- `just format` -> `uv run ruff format .`
- `just type` -> `uv run ty .`

## Crate-Scoped Examples

Run a single crate's tests:

```bash
uv run pytest crates/qore-factor/tests
```

Run a module with workspace imports enabled:

```bash
uv run python -m qore_data.fetch
```

Run a one-off Python snippet inside the workspace environment:

```bash
uv run python -c "from qore_core import QoreConfig; print(QoreConfig())"
```

## Agent Working Rules

- Read `docs/design.md` before structural work.
- Update `docs/roadmap.md` when a roadmap item materially changes state.
- Prefer focused crate tests after changes, then broader checks.
- Keep new code ASCII unless the file already requires non-ASCII content.
- Do not treat legacy `src/quant_trade` as the target architecture.
