# Qore Agent Introduction

This repository is the `Qore` rewrite workspace.

- Treat `docs/design.md` as the canonical architecture specification.
- Treat `docs/roadmap.md` as the execution order and status tracker.
- Implement new work only under `crates/qore-*` unless migration support is required.
- Keep the stack aligned with the design: Python 3.13, uv workspace, Polars lazy pipelines, DuckDB + Parquet, strict typing.
- Never import `akshare` in crate runtime code; use `.ai/refs/akshare/` only for endpoint reverse engineering.
- Use `singledispatch` for instrument-specific behavior; do not add `isinstance` routing.
- Prefer crate-local typed config structures; do not introduce new cross-crate `QoreConfig` coupling.
- Construct dependencies from each crate's own config type; unify config surfaces only at workflow/CLI entry layers.

## Principle

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

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

Code quality:

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check .
```

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

## Agent Working Rules

- Read `docs/design.md` before structural work.
- Update `docs/roadmap.md` when a roadmap item materially changes state.
- Prefer focused crate tests after changes, then broader checks.
- Keep new code ASCII unless the file already requires non-ASCII content.
- Do not treat legacy `src/quant_trade` as the target architecture.
- For config work, avoid adding `from_config(QoreConfig)` in crate internals; prefer `from_settings(...)` or direct typed constructor inputs from the owning crate.
