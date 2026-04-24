# Qore Configuration (Reference)

Primary user-facing config/workflow introduction moved to `docs/introduction.md`.

This file is kept as a short boundary reference.

## Configuration boundary

- Use crate-local typed settings in crate runtime APIs (`DataSettings`, `IntelligenceSettings`, `RunnerSettings`, `BacktestSettings`).
- Keep `QoreConfig` (if used) at workflow composition boundaries only.
- Do not couple crate internals to cross-crate global config objects.

## What belongs in config

- filesystem/storage locations
- source runtime knobs (timeout, retries, concurrency, cooldown)
- runtime budgets and default behavior

## What does not belong in config

- trained model payload state
- learned weights/schema and training summaries
- model-family learned internals produced by fitting

## Rule for new code

- map boundary config to crate-local settings in workflow/example code
- keep crate internals typed and config-decoupled
- keep crates library-first; do not introduce product CLI entrypoints in crates
