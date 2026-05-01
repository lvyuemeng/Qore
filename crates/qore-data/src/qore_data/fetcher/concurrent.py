"""Concurrent fetch dispatcher with ProcessPoolExecutor.

Provides a generic ``batch_fetch`` for CPU-bound / IO-bound fetch workers
that must run in separate processes (e.g. BaoStock TCP sessions).
Each worker is a module-level callable (pickleable by dotted-path reference).

Example::

    from qore_data.fetcher.concurrent import BatchConfig, batch_fetch

    def _fetch_one(symbol: str) -> pl.DataFrame: ...

    results = batch_fetch(BatchConfig.process(), _fetch_one, symbols)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class BatchConfig:
    max_workers: int | None = field(default=None)

    @classmethod
    def process(cls, workers: int | None = None) -> BatchConfig:
        return cls(max_workers=workers)


def _optimal_workers(n_tasks: int) -> int:
    import os

    cpu_count = os.cpu_count() or 4
    if n_tasks <= 0:
        return cpu_count
    return min(cpu_count, n_tasks)


def batch_fetch[T](
    config: BatchConfig,
    worker: Any,
    items: list[T],
) -> list[pl.DataFrame]:
    """Dispatch ``worker(item)`` across a process pool, preserving order.

    Args:
        config: BatchConfig controlling parallelism.
        worker: Module-level callable — must be pickleable by dotted path.
        items: Inputs, one per task. Each is passed as single positional arg.

    Returns:
        List of DataFrames in the same order as ``items``.
        Empty DataFrame for failed tasks.
    """
    if not items:
        return []

    n = len(items)
    workers = config.max_workers or _optimal_workers(n)
    results: list[pl.DataFrame] = [pl.DataFrame()] * n

    with ProcessPoolExecutor(max_workers=workers) as pool:
        fut_map: dict[Any, int] = {}
        for idx, item in enumerate(items):
            fut = pool.submit(worker, item)
            fut_map[fut] = idx

        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = pl.DataFrame()

    return results
