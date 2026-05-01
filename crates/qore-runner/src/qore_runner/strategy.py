from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Strategy(Protocol):
    name: str

    @property
    def required_columns(self) -> frozenset[str]: ...

    def generate(self, factor_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Returns a LazyFrame with ``symbol`` and ``signal`` columns."""
        ...
