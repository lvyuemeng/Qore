from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Factor(Protocol):
    name: str
    produces: str
    requires: frozenset[str]

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Append `self.produces` to the input LazyFrame."""


class OHLCVFactor:
    """Marker for factors derived from price/volume data."""


class FundamentalFactor:
    """Marker for factors derived from point-in-time fundamentals."""


class CrossSectionalFactor:
    """Marker for factors derived from cross-sectional normalization/ranking."""


class EventFactor:
    """Marker for factors derived from event or status overlays."""
