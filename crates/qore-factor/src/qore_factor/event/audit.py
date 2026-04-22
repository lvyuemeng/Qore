from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qore_factor.base import EventFactor


@dataclass(slots=True)
class AdverseAuditOpinionFlagFactor(EventFactor):
    name: str = "has_adverse_audit_opinion"
    produces: str = "has_adverse_audit_opinion"
    requires: frozenset[str] = frozenset({"has_adverse_audit_opinion"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("has_adverse_audit_opinion")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias(self.produces)
        )


@dataclass(slots=True)
class ActiveAuditExclusionFactor(EventFactor):
    name: str = "active_audit_exclusion"
    produces: str = "active_audit_exclusion"
    requires: frozenset[str] = frozenset({"active_audit_exclusion"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("active_audit_exclusion")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias(self.produces)
        )


@dataclass(slots=True)
class AdverseAuditOpinionAgeFactor(EventFactor):
    name: str = "adverse_audit_opinion_age_days"
    produces: str = "adverse_audit_opinion_age_days"
    requires: frozenset[str] = frozenset({"adverse_audit_opinion_age_days"})

    def compute(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("adverse_audit_opinion_age_days")
            .cast(pl.Float64)
            .alias(self.produces)
        )
