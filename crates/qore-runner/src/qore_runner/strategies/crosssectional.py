from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.universe import Universe


@dataclass(slots=True)
class CrossSectionalScreener:
    factor_weights: dict[str, float]
    name: str = "cross_sectional_screener"
    compatible_sessions: frozenset[str] = frozenset({"nav", "auction"})
    signal_freq: str = "monthly"

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset(self.factor_weights)

    def generate(
        self,
        lf: pl.LazyFrame,
        news_scores: dict[str, float] | None,
        universe: Universe,
        date: date,
        calendar: TradingCalendar,
    ) -> pl.Series:
        del news_scores
        df = lf.collect()
        weighted_terms = [
            pl.col(column) * weight for column, weight in self.factor_weights.items()
        ]
        if not weighted_terms:
            return pl.Series(name="signal", values=[float("nan")] * len(universe))
        expr = weighted_terms[0]
        for term in weighted_terms[1:]:
            expr = expr + term
        scored = df.select("symbol", expr.alias("signal"))
        mapping = dict(
            zip(
                scored.get_column("symbol").to_list(),
                scored.get_column("signal").to_list(),
                strict=False,
            )
        )
        values = [
            float(mapping.get(symbol, float("nan")))
            if universe.get(symbol).session in self.compatible_sessions
            and not universe.is_suspended(
                symbol, calendar.fill_date(date, universe.get(symbol))
            )
            else float("nan")
            for symbol in universe.symbols()
        ]
        return pl.Series(name="signal", values=values)
