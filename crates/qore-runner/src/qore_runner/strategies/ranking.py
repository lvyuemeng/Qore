from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.config import QoreConfig
from qore_core.universe import Universe
from qore_intelligence.combine import SignalCombiner
from qore_intelligence.model.pipeline import ModelPipeline


@dataclass(slots=True)
class RankingStrategy:
    pipeline: ModelPipeline
    combiner: SignalCombiner
    name: str = "ranking"
    compatible_sessions: frozenset[str] = frozenset({"auction", "nav", "continuous"})
    signal_freq: str = "weekly"
    required_columns: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, config: QoreConfig) -> RankingStrategy:
        return cls(
            pipeline=ModelPipeline.load("stock_ranker", config),
            combiner=SignalCombiner(news_alpha=0.0),
        )

    def generate(
        self,
        lf: pl.LazyFrame,
        news_scores: dict[str, float] | None,
        universe: Universe,
        date: date,
        calendar: TradingCalendar,
    ) -> pl.Series:
        df = lf.collect()
        scores = self.pipeline.predict_score(df.lazy())
        combined = self.combiner.combine(
            scores,
            news_scores or {},
            symbols=df.get_column("symbol").to_list(),
        )
        mapping = dict(
            zip(
                df.get_column("symbol").to_list(),
                combined.to_list(),
                strict=False,
            )
        )
        values = [
            float(mapping.get(symbol, float("nan")))
            if universe.get(symbol)
            and universe.get(symbol).session in self.compatible_sessions
            and not universe.is_suspended(
                symbol, calendar.fill_date(date, universe.get(symbol))
            )
            else float("nan")
            for symbol in universe.symbols()
        ]
        return pl.Series(name="signal", values=values)
