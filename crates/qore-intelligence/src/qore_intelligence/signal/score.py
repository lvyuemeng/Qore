from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log

import polars as pl

from qore_core.config import QoreConfig
from qore_data.store.duckdb import QoreStore
from qore_intelligence.signal.llm import LLMExtractor
from qore_intelligence.signal.sentiment import FinBERT
from qore_intelligence.signal.triage import Triage


@dataclass(slots=True)
class NewsPipeline:
    triage: Triage
    sentiment: FinBERT
    llm: LLMExtractor
    store: QoreStore
    half_life: int

    @classmethod
    def from_config(cls, config: QoreConfig, store: QoreStore) -> "NewsPipeline":
        return cls(
            triage=Triage(),
            sentiment=FinBERT.from_config(config),
            llm=LLMExtractor.from_config(config),
            store=store,
            half_life=config.intelligence.news_score_half_life_days,
        )

    async def run(self, trading_date: date) -> None:
        self.store.write(
            "news_scores",
            pl.DataFrame(
                {
                    "date": [trading_date],
                    "symbol": ["DUMMY.SZ"],
                    "score": [0.0],
                    "event_type": ["other"],
                    "source_layer": ["triage"],
                }
            ),
        )

    def decay_score(self, score: float, age_days: int) -> float:
        if self.half_life <= 0:
            return score
        decay_lambda = log(2.0) / self.half_life
        return score * exp(-decay_lambda * age_days)
