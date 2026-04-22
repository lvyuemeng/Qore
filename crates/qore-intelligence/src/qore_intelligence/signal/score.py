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


@dataclass(slots=True, frozen=True)
class NewsArticle:
    symbol: str
    published_at: date
    text: str


@dataclass(slots=True)
class NewsPipeline:
    triage: Triage
    sentiment: FinBERT
    llm: LLMExtractor
    store: QoreStore
    half_life: int

    @classmethod
    def from_config(cls, config: QoreConfig, store: QoreStore) -> NewsPipeline:
        return cls(
            triage=Triage(),
            sentiment=FinBERT.from_config(config),
            llm=LLMExtractor.from_config(config),
            store=store,
            half_life=config.intelligence.news_score_half_life_days,
        )

    async def run(self, trading_date: date) -> None:
        await self.process_articles(trading_date, [])

    async def process_articles(
        self,
        trading_date: date,
        articles: list[NewsArticle],
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for article in articles:
            event_type = self.triage.classify(article.text)
            if not self.triage.trading_relevant(article.text):
                continue

            sentiment_score = self.sentiment.score(article.text)
            llm_result = await self.llm.extract(article.text)
            layer = "triage"
            score = sentiment_score

            if llm_result is not None and llm_result.trading_relevant:
                layer = "llm"
                score = self._score_extraction(
                    sentiment_score, llm_result.direction, llm_result.magnitude
                )
                event_type = llm_result.event_type
            elif sentiment_score != 0.0:
                layer = "finbert"

            age_days = max((trading_date - article.published_at).days, 0)
            rows.append(
                {
                    "date": trading_date,
                    "symbol": article.symbol,
                    "score": self.decay_score(score, age_days),
                    "event_type": event_type,
                    "source_layer": layer,
                }
            )

        if not rows:
            return pl.DataFrame(schema=self._news_schema())

        output = (
            pl.DataFrame(rows)
            .group_by("date", "symbol")
            .agg(
                pl.col("score").mean().alias("score"),
                pl.col("event_type").last().alias("event_type"),
                pl.col("source_layer").last().alias("source_layer"),
            )
            .sort(["date", "symbol"])
        )
        self.store.write("news_scores", output)
        return output

    def decay_score(self, score: float, age_days: int) -> float:
        if self.half_life <= 0:
            return score
        decay_lambda = log(2.0) / self.half_life
        return score * exp(-decay_lambda * age_days)

    def _score_extraction(
        self,
        sentiment_score: float,
        direction: str,
        magnitude: str,
    ) -> float:
        direction_score = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0,
        }[direction]
        magnitude_score = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
        }[magnitude]
        blended = 0.5 * sentiment_score + 0.5 * direction_score
        return max(min(blended * magnitude_score, 1.0), -1.0)

    def _news_schema(self) -> dict[str, pl.DataType | type[pl.DataType] | None]:
        return {
            "date": pl.Date,
            "symbol": pl.String,
            "score": pl.Float64,
            "event_type": pl.String,
            "source_layer": pl.String,
        }
