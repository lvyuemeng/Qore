from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_data import DataSettings
from qore_data.store.duckdb import QoreStore
from qore_intelligence.combine import SignalCombiner
from qore_intelligence import IntelligenceSettings
from qore_intelligence.signal.llm import LLMExtractor
from qore_intelligence.signal.score import NewsArticle, NewsPipeline
from qore_intelligence.signal.triage import Triage


def test_signal_combiner_returns_model_scores_when_disabled() -> None:
    series = pl.Series(name="score", values=[0.1, 0.2])
    combiner = SignalCombiner(news_alpha=0.0)
    result = combiner.combine(series, {"0": 1.0})
    assert result.to_list() == [0.1, 0.2]


def test_triage_classifies_regulatory_news() -> None:
    triage = Triage()
    assert triage.classify("公司收到监管问询函") == "regulatory"


@pytest.mark.asyncio
async def test_llm_extractor_respects_budget() -> None:
    extractor = LLMExtractor(model="demo", daily_budget=1)
    assert await extractor.extract("公司盈利增长") is not None
    assert await extractor.extract("公司盈利增长") is None


def test_news_pipeline_decay_score(tmp_path: Path) -> None:
    data_settings = DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )
    pipeline = NewsPipeline.from_settings(
        IntelligenceSettings(),
        QoreStore.from_settings(data_settings),
    )
    assert pipeline.decay_score(1.0, 5) < 1.0


@pytest.mark.asyncio
async def test_news_pipeline_processes_and_persists_articles(tmp_path: Path) -> None:
    data_settings = DataSettings(
        db_path=str(tmp_path / "qore.duckdb"),
        parquet_root=str(tmp_path / "raw"),
    )
    intelligence_settings = IntelligenceSettings(
        news_score_half_life_days=5,
        news_llm_daily_budget=5,
    )
    store = QoreStore.from_settings(data_settings)
    pipeline = NewsPipeline.from_settings(intelligence_settings, store)

    result = await pipeline.process_articles(
        trading_date=date(2026, 4, 17),
        articles=[
            NewsArticle(
                symbol="600519.SH",
                published_at=date(2026, 4, 17),
                text="公司盈利增长并上调全年指引",
            ),
            NewsArticle(
                symbol="000001.SZ",
                published_at=date(2026, 4, 16),
                text="公司收到监管处罚决定书",
            ),
        ],
    )

    assert result.height == 2
    persisted = pl.DataFrame(store.read("news_scores").collect()).sort(
        ["date", "symbol"]
    )
    assert persisted.height == 2
    assert persisted.get_column("symbol").to_list() == ["000001.SZ", "600519.SH"]
    assert persisted.get_column("source_layer").to_list() == ["llm", "llm"]
