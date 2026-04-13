from __future__ import annotations

from pathlib import Path

import pytest
import polars as pl

from qore_core import QoreConfig
from qore_data.store.duckdb import QoreStore
from qore_intelligence.combine import SignalCombiner
from qore_intelligence.signal.llm import LLMExtractor
from qore_intelligence.signal.score import NewsPipeline
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
    config = QoreConfig.model_validate(
        {
            "data": {
                "db_path": str(tmp_path / "qore.duckdb"),
                "parquet_root": str(tmp_path / "raw"),
            }
        }
    )
    pipeline = NewsPipeline.from_config(config, QoreStore.from_config(config))
    assert pipeline.decay_score(1.0, 5) < 1.0
