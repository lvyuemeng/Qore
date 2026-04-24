from __future__ import annotations

from dataclasses import dataclass

from qore_intelligence import IntelligenceSettings


@dataclass(slots=True)
class FinBERT:
    model_name: str

    @classmethod
    def from_settings(cls, settings: IntelligenceSettings) -> FinBERT:
        return cls(model_name=settings.news_finbert_model)

    def score(self, text: str) -> float:
        positive_words = ("增长", "盈利", "突破", "增持", "上调")
        negative_words = ("亏损", "处罚", "减持", "下调", "违约")
        pos = sum(1 for word in positive_words if word in text)
        neg = sum(1 for word in negative_words if word in text)
        if pos == neg:
            return 0.0
        return max(min((pos - neg) / 3.0, 1.0), -1.0)
