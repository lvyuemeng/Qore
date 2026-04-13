from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from qore_core.config import QoreConfig


class EventExtraction(BaseModel):
    event_type: Literal["earnings", "guidance", "regulatory", "ma", "other"]
    direction: Literal["positive", "negative", "neutral"]
    magnitude: Literal["high", "medium", "low"]
    certainty: float = Field(ge=0.0, le=1.0)
    trading_relevant: bool


@dataclass(slots=True)
class _DailyBudget:
    remaining: int

    def can_call(self) -> bool:
        return self.remaining > 0

    def record(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1


@dataclass(slots=True)
class LLMExtractor:
    model: str
    daily_budget: int
    _budget: _DailyBudget = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._budget = _DailyBudget(self.daily_budget)

    @classmethod
    def from_config(cls, config: QoreConfig) -> "LLMExtractor":
        return cls(
            model=config.intelligence.news_llm_model,
            daily_budget=config.intelligence.news_llm_daily_budget,
        )

    async def extract(self, text: str) -> EventExtraction | None:
        if not self._budget.can_call():
            return None
        self._budget.record()
        direction: Literal["positive", "negative", "neutral"]
        if any(word in text for word in ("增长", "盈利", "中标", "上调")):
            direction = "positive"
        elif any(word in text for word in ("亏损", "处罚", "违约", "下调")):
            direction = "negative"
        else:
            direction = "neutral"
        return EventExtraction(
            event_type="other",
            direction=direction,
            magnitude="medium",
            certainty=0.5,
            trading_relevant=direction != "neutral",
        )
