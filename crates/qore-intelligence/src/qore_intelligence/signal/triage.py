from __future__ import annotations

import re
from dataclasses import dataclass, field

import jieba


@dataclass(slots=True)
class Triage:
    keywords: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "earnings": ("业绩", "盈利", "利润", "预增", "预亏"),
            "guidance": ("指引", "展望", "预期"),
            "regulatory": ("监管", "问询", "处罚", "立案"),
            "ma": ("收购", "并购", "重组", "增持", "减持"),
        }
    )

    def classify(self, text: str) -> str:
        tokens = set(jieba.cut(text))
        for event_type, words in self.keywords.items():
            if any(word in text or word in tokens for word in words):
                return event_type
        return "other"

    def trading_relevant(self, text: str) -> bool:
        return self.classify(text) != "other" or bool(
            re.search(r"涨停|跌停|停牌", text)
        )
