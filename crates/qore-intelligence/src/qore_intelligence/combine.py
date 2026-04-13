from __future__ import annotations

import polars as pl


class SignalCombiner:
    def __init__(self, news_alpha: float = 0.0) -> None:
        self.news_alpha = news_alpha

    def combine(
        self,
        model_scores: pl.Series,
        news_scores: dict[str, float],
    ) -> pl.Series:
        if self.news_alpha == 0.0:
            return model_scores
        combined = [
            (1.0 - self.news_alpha) * float(score)
            + self.news_alpha * news_scores.get(str(idx), 0.0)
            for idx, score in enumerate(model_scores.to_list())
        ]
        return pl.Series(name=model_scores.name, values=combined)
