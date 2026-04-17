from __future__ import annotations

import polars as pl


class SignalCombiner:
    def __init__(self, news_alpha: float = 0.0) -> None:
        self.news_alpha = news_alpha

    def combine(
        self,
        model_scores: pl.Series,
        news_scores: dict[str, float],
        *,
        symbols: list[str] | None = None,
    ) -> pl.Series:
        if self.news_alpha == 0.0:
            return model_scores
        score_keys = symbols or [str(idx) for idx in range(len(model_scores))]
        combined = [
            (1.0 - self.news_alpha) * float(score)
            + self.news_alpha * news_scores.get(key, 0.0)
            for key, score in zip(score_keys, model_scores.to_list(), strict=False)
        ]
        return pl.Series(name=model_scores.name, values=combined)
