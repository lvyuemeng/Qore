from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from qore_core.config import QoreConfig


@dataclass(slots=True)
class RiskManager:
    max_single: float
    drawdown_stop: float

    @classmethod
    def from_config(cls, config: QoreConfig) -> RiskManager:
        return cls(
            max_single=config.stock.max_weight,
            drawdown_stop=config.backtest.drawdown_stop,
        )

    def apply(
        self,
        target: dict[str, float],
        current: dict[str, float],
        nav: pl.Series,
    ) -> dict[str, float]:
        del current
        if len(nav) > 1:
            peak_value = nav.max()
            latest_value = nav.tail(1).item()
            peak = float(peak_value) if isinstance(peak_value, (int, float)) else 0.0
            latest = (
                float(latest_value) if isinstance(latest_value, (int, float)) else 0.0
            )
            if peak > 0 and (peak - latest) / peak >= self.drawdown_stop:
                return {}
        capped = {
            symbol: min(weight, self.max_single) for symbol, weight in target.items()
        }
        total = sum(capped.values())
        if total > 1.0 and total > 0.0:
            return {symbol: weight / total for symbol, weight in capped.items()}
        return capped
