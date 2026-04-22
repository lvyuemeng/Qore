from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

AlertOperator = Literal["gt", "ge", "lt", "le", "eq", "ne"]


@dataclass(frozen=True, slots=True)
class AlertCondition:
    field: str
    operator: AlertOperator
    value: object

    def expr(self) -> pl.Expr:
        column = pl.col(self.field)
        value = pl.lit(self.value)
        if self.operator == "gt":
            return column > value
        if self.operator == "ge":
            return column >= value
        if self.operator == "lt":
            return column < value
        if self.operator == "le":
            return column <= value
        if self.operator == "eq":
            return column == value
        return column != value


@dataclass(frozen=True, slots=True)
class AlertRule:
    name: str
    conditions: tuple[AlertCondition, ...]
    action: str = "emit_alert"

    def expr(self) -> pl.Expr:
        if not self.conditions:
            return pl.lit(True)
        expr = self.conditions[0].expr()
        for condition in self.conditions[1:]:
            expr = expr & condition.expr()
        return expr


def build_alert_frame(
    lf: pl.LazyFrame,
    *,
    rules: tuple[AlertRule, ...],
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> pl.LazyFrame:
    schema_names = set(lf.collect_schema().names())
    required = {date_column, symbol_column}
    for rule in rules:
        required.update(condition.field for condition in rule.conditions)
    missing = sorted(required - schema_names)
    if missing:
        msg = f"Alert frame missing required columns: {missing}"
        raise ValueError(msg)
    if not rules:
        return pl.DataFrame(
            schema={
                date_column: pl.Date,
                symbol_column: pl.String,
                "alert_name": pl.String,
                "alert_action": pl.String,
            }
        ).lazy()

    frames: list[pl.LazyFrame] = []
    for rule in rules:
        frames.append(
            lf.filter(rule.expr()).select(
                pl.col(date_column),
                pl.col(symbol_column),
                pl.lit(rule.name).alias("alert_name"),
                pl.lit(rule.action).alias("alert_action"),
            )
        )
    return pl.concat(frames, how="vertical")
