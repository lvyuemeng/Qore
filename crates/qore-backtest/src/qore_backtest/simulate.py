from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import singledispatch
from typing import Literal

import polars as pl

from qore_core.calendar import TradingCalendar
from qore_core.config import BacktestConfig
from qore_core.instrument import (
    DerivativeInstrument,
    FundInstrument,
    Instrument,
    StockInstrument,
)


@dataclass(frozen=True, slots=True)
class Fill:
    status: Literal["filled", "rejected", "pending"]
    fill_date: date | None
    fill_price: float | None
    quantity: float
    reason: str | None = None


@singledispatch
def fill_order(
    inst: Instrument,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    price_data: pl.DataFrame,
    config: BacktestConfig,
    calendar: TradingCalendar,
) -> Fill:
    del order_date, direction, quantity, price_data, config, calendar
    raise TypeError(f"No fill logic for {type(inst).__name__}")


@fill_order.register(StockInstrument)
def _fill_stock_order(
    inst: StockInstrument,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    price_data: pl.DataFrame,
    config: BacktestConfig,
    calendar: TradingCalendar,
) -> Fill:
    fill_date = calendar.next_trading_day(order_date)
    row = _row_for_date(price_data, fill_date)
    if row is None:
        return Fill("rejected", None, None, quantity, "missing price data")
    if bool(row.get("is_suspended", False)):
        return Fill("rejected", fill_date, None, quantity, "suspended")
    if direction == "buy" and bool(row.get("limit_up", False)):
        return Fill("pending", fill_date, None, quantity, "limit up")
    if direction == "sell" and bool(row.get("limit_down", False)):
        return Fill("pending", fill_date, None, quantity, "limit down")
    open_price = _safe_float(row.get("open"))
    if open_price <= 0.0:
        return Fill("rejected", fill_date, None, quantity, "invalid open price")
    slip = 1.0 + config.slippage if direction == "buy" else 1.0 - config.slippage
    return Fill("filled", fill_date, open_price * slip, quantity)


@fill_order.register(FundInstrument)
def _fill_fund_order(
    inst: FundInstrument,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    price_data: pl.DataFrame,
    config: BacktestConfig,
    calendar: TradingCalendar,
) -> Fill:
    delay = inst.subscription_delay if direction == "buy" else inst.redemption_delay
    fill_date = calendar.next_trading_day(order_date, delay)
    row = _row_for_date(price_data, fill_date)
    if row is None:
        return Fill("rejected", None, None, quantity, "missing nav data")
    nav = _safe_float(row.get("nav"))
    if nav <= 0.0:
        return Fill("rejected", fill_date, None, quantity, "invalid nav")
    fee = 1.0 + config.commission if direction == "buy" else 1.0 - config.commission
    return Fill("filled", fill_date, nav * fee, quantity)


@fill_order.register(DerivativeInstrument)
def _fill_derivative_order(
    inst: DerivativeInstrument,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    price_data: pl.DataFrame,
    config: BacktestConfig,
    calendar: TradingCalendar,
) -> Fill:
    del inst, calendar
    row = _row_for_date(price_data, order_date)
    if row is None:
        return Fill("rejected", None, None, quantity, "missing derivative data")
    open_price = _safe_float(row.get("open"))
    if open_price <= 0.0:
        return Fill("rejected", order_date, None, quantity, "invalid open price")
    slip = 1.0 + config.slippage if direction == "buy" else 1.0 - config.slippage
    return Fill("filled", order_date, open_price * slip, quantity)


def _row_for_date(
    price_data: pl.DataFrame, fill_date: date
) -> dict[str, object] | None:
    if price_data.is_empty() or "date" not in price_data.columns:
        return None
    matched = price_data.filter(pl.col("date") == fill_date)
    if matched.is_empty():
        return None
    return matched.to_dicts()[0]


def _safe_float(value: object | None) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    try:
        return float(0.0 if value is None else str(value))
    except (TypeError, ValueError):
        return 0.0
