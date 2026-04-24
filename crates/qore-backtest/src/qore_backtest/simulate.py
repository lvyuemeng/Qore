from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, TypedDict

import polars as pl
from qore_runner.calendar import TradingCalendar

from qore_backtest import BacktestSettings

TradingSession = Literal["auction", "nav", "continuous"]


class MarketRow(TypedDict, total=False):
    open: float | int | None
    nav: float | int | None
    is_suspended: bool
    limit_up: bool
    limit_down: bool


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    status: Literal["filled", "rejected", "pending"]
    fill_date: date | None
    fill_price: float | None
    quantity: float
    reason: str | None = None


def fill_delay(
    session: TradingSession,
    direction: Literal["buy", "sell"],
    *,
    buy_delay: int | None = None,
    sell_delay: int | None = None,
) -> int:
    if session == "nav":
        if direction == "buy":
            return max(0, buy_delay if buy_delay is not None else 1)
        return max(0, sell_delay if sell_delay is not None else 2)
    if session == "continuous":
        return 0
    return 1


def expected_fill_date(
    session: TradingSession,
    order_date: date,
    direction: Literal["buy", "sell"],
    calendar: TradingCalendar,
    *,
    buy_delay: int | None = None,
    sell_delay: int | None = None,
) -> date:
    delay = fill_delay(
        session,
        direction,
        buy_delay=buy_delay,
        sell_delay=sell_delay,
    )
    if delay == 0:
        return order_date
    return calendar.next_trading_day(order_date, delay)


def fill_order(
    symbol: str,
    session: TradingSession,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    price_data: pl.DataFrame,
    config: BacktestSettings,
    calendar: TradingCalendar,
    *,
    buy_delay: int | None = None,
    sell_delay: int | None = None,
) -> Fill:
    fill_date = expected_fill_date(
        session,
        order_date,
        direction,
        calendar,
        buy_delay=buy_delay,
        sell_delay=sell_delay,
    )
    row = _row_for_date(price_data, fill_date)
    return _fill_from_market_row(
        symbol,
        session,
        fill_date,
        direction,
        quantity,
        row,
        config,
    )


def fill_order_from_market_row(
    symbol: str,
    session: TradingSession,
    order_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    market_row: MarketRow | None,
    config: BacktestSettings,
    calendar: TradingCalendar,
    *,
    buy_delay: int | None = None,
    sell_delay: int | None = None,
) -> Fill:
    fill_date = expected_fill_date(
        session,
        order_date,
        direction,
        calendar,
        buy_delay=buy_delay,
        sell_delay=sell_delay,
    )
    return _fill_from_market_row(
        symbol,
        session,
        fill_date,
        direction,
        quantity,
        market_row,
        config,
    )


def _fill_from_market_row(
    symbol: str,
    session: TradingSession,
    fill_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    market_row: MarketRow | None,
    config: BacktestSettings,
) -> Fill:
    if session == "nav":
        return _fill_nav_order(
            symbol,
            fill_date,
            direction,
            quantity,
            market_row,
            config,
        )
    if session == "continuous":
        return _fill_continuous_order(
            symbol,
            fill_date,
            direction,
            quantity,
            market_row,
            config,
        )
    return _fill_stock_order(
        symbol,
        fill_date,
        direction,
        quantity,
        market_row,
        config,
    )


def _fill_stock_order(
    symbol: str,
    fill_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    market_row: MarketRow | None,
    config: BacktestSettings,
) -> Fill:
    if market_row is None:
        return Fill(symbol, "rejected", None, None, quantity, "missing price data")
    if bool(market_row.get("is_suspended", False)):
        return Fill(symbol, "rejected", fill_date, None, quantity, "suspended")
    if direction == "buy" and bool(market_row.get("limit_up", False)):
        return Fill(symbol, "pending", fill_date, None, quantity, "limit up")
    if direction == "sell" and bool(market_row.get("limit_down", False)):
        return Fill(symbol, "pending", fill_date, None, quantity, "limit down")
    open_value = market_row.get("open")
    open_price = float(open_value) if isinstance(open_value, (int, float)) else 0.0
    if open_price <= 0.0:
        return Fill(symbol, "rejected", fill_date, None, quantity, "invalid open price")
    slip = 1.0 + config.slippage if direction == "buy" else 1.0 - config.slippage
    return Fill(symbol, "filled", fill_date, open_price * slip, quantity)


def _fill_nav_order(
    symbol: str,
    fill_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    market_row: MarketRow | None,
    config: BacktestSettings,
) -> Fill:
    if market_row is None:
        return Fill(symbol, "rejected", None, None, quantity, "missing nav data")
    nav_value = market_row.get("nav")
    nav = float(nav_value) if isinstance(nav_value, (int, float)) else 0.0
    if nav <= 0.0:
        return Fill(symbol, "rejected", fill_date, None, quantity, "invalid nav")
    fee = 1.0 + config.commission if direction == "buy" else 1.0 - config.commission
    return Fill(symbol, "filled", fill_date, nav * fee, quantity)


def _fill_continuous_order(
    symbol: str,
    fill_date: date,
    direction: Literal["buy", "sell"],
    quantity: float,
    market_row: MarketRow | None,
    config: BacktestSettings,
) -> Fill:
    if market_row is None:
        return Fill(symbol, "rejected", None, None, quantity, "missing derivative data")
    open_value = market_row.get("open")
    open_price = float(open_value) if isinstance(open_value, (int, float)) else 0.0
    if open_price <= 0.0:
        return Fill(symbol, "rejected", fill_date, None, quantity, "invalid open price")
    slip = 1.0 + config.slippage if direction == "buy" else 1.0 - config.slippage
    return Fill(symbol, "filled", fill_date, open_price * slip, quantity)


def _row_for_date(price_data: pl.DataFrame, fill_date: date) -> MarketRow | None:
    if price_data.is_empty() or "date" not in price_data.columns:
        return None
    matched = price_data.filter(pl.col("date") == fill_date)
    if matched.is_empty():
        return None
    row = matched.row(0, named=True)
    if not isinstance(row, dict):
        return None
    return {
        "open": row.get("open") if isinstance(row.get("open"), (int, float)) else None,
        "nav": row.get("nav") if isinstance(row.get("nav"), (int, float)) else None,
        "is_suspended": bool(row.get("is_suspended", False)),
        "limit_up": bool(row.get("limit_up", False)),
        "limit_down": bool(row.get("limit_down", False)),
    }
