"""Base shared utilities for all fetchers — helpers, BaoStock session, schemas."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl

EMPTY_SCHEMA: dict[str, dict[str, Any]] = {
    "stock_daily": {
        "date": pl.Date,
        "symbol": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "amount": pl.Float64,
        "adj_factor": pl.Float64,
        "is_suspended": pl.Boolean,
        "limit_up": pl.Boolean,
        "limit_down": pl.Boolean,
    },
    "capital_flow": {
        "date": pl.Date,
        "symbol": pl.String,
        "main_net": pl.Float64,
        "small_net": pl.Float64,
        "mid_net": pl.Float64,
        "large_net": pl.Float64,
        "xlarge_net": pl.Float64,
    },
    "fund_nav": {
        "date": pl.Date,
        "symbol": pl.String,
        "nav": pl.Float64,
        "acc_nav": pl.Float64,
        "daily_return": pl.Float64,
    },
    "fund_holdings": {
        "report_date": pl.Date,
        "symbol": pl.String,
        "stock_symbol": pl.String,
        "stock_name": pl.String,
        "shares": pl.Float64,
        "market_value": pl.Float64,
        "total_share_ratio": pl.Float64,
        "float_share_ratio": pl.Float64,
    },
    "analyst_forecast": {
        "as_of": pl.Date,
        "symbol": pl.String,
        "report_count": pl.Int64,
        "buy": pl.Int64,
        "overweight": pl.Int64,
        "neutral": pl.Int64,
        "underweight": pl.Int64,
        "sell": pl.Int64,
        "eps_year1": pl.Float64,
        "eps_year2": pl.Float64,
        "eps_year3": pl.Float64,
        "eps_year4": pl.Float64,
    },
    "announcements": {
        "symbol": pl.String,
        "short_name": pl.String,
        "title": pl.String,
        "notice_type": pl.String,
        "notice_date": pl.Date,
        "art_code": pl.String,
        "url": pl.String,
    },
    "audit_opinions": {
        "symbol": pl.String,
        "report_date": pl.Date,
        "announce_date": pl.Date,
        "opinion": pl.String,
        "opinion_code": pl.String,
        "source_notice_type": pl.String,
        "title": pl.String,
        "art_code": pl.String,
        "url": pl.String,
    },
    "stock_profile": {
        "symbol": pl.String,
        "short_name": pl.String,
        "exchange": pl.String,
        "industry": pl.String,
        "board": pl.String,
        "listing_date": pl.Date,
    },
    "fundamentals": {
        "report_date": pl.Date,
        "announce_date": pl.Date,
        "symbol": pl.String,
        "pe_ttm": pl.Float64,
        "pb": pl.Float64,
        "ps_ttm": pl.Float64,
        "roe": pl.Float64,
        "roa": pl.Float64,
        "gross_margin": pl.Float64,
        "net_margin": pl.Float64,
        "eps_ttm": pl.Float64,
        "revenue": pl.Float64,
        "net_income": pl.Float64,
        "total_shares": pl.Float64,
        "float_shares": pl.Float64,
        "equity_yoy": pl.Float64,
        "total_asset_yoy": pl.Float64,
        "net_profit_yoy": pl.Float64,
        "eps_basic_yoy": pl.Float64,
        "net_profit_parent_yoy": pl.Float64,
        "current_ratio": pl.Float64,
        "quick_ratio": pl.Float64,
        "cash_ratio": pl.Float64,
        "total_debt_yoy": pl.Float64,
        "debts_to_assets": pl.Float64,
        "assets_to_equity": pl.Float64,
        "current_assets_to_total_asset": pl.Float64,
        "non_current_assets_to_total_asset": pl.Float64,
        "tangible_assets_to_total_asset": pl.Float64,
        "ebit_to_interest": pl.Float64,
        "cfo_to_revenue": pl.Float64,
        "cfo_to_net_profit": pl.Float64,
        "receivable_turnover": pl.Float64,
        "receivable_turnover_days": pl.Float64,
        "inventory_turnover": pl.Float64,
        "inventory_turnover_days": pl.Float64,
        "current_assets_turnover": pl.Float64,
        "total_asset_turnover": pl.Float64,
        "parent_profit_ratio": pl.Float64,
        "tax_burden": pl.Float64,
        "interest_burden": pl.Float64,
        "ebit_margin": pl.Float64,
        "total_liabilities": pl.Float64,
        "total_assets": pl.Float64,
        "operating_cashflow": pl.Float64,
        "total_market_cap": pl.Float64,
        "float_market_cap": pl.Float64,
        "is_st": pl.Boolean,
    },
    "index_constituents": {
        "index_symbol": pl.String,
        "symbol": pl.String,
        "short_name": pl.String,
        "exchange": pl.String,
        "industry": pl.String,
        "weight": pl.Float64,
    },
}

_CSINDEX_URL_TEMPLATE = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{symbol}cons.xls"


# -- BaoStock session ---------------------------------------------------------


class _BaoStockSessionError(RuntimeError):
    pass


@contextlib.contextmanager
def _suppress_stdout():
    old_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)


class _BaoStockSession:
    def __enter__(self):
        import baostock as bs

        with _suppress_stdout():
            lg = bs.login()
        if lg.error_code != "0":
            raise _BaoStockSessionError(f"BaoStock login failed: {lg.error_msg}")
        return self

    def __exit__(self, *args):
        import baostock as bs

        with _suppress_stdout():
            bs.logout()
        return False


def _run_bao[T](fn: Callable[[], T]) -> T:
    with _BaoStockSession():
        return fn()


def _board_from_code(code: str) -> str:
    if code.startswith("688"):
        return "STAR"
    if code.startswith("300"):
        return "ChiNext"
    if code.startswith(("8", "4")):
        return "Beijing"
    return "MainBoard"


# -- helpers ------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        s = str(value).replace(",", "").strip()
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        s = str(value).replace(",", "").strip()
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> date | None:
    if value in (None, "", "-"):
        return None
    return date.fromisoformat(str(value)[:10])


def _parse_compact_date(value: Any) -> date | None:
    if value in (None, "", "-"):
        return None
    digits = str(value)
    if len(digits) != 8 or not digits.isdigit():
        return None
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


def _symbol_digits(symbol: str) -> str:
    return symbol.split(".", maxsplit=1)[0]


def _stock_secid(symbol: str) -> str:
    prefix_map = {"SH": "1", "SZ": "0", "BJ": "0"}
    code = _symbol_digits(symbol)
    exchange = _exchange_from_stock_code(code)
    return f"{prefix_map[exchange]}.{code}"


def _secid_referer(symbol: str, base: str) -> str:
    code = _symbol_digits(symbol)
    exchange = _exchange_from_stock_code(code)
    return f"{base}{exchange.lower()}{code}.html"


def _records(payload: dict[str, Any], *path: str) -> list[dict[str, Any]]:
    current: Any = payload
    for part in path:
        current = current.get(part) if isinstance(current, dict) else None
    if not isinstance(current, list):
        return []
    return [row for row in current if isinstance(row, dict)]


def _page_records(pages: list[dict[str, Any]], *path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        rows.extend(_records(page, *path))
    return rows


def _empty_frame(name: str) -> pl.DataFrame:
    return pl.DataFrame(schema=EMPTY_SCHEMA[name])


def _frame_from_rows(
    name: str, rows: list[dict[str, Any]], *, sort_by: str | None = None
) -> pl.DataFrame:
    if not rows:
        return _empty_frame(name)
    frame = _frame_from_records(rows, columns=tuple(EMPTY_SCHEMA[name]))
    return frame.sort(sort_by) if sort_by is not None else frame


def _frame_from_records(
    rows: list[dict[str, Any]], *, columns: list[str] | tuple[str, ...]
) -> pl.DataFrame:
    return pl.DataFrame(rows).select(list(columns))


def _exchange_from_stock_code(code: str) -> str:
    c = code.zfill(6)
    if c.startswith(("600", "601", "603", "605", "688", "900")):
        return "SH"
    if c.startswith(("430", "83", "87", "88")):
        return "BJ"
    if c.startswith(("60", "68", "51", "11", "90")):
        return "SH"
    if c.startswith(("8", "4", "920")):
        return "BJ"
    return "SZ"


def _extract_code(record: dict[str, Any], *, key: str = "SECURITY_CODE") -> str:
    return str(record.get(key, "")).zfill(6)
