"""Base shared utilities for EastMoney fetchers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import polars as pl

from qore_data import DataSettings
from qore_data.fetcher.http import (
    AsyncClosable,
    HardenedJsonFetcher,
    HeaderProfile,
    JsonFetcher,
    RequestHardening,
    RequestPolicy,
    RequestSpec,
)


class BlockedError(RuntimeError):
    pass


class ResponseGuard:
    def __init__(
        self,
        anti_crawl_status_codes: frozenset[int],
        anti_crawl_markers: tuple[str, ...],
    ) -> None:
        self._anti_crawl_status_codes = anti_crawl_status_codes
        self._anti_crawl_markers = anti_crawl_markers

    def should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, BlockedError):
            return True
        if isinstance(exc, RuntimeError):
            return False
        if isinstance(exc, httpx.TimeoutException | httpx.NetworkError | ValueError):
            return True
        if isinstance(exc, httpx.ProtocolError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        return False

    def is_blocked_response(self, response: httpx.Response) -> bool:
        if response.status_code in self._anti_crawl_status_codes:
            return True
        lowered = response.text.lower()
        return any(marker in lowered for marker in self._anti_crawl_markers)

    def blocked_error(self, endpoint: str) -> BaseException:
        return BlockedError(f"EastMoney anti-crawling response for {endpoint}")


HEADER_PROFILES = (
    HeaderProfile(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
        ),
        accept_language="zh-CN,zh;q=0.9,en;q=0.6",
        cache_control="no-cache",
        pragma="no-cache",
    ),
    HeaderProfile(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36"
        ),
        accept_language="zh-CN,zh;q=0.9,en-US;q=0.7,en;q=0.5",
        cache_control="max-age=0",
        pragma="no-cache",
    ),
    HeaderProfile(
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
        ),
        accept_language="zh-CN,zh;q=0.85,en;q=0.6",
        cache_control="no-store",
        pragma="no-cache",
    ),
)

ANTI_CRAWL_STATUS_CODES = frozenset({403, 412, 429})
ANTI_CRAWL_MARKERS = (
    "访问过于频繁",
    "访问受限",
    "请求过于频繁",
    "请稍后再试",
    "captcha",
    "forbidden",
    "deny",
)

# -- EastMoney API URLs -----------------------------------------------------

_FINANCIAL_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_PUSH2HIS_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
_CAPITAL_FLOW_URL = "https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get"
_ANNOUNCE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
_FUND_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
_FUNDZTAPI_URL = (
    "https://fundztapi.eastmoney.com/FundSpecialApiNew/FundSpecialZSB30ZSCFG"
)
_CSINDEX_URL_TEMPLATE = (
    "https://oss-ch.csindex.com.cn/static/"
    "html/csindex/public/uploads/file/autofile/cons/{symbol}cons.xls"
)

# -- EastMoney API tokens ---------------------------------------------------
_UT_KLINE = "7eea3edcaed734bea9cbfc24409ed989"
_UT_CAPITAL_FLOW = "b2884a393a59ad64002292a3e90d46a5"
_UT_CLIST = "bd1d9ddb04089700cf9c27f6f7426281"

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


class BaseJsonFetcher:
    def __init__(self, json_fetcher: JsonFetcher) -> None:
        self._json_fetcher = json_fetcher

    @classmethod
    def from_settings(cls, settings: DataSettings) -> BaseJsonFetcher:
        return cls(build_json_fetcher(settings))

    async def close(self) -> None:
        if isinstance(self._json_fetcher, AsyncClosable):
            await self._json_fetcher.close()

    async def _fetch_paginated(
        self,
        *,
        endpoint: str,
        url: str,
        build_params,
        total_count,
        page_size: int,
        max_pages: int | None = None,
        referer: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        first = await self._json_fetcher.fetch_json(
            RequestSpec(
                endpoint=endpoint,
                url=url,
                params=build_params(1),
                referer=referer,
                headers=headers,
            )
        )
        total = max((total_count(first) + page_size - 1) // page_size, 1)
        if max_pages is not None:
            total = min(total, max_pages)
        pages: list[dict[str, Any]] = [first]
        for i in range(2, total + 1):
            await asyncio.sleep(random.uniform(0.1, 0.3))
            try:
                page = await self._json_fetcher.fetch_json(
                    RequestSpec(
                        endpoint=endpoint,
                        url=url,
                        params=build_params(i),
                        referer=referer,
                        headers=headers,
                    )
                )
                pages.append(page)
            except Exception:
                continue
        return pages


def build_json_fetcher(settings: DataSettings) -> HardenedJsonFetcher:
    policy = RequestPolicy(
        delay_min=settings.delay_min,
        delay_max=settings.delay_max,
        max_retries=settings.max_retries,
        retry_budget=settings.retry_budget,
        retry_backoff_min=settings.retry_backoff_min,
        retry_backoff_max=settings.retry_backoff_max,
    )
    hardening = RequestHardening(
        header_profiles=HEADER_PROFILES,
        cooldown_min=settings.cooldown_min,
        cooldown_max=settings.cooldown_max,
    )
    client = httpx.AsyncClient(
        http2=False,
        headers={
            "Referer": "https://data.eastmoney.com/",
            "Connection": "close",
        },
        timeout=settings.timeout,
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
    )
    return HardenedJsonFetcher(
        client=client,
        semaphore=asyncio.Semaphore(settings.concurrency),
        policy=policy,
        hardening=hardening,
        guard=ResponseGuard(
            anti_crawl_status_codes=ANTI_CRAWL_STATUS_CODES,
            anti_crawl_markers=ANTI_CRAWL_MARKERS,
        ),
    )


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


# -- BaoStock session -------------------------------------------------------


class _BaoStockSessionError(RuntimeError):
    pass


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress stdout for the duration of the block (BaoStock SDK prints to stdout)."""
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
    """Context manager for BaoStock login/logout in single-sync paths.

    Not used in ProcessPoolExecutor workers (each worker self-manages).
    """

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
    """Execute `fn` inside a BaoStock session. For single-sync helper."""
    with _BaoStockSession():
        return fn()


# -- BaoStock shared helpers -------------------------------------------------


def _board_from_code(code: str) -> str:
    if code.startswith("688"):
        return "STAR"
    if code.startswith("300"):
        return "ChiNext"
    if code.startswith(("8", "4")):
        return "Beijing"
    return "MainBoard"


# (BaoStock sync helpers moved to qore_data.fetcher.baostock)


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
    name: str,
    rows: list[dict[str, Any]],
    *,
    sort_by: str | None = None,
) -> pl.DataFrame:
    if not rows:
        return _empty_frame(name)
    frame = _frame_from_records(rows, columns=tuple(EMPTY_SCHEMA[name]))
    return frame.sort(sort_by) if sort_by is not None else frame


def _frame_from_records(
    rows: list[dict[str, Any]],
    *,
    columns: list[str] | tuple[str, ...],
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
