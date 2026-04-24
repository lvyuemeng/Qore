from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import date
from typing import Any, Literal, cast

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
    RequestTelemetry,
    ResponseGuard,
    TelemetryReadable,
)
from qore_data.instrument import FundInstrument, StockInstrument


class EastMoneyBlockedError(RuntimeError):
    pass


class EastMoneyResponseGuard(ResponseGuard):
    def __init__(
        self,
        anti_crawl_status_codes: frozenset[int],
        anti_crawl_markers: tuple[str, ...],
    ) -> None:
        self._anti_crawl_status_codes = anti_crawl_status_codes
        self._anti_crawl_markers = anti_crawl_markers

    def should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, EastMoneyBlockedError):
            return True
        if isinstance(exc, RuntimeError):
            return False
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, ValueError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status_error = cast(httpx.HTTPStatusError, exc)
            return status_error.response.status_code in {
                408,
                409,
                425,
                429,
                500,
                502,
                503,
                504,
            }
        return False

    def is_blocked_response(self, response: httpx.Response) -> bool:
        if response.status_code in self._anti_crawl_status_codes:
            return True
        lowered = response.text.lower()
        return any(marker in lowered for marker in self._anti_crawl_markers)

    def blocked_error(self, endpoint: str) -> BaseException:
        return EastMoneyBlockedError(f"EastMoney anti-crawling response for {endpoint}")


class EastMoneyFetcher:
    _PREFIX = {"SH": "1", "SZ": "0", "BJ": "0"}
    _STOCK_INFO_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    _KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    _FUND_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
    _CONSTITUENT_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    _FINANCIAL_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    _ANALYST_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    _ANNOUNCE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    _FINANCIAL_PAGE_SIZE = 500
    _FUND_HOLDINGS_PAGE_SIZE = 500
    _ANALYST_PAGE_SIZE = 500
    _FINANCIAL_REPORTS = {
        "balance": "RPT_DMSK_FN_BALANCE",
        "income": "RPT_DMSK_FN_INCOME",
        "cashflow": "RPT_DMSK_FN_CASHFLOW",
    }
    _HEADER_PROFILES = (
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
    _ANTI_CRAWL_STATUS_CODES = frozenset({403, 412, 429})
    _ANTI_CRAWL_MARKERS = (
        "访问过于频繁",
        "访问受限",
        "请求过于频繁",
        "请稍后再试",
        "captcha",
        "forbidden",
        "deny",
    )
    _EMPTY_SCHEMA = {
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
            "as_of": pl.Date,
            "symbol": pl.String,
            "short_name": pl.String,
            "exchange": pl.String,
            "industry": pl.String,
            "board": pl.String,
            "listing_date": pl.Date,
            "total_market_cap": pl.Float64,
            "float_market_cap": pl.Float64,
            "total_shares": pl.Float64,
            "float_shares": pl.Float64,
            "is_st": pl.Boolean,
        },
        "fundamentals": {
            "report_date": pl.Date,
            "announce_date": pl.Date,
            "symbol": pl.String,
            "pe_ttm": pl.Float64,
            "pb": pl.Float64,
            "ps_ttm": pl.Float64,
            "ev_ebitda": pl.Float64,
            "roe": pl.Float64,
            "roa": pl.Float64,
            "gross_margin": pl.Float64,
            "revenue": pl.Float64,
            "net_income": pl.Float64,
            "total_liabilities": pl.Float64,
            "total_assets": pl.Float64,
            "operating_cashflow": pl.Float64,
        },
    }

    def __init__(self, json_fetcher: JsonFetcher) -> None:
        self._json_fetcher = json_fetcher

    @classmethod
    def from_settings(cls, settings: DataSettings) -> EastMoneyFetcher:
        policy = RequestPolicy(
            delay_min=settings.eastmoney_delay_min,
            delay_max=settings.eastmoney_delay_max,
            max_retries=settings.eastmoney_max_retries,
            retry_budget=settings.eastmoney_retry_budget,
            retry_backoff_min=settings.eastmoney_retry_backoff_min,
            retry_backoff_max=settings.eastmoney_retry_backoff_max,
        )
        hardening = RequestHardening(
            telemetry=RequestTelemetry(),
            header_profiles=cls._HEADER_PROFILES,
            cooldown_min=settings.eastmoney_cooldown_min,
            cooldown_max=settings.eastmoney_cooldown_max,
        )
        client = httpx.AsyncClient(
            http2=True,
            headers={"Referer": "https://finance.eastmoney.com/"},
            timeout=settings.eastmoney_timeout,
        )
        fetcher = HardenedJsonFetcher(
            client=client,
            semaphore=asyncio.Semaphore(settings.eastmoney_concurrency),
            policy=policy,
            hardening=hardening,
            guard=EastMoneyResponseGuard(
                anti_crawl_status_codes=cls._ANTI_CRAWL_STATUS_CODES,
                anti_crawl_markers=cls._ANTI_CRAWL_MARKERS,
            ),
        )
        return cls(fetcher)

    async def close(self) -> None:
        if isinstance(self._json_fetcher, AsyncClosable):
            await self._json_fetcher.close()

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]:
        if isinstance(self._json_fetcher, TelemetryReadable):
            return self._json_fetcher.telemetry_snapshot()
        return {}

    def telemetry_frame(self) -> pl.DataFrame:
        if isinstance(self._json_fetcher, TelemetryReadable):
            return self._json_fetcher.telemetry_frame()
        return RequestTelemetry().frame()

    async def stock_daily(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        payload = await self._json_request(
            RequestSpec(
                endpoint="stock_daily",
                url=self._KLINE_URL,
                params={
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
                    "ut": "7eea3edcaed734bea9cbfc24409ed989",
                    "klt": "101",
                    "fqt": "0",
                    "secid": self._stock_secid(inst),
                    "beg": start.strftime("%Y%m%d"),
                    "end": end.strftime("%Y%m%d"),
                },
                referer=f"https://quote.eastmoney.com/{inst.symbol.lower()}.html",
            )
        )
        data = payload.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return self._empty_frame("stock_daily")
        rows: list[dict[str, Any]] = []
        for item in klines:
            parts = item.split(",")
            pct_change = self._to_float(parts[8])
            rows.append(
                {
                    "date": date.fromisoformat(parts[0]),
                    "symbol": inst.symbol,
                    "open": self._to_float(parts[1]),
                    "close": self._to_float(parts[2]),
                    "high": self._to_float(parts[3]),
                    "low": self._to_float(parts[4]),
                    "volume": self._to_int(parts[5]),
                    "amount": self._to_float(parts[6]),
                    "adj_factor": 1.0,
                    "is_suspended": False,
                    "limit_up": pct_change is not None
                    and pct_change >= inst.price_limit_pct * 100.0 - 1e-6,
                    "limit_down": pct_change is not None
                    and pct_change <= -inst.price_limit_pct * 100.0 + 1e-6,
                }
            )
        return self._frame_from_rows("stock_daily", rows)

    async def fundamentals(
        self,
        inst: StockInstrument,
        fields: list[str],
        as_of: date,
    ) -> pl.DataFrame:
        value_row = await self._fetch_value_analysis_row(inst, as_of)
        report_date = self._latest_report_date(as_of)
        financial_rows = await self._fetch_financial_rows(inst, report_date)
        balance_row = financial_rows["balance"]
        income_row = financial_rows["income"]
        cashflow_row = financial_rows["cashflow"]
        announce_date = self._first_date(
            balance_row.get("NOTICE_DATE"),
            income_row.get("NOTICE_DATE"),
            cashflow_row.get("NOTICE_DATE"),
            value_row.get("TRADE_DATE"),
        )
        row = {
            "report_date": report_date,
            "announce_date": announce_date or report_date,
            "symbol": inst.symbol,
            "pe_ttm": self._to_float(value_row.get("PE_TTM")),
            "pb": self._to_float(value_row.get("PB_MRQ")),
            "ps_ttm": self._to_float(value_row.get("PS_TTM")),
            "ev_ebitda": None,
            "roe": self._to_float(balance_row.get("ROE_WEIGHTED"))
            or self._to_float(income_row.get("ROEJQ")),
            "roa": self._ratio(
                income_row.get("NETPROFIT"),
                balance_row.get("TOTAL_ASSETS"),
            ),
            "gross_margin": self._ratio(
                self._difference(
                    income_row.get("TOTAL_OPERATE_INCOME"),
                    income_row.get("OPERATE_COST"),
                ),
                income_row.get("TOTAL_OPERATE_INCOME"),
            ),
            "revenue": self._to_float(income_row.get("TOTAL_OPERATE_INCOME")),
            "net_income": self._to_float(income_row.get("NETPROFIT")),
            "total_liabilities": self._to_float(balance_row.get("TOTAL_LIABILITIES")),
            "total_assets": self._to_float(balance_row.get("TOTAL_ASSETS")),
            "operating_cashflow": self._to_float(cashflow_row.get("NETCASH_OPERATE")),
        }
        selected = [
            column
            for column in ["report_date", "announce_date", "symbol", *fields]
            if column in row
        ]
        return self._frame_from_records([row], columns=selected)

    async def index_constituents(
        self,
        index_symbol: str,
        as_of: date,
    ) -> list[StockInstrument]:
        del as_of
        prefix, symbol = self._split_symbol(index_symbol)
        params = {
            "pn": "1",
            "pz": "500",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": f"i:{prefix}.{symbol}",
            "fields": "f12,f14,f100,f13",
        }
        payload = await self._json_request(
            RequestSpec(
                endpoint="index_constituents", url=self._CONSTITUENT_URL, params=params
            )
        )
        diff = ((payload.get("data") or {}).get("diff")) or []
        instruments: list[StockInstrument] = []
        for row in diff:
            exchange = self._exchange_from_market_code(str(row.get("f13", "1")))
            code = str(row.get("f12", "")).zfill(6)
            instruments.append(
                StockInstrument(
                    symbol=f"{code}.{exchange}",
                    exchange=exchange,
                    industry=str(row.get("f100") or "unknown"),
                )
            )
        return instruments

    async def stock_profile(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame:
        payload = await self._json_request(
            RequestSpec(
                endpoint="stock_profile",
                url=self._STOCK_INFO_URL,
                params=self._stock_profile_params(inst),
                referer=(
                    "https://quote.eastmoney.com/concept/"
                    f"{inst.exchange.lower()}{self._symbol_digits(inst.symbol)}.html?from=classic"
                ),
            )
        )
        data = payload.get("data") or {}
        if not data:
            return self._empty_frame("stock_profile")
        short_name = str(data.get("f58") or "")
        row = {
            "as_of": as_of,
            "symbol": inst.symbol,
            "short_name": short_name,
            "exchange": inst.exchange,
            "industry": str(data.get("f127") or inst.industry),
            "board": self._board_from_symbol(inst.symbol),
            "listing_date": self._parse_compact_date(data.get("f189")),
            "total_market_cap": self._to_float(data.get("f116")),
            "float_market_cap": self._to_float(data.get("f117")),
            "total_shares": self._to_float(data.get("f84")),
            "float_shares": self._to_float(data.get("f85")),
            "is_st": short_name.startswith(("ST", "*ST")),
        }
        return self._frame_from_rows("stock_profile", [row])

    async def fund_nav(
        self,
        inst: FundInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated_json(
            endpoint="fund_nav",
            url=self._FUND_NAV_URL,
            build_params=lambda page: self._fund_nav_params(
                inst.symbol,
                start,
                end,
                page_index=page,
            ),
            total_count=lambda payload: int(payload.get("TotalCount") or 0),
            page_size=200,
            referer=f"https://fundf10.eastmoney.com/jjjz_{inst.symbol}.html",
            headers={"Host": "api.fund.eastmoney.com"},
        )

        rows: list[dict[str, Any]] = []
        for item in self._page_records(pages, "Data", "LSJZList"):
            nav_date = date.fromisoformat(str(item.get("FSRQ")))
            if nav_date < start or nav_date > end:
                continue
            daily_return = self._to_float(item.get("JZZZL"))
            rows.append(
                {
                    "date": nav_date,
                    "symbol": inst.symbol,
                    "nav": self._to_float(item.get("DWJZ")),
                    "acc_nav": self._to_float(item.get("LJJZ")),
                    "daily_return": None
                    if daily_return is None
                    else daily_return / 100.0,
                }
            )
        return self._frame_from_rows("fund_nav", rows, sort_by="date")

    async def fund_holdings(
        self,
        inst: FundInstrument,
        report_date: date,
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated_json(
            endpoint="fund_holdings",
            url=self._FINANCIAL_URL,
            build_params=lambda page: self._fund_holdings_params(
                inst.symbol,
                report_date,
                page_index=page,
            ),
            total_count=lambda payload: int(
                ((payload.get("result") or {}).get("pages")) or 0
            ),
            page_size=1,
            referer=(
                "https://data.eastmoney.com/zlsj/ccjj/"
                f"{report_date.isoformat()}-{inst.symbol}.html"
            ),
        )
        rows: list[dict[str, Any]] = []
        for item in self._page_records(pages, "result", "data"):
            code = str(item.get("SECURITY_CODE") or "").strip()
            if not code:
                continue
            rows.append(
                {
                    "report_date": report_date,
                    "symbol": inst.symbol,
                    "stock_symbol": code,
                    "stock_name": str(item.get("SECURITY_NAME_ABBR") or ""),
                    "shares": self._to_float(item.get("HOLD_SHARES")),
                    "market_value": self._to_float(item.get("HOLD_MV")),
                    "total_share_ratio": self._to_float(item.get("TOTAL_SHARES_RATIO")),
                    "float_share_ratio": self._to_float(item.get("FREE_SHARES_RATIO")),
                }
            )
        return self._frame_from_rows("fund_holdings", rows)

    async def analyst_forecast(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame:
        payload = await self._json_request(
            RequestSpec(
                endpoint="analyst_forecast",
                url=self._ANALYST_URL,
                params=self._analyst_forecast_params(inst),
                referer="https://data.eastmoney.com/report/profitforecast.jshtml",
            )
        )
        records = ((payload.get("result") or {}).get("data")) or []
        matched = self._match_security_code(records, inst.symbol)
        if not matched:
            return self._empty_frame("analyst_forecast")

        year_columns = [self._to_float(matched.get(f"EPS{i}")) for i in range(1, 5)]
        row = {
            "as_of": as_of,
            "symbol": inst.symbol,
            "report_count": self._to_int(matched.get("RATING_ORG_NUM")),
            "buy": self._to_int(matched.get("BUY")),
            "overweight": self._to_int(matched.get("HOLD")),
            "neutral": self._to_int(matched.get("NEUTRAL")),
            "underweight": self._to_int(matched.get("SELL")),
            "sell": self._to_int(matched.get("STRONG_SELL")),
            "eps_year1": year_columns[0],
            "eps_year2": year_columns[1],
            "eps_year3": year_columns[2],
            "eps_year4": year_columns[3],
        }
        return self._frame_from_rows("analyst_forecast", [row])

    async def announcements(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated_json(
            endpoint="announcements",
            url=self._ANNOUNCE_URL,
            build_params=lambda page: self._announcement_params(
                inst, start, end, page_index=page
            ),
            total_count=lambda payload: int(
                ((payload.get("data") or {}).get("total_hits")) or 0
            ),
            page_size=100,
            referer=f"https://data.eastmoney.com/notices/stock/{self._symbol_digits(inst.symbol)}.html",
        )
        rows: list[dict[str, Any]] = []
        for item in self._page_records(pages, "data", "list"):
            codes = item.get("codes") or []
            matched = next(
                (
                    code
                    for code in codes
                    if str(code.get("stock_code", "")).zfill(6)
                    == self._symbol_digits(inst.symbol)
                ),
                None,
            )
            if matched is None:
                continue
            notice_date = self._parse_date(item.get("notice_date"))
            if notice_date is None or notice_date < start or notice_date > end:
                continue
            columns = item.get("columns") or []
            first_column = columns[0] if columns else {}
            art_code = str(item.get("art_code") or "")
            rows.append(
                {
                    "symbol": inst.symbol,
                    "short_name": str(matched.get("short_name") or ""),
                    "title": str(item.get("title") or ""),
                    "notice_type": str(first_column.get("column_name") or ""),
                    "notice_date": notice_date,
                    "art_code": art_code,
                    "url": f"https://data.eastmoney.com/notices/detail/{self._symbol_digits(inst.symbol)}/{art_code}.html",
                }
            )
        return self._frame_from_rows("announcements", rows)

    async def audit_opinions(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated_json(
            endpoint="audit_opinions",
            url=self._ANNOUNCE_URL,
            build_params=lambda page: self._announcement_params(
                inst, start, end, page_index=page
            ),
            total_count=lambda payload: int(
                ((payload.get("data") or {}).get("total_hits")) or 0
            ),
            page_size=100,
            referer=f"https://data.eastmoney.com/notices/stock/{self._symbol_digits(inst.symbol)}.html",
        )
        rows: list[dict[str, Any]] = []
        for item in self._page_records(pages, "data", "list"):
            codes = item.get("codes") or []
            matched = next(
                (
                    code
                    for code in codes
                    if str(code.get("stock_code", "")).zfill(6)
                    == self._symbol_digits(inst.symbol)
                ),
                None,
            )
            if matched is None:
                continue
            notice_date = self._parse_date(item.get("notice_date"))
            if notice_date is None or notice_date < start or notice_date > end:
                continue
            columns = item.get("columns") or []
            first_column = columns[0] if columns else {}
            title = str(item.get("title") or "")
            opinion = self._audit_opinion_from_title(title)
            if opinion is None:
                continue
            art_code = str(item.get("art_code") or "")
            rows.append(
                {
                    "symbol": inst.symbol,
                    "report_date": self._audit_report_date(title, notice_date),
                    "announce_date": notice_date,
                    "opinion": opinion[0],
                    "opinion_code": opinion[1],
                    "source_notice_type": str(first_column.get("column_name") or ""),
                    "title": title,
                    "art_code": art_code,
                    "url": f"https://data.eastmoney.com/notices/detail/{self._symbol_digits(inst.symbol)}/{art_code}.html",
                }
            )
        return self._frame_from_rows("audit_opinions", rows)

    async def _fetch_value_analysis_row(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> dict[str, Any]:
        params = {
            "sortColumns": "TRADE_DATE",
            "sortTypes": "-1",
            "pageSize": "5000",
            "pageNumber": "1",
            "reportName": "RPT_VALUEANALYSIS_DET",
            "columns": "ALL",
            "quoteColumns": "",
            "source": "WEB",
            "client": "WEB",
            "filter": f'(SECURITY_CODE="{self._symbol_digits(inst.symbol)}")',
        }
        payload = await self._json_request(
            RequestSpec(
                endpoint="value_analysis", url=self._FINANCIAL_URL, params=params
            )
        )
        for record in self._records(payload, "result", "data"):
            trade_date = self._parse_date(record.get("TRADE_DATE"))
            if trade_date is not None and trade_date <= as_of:
                return record
        return {}

    async def _fetch_financial_rows(
        self,
        inst: StockInstrument,
        report_date: date,
    ) -> dict[str, dict[str, Any]]:
        rows = await asyncio.gather(
            *(
                self._fetch_financial_report_row(
                    inst=inst,
                    report_name=report_name,
                    report_date=report_date,
                )
                for report_name in self._FINANCIAL_REPORTS.values()
            )
        )
        return dict(zip(self._FINANCIAL_REPORTS, rows, strict=True))

    async def _fetch_financial_report_row(
        self,
        *,
        inst: StockInstrument,
        report_name: str,
        report_date: date,
    ) -> dict[str, Any]:
        params = {
            "sortColumns": "NOTICE_DATE,SECURITY_CODE",
            "sortTypes": "-1,-1",
            "pageSize": str(self._FINANCIAL_PAGE_SIZE),
            "pageNumber": "1",
            "reportName": report_name,
            "columns": "ALL",
            "filter": self._financial_filter(inst, report_date),
        }
        pages = await self._fetch_paginated_json(
            endpoint=f"financial_report:{report_name}",
            url=self._FINANCIAL_URL,
            build_params=lambda page: {**params, "pageNumber": str(page)},
            total_count=lambda payload: int(
                ((payload.get("result") or {}).get("pages")) or 0
            ),
            page_size=1,
        )
        for payload in pages:
            records = self._records(payload, "result", "data")
            matched = self._match_security_code(records, inst.symbol)
            if matched:
                return matched
        return {}

    @staticmethod
    def _should_retry_request(exc: BaseException) -> bool:
        if isinstance(exc, EastMoneyBlockedError):
            return True
        if isinstance(exc, RuntimeError):
            return False
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, ValueError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status_error = cast(httpx.HTTPStatusError, exc)
            return status_error.response.status_code in {
                408,
                409,
                425,
                429,
                500,
                502,
                503,
                504,
            }
        return False

    async def _json_request(self, spec: RequestSpec) -> dict[str, Any]:
        return await self._json_fetcher.fetch_json(spec)

    async def _fetch_paginated_json(
        self,
        *,
        endpoint: str,
        url: str,
        build_params: Callable[[int], dict[str, Any]],
        total_count: Callable[[dict[str, Any]], int],
        page_size: int,
        referer: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        first_page = await self._json_request(
            RequestSpec(
                endpoint=endpoint,
                url=url,
                params=build_params(1),
                referer=referer,
                headers=headers,
            )
        )
        total_pages = max((total_count(first_page) + page_size - 1) // page_size, 1)
        if total_pages == 1:
            return [first_page]
        remaining_pages = await asyncio.gather(
            *(
                self._json_request(
                    RequestSpec(
                        endpoint=endpoint,
                        url=url,
                        params=build_params(page_index),
                        referer=referer,
                        headers=headers,
                    )
                )
                for page_index in range(2, total_pages + 1)
            )
        )
        return [first_page, *remaining_pages]

    def _is_anti_crawl_response(self, response: httpx.Response) -> bool:
        if response.status_code in self._ANTI_CRAWL_STATUS_CODES:
            return True
        lowered = response.text.lower()
        return any(marker in lowered for marker in self._ANTI_CRAWL_MARKERS)

    def _fund_nav_params(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        page_index: int,
    ) -> dict[str, Any]:
        return {
            "fundCode": symbol,
            "pageIndex": str(page_index),
            "pageSize": "200",
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "_": "0",
        }

    def _stock_profile_params(self, inst: StockInstrument) -> dict[str, Any]:
        return {
            "fltt": "2",
            "invt": "2",
            "fields": "f57,f58,f84,f85,f116,f117,f127,f189",
            "secid": self._stock_secid(inst),
        }

    def _fund_holdings_params(
        self,
        symbol: str,
        report_date: date,
        *,
        page_index: int,
    ) -> dict[str, Any]:
        return {
            "sortColumns": "SECURITY_CODE",
            "sortTypes": "-1",
            "pageSize": str(self._FUND_HOLDINGS_PAGE_SIZE),
            "pageNumber": str(page_index),
            "reportName": "RPT_MAINDATA_MAIN_POSITIONDETAILS",
            "columns": "ALL",
            "quoteColumns": "",
            "source": "WEB",
            "client": "WEB",
            "filter": (
                f"(HOLDER_CODE=\"{symbol}\")(REPORT_DATE='{report_date.isoformat()}')"
            ),
        }

    def _financial_filter(self, inst: StockInstrument, report_date: date) -> str:
        return (
            '(SECURITY_TYPE_CODE in ("058001001","058001008"))'
            '(TRADE_MARKET_CODE!="069001017")'
            f'(REPORT_DATE="{report_date.isoformat()}")'
        )

    def _analyst_forecast_params(self, inst: StockInstrument) -> dict[str, Any]:
        return {
            "reportName": "RPT_WEB_RESPREDICT",
            "columns": "WEB_RESPREDICT",
            "pageNumber": "1",
            "pageSize": str(self._ANALYST_PAGE_SIZE),
            "sortTypes": "-1",
            "sortColumns": "RATING_ORG_NUM",
            "p": "1",
            "pageNo": "1",
            "pageNum": "1",
            "filter": f'(SECURITY_CODE="{self._symbol_digits(inst.symbol)}")',
        }

    def _announcement_params(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
        *,
        page_index: int,
    ) -> dict[str, Any]:
        return {
            "sr": "-1",
            "page_size": "100",
            "page_index": str(page_index),
            "ann_type": "A",
            "client_source": "web",
            "f_node": "0",
            "s_node": "0",
            "stock_list": self._symbol_digits(inst.symbol),
            "begin_time": start.isoformat(),
            "end_time": end.isoformat(),
        }

    def _match_security_code(
        self,
        records: list[dict[str, Any]],
        symbol: str,
    ) -> dict[str, Any]:
        digits = self._symbol_digits(symbol)
        for record in records:
            if str(record.get("SECURITY_CODE", "")).zfill(6) == digits:
                return record
        return {}

    def _latest_report_date(self, as_of: date) -> date:
        quarter_end = ((as_of.month - 1) // 3) * 3
        if quarter_end == 0:
            return date(as_of.year - 1, 12, 31)
        if quarter_end == 3:
            return date(as_of.year, 3, 31)
        if quarter_end == 6:
            return date(as_of.year, 6, 30)
        return date(as_of.year, 9, 30)

    def _parse_date(self, value: Any) -> date | None:
        if value in (None, "", "-"):
            return None
        return date.fromisoformat(str(value)[:10])

    def _first_date(self, *values: Any) -> date | None:
        for value in values:
            parsed = self._parse_date(value)
            if parsed is not None:
                return parsed
        return None

    def _parse_compact_date(self, value: Any) -> date | None:
        if value in (None, "", "-"):
            return None
        digits = str(value)
        if len(digits) != 8 or not digits.isdigit():
            return None
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))

    def _audit_opinion_from_title(self, title: str) -> tuple[str, str] | None:
        normalized = title.replace(" ", "")
        if "无法表示意见" in normalized:
            return ("无法表示意见", "disclaimer")
        if "否定意见" in normalized:
            return ("否定意见", "adverse")
        if "保留意见" in normalized:
            return ("保留意见", "qualified")
        if "无保留意见" in normalized and "审计" in normalized:
            return ("无保留意见", "unqualified")
        return None

    def _audit_report_date(self, title: str, announce_date: date) -> date:
        matched = re.search(r"(20\d{2})年", title)
        if matched is None:
            return date(announce_date.year - 1, 12, 31)
        return date(int(matched.group(1)), 12, 31)

    def _difference(self, left: Any, right: Any) -> float | None:
        left_value = self._to_float(left)
        right_value = self._to_float(right)
        if left_value is None or right_value is None:
            return None
        return left_value - right_value

    def _ratio(self, numerator: Any, denominator: Any) -> float | None:
        numerator_value = self._to_float(numerator)
        denominator_value = self._to_float(denominator)
        if numerator_value is None or denominator_value in (None, 0.0):
            return None
        return numerator_value / denominator_value

    def _stock_secid(self, inst: StockInstrument) -> str:
        return f"{self._PREFIX[inst.exchange]}.{self._symbol_digits(inst.symbol)}"

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        if "." in symbol:
            digits, exchange = symbol.split(".", maxsplit=1)
            prefix = self._PREFIX.get(exchange.upper(), "1")
            return prefix, digits
        return "1", self._symbol_digits(symbol)

    def _exchange_from_market_code(self, code: str) -> Literal["SH", "SZ", "BJ"]:
        if code == "0":
            return "SZ"
        if code == "1":
            return "SH"
        return "BJ"

    def _symbol_digits(self, symbol: str) -> str:
        return symbol.split(".", maxsplit=1)[0]

    def _board_from_symbol(self, symbol: str) -> str:
        digits = self._symbol_digits(symbol)
        if digits.startswith("688"):
            return "STAR"
        if digits.startswith("300"):
            return "ChiNext"
        if digits.startswith(("8", "4")):
            return "Beijing"
        return "MainBoard"

    def _to_float(self, value: Any) -> float | None:
        if value in (None, "", "-"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int(self, value: Any) -> int | None:
        if value in (None, "", "-"):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _empty_frame(self, name: str) -> pl.DataFrame:
        return pl.DataFrame(schema=self._EMPTY_SCHEMA[name])

    def _frame_from_rows(
        self,
        name: str,
        rows: list[dict[str, Any]],
        *,
        sort_by: str | None = None,
    ) -> pl.DataFrame:
        if not rows:
            return self._empty_frame(name)
        frame = self._frame_from_records(rows, columns=tuple(self._EMPTY_SCHEMA[name]))
        return frame.sort(sort_by) if sort_by is not None else frame

    @staticmethod
    def _frame_from_records(
        rows: list[dict[str, Any]],
        *,
        columns: list[str] | tuple[str, ...],
    ) -> pl.DataFrame:
        return pl.DataFrame(rows).select(list(columns))

    @staticmethod
    def _records(payload: dict[str, Any], *path: str) -> list[dict[str, Any]]:
        current: Any = payload
        for part in path:
            current = current.get(part) if isinstance(current, dict) else None
        if not isinstance(current, list):
            return []
        return [row for row in current if isinstance(row, dict)]

    def _page_records(
        self, pages: list[dict[str, Any]], *path: str
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in pages:
            rows.extend(self._records(page, *path))
        return rows

    def _empty_fundamentals_frame(self, fields: list[str]) -> pl.DataFrame:
        schema: dict[str, pl.DataType | type[pl.DataType] | None] = {
            "report_date": pl.Date,
            "announce_date": pl.Date,
            "symbol": pl.String,
        }
        for column in fields:
            schema[column] = pl.Float64
        return pl.DataFrame(schema=schema)
