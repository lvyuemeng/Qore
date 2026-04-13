from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import polars as pl
from qore_core.config import QoreConfig
from qore_core.instrument import FundInstrument, StockInstrument


class EastMoneyFetcher:
    _PREFIX = {"SH": "1", "SZ": "0", "BJ": "0"}
    _KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    _FUND_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
    _CONSTITUENT_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    _FINANCIAL_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    _ANALYST_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    _ANNOUNCE_URL = "https://np-anotice.eastmoney.com/anlist/gglist.aspx"
    _FINANCIAL_PAGE_SIZE = 500
    _FUND_HOLDINGS_PAGE_SIZE = 500
    _FINANCIAL_REPORTS = {
        "balance": "RPT_DMSK_FN_BALANCE",
        "income": "RPT_DMSK_FN_INCOME",
        "cashflow": "RPT_DMSK_FN_CASHFLOW",
    }
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
    }

    def __init__(self, concurrency: int, delay_min: float, delay_max: float) -> None:
        self._sem = asyncio.Semaphore(concurrency)
        self._delay = (delay_min, delay_max)
        self._client = httpx.AsyncClient(
            http2=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://finance.eastmoney.com/",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
            timeout=15.0,
        )

    @classmethod
    def from_config(cls, config: QoreConfig) -> EastMoneyFetcher:
        return cls(
            concurrency=config.data.eastmoney_concurrency,
            delay_min=config.data.eastmoney_delay_min,
            delay_max=config.data.eastmoney_delay_max,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def stock_daily(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        payload = await self._request_json(
            self._KLINE_URL,
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
        return pl.DataFrame(rows).select(self._empty_frame("stock_daily").columns)

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
            "total_assets": self._to_float(balance_row.get("TOTAL_ASSETS")),
            "operating_cashflow": self._to_float(cashflow_row.get("NETCASH_OPERATE")),
        }
        selected = [
            column
            for column in ["report_date", "announce_date", "symbol", *fields]
            if column in row
        ]
        return pl.DataFrame([row]).select(selected)

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
        payload = await self._request_json(self._CONSTITUENT_URL, params=params)
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

    async def fund_nav(
        self,
        inst: FundInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated_json(
            self._FUND_NAV_URL,
            build_params=lambda page: self._fund_nav_params(
                inst.symbol,
                start,
                end,
                page_index=page,
            ),
            total_count=lambda payload: int(payload.get("TotalCount") or 0),
            page_size=200,
            referer=f"https://fundf10.eastmoney.com/jjjz_{inst.symbol}.html",
            extra_headers={"Host": "api.fund.eastmoney.com"},
        )

        rows: list[dict[str, Any]] = []
        for page in pages:
            records = ((page.get("Data") or {}).get("LSJZList")) or []
            for item in records:
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
        if not rows:
            return self._empty_frame("fund_nav")
        return pl.DataFrame(rows).sort("date")

    async def fund_holdings(
        self,
        inst: FundInstrument,
        report_date: date,
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated_json(
            self._FINANCIAL_URL,
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
        for page in pages:
            records = ((page.get("result") or {}).get("data")) or []
            for item in records:
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
                        "total_share_ratio": self._to_float(
                            item.get("TOTAL_SHARES_RATIO")
                        ),
                        "float_share_ratio": self._to_float(
                            item.get("FREE_SHARES_RATIO")
                        ),
                    }
                )
        if not rows:
            return self._empty_frame("fund_holdings")
        return pl.DataFrame(rows).select(self._empty_frame("fund_holdings").columns)

    async def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        referer: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if referer is not None:
            headers["Referer"] = referer
        if extra_headers is not None:
            headers.update(extra_headers)
        async with self._sem:
            await asyncio.sleep(random.uniform(*self._delay))
            response = await self._client.get(
                url, params=params, headers=headers or None
            )
        response.raise_for_status()
        return response.json()

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
        payload = await self._request_json(self._FINANCIAL_URL, params=params)
        records = ((payload.get("result") or {}).get("data")) or []
        filtered: list[dict[str, Any]] = []
        for record in records:
            trade_date = self._parse_date(record.get("TRADE_DATE"))
            if trade_date is not None and trade_date <= as_of:
                filtered.append(record)
        return filtered[0] if filtered else {}

    async def _fetch_financial_rows(
        self,
        inst: StockInstrument,
        report_date: date,
    ) -> dict[str, dict[str, Any]]:
        return {
            name: await self._fetch_financial_report_row(
                inst=inst,
                report_name=report_name,
                report_date=report_date,
            )
            for name, report_name in self._FINANCIAL_REPORTS.items()
        }

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
            self._FINANCIAL_URL,
            build_params=lambda page: {**params, "pageNumber": str(page)},
            total_count=lambda payload: int(
                ((payload.get("result") or {}).get("pages")) or 0
            ),
            page_size=1,
        )
        for payload in pages:
            records = ((payload.get("result") or {}).get("data")) or []
            matched = self._match_security_code(records, inst.symbol)
            if matched:
                return matched
        return {}

    async def _fetch_paginated_json(
        self,
        url: str,
        *,
        build_params: Callable[[int], dict[str, Any]],
        total_count: Callable[[dict[str, Any]], int],
        page_size: int,
        referer: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        first_page = await self._request_json(
            url,
            params=build_params(1),
            referer=referer,
            extra_headers=extra_headers,
        )
        total_pages = max((total_count(first_page) + page_size - 1) // page_size, 1)
        pages = [first_page]
        for page_index in range(2, total_pages + 1):
            pages.append(
                await self._request_json(
                    url,
                    params=build_params(page_index),
                    referer=referer,
                    extra_headers=extra_headers,
                )
            )
        return pages

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

    def _exchange_from_market_code(self, code: str) -> str:
        if code == "0":
            return "SZ"
        if code == "1":
            return "SH"
        return "BJ"

    def _symbol_digits(self, symbol: str) -> str:
        return symbol.split(".", maxsplit=1)[0]

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

    def _empty_fundamentals_frame(self, fields: list[str]) -> pl.DataFrame:
        schema = {"report_date": pl.Date, "announce_date": pl.Date, "symbol": pl.String}
        for field in fields:
            schema[field] = pl.Float64
        return pl.DataFrame(schema=schema)
