from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from qore_data.fetcher._base import (
    _FINANCIAL_URL,
    _FUND_NAV_URL,
    BaseJsonFetcher,
    _frame_from_rows,
    _page_records,
    _to_float,
)
from qore_data.fetcher.xueqiu import _xq_symbol, _XueqiuSession

# ── Xueqiu fund source ──────────────────────────────────────────────────


_XQ_FUND_NAV_URL = "https://stock.xueqiu.com/v5/stock/fund/nav/history.json"
_XQ_FUND_ASSET_URL = "https://stock.xueqiu.com/v5/stock/fund/nav/asset.json"


class _XueqiuFundSource:
    def __init__(self, session: _XueqiuSession) -> None:
        self._session = session

    async def fund_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        try:
            data = await self._session.get_json(
                _XQ_FUND_NAV_URL,
                {"symbol": _xq_symbol(fund_code), "count": "500", "page": "1"},
            )
        except Exception:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "symbol": pl.String,
                    "nav": pl.Float64,
                    "acc_nav": pl.Float64,
                    "daily_return": pl.Float64,
                }
            )
        items = (data.get("data") or {}).get("list") or []
        rows: list[dict[str, Any]] = []
        for item in items:
            nav_date = date.fromisoformat(str(item.get("nav_date", ""))[:10])
            if nav_date < start or nav_date > end:
                continue
            daily_return = _to_float(item.get("nav_yield"))
            rows.append(
                {
                    "date": nav_date,
                    "symbol": fund_code,
                    "nav": _to_float(item.get("unit_nav")),
                    "acc_nav": _to_float(item.get("acc_nav")),
                    "daily_return": daily_return / 100.0
                    if daily_return is not None
                    else None,
                }
            )
        return _frame_from_rows("fund_nav", rows, sort_by="date")

    async def fund_holdings(self, fund_code: str, report_date: date) -> pl.DataFrame:
        try:
            data = await self._session.get_json(
                _XQ_FUND_ASSET_URL, {"symbol": _xq_symbol(fund_code), "count": "4"}
            )
        except Exception:
            return pl.DataFrame(
                schema={
                    "report_date": pl.Date,
                    "symbol": pl.String,
                    "stock_symbol": pl.String,
                    "stock_name": pl.String,
                    "shares": pl.Float64,
                    "market_value": pl.Float64,
                    "total_share_ratio": pl.Float64,
                    "float_share_ratio": pl.Float64,
                }
            )
        items = (data.get("data") or {}).get("list") or []
        rows: list[dict[str, Any]] = []
        for item in items:
            rpt = date.fromisoformat(str(item.get("report_date", ""))[:10])
            if rpt != report_date:
                continue
            rows.append(
                {
                    "report_date": rpt,
                    "symbol": fund_code,
                    "stock_symbol": str(item.get("stock_symbol") or ""),
                    "stock_name": str(item.get("stock_name") or ""),
                    "shares": _to_float(item.get("hold_shares")),
                    "market_value": _to_float(item.get("hold_market_value")),
                    "total_share_ratio": _to_float(item.get("hold_ratio")),
                    "float_share_ratio": None,
                }
            )
        return _frame_from_rows("fund_holdings", rows)

    async def close(self) -> None:
        pass


# ── FundFetcher ─────────────────────────────────────────────────────────


_FUND_HOLDINGS_PAGE_SIZE = 500


class FundFetcher(BaseJsonFetcher):
    """Fund data. Xueqiu (primary) -> EastMoney api.fund / datacenter-web (fallback)."""

    def __init__(self, json_fetcher) -> None:
        super().__init__(json_fetcher)
        self._xueqiu_session = _XueqiuSession()
        self._xueqiu = _XueqiuFundSource(self._xueqiu_session)

    async def close(self) -> None:
        await self._xueqiu_session.close()
        await super().close()

    async def fund_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        result = await self._xueqiu.fund_nav(fund_code, start, end)
        if not result.is_empty():
            return result
        return await self._em_fund_nav(fund_code, start, end)

    async def fund_holdings(self, fund_code: str, report_date: date) -> pl.DataFrame:
        result = await self._xueqiu.fund_holdings(fund_code, report_date)
        if not result.is_empty():
            return result
        return await self._em_fund_holdings(fund_code, report_date)

    # -- EastMoney fallback --------------------------------------------------

    async def _em_fund_nav(
        self, fund_code: str, start: date, end: date
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated(
            endpoint="fund_nav",
            url=_FUND_NAV_URL,
            build_params=lambda page: {
                "fundCode": fund_code,
                "pageIndex": str(page),
                "pageSize": "200",
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "_": "0",
            },
            total_count=lambda payload: int(payload.get("TotalCount") or 0),
            page_size=200,
            referer=f"https://fundf10.eastmoney.com/jjjz_{fund_code}.html",
            headers={"Host": "api.fund.eastmoney.com"},
        )
        rows: list[dict[str, Any]] = []
        for item in _page_records(pages, "Data", "LSJZList"):
            nav_date = date.fromisoformat(str(item.get("FSRQ")))
            if nav_date < start or nav_date > end:
                continue
            daily_return = _to_float(item.get("JZZZL"))
            rows.append(
                {
                    "date": nav_date,
                    "symbol": fund_code,
                    "nav": _to_float(item.get("DWJZ")),
                    "acc_nav": _to_float(item.get("LJJZ")),
                    "daily_return": None
                    if daily_return is None
                    else daily_return / 100.0,
                }
            )
        return _frame_from_rows("fund_nav", rows, sort_by="date")

    async def _em_fund_holdings(
        self, fund_code: str, report_date: date
    ) -> pl.DataFrame:
        pages = await self._fetch_paginated(
            endpoint="fund_holdings",
            url=_FINANCIAL_URL,
            build_params=lambda page: {
                "sortColumns": "SECURITY_CODE",
                "sortTypes": "-1",
                "pageSize": str(_FUND_HOLDINGS_PAGE_SIZE),
                "pageNumber": str(page),
                "reportName": "RPT_MAINDATA_MAIN_POSITIONDETAILS",
                "columns": "ALL",
                "quoteColumns": "",
                "source": "WEB",
                "client": "WEB",
                "filter": f"(HOLDER_CODE=\"{fund_code}\")(REPORT_DATE='{report_date.isoformat()}')",
            },
            total_count=lambda payload: int(
                ((payload.get("result") or {}).get("pages")) or 0
            ),
            page_size=1,
            referer=(
                "https://data.eastmoney.com/zlsj/ccjj/"
                f"{report_date.isoformat()}-{fund_code}.html"
            ),
        )
        rows: list[dict[str, Any]] = []
        for item in _page_records(pages, "result", "data"):
            code = str(item.get("SECURITY_CODE") or "").zfill(6)
            rows.append(
                {
                    "report_date": report_date,
                    "symbol": fund_code,
                    "stock_symbol": code,
                    "stock_name": str(item.get("SECURITY_NAME_ABBR") or ""),
                    "shares": _to_float(item.get("HOLD_SHARES")),
                    "market_value": _to_float(item.get("HOLD_MV")),
                    "total_share_ratio": _to_float(item.get("TOTAL_SHARES_RATIO")),
                    "float_share_ratio": _to_float(item.get("FREE_SHARES_RATIO")),
                }
            )
        return _frame_from_rows("fund_holdings", rows)
