from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from qore_data.fetcher._base import (
    _FINANCIAL_URL,
    BaseJsonFetcher,
    _extract_code,
    _frame_from_records,
    _symbol_digits,
    _to_float,
    _to_int,
)
from qore_data.fetcher.http import RequestSpec
from qore_data.fetcher.xueqiu import _xq_symbol, _XueqiuSession

# ── Xueqiu analyst source (fallback, EPS-only) ──────────────────────────


_XQ_EARN_URL = "https://stock.xueqiu.com/v5/stock/finance/cn/earningforecast.json"


class _XueqiuAnalystSource:
    def __init__(self, session: _XueqiuSession) -> None:
        self._session = session

    async def batch_analyst_forecasts(
        self, symbols: list[str], as_of: date
    ) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for sym in symbols:
            try:
                data = await self._session.get_json(
                    _XQ_EARN_URL, {"symbol": _xq_symbol(sym), "count": "6"}
                )
            except Exception:
                continue
            items = (data.get("data") or {}).get("list") or []
            if not items:
                continue
            eps_by_year: dict[str, float | None] = {}
            org_count = 0
            for item in items:
                fy = str(item.get("forecast_year", ""))
                eps = _to_float(item.get("eps"))
                eps_by_year[fy] = eps
                oc = item.get("org_num") or 0
                if oc > org_count:
                    org_count = oc

            now = as_of.year
            row: dict[str, Any] = {
                "as_of": as_of,
                "symbol": sym,
                "report_count": org_count,
                "buy": None,
                "overweight": None,
                "neutral": None,
                "underweight": None,
                "sell": None,
                "eps_year1": eps_by_year.get(str(now)),
                "eps_year2": eps_by_year.get(str(now + 1)),
                "eps_year3": eps_by_year.get(str(now + 2)),
                "eps_year4": eps_by_year.get(str(now + 3)),
            }
            rows.append(row)

        if rows:
            return _frame_from_records(
                rows,
                columns=(
                    "as_of",
                    "symbol",
                    "report_count",
                    "buy",
                    "overweight",
                    "neutral",
                    "underweight",
                    "sell",
                    "eps_year1",
                    "eps_year2",
                    "eps_year3",
                    "eps_year4",
                ),
            )
        return pl.DataFrame(
            schema={"as_of": pl.Date, "symbol": pl.String, "report_count": pl.Int64}
        )

    async def close(self) -> None:
        pass


# ── AnalystFetcher ──────────────────────────────────────────────────────


class AnalystFetcher(BaseJsonFetcher):
    """Analyst consensus forecasts. Priority: Xueqiu (EPS) -> EastMoney (full rating)."""

    def __init__(self, json_fetcher) -> None:
        super().__init__(json_fetcher)
        self._xueqiu_session = _XueqiuSession()
        self._xueqiu = _XueqiuAnalystSource(self._xueqiu_session)

    async def close(self) -> None:
        await self._xueqiu_session.close()
        await super().close()

    async def analyst_forecast(self, symbol: str, as_of: date) -> pl.DataFrame:
        return await self.batch_analyst_forecasts([symbol], as_of)

    async def batch_analyst_forecasts(
        self, symbols: list[str], as_of: date
    ) -> pl.DataFrame:
        result = await self._xueqiu.batch_analyst_forecasts(symbols, as_of)
        if not result.is_empty():
            return result
        return await self._batch_eastmoney(symbols, as_of)

    async def _batch_eastmoney(self, symbols: list[str], as_of: date) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame(
                schema={"as_of": pl.Date, "symbol": pl.String, "report_count": pl.Int64}
            )
        instrument_codes = [_symbol_digits(s) for s in symbols]
        code_set = frozenset(instrument_codes)

        all_records = await self._gather_chunked(
            codes=instrument_codes,
            report_name="RPT_WEB_RESPREDICT",
            sort_col="RATING_ORG_NUM",
            chunk_size=100,
            referer="https://data.eastmoney.com/report/profitforecast.jshtml",
        )
        rows: list[dict[str, Any]] = []
        for rec in all_records:
            code = _extract_code(rec)
            if code not in code_set:
                continue
            rows.append(
                {
                    "as_of": as_of,
                    "symbol": symbols[0] if len(symbols) == 1 else code,
                    "report_count": _to_int(rec.get("RATING_ORG_NUM")),
                    "buy": _to_int(rec.get("BUY")),
                    "overweight": _to_int(rec.get("HOLD")),
                    "neutral": _to_int(rec.get("NEUTRAL")),
                    "underweight": _to_int(rec.get("SELL")),
                    "sell": _to_int(rec.get("STRONG_SELL")),
                    "eps_year1": _to_float(rec.get("EPS1")),
                    "eps_year2": _to_float(rec.get("EPS2")),
                    "eps_year3": _to_float(rec.get("EPS3")),
                    "eps_year4": _to_float(rec.get("EPS4")),
                }
            )
        return _frame_from_records(
            rows,
            columns=(
                "as_of",
                "symbol",
                "report_count",
                "buy",
                "overweight",
                "neutral",
                "underweight",
                "sell",
                "eps_year1",
                "eps_year2",
                "eps_year3",
                "eps_year4",
            ),
        )

    async def _gather_chunked(
        self, *, codes, report_name, sort_col, chunk_size=100, referer=None
    ):
        all_records: list[dict] = []
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i : i + chunk_size]
            chunk_filter = ",".join(f'"{c}"' for c in chunk)
            payload = await self._json_fetcher.fetch_json(
                RequestSpec(
                    endpoint="analyst_forecast_batch",
                    url=_FINANCIAL_URL,
                    params={
                        "reportName": report_name,
                        "columns": "ALL",
                        "pageSize": "5000",
                        "pageNumber": "1",
                        "sortTypes": "-1",
                        "sortColumns": sort_col,
                        "source": "WEB",
                        "client": "WEB",
                        "filter": f"(SECURITY_CODE in ({chunk_filter}))",
                    },
                    referer=referer,
                )
            )
            all_records.extend(((payload.get("result") or {}).get("data")) or [])
        return all_records
