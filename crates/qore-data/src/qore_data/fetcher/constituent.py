"""Unified index constituent fetcher.

Sources in priority order:
1. CSI official XLS (verified working)
2. EastMoney fundztapi (rate-limited fallback)
3. BaoStock query (fallback for supported indices)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
import polars as pl
import xlrd

from qore_data.fetcher._base import (
    _CSINDEX_URL_TEMPLATE,
    _FUNDZTAPI_URL,
    BaseJsonFetcher,
    _empty_frame,
    _exchange_from_stock_code,
    _frame_from_rows,
    _symbol_digits,
    _to_float,
)
from qore_data.fetcher.http import RequestSpec
from qore_data.fetcher.xueqiu import _xq_symbol, _XueqiuSession

logger = logging.getLogger(__name__)

_CSI_CODE_COLUMNS = (
    "成分券代码",
    "成份券代码",
    "成分券代码Constituent Code",
    "成份券代码Constituent Code",
)


# ── BaoStock constituents worker (pickleable for ProcessPoolExecutor) ────


def _constituents_worker(index_symbol: str) -> pl.Series:
    import baostock as bs

    from qore_data.fetcher._base import _suppress_stdout

    code_map = {
        "000016": "query_sz50_stocks",
        "000300": "query_hs300_stocks",
        "000905": "query_zz500_stocks",
        "000852": "query_zz500_stocks",
    }
    digits = "".join(ch for ch in index_symbol if ch.isdigit())
    method_name = code_map.get(digits, "")
    if not method_name:
        return pl.Series("symbol", [], dtype=pl.String)

    with _suppress_stdout():
        lg = bs.login()
    if lg.error_code != "0":
        return pl.Series("symbol", [], dtype=pl.String)
    try:
        func = getattr(bs, method_name)
        rs = func()
        if rs is None or rs.error_code != "0":
            return pl.Series("symbol", [], dtype=pl.String)
        fields = getattr(rs, "fields", None)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not fields or not rows:
            return pl.Series("symbol", [], dtype=pl.String)
        df = pl.DataFrame(rows, schema=fields, orient="row")
        if "code" not in df.columns:
            return pl.Series("symbol", [], dtype=pl.String)
        codes = [
            f"{c.zfill(6)}.{_exchange_from_stock_code(c)}"
            for c in df.get_column("code").to_list()
        ]
        return pl.Series("symbol", codes, dtype=pl.String)
    finally:
        with _suppress_stdout():
            bs.logout()


@dataclass(frozen=True, slots=True)
class _CSISource:
    _client: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={
                "Referer": "https://www.csindex.com.cn/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
        ),
        hash=False,
        compare=False,
    )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch(self, digits: str) -> pl.Series | None:
        if len(digits) != 6:
            return None
        url = _CSINDEX_URL_TEMPLATE.format(symbol=digits)
        resp = await self._client.get(url)
        resp.raise_for_status()
        return _parse_csi_xls(resp.content)


class ConstituentFetcher(BaseJsonFetcher):
    """Index constituents. Priority: CSI → EastMoney → Xueqiu → BaoStock."""

    def __init__(self, json_fetcher) -> None:
        super().__init__(json_fetcher)
        self._csi = _CSISource()
        self._xueqiu_session = _XueqiuSession()

    async def close(self) -> None:
        await self._csi.close()
        await self._xueqiu_session.close()
        await super().close()

    async def _xq_constituents(self, index_symbol: str) -> pl.Series:
        try:
            data = await self._xueqiu_session.get_json(
                "https://stock.xueqiu.com/v5/stock/index/detail/quote.json",
                {"symbol": _xq_symbol(index_symbol), "size": "500", "page": "1"},
            )
        except Exception:
            return pl.Series("symbol", [], dtype=pl.String)
        items = (data.get("data") or {}).get("list") or []
        codes: list[str] = []
        for item in items:
            sym = str(item.get("symbol", ""))
            if not sym:
                continue
            codes.append(f"{sym[2:]}." + ("SH" if sym.startswith("SH") else "SZ"))
        return pl.Series("symbol", codes, dtype=pl.String)

    async def index_constituents(self, index_symbol: str, as_of: date) -> pl.Series:
        del as_of
        digits = _symbol_digits(index_symbol)

        result = await self._csi.fetch(digits)
        if result is not None and len(result) > 0:
            logger.info(
                "constituents symbol=%s count=%d source=csi",
                index_symbol,
                len(result),
            )
            return result

        try:
            frame = await self.index_constituents_with_weight(index_symbol)
            if not frame.is_empty():
                result = frame.get_column("symbol")
                logger.info(
                    "constituents symbol=%s count=%d source=eastmoney",
                    index_symbol,
                    len(result),
                )
                return result
        except Exception:
            pass

        try:
            result = await self._xq_constituents(index_symbol)
            if len(result) > 0:
                logger.info(
                    "constituents symbol=%s count=%d source=xueqiu",
                    index_symbol,
                    len(result),
                )
                return result
        except Exception:
            pass

        try:
            result = await asyncio.to_thread(_constituents_worker, index_symbol)
            if len(result) > 0:
                logger.info(
                    "constituents symbol=%s count=%d source=baostock",
                    index_symbol,
                    len(result),
                )
                return result
        except Exception:
            pass

        return pl.Series("symbol", [], dtype=pl.String)

    async def index_constituents_with_weight(self, index_symbol: str) -> pl.DataFrame:
        digits = _symbol_digits(index_symbol)
        payload = await self._json_fetcher.fetch_json(
            RequestSpec(
                endpoint="index_constituents_weight",
                url=_FUNDZTAPI_URL,
                params={
                    "IndexCode": digits,
                    "pageIndex": "1",
                    "pageSize": "10000",
                    "product": "EFund",
                    "plat": "Iphone",
                },
                referer="https://fund.eastmoney.com/",
            )
        )
        records = payload.get("data") or []
        if not records:
            return _empty_frame("index_constituents")
        rows: list[dict[str, Any]] = []
        for r in records:
            code = str(r.get("ZQDM") or "").zfill(6)
            exchange = _exchange_from_stock_code(code)
            rows.append(
                {
                    "index_symbol": index_symbol,
                    "symbol": f"{code}.{exchange}",
                    "short_name": str(r.get("ZQMC") or ""),
                    "exchange": exchange,
                    "industry": str(r.get("HYMC") or "unknown"),
                    "weight": _to_float(r.get("ZJBL")),
                }
            )
        return _frame_from_rows("index_constituents", rows)


def _parse_csi_xls(content: bytes) -> pl.Series | None:
    wb = xlrd.open_workbook(file_contents=content)
    for si in range(wb.nsheets):
        sheet = wb.sheet_by_index(si)
        if sheet.nrows < 2:
            continue
        header = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
        code_col = next(
            (i for i, h in enumerate(header) if any(c in h for c in _CSI_CODE_COLUMNS)),
            None,
        )
        if code_col is None:
            continue
        codes: list[str] = []
        for r in range(1, sheet.nrows):
            raw = str(sheet.cell_value(r, code_col)).strip().split(".")[0].zfill(6)
            if raw.isdigit() and len(raw) == 6:
                codes.append(f"{raw}.{_exchange_from_stock_code(raw)}")
        if codes:
            return pl.Series("symbol", codes, dtype=pl.String)
    return None
