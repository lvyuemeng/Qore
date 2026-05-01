"""Quote data fetcher with source priority: BaoStock -> NetEase -> EastMoney."""

from __future__ import annotations

import asyncio
import csv
import io
import random
import time
from collections.abc import Sequence
from datetime import date
from typing import Any, Protocol

import httpx
import polars as pl

from qore_data.fetcher._base import (
    _CAPITAL_FLOW_URL,
    _CLIST_URL,
    _PUSH2HIS_URL,
    _UT_CAPITAL_FLOW,
    _UT_CLIST,
    _UT_KLINE,
    _board_from_code,
    _empty_frame,
    _exchange_from_stock_code,
    _frame_from_rows,
    _parse_compact_date,
    _secid_referer,
    _stock_secid,
    _symbol_digits,
    _to_float,
    _to_int,
)
from qore_data.fetcher.concurrent import BatchConfig, batch_fetch
from qore_data.fetcher.http import RequestSpec
from qore_data.fetcher.xueqiu import (
    _kline_timestamp,
    _ts_to_date,
    _xq_symbol,
    _XueqiuSession,
)

# ── BaoStock kline worker (pickleable for ProcessPoolExecutor) ────────────


def _kline_worker(symbol: str, start: date, end: date) -> pl.DataFrame:
    import baostock as bs

    from qore_data.fetcher._base import _suppress_stdout

    s = start.isoformat()
    e = end.isoformat()
    with _suppress_stdout():
        lg = bs.login()
    if lg.error_code != "0":
        return pl.DataFrame()
    try:
        code, exchange = symbol.upper().split(".", maxsplit=1)
        bs_code = f"{exchange.lower()}.{code}"
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,tradestatus,pctChg",
            s,
            e,
            "d",
            "2",
        )
        if rs is None or rs.error_code != "0":
            return pl.DataFrame()
        fields = getattr(rs, "fields", None)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not fields or not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows, schema=fields, orient="row")
    finally:
        with _suppress_stdout():
            bs.logout()


def _kline_batch_worker(item: tuple[str, date, date]) -> pl.DataFrame:
    symbol, start, end = item
    return _kline_worker(symbol, start, end)


def _kline_chunk_worker(chunk: list[tuple[str, date, date]]) -> list[pl.DataFrame]:
    """Process multiple symbols in a single BaoStock login session."""
    import baostock as bs

    from qore_data.fetcher._base import _suppress_stdout

    with _suppress_stdout():
        lg = bs.login()
    if lg.error_code != "0":
        return [pl.DataFrame()] * len(chunk)
    try:
        results: list[pl.DataFrame] = []
        for symbol, start, end in chunk:
            s = start.isoformat()
            e = end.isoformat()
            code, exchange = symbol.upper().split(".", maxsplit=1)
            bs_code = f"{exchange.lower()}.{code}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,volume,amount,tradestatus,pctChg",
                s,
                e,
                "d",
                "2",
            )
            if rs is None or rs.error_code != "0":
                results.append(pl.DataFrame())
                continue
            fields = getattr(rs, "fields", None)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not fields or not rows:
                results.append(pl.DataFrame())
            else:
                results.append(pl.DataFrame(rows, schema=fields, orient="row"))
        return results
    finally:
        with _suppress_stdout():
            bs.logout()


# ── BaoStock stock_info worker (pickleable) ──────────────────────────────


def _stock_info_worker(symbols: Sequence[str]) -> pl.DataFrame:
    import baostock as bs

    from qore_data.fetcher._base import _suppress_stdout

    with _suppress_stdout():
        lg = bs.login()
    if lg.error_code != "0":
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "short_name": pl.String,
                "exchange": pl.String,
                "industry": pl.String,
                "board": pl.String,
                "listing_date": pl.Date,
            }
        )
    try:
        rs1 = bs.query_stock_basic()
        basics = {}
        while rs1.next():
            row = rs1.get_row_data()
            basics[row[0]] = row

        rs2 = bs.query_stock_industry()
        industries = {}
        while rs2.next():
            row = rs2.get_row_data()
            if len(row) > 1:
                industries[row[1]] = row[3] if len(row) > 3 else "unknown"

        output: list[dict[str, Any]] = []
        for sym in symbols:
            code = _symbol_digits(sym)
            exchange = _exchange_from_stock_code(code)
            br = basics.get(f"{exchange.lower()}.{code}") or basics.get(code)
            if br is None or (len(br) > 3 and br[3] != ""):
                continue
            name = br[1] if len(br) > 1 else ""
            ipo = br[2] if len(br) > 2 and br[2] and br[2] != "0" else None
            output.append(
                {
                    "symbol": f"{code}.{exchange}",
                    "short_name": name,
                    "exchange": exchange,
                    "industry": industries.get(
                        f"{exchange.lower()}.{code}",
                        industries.get(code, "unknown"),
                    ),
                    "board": _board_from_code(code),
                    "listing_date": date.fromisoformat(ipo) if ipo else None,
                }
            )
        return pl.DataFrame(
            output,
            schema={
                "symbol": pl.String,
                "short_name": pl.String,
                "exchange": pl.String,
                "industry": pl.String,
                "board": pl.String,
                "listing_date": pl.Date,
            },
        )
    finally:
        with _suppress_stdout():
            bs.logout()


# ── source protocols ────────────────────────────────────────────────────


class QuoteDaySource(Protocol):
    async def stock_daily(
        self, symbol: str, start: date, end: date
    ) -> pl.DataFrame: ...

    async def batch_stock_daily(
        self, symbols: list[str], start: date, end: date
    ) -> list[pl.DataFrame]: ...

    async def close(self) -> None: ...


class QuoteProfileSource(Protocol):
    async def batch_stock_profiles(
        self, symbols: list[str], as_of: date
    ) -> pl.DataFrame: ...

    async def close(self) -> None: ...


class CapitalFlowSource(Protocol):
    async def capital_flow(
        self, symbol: str, start: date, end: date
    ) -> pl.DataFrame: ...

    async def close(self) -> None: ...


# ── BaoStock source (top priority) ──────────────────────────────────────


class _BaoStockQuoteSource:
    async def stock_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        result = await asyncio.to_thread(_kline_worker, symbol, start, end)
        if result.is_empty():
            return _empty_frame("stock_daily")
        return (
            result.with_columns(
                [
                    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
                    pl.lit(symbol).alias("symbol"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                    pl.col("amount").cast(pl.Float64),
                    pl.lit(1.0).alias("adj_factor"),
                    (pl.col("tradestatus") == "0").alias("is_suspended"),
                    (pl.col("pctChg").cast(pl.Float64) >= 9.9).alias("limit_up"),
                    (pl.col("pctChg").cast(pl.Float64) <= -9.9).alias("limit_down"),
                ]
            )
            .select(
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "adj_factor",
                "is_suspended",
                "limit_up",
                "limit_down",
            )
            .sort("date")
        )

    async def batch_stock_daily(
        self, symbols: list[str], start: date, end: date
    ) -> list[pl.DataFrame]:
        from qore_data.fetcher.concurrent import _chunked

        items = [(s, start, end) for s in symbols]
        chunks = _chunked(items, 50)
        chunk_results = await asyncio.to_thread(
            batch_fetch, BatchConfig.process(), _kline_chunk_worker, chunks
        )
        flat = [r for cr in chunk_results for r in cr]
        parsed: list[pl.DataFrame] = []
        for sym, raw in zip(symbols, flat, strict=False):
            if raw.is_empty():
                parsed.append(_empty_frame("stock_daily"))
            else:
                parsed.append(
                    raw.with_columns(
                        [
                            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
                            pl.lit(sym).alias("symbol"),
                            pl.col("open").cast(pl.Float64),
                            pl.col("high").cast(pl.Float64),
                            pl.col("low").cast(pl.Float64),
                            pl.col("close").cast(pl.Float64),
                            pl.col("volume").cast(pl.Int64),
                            pl.col("amount").cast(pl.Float64),
                            pl.lit(1.0).alias("adj_factor"),
                            (pl.col("tradestatus") == "0").alias("is_suspended"),
                            (pl.col("pctChg").cast(pl.Float64) >= 9.9).alias(
                                "limit_up"
                            ),
                            (pl.col("pctChg").cast(pl.Float64) <= -9.9).alias(
                                "limit_down"
                            ),
                        ]
                    )
                    .select(
                        "date",
                        "symbol",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                        "adj_factor",
                        "is_suspended",
                        "limit_up",
                        "limit_down",
                    )
                    .sort("date")
                )
        return parsed

    async def batch_stock_profiles(
        self, symbols: list[str], as_of: date
    ) -> pl.DataFrame:
        result = await asyncio.to_thread(_stock_info_worker, symbols)
        if not result.is_empty():
            return result
        return _empty_frame("stock_profile")

    async def close(self) -> None:
        pass


# ── Xueqiu source (second priority) ────────────────────────────────────


class _XueqiuQuoteDaySource:
    _KLINE_URL = "https://stock.xueqiu.com/v5/stock/chart/kline.json"

    def __init__(self, session: _XueqiuSession) -> None:
        self._session = session

    async def stock_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        params = {
            "symbol": _xq_symbol(symbol),
            "begin": str(_kline_timestamp(end)),
            "period": "day",
            "type": "before",
            "count": "-5000",
            "indicator": "kline",
        }
        try:
            data = await self._session.get_json(self._KLINE_URL, params)
        except Exception:
            return _empty_frame("stock_daily")
        return _xq_kline_to_frame(data, symbol, start, end)

    async def batch_stock_daily(
        self, symbols: list[str], start: date, end: date
    ) -> list[pl.DataFrame]:
        results: list[pl.DataFrame] = []
        for sym in symbols:
            try:
                r = await self.stock_daily(sym, start, end)
                results.append(r)
            except Exception:
                results.append(_empty_frame("stock_daily"))
        return results

    async def close(self) -> None:
        pass


def _xq_kline_to_frame(
    data: dict[str, Any], symbol: str, start: date, end: date
) -> pl.DataFrame:
    item_data = (data.get("data") or {}).get("item") or []
    if not item_data:
        return _empty_frame("stock_daily")
    rows: list[dict[str, Any]] = []
    for item in item_data:
        ts = item[0]
        bar_date = _ts_to_date(ts)
        if bar_date < start or bar_date > end:
            continue
        vol = item[1]
        open_p, high_p, low_p, close_p = item[2], item[3], item[4], item[5]
        if any(v is None for v in (open_p, high_p, low_p, close_p, vol)):
            continue
        amount = item[9] if len(item) > 9 and item[9] is not None else 0.0
        pct = ((close_p - open_p) / open_p * 100) if open_p else 0.0
        rows.append(
            {
                "date": bar_date,
                "symbol": symbol,
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": int(float(vol)),
                "amount": float(amount),
                "adj_factor": 1.0,
                "is_suspended": False,
                "limit_up": pct >= 9.9,
                "limit_down": pct <= -9.9,
            }
        )
    if not rows:
        return _empty_frame("stock_daily")
    return pl.DataFrame(rows).sort("date")


# ── NetEase source (mid priority) ───────────────────────────────────────


class _NeteaseQuoteSource:
    _URL = "http://quotes.money.163.com/service/chddata.html"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "Referer": "http://quotes.money.163.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                ),
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def stock_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        code = symbol.split(".", maxsplit=1)[0]
        exchange = symbol.upper().split(".")[-1] if "." in symbol else ""
        prefix = (
            "0"
            if exchange == "SH" or code.startswith(("60", "68", "51", "11"))
            else "1"
        )
        resp = await self._client.get(
            self._URL,
            params={
                "code": f"{prefix}{code}",
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "fields": "TCLOSE;HIGH;LOW;TOPEN;VOTURNOVER;VATURNOVER",
            },
        )
        resp.raise_for_status()
        text = resp.content.decode("gbk", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows: list[dict[str, Any]] = []
        for row in reader:
            raw_date = row.get("日期", "").strip()
            if not raw_date or raw_date == "日期":
                continue
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError:
                continue
            if parsed_date < start or parsed_date > end:
                continue
            rows.append(
                {
                    "date": parsed_date,
                    "symbol": symbol,
                    "open": _to_float(row.get("开盘价")),
                    "high": _to_float(row.get("最高价")),
                    "low": _to_float(row.get("最低价")),
                    "close": _to_float(row.get("收盘价")),
                    "volume": _to_int(row.get("成交量(手)")),
                    "amount": _to_float(row.get("成交金额(元)")),
                    "adj_factor": 1.0,
                    "is_suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                }
            )
        return pl.DataFrame(rows).sort("date") if rows else _empty_frame("stock_daily")

    async def batch_stock_daily(
        self, symbols: list[str], start: date, end: date
    ) -> list[pl.DataFrame]:
        results: list[pl.DataFrame] = []
        for sym in symbols:
            try:
                r = await self.stock_daily(sym, start, end)
                results.append(r)
            except Exception:
                results.append(_empty_frame("stock_daily"))
        return results


# ── EastMoney source (lowest priority) ──────────────────────────────────


class _EastMoneyQuoteSource:
    _ALL_A_SHARE_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    _CLIST_FIELDS = "f12,f14,f20,f21,f84,f85,f100,f26,f37,f3"

    def __init__(self, json_fetcher):
        self._json_fetcher = json_fetcher

    async def close(self) -> None:
        pass

    async def stock_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        payload = await self._json_fetcher.fetch_json(
            RequestSpec(
                endpoint="stock_daily",
                url=_PUSH2HIS_URL,
                params={
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
                    "ut": _UT_KLINE,
                    "klt": "101",
                    "fqt": "0",
                    "secid": _stock_secid(symbol),
                    "beg": start.strftime("%Y%m%d"),
                    "end": end.strftime("%Y%m%d"),
                },
                referer=_secid_referer(symbol, "https://quote.eastmoney.com/"),
            )
        )
        klines = (payload.get("data") or {}).get("klines") or []
        if not klines:
            return _empty_frame("stock_daily")
        rows: list[dict[str, Any]] = []
        for item in klines:
            parts = item.split(",")
            pct_change = _to_float(parts[8])
            limit_pct = 0.05 if symbol.startswith(("ST", "*ST")) else 0.10
            rows.append(
                {
                    "date": date.fromisoformat(parts[0]),
                    "symbol": symbol,
                    "open": _to_float(parts[1]),
                    "close": _to_float(parts[2]),
                    "high": _to_float(parts[3]),
                    "low": _to_float(parts[4]),
                    "volume": _to_int(parts[5]),
                    "amount": _to_float(parts[6]),
                    "adj_factor": 1.0,
                    "is_suspended": False,
                    "limit_up": pct_change is not None
                    and pct_change >= limit_pct * 100.0 - 1e-6,
                    "limit_down": pct_change is not None
                    and pct_change <= -limit_pct * 100.0 + 1e-6,
                }
            )
        return _frame_from_rows("stock_daily", rows)

    async def batch_stock_daily(
        self, symbols: list[str], start: date, end: date
    ) -> list[pl.DataFrame]:
        results: list[pl.DataFrame] = []
        for sym in symbols:
            try:
                r = await self.stock_daily(sym, start, end)
                results.append(r)
            except Exception:
                results.append(_empty_frame("stock_daily"))
        return results

    async def capital_flow(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        payload = await self._json_fetcher.fetch_json(
            RequestSpec(
                endpoint="capital_flow",
                url=_CAPITAL_FLOW_URL,
                params={
                    "lmt": "0",
                    "klt": "101",
                    "secid": _stock_secid(symbol),
                    "fields1": "f1,f2,f3,f7",
                    "fields2": "f51,f52,f53,f54,f55,f56",
                    "ut": _UT_CAPITAL_FLOW,
                    "_": str(int(time.time() * 1000)),
                },
                referer=_secid_referer(symbol, "https://data.eastmoney.com/zjlx/"),
            )
        )
        klines = (payload.get("data") or {}).get("klines") or []
        if not klines:
            return _empty_frame("capital_flow")
        rows: list[dict[str, Any]] = []
        for item in klines:
            parts = item.split(",")
            try:
                row_date = date.fromisoformat(parts[0])
            except (ValueError, IndexError):
                continue
            if row_date < start or row_date > end:
                continue
            rows.append(
                {
                    "date": row_date,
                    "symbol": symbol,
                    "main_net": _to_float(parts[1]),
                    "small_net": _to_float(parts[2]),
                    "mid_net": _to_float(parts[3]),
                    "large_net": _to_float(parts[4]),
                    "xlarge_net": _to_float(parts[5]) if len(parts) > 5 else None,
                }
            )
        return _frame_from_rows("capital_flow", rows, sort_by="date")

    async def batch_stock_profiles(
        self, symbols: list[str], as_of: date
    ) -> pl.DataFrame:
        symbol_set = frozenset(_symbol_digits(s) for s in symbols)
        all_items: list[dict[str, Any]] = []
        first = await self._json_fetcher.fetch_json(
            RequestSpec(
                endpoint="batch_profiles",
                url=_CLIST_URL,
                params={
                    "pn": "1",
                    "pz": "100",
                    "po": "1",
                    "np": "1",
                    "ut": _UT_CLIST,
                    "fltt": "2",
                    "invt": "2",
                    "fid": "f3",
                    "fs": self._ALL_A_SHARE_FS,
                    "fields": self._CLIST_FIELDS,
                },
            )
        )
        data_block = first.get("data") or {}
        total = int(data_block.get("total") or 0)
        all_items.extend(data_block.get("diff") or [])
        total_pages = min((total + 99) // 100, 200)
        for p in range(2, total_pages + 1):
            await asyncio.sleep(random.uniform(0.1, 0.3))
            payload = await self._json_fetcher.fetch_json(
                RequestSpec(
                    endpoint="batch_profiles",
                    url=_CLIST_URL,
                    params={
                        "pn": str(p),
                        "pz": "100",
                        "po": "1",
                        "np": "1",
                        "ut": _UT_CLIST,
                        "fltt": "2",
                        "invt": "2",
                        "fid": "f3",
                        "fs": self._ALL_A_SHARE_FS,
                        "fields": self._CLIST_FIELDS,
                    },
                )
            )
            page_data = payload.get("data") or {}
            all_items.extend(page_data.get("diff") or [])
        rows: list[dict[str, Any]] = []
        for item in all_items:
            code = str(item.get("f12") or "").zfill(6)
            if code not in symbol_set:
                continue
            exchange = _exchange_from_stock_code(code)
            full_symbol = f"{code}.{exchange}"
            short_name = str(item.get("f14") or "")
            listing_date = _parse_compact_date(item.get("f26"))
            rows.append(
                {
                    "symbol": full_symbol,
                    "short_name": short_name,
                    "exchange": exchange,
                    "industry": str(item.get("f100") or "unknown"),
                    "board": _board_from_symbol(full_symbol),
                    "listing_date": listing_date,
                }
            )
        return _frame_from_rows("stock_profile", rows)


# ── QuoteFetcher ────────────────────────────────────────────────────────


class QuoteFetcher:
    """Quote data. Priority: BaoStock -> Xueqiu -> NetEase -> EastMoney."""

    def __init__(self, json_fetcher) -> None:
        self._json_fetcher = json_fetcher
        self._xueqiu = _XueqiuSession()
        em = _EastMoneyQuoteSource(json_fetcher)
        self._day_sources: list[QuoteDaySource] = [
            _BaoStockQuoteSource(),
            _XueqiuQuoteDaySource(self._xueqiu),
            _NeteaseQuoteSource(),
            em,
        ]
        self._profile_sources: list[QuoteProfileSource] = [
            _BaoStockQuoteSource(),
            em,
        ]
        self._capital_sources: list[CapitalFlowSource] = [em]

    @classmethod
    def from_settings(cls, settings) -> QuoteFetcher:
        from qore_data.fetcher._base import build_json_fetcher

        return cls(build_json_fetcher(settings))

    async def close(self) -> None:
        seen: set[int] = set()
        for src in (
            *self._day_sources,
            *self._profile_sources,
            *self._capital_sources,
        ):
            if id(src) not in seen:
                seen.add(id(src))
                await src.close()
        await self._xueqiu.close()
        if hasattr(self._json_fetcher, "close"):
            await self._json_fetcher.close()

    # -- stock_daily ------------------------------------------------------------

    async def stock_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        for src in self._day_sources:
            try:
                result = await src.stock_daily(symbol, start, end)
                if not result.is_empty():
                    return result
            except Exception:
                continue
        return _empty_frame("stock_daily")

    async def batch_stock_daily(
        self, symbols: list[str], start: date, end: date
    ) -> list[pl.DataFrame]:
        for src in self._day_sources:
            try:
                results = await src.batch_stock_daily(symbols, start, end)
                if any(not r.is_empty() for r in results):
                    return results
            except Exception:
                continue
        return [_empty_frame("stock_daily")] * len(symbols)

    # -- capital_flow ----------------------------------------------------------

    async def capital_flow(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        for src in self._capital_sources:
            try:
                result = await src.capital_flow(symbol, start, end)
                if not result.is_empty():
                    return result
            except Exception:
                continue
        return _empty_frame("capital_flow")

    # -- stock_profile (single) -------------------------------------------------

    async def stock_profile(self, symbol: str, as_of: date) -> pl.DataFrame:
        return await self.batch_stock_profiles([symbol], as_of)

    # -- batch stock profiles ---------------------------------------------------

    async def batch_stock_profiles(
        self, symbols: list[str], as_of: date
    ) -> pl.DataFrame:
        for src in self._profile_sources:
            try:
                result = await src.batch_stock_profiles(symbols, as_of)
                if not result.is_empty():
                    return result
            except Exception:
                continue
        return _empty_frame("stock_profile")


def _board_from_symbol(symbol: str) -> str:
    return _board_from_code(_symbol_digits(symbol))
