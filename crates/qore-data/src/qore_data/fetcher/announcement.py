from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import httpx
import polars as pl

from qore_data.fetcher._base import (
    _ANNOUNCE_URL,
    BaseJsonFetcher,
    _frame_from_records,
    _parse_date,
    _symbol_digits,
)

logger = logging.getLogger(__name__)

_MAX_PAGES = 500
_PAGE_SIZE = 100


class _CNInfoFetcher:
    _ANNOUNCE_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    _ORG_URL = "http://www.cninfo.com.cn/new/data/query_stock_search.json"
    _PDF_BASE = "https://static.cninfo.com.cn/"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                ),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=20.0,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def announcements(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        audit_only: bool,
    ) -> pl.DataFrame:
        org_cache: dict[str, str] = {}
        announcements: list[dict[str, Any]] = []
        audit_opinions: list[dict[str, Any]] = []

        for sym in symbols:
            code = _symbol_digits(sym)
            org_id = await self._org_id(code, org_cache)
            if not org_id:
                continue
            items = await self._fetch_symbol(code, org_id, start, end, audit_only)
            for item in items:
                announcements.append(item)
                op = _audit_opinion_from_title(item.get("title", ""))
                if op is not None:
                    audit_opinions.append(
                        {
                            "symbol": item["symbol"],
                            "report_date": _audit_report_date(
                                item["title"], item["notice_date"]
                            ),
                            "announce_date": item["notice_date"],
                            "opinion": op[0],
                            "opinion_code": op[1],
                            "source_notice_type": item["notice_type"],
                            "title": item["title"],
                            "art_code": item["art_code"],
                            "url": item["url"],
                        }
                    )

        if audit_only:
            return (
                _frame_from_records(audit_opinions, columns=_AUDIT_OPINION_COLUMNS)
                if audit_opinions
                else pl.DataFrame(
                    schema={
                        "symbol": pl.String,
                        "report_date": pl.Date,
                        "announce_date": pl.Date,
                        "opinion": pl.String,
                        "opinion_code": pl.String,
                        "source_notice_type": pl.String,
                        "title": pl.String,
                        "art_code": pl.String,
                        "url": pl.String,
                    }
                )
            )
        return (
            _frame_from_records(announcements, columns=_ANNOUNCEMENT_COLUMNS)
            if announcements
            else pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "short_name": pl.String,
                    "title": pl.String,
                    "notice_type": pl.String,
                    "notice_date": pl.Date,
                    "art_code": pl.String,
                    "url": pl.String,
                }
            )
        )

    async def _org_id(self, code: str, cache: dict[str, str]) -> str:
        if code in cache:
            return cache[code]
        resp = await self._client.get(self._ORG_URL, params={"keyWord": code})
        resp.raise_for_status()
        for item in resp.json() or []:
            if str(item.get("code", "")).zfill(6) == code.zfill(6):
                org_id = str(item.get("orgId", ""))
                cache[code] = org_id
                return org_id
        return ""

    async def _fetch_symbol(
        self,
        code: str,
        org_id: str,
        start: date,
        end: date,
        audit_only: bool,
    ) -> list[dict[str, Any]]:
        category = "category_sjdbg_szsh" if audit_only else ""
        plate = "sse" if code.startswith(("60", "68", "51", "11")) else "szse"
        page = 1
        items: list[dict[str, Any]] = []
        while True:
            data = {
                "stock": f"{code},{org_id}",
                "tabName": "fulltext",
                "pageSize": "30",
                "pageNum": str(page),
                "column": plate,
                "category": category,
                "plate": "",
                "seDate": f"{start.isoformat()} ~ {end.isoformat()}",
                "searchkey": "",
                "secid": "",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }
            resp = await self._client.post(self._ANNOUNCE_URL, data=data)
            resp.raise_for_status()
            payload = resp.json()
            page_items = payload.get("announcements") or []
            if not page_items:
                break
            for item in page_items:
                ann_time = item.get("announcementTime", 0)
                ann_date = date.fromtimestamp(ann_time / 1000) if ann_time else None
                if ann_date is None or ann_date < start or ann_date > end:
                    continue
                adj_url = item.get("adjunctUrl", "")
                items.append(
                    {
                        "symbol": f"{code}.{_cninfo_exchange(code)}",
                        "short_name": str(item.get("secName") or ""),
                        "title": str(item.get("announcementTitle") or ""),
                        "notice_type": str(item.get("announcementTypeName") or ""),
                        "notice_date": ann_date,
                        "art_code": str(item.get("announcementId") or ""),
                        "url": self._PDF_BASE + adj_url if adj_url else "",
                    }
                )
            page += 1
            if page > 100:
                break
        logger.info(
            "cninfo_fetch symbol=%s pages=%d items=%d audit_only=%s",
            code,
            page - 1,
            len(items),
            audit_only,
        )
        return items


# -- AnnouncementFetcher ----------------------------------------------------


class AnnouncementFetcher(BaseJsonFetcher):
    """Announcements. Priority: CNInfo → EastMoney."""

    def __init__(self, json_fetcher) -> None:
        super().__init__(json_fetcher)
        self._cninfo = _CNInfoFetcher()

    async def close(self) -> None:
        await super().close()
        await self._cninfo.close()

    async def announcements(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        try:
            return await self._cninfo.announcements(
                [symbol], start, end, audit_only=False
            )
        except Exception:
            pass
        raw = await self._fetch_all_raw(start, end)
        return _announcements_frame({_symbol_digits(symbol)}, raw)

    async def audit_opinions(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        try:
            return await self._cninfo.announcements(
                [symbol], start, end, audit_only=True
            )
        except Exception:
            pass
        raw = await self._fetch_all_raw(start, end)
        return _audit_opinions_frame({_symbol_digits(symbol)}, raw)

    async def batch_announcements(
        self, symbols: list[str], start: date, end: date
    ) -> pl.DataFrame:
        try:
            return await self._cninfo.announcements(
                symbols, start, end, audit_only=False
            )
        except Exception:
            logger.debug(
                "cninfo_fallback_to_eastmoney dataset=announcements", exc_info=True
            )
        raw = await self._fetch_all_raw(start, end)
        codes = {_symbol_digits(s) for s in symbols}
        return _announcements_frame(codes, raw)

    async def batch_audit_opinions(
        self, symbols: list[str], start: date, end: date
    ) -> pl.DataFrame:
        try:
            return await self._cninfo.announcements(
                symbols, start, end, audit_only=True
            )
        except Exception:
            logger.debug(
                "cninfo_fallback_to_eastmoney dataset=audit_opinions", exc_info=True
            )
        raw = await self._fetch_all_raw(start, end)
        codes = {_symbol_digits(s) for s in symbols}
        return _audit_opinions_frame(codes, raw)

    # -- EastMoney fallback --------------------------------------------------

    async def _fetch_all_raw(self, start: date, end: date) -> list[dict[str, Any]]:
        pages = await self._fetch_paginated(
            endpoint="announcements",
            url=_ANNOUNCE_URL,
            build_params=lambda p: {
                "sr": "-1",
                "page_size": str(_PAGE_SIZE),
                "page_index": str(p),
                "ann_type": "A",
                "client_source": "web",
                "f_node": "0",
                "s_node": "0",
                "begin_time": start.isoformat(),
                "end_time": end.isoformat(),
            },
            total_count=lambda payload: int(
                ((payload.get("data") or {}).get("total_hits")) or 0
            ),
            page_size=_PAGE_SIZE,
            max_pages=_MAX_PAGES,
        )
        records: list[dict[str, Any]] = []
        for page in pages:
            data = page.get("data") or {}
            records.extend(data.get("list") or [])
        logger.info(
            "announcements_fetch_eastmoney pages=%d records=%d",
            len(pages),
            len(records),
        )
        return records


# -- CNInfo helpers ---------------------------------------------------------


def _cninfo_exchange(code: str) -> str:
    c = code.zfill(6)
    if c.startswith(("60", "68", "51", "11")):
        return "SH"
    if c.startswith(("430", "83", "87", "88")):
        return "BJ"
    return "SZ"


# -- shared helpers ----------------------------------------------------------


def _unnest_items(
    raw_records: list[dict[str, Any]],
    start: date,
    end: date,
    target_codes: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_records:
        codes = item.get("codes") or []
        for code_info in codes:
            sym = str(code_info.get("stock_code", "")).zfill(6)
            if target_codes is not None and sym not in target_codes:
                continue
            notice_date = _parse_date(item.get("notice_date"))
            if notice_date is None or notice_date < start or notice_date > end:
                continue
            title = str(item.get("title") or "")
            cols = item.get("columns") or []
            first_col = cols[0] if cols else {}
            art_code = str(item.get("art_code") or "")
            rows.append(
                {
                    "symbol": sym,
                    "short_name": str(code_info.get("short_name") or ""),
                    "title": title,
                    "notice_type": str(first_col.get("column_name") or ""),
                    "notice_date": notice_date,
                    "art_code": art_code,
                    "url": f"https://data.eastmoney.com/notices/detail/{sym}/{art_code}.html",
                }
            )
    return rows


_ANNOUNCEMENT_COLUMNS = (
    "symbol",
    "short_name",
    "title",
    "notice_type",
    "notice_date",
    "art_code",
    "url",
)

_AUDIT_OPINION_COLUMNS = (
    "symbol",
    "report_date",
    "announce_date",
    "opinion",
    "opinion_code",
    "source_notice_type",
    "title",
    "art_code",
    "url",
)


def _announcements_frame(
    target_codes: set[str], raw_records: list[dict[str, Any]]
) -> pl.DataFrame:
    rows = _unnest_items(
        raw_records, date(2000, 1, 1), date(2099, 12, 31), target_codes
    )
    return (
        _frame_from_records(rows, columns=_ANNOUNCEMENT_COLUMNS)
        if rows
        else pl.DataFrame(
            schema={
                "symbol": pl.String,
                "short_name": pl.String,
                "title": pl.String,
                "notice_type": pl.String,
                "notice_date": pl.Date,
                "art_code": pl.String,
                "url": pl.String,
            }
        )
    )


def _audit_opinions_frame(
    target_codes: set[str], raw_records: list[dict[str, Any]]
) -> pl.DataFrame:
    rows = _unnest_items(
        raw_records, date(2000, 1, 1), date(2099, 12, 31), target_codes
    )
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        opinion = _audit_opinion_from_title(row["title"])
        if opinion is None:
            continue
        audit_rows.append(
            {
                "symbol": row["symbol"],
                "report_date": _audit_report_date(row["title"], row["notice_date"]),
                "announce_date": row["notice_date"],
                "opinion": opinion[0],
                "opinion_code": opinion[1],
                "source_notice_type": row["notice_type"],
                "title": row["title"],
                "art_code": row["art_code"],
                "url": row["url"],
            }
        )
    return (
        _frame_from_records(audit_rows, columns=_AUDIT_OPINION_COLUMNS)
        if audit_rows
        else pl.DataFrame(
            schema={
                "symbol": pl.String,
                "report_date": pl.Date,
                "announce_date": pl.Date,
                "opinion": pl.String,
                "opinion_code": pl.String,
                "source_notice_type": pl.String,
                "title": pl.String,
                "art_code": pl.String,
                "url": pl.String,
            }
        )
    )


def _audit_opinion_from_title(title: str) -> tuple[str, str] | None:
    norm = title.replace(" ", "")
    if "无法表示意见" in norm:
        return ("无法表示意见", "disclaimer")
    if "否定意见" in norm:
        return ("否定意见", "adverse")
    if "保留意见" in norm:
        return ("保留意见", "qualified")
    if "无保留意见" in norm and "审计" in norm:
        return ("无保留意见", "unqualified")
    return None


def _audit_report_date(title: str, announce_date: date) -> date:
    matched = re.search(r"(20\d{2})年", title)
    if matched is None:
        return date(announce_date.year - 1, 12, 31)
    return date(int(matched.group(1)), 12, 31)
