from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class _XueqiuTokenError(RuntimeError):
    pass


class _XueqiuSession:
    """Guest token lifecycle for Xueqiu API.

    Acquires ``xq_a_token`` by hitting the Xueqiu homepage.
    Lazily fetched on first request; refreshed on 400/401.
    Thread-safe for asyncio usage (single connection).
    """

    _BASE = "https://xueqiu.com"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._token: str = ""
        self._uid: str = ""
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=20.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                    ),
                },
            )
        return self._client

    async def _acquire_token(self) -> None:
        client = await self._ensure_client()
        resp = await client.get(self._BASE)
        resp.raise_for_status()
        for cookie in client.cookies.jar:
            if cookie.name == "xq_a_token":
                self._token = cookie.value
            elif cookie.name == "u":
                self._uid = cookie.value
        if not self._token:
            raise _XueqiuTokenError("xq_a_token not set by Xueqiu homepage")

    async def _headers(self) -> dict[str, str]:
        async with self._lock:
            if not self._token:
                await self._acquire_token()
            return {
                "Cookie": f"xq_a_token={self._token}; u={self._uid}",
                "Referer": f"{self._BASE}/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                ),
            }

    async def get_json(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        headers = await self._headers()
        client = await self._ensure_client()
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code in (400, 401):
            logger.info("xueqiu_token_expired refreshing")
            async with self._lock:
                self._token = ""
                self._uid = ""
                headers = await self._headers()
            resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error_code"):
            raise _XueqiuTokenError(
                f"Xueqiu API error: {data.get('error_description', 'unknown')}"
            )
        return data

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ── kline helpers ───────────────────────────────────────────────────────


def _xq_symbol(symbol: str) -> str:
    code, exchange = symbol.upper().split(".", maxsplit=1)
    return f"{exchange}{code}"


def _kline_timestamp(d: date) -> int:
    return int(d.timestamp() * 1000)


def _ts_to_date(ms: int) -> date:
    return date.fromtimestamp(ms / 1000)


# ── financial period helpers ─────────────────────────────────────────────


def _xq_period(year: int, quarter: int) -> str:
    if quarter <= 0 or quarter > 4:
        return "annual"
    return f"Q{quarter}"


def _latest_financial_quarter(as_of: date) -> tuple[int, int]:
    effective = as_of - timedelta(days=90)
    m = effective.month
    q = ((m - 1) // 3) + 1
    return effective.year, q
