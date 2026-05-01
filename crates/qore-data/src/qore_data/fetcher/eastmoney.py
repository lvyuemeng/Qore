"""EastMoney HTTP fetcher infrastructure — URLs, tokens, anti-crawl, BaseJsonFetcher."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

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

# URLs
_FINANCIAL_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_PUSH2HIS_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
_CAPITAL_FLOW_URL = "https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get"
_ANNOUNCE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
_FUND_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
_FUNDZTAPI_URL = (
    "https://fundztapi.eastmoney.com/FundSpecialApiNew/FundSpecialZSB30ZSCFG"
)

# Tokens
_UT_KLINE = "7eea3edcaed734bea9cbfc24409ed989"
_UT_CAPITAL_FLOW = "b2884a393a59ad64002292a3e90d46a5"
_UT_CLIST = "bd1d9ddb04089700cf9c27f6f7426281"


class BaseJsonFetcher:
    """Base for EastMoney HTTP JSON fetchers."""

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
                pages.append(
                    await self._json_fetcher.fetch_json(
                        RequestSpec(
                            endpoint=endpoint,
                            url=url,
                            params=build_params(i),
                            referer=referer,
                            headers=headers,
                        )
                    )
                )
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
        headers={"Referer": "https://data.eastmoney.com/", "Connection": "close"},
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
