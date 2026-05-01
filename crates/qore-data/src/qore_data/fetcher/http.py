from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


@dataclass(frozen=True, slots=True)
class HeaderProfile:
    user_agent: str
    accept_language: str
    cache_control: str
    pragma: str


@dataclass(frozen=True, slots=True)
class RequestSpec:
    endpoint: str
    url: str
    params: dict[str, Any]
    referer: str | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    delay_min: float
    delay_max: float
    max_retries: int
    retry_budget: int
    retry_backoff_min: float
    retry_backoff_max: float


@runtime_checkable
class JsonFetcher(Protocol):
    async def fetch_json(self, spec: RequestSpec) -> dict[str, Any]: ...


@runtime_checkable
class AsyncClosable(Protocol):
    async def close(self) -> None: ...


class ResponseGuard(Protocol):
    def should_retry(self, exc: BaseException) -> bool: ...

    def is_blocked_response(self, response: httpx.Response) -> bool: ...

    def blocked_error(self, endpoint: str) -> BaseException: ...


@dataclass(slots=True)
class RequestHardening:
    header_profiles: tuple[HeaderProfile, ...]
    cooldown_min: float
    cooldown_max: float
    _cooldown_until_by_host: dict[str, float] = field(default_factory=dict)

    def headers_for(
        self, url: str, extra_headers: dict[str, str] | None
    ) -> dict[str, str]:
        profile = random.choice(self.header_profiles)
        headers = {
            "User-Agent": profile.user_agent,
            "Accept-Language": profile.accept_language,
            "Cache-Control": profile.cache_control,
            "Pragma": profile.pragma,
            "Accept": "application/json, text/plain, */*",
            "Origin": f"{urlsplit(url).scheme}://{self.host_name(url)}",
        }
        if extra_headers is not None:
            headers.update(extra_headers)
        return headers

    @staticmethod
    def host_name(url: str) -> str:
        return urlsplit(url).netloc

    async def wait_for_host_cooldown(self, host: str) -> float:
        now = monotonic()
        cooldown_until = self._cooldown_until_by_host.get(host, 0.0)
        wait_seconds = max(cooldown_until - now, 0.0)
        if wait_seconds > 0.0:
            await asyncio.sleep(wait_seconds)
        return wait_seconds

    def apply_host_cooldown(self, host: str) -> float:
        duration = random.uniform(self.cooldown_min, self.cooldown_max)
        current = monotonic()
        base = max(self._cooldown_until_by_host.get(host, current), current)
        self._cooldown_until_by_host[host] = base + duration
        return duration


@dataclass(slots=True)
class HardenedJsonFetcher(JsonFetcher, AsyncClosable):
    client: httpx.AsyncClient
    semaphore: asyncio.Semaphore
    policy: RequestPolicy
    hardening: RequestHardening
    guard: ResponseGuard
    _retry_budget_used: dict[str, int] = field(default_factory=dict)

    async def fetch_json(self, spec: RequestSpec) -> dict[str, Any]:
        endpoint = spec.endpoint
        host = self.hardening.host_name(spec.url)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.policy.max_retries),
            wait=wait_random_exponential(
                multiplier=self.policy.retry_backoff_min,
                max=self.policy.retry_backoff_max,
            ),
            retry=retry_if_exception(self.guard.should_retry),
            reraise=True,
        ):
            with attempt:
                return await self._fetch_once(spec, endpoint, host)
        msg = f"Exhausted retries for {endpoint}"
        raise RuntimeError(msg)

    async def close(self) -> None:
        await self.client.aclose()

    async def _fetch_once(
        self, spec: RequestSpec, endpoint: str, host: str
    ) -> dict[str, Any]:
        await self.hardening.wait_for_host_cooldown(host)
        try:
            async with self.semaphore:
                await asyncio.sleep(
                    random.uniform(self.policy.delay_min, self.policy.delay_max)
                )
                response = await self.client.get(
                    spec.url,
                    params=spec.params,
                    headers=self.hardening.headers_for(
                        spec.url, self._merged_headers(spec)
                    ),
                )
            if self.guard.is_blocked_response(response):
                self.hardening.apply_host_cooldown(host)
                raise self.guard.blocked_error(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if isinstance(exc, httpx.ProtocolError | httpx.NetworkError):
                self.hardening.apply_host_cooldown(host)
            raise

    def _consume_retry_budget(self, endpoint: str) -> bool:
        used = self._retry_budget_used.get(endpoint, 0)
        if used >= self.policy.retry_budget:
            return False
        self._retry_budget_used[endpoint] = used + 1
        return True

    @staticmethod
    def _merged_headers(spec: RequestSpec) -> dict[str, str] | None:
        headers: dict[str, str] = {}
        if spec.referer is not None:
            headers["Referer"] = spec.referer
        if spec.headers is not None:
            headers.update(spec.headers)
        return headers or None
