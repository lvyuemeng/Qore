from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field
from time import monotonic, perf_counter
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
import polars as pl
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


@dataclass(slots=True)
class EndpointStats:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    retry_budget_exhaustions: int = 0
    anti_crawl_hits: int = 0
    cooldown_wait_seconds: float = 0.0
    total_latency_seconds: float = 0.0


@dataclass(slots=True)
class RequestTelemetry:
    endpoints: dict[str, EndpointStats] = field(default_factory=dict)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                **asdict(stats),
                "avg_latency_seconds": (
                    stats.total_latency_seconds / stats.successes
                    if stats.successes > 0
                    else 0.0
                ),
            }
            for name, stats in self.endpoints.items()
        }

    def frame(self) -> pl.DataFrame:
        if not self.endpoints:
            return pl.DataFrame(
                schema={
                    "endpoint": pl.String,
                    "requests": pl.Int64,
                    "successes": pl.Int64,
                    "failures": pl.Int64,
                    "retries": pl.Int64,
                    "retry_budget_exhaustions": pl.Int64,
                    "anti_crawl_hits": pl.Int64,
                    "cooldown_wait_seconds": pl.Float64,
                    "total_latency_seconds": pl.Float64,
                    "avg_latency_seconds": pl.Float64,
                }
            )
        return pl.DataFrame(
            [{"endpoint": name, **values} for name, values in self.snapshot().items()]
        ).sort("endpoint")


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


@runtime_checkable
class TelemetryReadable(Protocol):
    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]: ...

    def telemetry_frame(self) -> pl.DataFrame: ...


class ResponseGuard(Protocol):
    def should_retry(self, exc: BaseException) -> bool: ...

    def is_blocked_response(self, response: httpx.Response) -> bool: ...

    def blocked_error(self, endpoint: str) -> BaseException: ...


@dataclass(slots=True)
class RequestHardening:
    telemetry: RequestTelemetry
    header_profiles: tuple[HeaderProfile, ...]
    cooldown_min: float
    cooldown_max: float
    _cooldown_until_by_host: dict[str, float] = field(default_factory=dict)

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]:
        return self.telemetry.snapshot()

    def telemetry_frame(self) -> pl.DataFrame:
        return self.telemetry.frame()

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
class HardenedJsonFetcher(JsonFetcher, AsyncClosable, TelemetryReadable):
    client: httpx.AsyncClient
    semaphore: asyncio.Semaphore
    policy: RequestPolicy
    hardening: RequestHardening
    guard: ResponseGuard
    _retry_budget_used: dict[str, int] = field(default_factory=dict)

    async def fetch_json(self, spec: RequestSpec) -> dict[str, Any]:
        endpoint = spec.endpoint
        host = self.hardening.host_name(spec.url)
        stats = self.hardening.telemetry.endpoints.setdefault(endpoint, EndpointStats())
        attempt_number = 0
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
                attempt_number += 1
                if attempt_number > 1:
                    if not self._consume_retry_budget(endpoint):
                        stats.retry_budget_exhaustions += 1
                        msg = f"Retry budget exhausted for {endpoint}"
                        raise RuntimeError(msg)
                    stats.retries += 1
                return await self._fetch_once(spec, endpoint, host)
        msg = f"Exhausted retries for {endpoint}"
        raise RuntimeError(msg)

    async def close(self) -> None:
        await self.client.aclose()

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]:
        return self.hardening.telemetry_snapshot()

    def telemetry_frame(self) -> pl.DataFrame:
        return self.hardening.telemetry_frame()

    async def _fetch_once(
        self, spec: RequestSpec, endpoint: str, host: str
    ) -> dict[str, Any]:
        stats = self.hardening.telemetry.endpoints.setdefault(endpoint, EndpointStats())
        stats.requests += 1
        started = perf_counter()
        try:
            waited = await self.hardening.wait_for_host_cooldown(host)
            stats.cooldown_wait_seconds += waited
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
                stats.anti_crawl_hits += 1
                self.hardening.apply_host_cooldown(host)
                raise self.guard.blocked_error(endpoint)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            stats.failures += 1
            raise
        stats.successes += 1
        stats.total_latency_seconds += perf_counter() - started
        return payload

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
