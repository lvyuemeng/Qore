from __future__ import annotations

import pytest
from qore_data.fetcher.http import (
    EndpointStats,
    HeaderProfile,
    RequestHardening,
    RequestTelemetry,
)


def test_request_hardening_headers_are_generic() -> None:
    hardening = RequestHardening(
        telemetry=RequestTelemetry(),
        header_profiles=(
            HeaderProfile(
                user_agent="Mozilla/5.0 Example",
                accept_language="zh-CN,zh;q=0.9",
                cache_control="no-cache",
                pragma="no-cache",
            ),
        ),
        cooldown_min=1.0,
        cooldown_max=2.0,
    )

    headers = hardening.headers_for(
        "https://api.example.com/data",
        {"Referer": "https://example.com/page"},
    )

    assert headers["Origin"] == "https://api.example.com"
    assert headers["Referer"] == "https://example.com/page"
    assert headers["User-Agent"] == "Mozilla/5.0 Example"


@pytest.mark.asyncio
async def test_request_hardening_host_cooldown_waits() -> None:
    hardening = RequestHardening(
        telemetry=RequestTelemetry(),
        header_profiles=(
            HeaderProfile(
                user_agent="Mozilla/5.0 Example",
                accept_language="zh-CN,zh;q=0.9",
                cache_control="no-cache",
                pragma="no-cache",
            ),
        ),
        cooldown_min=0.0,
        cooldown_max=0.0,
    )

    applied = hardening.apply_host_cooldown("api.example.com")
    waited = await hardening.wait_for_host_cooldown("api.example.com")

    assert applied == 0.0
    assert waited >= 0.0


def test_request_telemetry_frame_is_generic() -> None:
    telemetry = RequestTelemetry(
        endpoints={
            "sample": EndpointStats(
                requests=2,
                successes=1,
                failures=1,
                retries=1,
                retry_budget_exhaustions=0,
                anti_crawl_hits=1,
                cooldown_wait_seconds=0.5,
                total_latency_seconds=0.25,
            )
        }
    )

    frame = telemetry.frame()

    assert frame.get_column("endpoint").to_list() == ["sample"]
    assert frame.get_column("anti_crawl_hits").to_list() == [1]
    assert frame.get_column("avg_latency_seconds").to_list() == [0.25]
