from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import polars as pl
import pytest
from qore_data import DataSettings, FundInstrument, StockInstrument
from qore_data.fetch import (
    fetch_analyst_forecast,
    fetch_announcements,
    fetch_audit_opinions,
    fetch_profile,
)
from qore_data.fetcher.eastmoney import EastMoneyFetcher
from qore_data.fetcher.http import (
    EndpointStats,
    HardenedJsonFetcher,
    JsonFetcher,
    RequestSpec,
    TelemetryReadable,
)


class StubJsonFetcher(JsonFetcher, TelemetryReadable):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses

    async def fetch_json(self, spec: RequestSpec) -> dict[str, Any]:
        del spec
        return self._responses.pop(0)

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {}

    def telemetry_frame(self) -> pl.DataFrame:
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


class RetryJsonFetcher(JsonFetcher, TelemetryReadable):
    def __init__(self) -> None:
        self.calls = 0
        self.retry_budget = 1
        self.telemetry: dict[str, EndpointStats] = {}

    async def fetch_json(self, spec: RequestSpec) -> dict[str, Any]:
        endpoint = spec.endpoint
        stats = self.telemetry.setdefault(endpoint, EndpointStats())
        self.calls += 1
        stats.requests += 1
        stats.failures += 1
        if self.retry_budget <= 0:
            stats.retry_budget_exhaustions += 1
            raise RuntimeError(f"Retry budget exhausted for {endpoint}")
        stats.retries += 1
        self.calls += 1
        stats.requests += 1
        stats.successes += 1
        return {"data": {"klines": []}}

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "requests": stats.requests,
                "successes": stats.successes,
                "failures": stats.failures,
                "retries": stats.retries,
                "retry_budget_exhaustions": stats.retry_budget_exhaustions,
                "anti_crawl_hits": stats.anti_crawl_hits,
                "cooldown_wait_seconds": stats.cooldown_wait_seconds,
                "total_latency_seconds": stats.total_latency_seconds,
                "avg_latency_seconds": (
                    stats.total_latency_seconds / stats.successes
                    if stats.successes > 0
                    else 0.0
                ),
            }
            for name, stats in self.telemetry.items()
        }

    def telemetry_frame(self) -> pl.DataFrame:
        if not self.telemetry:
            return StubJsonFetcher([]).telemetry_frame()
        return pl.DataFrame(
            [
                {"endpoint": name, **values}
                for name, values in self.telemetry_snapshot().items()
            ]
        )


def make_stub_fetcher(responses: list[dict[str, Any]]) -> EastMoneyFetcher:
    return EastMoneyFetcher(StubJsonFetcher(responses))


def make_retry_fetcher() -> tuple[EastMoneyFetcher, RetryJsonFetcher]:
    json_fetcher = RetryJsonFetcher()
    return EastMoneyFetcher(json_fetcher), json_fetcher


@pytest.mark.asyncio
async def test_stock_daily_parses_kline_payload() -> None:
    fetcher = make_stub_fetcher(
        [{"data": {"klines": ["2026-04-10,10,11,12,9,1000,2000,1.0,10.0,1.0,5.0,1.0"]}}]
    )
    result = await fetcher.stock_daily(
        StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
        date(2026, 4, 1),
        date(2026, 4, 30),
    )
    assert result.get_column("symbol").to_list() == ["600519.SH"]
    assert result.get_column("limit_up").to_list() == [True]


@pytest.mark.asyncio
async def test_fund_nav_parses_lsjz_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "TotalCount": 1,
                "Data": {
                    "LSJZList": [
                        {
                            "FSRQ": "2026-04-10",
                            "DWJZ": "1.234",
                            "LJJZ": "2.345",
                            "JZZZL": "1.50",
                        }
                    ]
                },
            }
        ]
    )
    result = await fetcher.fund_nav(
        FundInstrument(symbol="110022", fund_type="active"),
        date(2026, 4, 1),
        date(2026, 4, 30),
    )
    assert result.get_column("nav").to_list() == [1.234]
    assert result.get_column("daily_return").to_list() == [0.015]


@pytest.mark.asyncio
async def test_fund_holdings_parse_position_detail_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "result": {
                    "pages": 1,
                    "data": [
                        {
                            "SECURITY_CODE": "600519",
                            "SECURITY_NAME_ABBR": "贵州茅台",
                            "HOLD_SHARES": "12345",
                            "HOLD_MV": "67890.5",
                            "TOTAL_SHARES_RATIO": "1.25",
                            "FREE_SHARES_RATIO": "2.50",
                        }
                    ],
                }
            }
        ]
    )
    result = await fetcher.fund_holdings(
        FundInstrument(symbol="008286", fund_type="active"),
        date(2026, 3, 31),
    )
    assert result.get_column("symbol").to_list() == ["008286"]
    assert result.get_column("stock_symbol").to_list() == ["600519"]
    assert result.get_column("market_value").to_list() == [67890.5]
    assert result.get_column("float_share_ratio").to_list() == [2.5]


@pytest.mark.asyncio
async def test_fundamentals_parse_value_analysis_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "result": {
                    "data": [
                        {
                            "TRADE_DATE": "2026-04-10",
                            "PE_TTM": "15.0",
                            "PB_MRQ": "2.5",
                            "PS_TTM": "3.5",
                        }
                    ]
                }
            },
            {
                "result": {
                    "pages": 1,
                    "data": [
                        {
                            "SECURITY_CODE": "600519",
                            "NOTICE_DATE": "2026-04-15",
                            "TOTAL_LIABILITIES": "350.0",
                            "TOTAL_ASSETS": "1000.0",
                            "ROE_WEIGHTED": "0.15",
                        }
                    ],
                }
            },
            {
                "result": {
                    "pages": 1,
                    "data": [
                        {
                            "SECURITY_CODE": "600519",
                            "NOTICE_DATE": "2026-04-15",
                            "NETPROFIT": "100.0",
                            "TOTAL_OPERATE_INCOME": "400.0",
                            "OPERATE_COST": "250.0",
                        }
                    ],
                }
            },
            {
                "result": {
                    "pages": 1,
                    "data": [
                        {
                            "SECURITY_CODE": "600519",
                            "NOTICE_DATE": "2026-04-15",
                            "NETCASH_OPERATE": "120.0",
                        }
                    ],
                }
            },
        ]
    )
    result = await fetcher.fundamentals(
        StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
        [
            "pe_ttm",
            "pb",
            "ps_ttm",
            "roe",
            "gross_margin",
            "total_liabilities",
            "total_assets",
        ],
        date(2026, 4, 30),
    )
    assert result.get_column("pe_ttm").to_list() == [15.0]
    assert result.get_column("pb").to_list() == [2.5]
    assert result.get_column("roe").to_list() == [0.15]
    assert result.get_column("gross_margin").to_list() == [0.375]
    assert result.get_column("total_liabilities").to_list() == [350.0]
    assert result.get_column("total_assets").to_list() == [1000.0]


@pytest.mark.asyncio
async def test_analyst_forecast_parses_profit_prediction_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "result": {
                    "data": [
                        {
                            "SECURITY_CODE": "600519",
                            "RATING_ORG_NUM": "12",
                            "BUY": "5",
                            "HOLD": "4",
                            "NEUTRAL": "2",
                            "SELL": "1",
                            "STRONG_SELL": "0",
                            "EPS1": "3.21",
                            "EPS2": "3.55",
                            "EPS3": "3.88",
                            "EPS4": "4.12",
                        }
                    ]
                }
            }
        ]
    )
    result = await fetcher.analyst_forecast(
        StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
        date(2026, 4, 30),
    )

    assert result.get_column("symbol").to_list() == ["600519.SH"]
    assert result.get_column("as_of").to_list() == [date(2026, 4, 30)]
    assert result.get_column("report_count").to_list() == [12]
    assert result.get_column("buy").to_list() == [5]
    assert result.get_column("overweight").to_list() == [4]
    assert result.get_column("eps_year1").to_list() == [3.21]
    assert result.get_column("eps_year4").to_list() == [4.12]


@pytest.mark.asyncio
async def test_announcements_parse_notice_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "data": {
                    "total_hits": 1,
                    "list": [
                        {
                            "art_code": "AN202604171234567890",
                            "notice_date": "2026-04-17",
                            "title": "2025年年度报告",
                            "columns": [{"column_name": "财务报告"}],
                            "codes": [
                                {
                                    "stock_code": "600519",
                                    "short_name": "贵州茅台",
                                    "ann_type": "A",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    )
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetcher.announcements(inst, date(2026, 4, 1), date(2026, 4, 30))

    assert result.get_column("symbol").to_list() == ["600519.SH"]
    assert result.get_column("notice_type").to_list() == ["财务报告"]
    assert result.get_column("title").to_list() == ["2025年年度报告"]


@pytest.mark.asyncio
async def test_audit_opinions_parse_announcement_derived_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "data": {
                    "total_hits": 2,
                    "list": [
                        {
                            "art_code": "AN202604170001",
                            "notice_date": "2026-04-17",
                            "title": "2025年年度审计报告(否定意见)",
                            "columns": [{"column_name": "财务报告"}],
                            "codes": [
                                {
                                    "stock_code": "600519",
                                    "short_name": "贵州茅台",
                                    "ann_type": "A",
                                }
                            ],
                        },
                        {
                            "art_code": "AN202604170002",
                            "notice_date": "2026-04-17",
                            "title": "2025年年度报告摘要",
                            "columns": [{"column_name": "财务报告"}],
                            "codes": [
                                {
                                    "stock_code": "600519",
                                    "short_name": "贵州茅台",
                                    "ann_type": "A",
                                }
                            ],
                        },
                    ],
                }
            }
        ]
    )
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetcher.audit_opinions(inst, date(2026, 4, 1), date(2026, 4, 30))

    assert result.get_column("symbol").to_list() == ["600519.SH"]
    assert result.get_column("report_date").to_list() == [date(2025, 12, 31)]
    assert result.get_column("announce_date").to_list() == [date(2026, 4, 17)]
    assert result.get_column("opinion").to_list() == ["否定意见"]
    assert result.get_column("opinion_code").to_list() == ["adverse"]


@pytest.mark.asyncio
async def test_stock_profile_parses_metadata_payload() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "data": {
                    "f57": "688001",
                    "f58": "*ST样本",
                    "f84": "1000000",
                    "f85": "800000",
                    "f116": "123456789",
                    "f117": "100000000",
                    "f127": "半导体",
                    "f189": "20190722",
                }
            }
        ]
    )
    inst = StockInstrument(symbol="688001.SH", exchange="SH", industry="unknown")
    result = await fetcher.stock_profile(inst, date(2026, 4, 30))

    assert result.get_column("symbol").to_list() == ["688001.SH"]
    assert result.get_column("board").to_list() == ["STAR"]
    assert result.get_column("industry").to_list() == ["半导体"]
    assert result.get_column("is_st").to_list() == [True]


@pytest.mark.asyncio
async def test_fetch_profile_dispatches_for_stock() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "data": {
                    "f57": "600519",
                    "f58": "贵州茅台",
                    "f84": "1256197800",
                    "f85": "1256197800",
                    "f116": "2100000000000",
                    "f117": "2100000000000",
                    "f127": "酿酒行业",
                    "f189": "20010827",
                }
            }
        ]
    )
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetch_profile(inst, date(2026, 4, 30), fetcher)

    assert result.height == 1
    assert result.get_column("short_name").to_list() == ["贵州茅台"]


@pytest.mark.asyncio
async def test_fetch_analyst_forecast_dispatches_for_stock() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "result": {
                    "data": [
                        {
                            "SECURITY_CODE": "600519",
                            "RATING_ORG_NUM": "12",
                            "BUY": "5",
                            "HOLD": "4",
                            "NEUTRAL": "2",
                            "SELL": "1",
                            "STRONG_SELL": "0",
                            "EPS1": "3.21",
                            "EPS2": "3.55",
                            "EPS3": "3.88",
                            "EPS4": "4.12",
                        }
                    ]
                }
            }
        ]
    )
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetch_analyst_forecast(inst, date(2026, 4, 30), fetcher)

    assert result.height == 1
    assert result.get_column("report_count").to_list() == [12]


@pytest.mark.asyncio
async def test_fetch_announcements_dispatches_for_stock() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "data": {
                    "total_hits": 1,
                    "list": [
                        {
                            "art_code": "AN202604171234567890",
                            "notice_date": "2026-04-17",
                            "title": "2025年年度报告",
                            "columns": [{"column_name": "财务报告"}],
                            "codes": [
                                {
                                    "stock_code": "600519",
                                    "short_name": "贵州茅台",
                                    "ann_type": "A",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    )
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetch_announcements(
        inst, date(2026, 4, 1), date(2026, 4, 30), fetcher
    )
    assert result.height == 1


@pytest.mark.asyncio
async def test_fetch_audit_opinions_dispatches_for_stock() -> None:
    fetcher = make_stub_fetcher(
        [
            {
                "data": {
                    "total_hits": 1,
                    "list": [
                        {
                            "art_code": "AN202604170001",
                            "notice_date": "2026-04-17",
                            "title": "2025年年度审计报告(无法表示意见)",
                            "columns": [{"column_name": "财务报告"}],
                            "codes": [
                                {
                                    "stock_code": "600519",
                                    "short_name": "贵州茅台",
                                    "ann_type": "A",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    )
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetch_audit_opinions(
        inst,
        date(2026, 4, 1),
        date(2026, 4, 30),
        fetcher,
    )

    assert result.height == 1
    assert result.get_column("opinion_code").to_list() == ["disclaimer"]


@pytest.mark.asyncio
async def test_request_retry_updates_telemetry() -> None:
    fetcher, json_fetcher = make_retry_fetcher()

    result = await fetcher.stock_daily(
        StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
        date(2026, 4, 1),
        date(2026, 4, 30),
    )

    telemetry = fetcher.telemetry_snapshot()
    assert result.is_empty()
    assert json_fetcher.calls == 2
    assert telemetry["stock_daily"]["requests"] == 2
    assert telemetry["stock_daily"]["failures"] == 1
    assert telemetry["stock_daily"]["successes"] == 1
    assert telemetry["stock_daily"]["retries"] == 1


@pytest.mark.asyncio
async def test_retry_budget_exhaustion_stops_additional_retries() -> None:
    fetcher, json_fetcher = make_retry_fetcher()
    json_fetcher.retry_budget = 0

    with pytest.raises(RuntimeError, match="Retry budget exhausted"):
        await fetcher.stock_daily(
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            date(2026, 4, 1),
            date(2026, 4, 30),
        )

    telemetry = fetcher.telemetry_snapshot()
    assert telemetry["stock_daily"]["retry_budget_exhaustions"] == 1


def test_telemetry_frame_contains_endpoint_rows() -> None:
    fetcher, json_fetcher = make_retry_fetcher()
    json_fetcher.telemetry["stock_daily"] = EndpointStats(
        requests=2,
        successes=1,
        failures=1,
        retries=1,
        retry_budget_exhaustions=0,
        total_latency_seconds=0.25,
    )

    frame = fetcher.telemetry_frame()

    assert frame.get_column("endpoint").to_list() == ["stock_daily"]
    assert frame.get_column("avg_latency_seconds").to_list() == [0.25]


def test_anti_crawl_detection_handles_status_and_body_markers() -> None:
    fetcher, _ = make_retry_fetcher()
    blocked_status = httpx.Response(
        429,
        request=httpx.Request("GET", "https://push2.eastmoney.com/api/qt/stock/get"),
    )
    blocked_body = httpx.Response(
        200,
        text="访问过于频繁, 请稍后再试",
        request=httpx.Request("GET", "https://push2.eastmoney.com/api/qt/stock/get"),
    )

    assert fetcher._is_anti_crawl_response(blocked_status)
    assert fetcher._is_anti_crawl_response(blocked_body)


def test_request_headers_include_rotating_profile_and_origin() -> None:
    fetcher = EastMoneyFetcher.from_settings(DataSettings())

    assert isinstance(fetcher._json_fetcher, HardenedJsonFetcher)

    headers = fetcher._json_fetcher.hardening.headers_for(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        {"Referer": "https://quote.eastmoney.com/600519.html"},
    )

    assert "Mozilla/5.0" in headers["User-Agent"]
    assert headers["Origin"] == "https://push2his.eastmoney.com"
    assert headers["Referer"] == "https://quote.eastmoney.com/600519.html"


def test_from_settings_applies_eastmoney_hardening_settings() -> None:
    fetcher = EastMoneyFetcher.from_settings(
        DataSettings(
            eastmoney_concurrency=3,
            eastmoney_delay_min=0.1,
            eastmoney_delay_max=0.2,
            eastmoney_timeout=9.0,
            eastmoney_max_retries=4,
            eastmoney_retry_budget=11,
            eastmoney_cooldown_min=2.0,
            eastmoney_cooldown_max=7.0,
            eastmoney_retry_backoff_min=0.25,
            eastmoney_retry_backoff_max=1.25,
        )
    )

    assert isinstance(fetcher._json_fetcher, HardenedJsonFetcher)
    assert fetcher._json_fetcher.client.timeout.connect == 9.0
    assert fetcher._json_fetcher.policy.max_retries == 4
    assert fetcher._json_fetcher.policy.retry_budget == 11
    assert fetcher._json_fetcher.hardening.cooldown_min == 2.0
    assert fetcher._json_fetcher.hardening.cooldown_max == 7.0
    assert fetcher._json_fetcher.policy.retry_backoff_min == 0.25
    assert fetcher._json_fetcher.policy.retry_backoff_max == 1.25
