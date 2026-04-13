from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from qore_core import FundInstrument, StockInstrument
from qore_data.fetcher.eastmoney import EastMoneyFetcher


class StubEastMoneyFetcher(EastMoneyFetcher):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(concurrency=1, delay_min=0.0, delay_max=0.0)
        self._responses = responses

    async def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        referer: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, params, referer, extra_headers
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_stock_daily_parses_kline_payload() -> None:
    fetcher = StubEastMoneyFetcher(
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
    fetcher = StubEastMoneyFetcher(
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
    fetcher = StubEastMoneyFetcher(
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
    fetcher = StubEastMoneyFetcher(
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
        ["pe_ttm", "pb", "ps_ttm", "roe", "gross_margin", "total_assets"],
        date(2026, 4, 30),
    )
    assert result.get_column("pe_ttm").to_list() == [15.0]
    assert result.get_column("pb").to_list() == [2.5]
    assert result.get_column("roe").to_list() == [0.15]
    assert result.get_column("gross_margin").to_list() == [0.375]
    assert result.get_column("total_assets").to_list() == [1000.0]
