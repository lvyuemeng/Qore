from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_data import DerivativeInstrument, FundInstrument, StockInstrument
from qore_data.fetch import (
    fetch_audit_opinions,
    fetch_daily,
    fetch_minute,
    fetch_profile,
)


class StubFundSource:
    async def fund_nav(
        self,
        inst: FundInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": [inst.symbol],
                "start": [start],
                "end": [end],
            }
        )


class StubStockSource:
    async def stock_profile(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": [inst.symbol],
                "as_of": [as_of],
            }
        )

    async def audit_opinions(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": [inst.symbol],
                "report_date": [date(2025, 12, 31)],
                "announce_date": [end],
                "opinion": ["否定意见"],
                "opinion_code": ["adverse"],
                "source_notice_type": ["财务报告"],
                "title": ["2025年年度审计报告(否定意见)"],
                "art_code": ["AUD-1"],
                "url": [f"https://example.test/{start.isoformat()}"],
            }
        )


class StubDerivativeSource:
    async def derivative_minute(
        self,
        inst: DerivativeInstrument,
        start: date,
        end: date,
        freq_minutes: int = 1,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": [inst.symbol],
                "freq_minutes": [freq_minutes],
                "start": [start],
                "end": [end],
            }
        )


@pytest.mark.asyncio
async def test_fetch_daily_routes_fund_to_fund_nav() -> None:
    inst = FundInstrument(symbol="110022", fund_type="active")
    result = await fetch_daily(
        inst, date(2026, 1, 1), date(2026, 1, 31), StubFundSource()
    )
    assert result.get_column("symbol").to_list() == ["110022"]


@pytest.mark.asyncio
async def test_fetch_minute_rejects_stock() -> None:
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    with pytest.raises(TypeError):
        await fetch_minute(inst, date(2026, 1, 1), date(2026, 1, 31), object())


@pytest.mark.asyncio
async def test_fetch_minute_accepts_derivative() -> None:
    inst = DerivativeInstrument(
        symbol="IF2503",
        exchange="CFFEX",
        underlying="IF",
        derivative_type="futures",
        contract_size=300.0,
        margin_rate=0.12,
    )
    result = await fetch_minute(
        inst,
        date(2026, 1, 1),
        date(2026, 1, 31),
        StubDerivativeSource(),
        freq_minutes=5,
    )
    assert result.get_column("freq_minutes").to_list() == [5]


@pytest.mark.asyncio
async def test_fetch_profile_routes_stock_to_stock_profile() -> None:
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetch_profile(inst, date(2026, 1, 31), StubStockSource())
    assert result.get_column("symbol").to_list() == ["600519.SH"]


@pytest.mark.asyncio
async def test_fetch_audit_opinions_routes_stock_source() -> None:
    inst = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    result = await fetch_audit_opinions(
        inst,
        date(2026, 1, 1),
        date(2026, 4, 30),
        StubStockSource(),
    )

    assert result.get_column("opinion_code").to_list() == ["adverse"]
