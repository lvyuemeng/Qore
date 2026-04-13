from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_core import DerivativeInstrument, FundInstrument, StockInstrument
from qore_data.fetch import fetch_daily, fetch_minute


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
