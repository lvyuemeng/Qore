from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl
from qore_core.instrument import DerivativeInstrument, FundInstrument, StockInstrument


class StockSource(Protocol):
    async def stock_daily(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame: ...

    async def fundamentals(
        self,
        inst: StockInstrument,
        fields: list[str],
        as_of: date,
    ) -> pl.DataFrame: ...

    async def index_constituents(
        self,
        index_symbol: str,
        as_of: date,
    ) -> list[StockInstrument]: ...

    async def stock_profile(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame: ...

    async def analyst_forecast(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame: ...

    async def announcements(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame: ...

    async def audit_opinions(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame: ...


class FundSource(Protocol):
    async def fund_nav(
        self,
        inst: FundInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame: ...

    async def fund_holdings(
        self,
        inst: FundInstrument,
        report_date: date,
    ) -> pl.DataFrame: ...


class DerivativeSource(Protocol):
    async def derivative_daily(
        self,
        inst: DerivativeInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame: ...

    async def derivative_minute(
        self,
        inst: DerivativeInstrument,
        start: date,
        end: date,
        freq_minutes: int = 1,
    ) -> pl.DataFrame: ...

    async def derivative_tick(
        self,
        inst: DerivativeInstrument,
        trading_date: date,
    ) -> pl.DataFrame: ...
