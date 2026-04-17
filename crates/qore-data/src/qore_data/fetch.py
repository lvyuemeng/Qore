from __future__ import annotations

from datetime import date
from functools import singledispatch

import polars as pl
from qore_core.instrument import (
    DerivativeInstrument,
    FundInstrument,
    Instrument,
    StockInstrument,
)

from qore_data.source import DerivativeSource, FundSource, StockSource


@singledispatch
async def fetch_daily(
    inst: Instrument,
    start: date,
    end: date,
    source: object,
) -> pl.DataFrame:
    raise TypeError(f"No daily fetch registered for {type(inst).__name__}")


@fetch_daily.register(StockInstrument)
async def _fetch_stock_daily(
    inst: StockInstrument,
    start: date,
    end: date,
    source: StockSource,
) -> pl.DataFrame:
    return await source.stock_daily(inst, start, end)


@fetch_daily.register(FundInstrument)
async def _fetch_fund_daily(
    inst: FundInstrument,
    start: date,
    end: date,
    source: FundSource,
) -> pl.DataFrame:
    return await source.fund_nav(inst, start, end)


@fetch_daily.register(DerivativeInstrument)
async def _fetch_derivative_daily(
    inst: DerivativeInstrument,
    start: date,
    end: date,
    source: DerivativeSource,
) -> pl.DataFrame:
    return await source.derivative_daily(inst, start, end)


@singledispatch
async def fetch_minute(
    inst: Instrument,
    start: date,
    end: date,
    source: object,
    freq_minutes: int = 1,
) -> pl.DataFrame:
    del start, end, source, freq_minutes
    raise TypeError(
        f"{type(inst).__name__} does not support sub-daily data. "
        "Only DerivativeInstrument supports fetch_minute()."
    )


@fetch_minute.register(DerivativeInstrument)
async def _fetch_derivative_minute(
    inst: DerivativeInstrument,
    start: date,
    end: date,
    source: DerivativeSource,
    freq_minutes: int = 1,
) -> pl.DataFrame:
    return await source.derivative_minute(inst, start, end, freq_minutes)


@singledispatch
async def fetch_tick(
    inst: Instrument,
    trading_date: date,
    source: object,
) -> pl.DataFrame:
    del trading_date, source
    raise TypeError(f"{type(inst).__name__} does not support tick data.")


@fetch_tick.register(DerivativeInstrument)
async def _fetch_derivative_tick(
    inst: DerivativeInstrument,
    trading_date: date,
    source: DerivativeSource,
) -> pl.DataFrame:
    return await source.derivative_tick(inst, trading_date)


@singledispatch
async def fetch_fundamentals(
    inst: Instrument,
    fields: list[str],
    as_of: date,
    source: object,
) -> pl.DataFrame:
    del fields, as_of, source
    raise TypeError(f"Fundamentals not available for {type(inst).__name__}.")


@fetch_fundamentals.register(StockInstrument)
async def _fetch_stock_fundamentals(
    inst: StockInstrument,
    fields: list[str],
    as_of: date,
    source: StockSource,
) -> pl.DataFrame:
    return await source.fundamentals(inst, fields, as_of)


@singledispatch
async def fetch_announcements(
    inst: Instrument,
    start: date,
    end: date,
    source: object,
) -> pl.DataFrame:
    del start, end, source
    raise TypeError(f"Announcements not available for {type(inst).__name__}.")


@fetch_announcements.register(StockInstrument)
async def _fetch_stock_announcements(
    inst: StockInstrument,
    start: date,
    end: date,
    source: StockSource,
) -> pl.DataFrame:
    return await source.announcements(inst, start, end)
