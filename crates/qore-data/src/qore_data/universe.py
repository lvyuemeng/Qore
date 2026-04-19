from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl
from qore_core.instrument import StockInstrument
from qore_core.universe import Universe

from qore_data.source import StockSource
from qore_data.store.duckdb import QoreStore


async def snapshot_index_constituents(
    source: StockSource,
    store: QoreStore,
    *,
    index_symbol: str,
    as_of: date,
) -> pl.DataFrame:
    instruments = await source.index_constituents(index_symbol, as_of)
    frame = _index_constituents_frame(
        instruments, index_symbol=index_symbol, as_of=as_of
    )
    store.write("index_constituents", frame)
    return frame


async def snapshot_stock_profiles(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    as_of: date,
) -> pl.DataFrame:
    frames = [await source.stock_profile(inst, as_of) for inst in instruments]
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return pl.DataFrame(
            schema={
                "as_of": pl.Date,
                "symbol": pl.String,
                "short_name": pl.String,
                "exchange": pl.String,
                "industry": pl.String,
                "board": pl.String,
                "listing_date": pl.Date,
                "total_market_cap": pl.Float64,
                "float_market_cap": pl.Float64,
                "total_shares": pl.Float64,
                "float_shares": pl.Float64,
                "is_st": pl.Boolean,
            }
        )
    combined = pl.concat(non_empty, how="vertical")
    store.write("stock_profiles", combined)
    return combined


async def snapshot_stock_analyst_forecasts(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    as_of: date,
) -> pl.DataFrame:
    frames = [await source.analyst_forecast(inst, as_of) for inst in instruments]
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return pl.DataFrame(
            schema={
                "as_of": pl.Date,
                "symbol": pl.String,
                "report_count": pl.Int64,
                "buy": pl.Int64,
                "overweight": pl.Int64,
                "neutral": pl.Int64,
                "underweight": pl.Int64,
                "sell": pl.Int64,
                "eps_year1": pl.Float64,
                "eps_year2": pl.Float64,
                "eps_year3": pl.Float64,
                "eps_year4": pl.Float64,
            }
        )
    combined = pl.concat(non_empty, how="vertical")
    store.write("analyst_forecasts", combined)
    return combined


async def snapshot_stock_announcements(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    start: date,
    end: date,
) -> pl.DataFrame:
    frames = [await source.announcements(inst, start, end) for inst in instruments]
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "short_name": pl.String,
                "title": pl.String,
                "notice_type": pl.String,
                "notice_date": pl.Date,
                "art_code": pl.String,
                "url": pl.String,
            }
        )
    combined = pl.concat(non_empty, how="vertical")
    store.write("announcements", combined)
    return combined


async def build_stock_universe_from_index(
    source: StockSource,
    store: QoreStore,
    *,
    index_symbol: str,
    as_of: date,
) -> Universe:
    instruments = await source.index_constituents(index_symbol, as_of)
    snapshot_frame = _index_constituents_frame(
        instruments,
        index_symbol=index_symbol,
        as_of=as_of,
    )
    store.write("index_constituents", snapshot_frame)
    profiles = await snapshot_stock_profiles(
        source,
        store,
        instruments=instruments,
        as_of=as_of,
    )
    profile_map = {row["symbol"]: row for row in profiles.to_dicts()}
    enriched = [
        StockInstrument(
            symbol=inst.symbol,
            exchange=inst.exchange,
            industry=str(
                profile_map.get(inst.symbol, {}).get("industry") or inst.industry
            ),
            price_limit_pct=inst.price_limit_pct,
            session=inst.session,
        )
        for inst in instruments
    ]
    return Universe(enriched)


def evaluate_stock_categories(
    store: QoreStore,
    *,
    index_symbol: str,
    as_of: date,
    start: date,
    end: date,
) -> pl.DataFrame:
    constituents = store.read(
        "index_constituents",
        filters={"index_symbol": index_symbol, "as_of": as_of},
    )
    profiles = store.read("stock_profiles", filters={"as_of": as_of})
    forecasts = store.read("analyst_forecasts", filters={"as_of": as_of})
    announcement_counts = (
        store.read("announcements")
        .filter(pl.col("notice_date").is_between(start, end))
        .group_by("symbol")
        .agg(pl.len().alias("announcement_count"))
    )
    return (
        constituents.join(profiles, on=["symbol", "as_of"], how="left")
        .join(forecasts, on=["symbol", "as_of"], how="left")
        .join(announcement_counts, on="symbol", how="left")
        .with_columns(pl.col("announcement_count").fill_null(0))
        .group_by("industry", "board")
        .agg(
            pl.len().alias("symbol_count"),
            pl.col("total_market_cap").mean().alias("avg_total_market_cap"),
            pl.col("report_count").mean().alias("avg_report_count"),
            pl.col("announcement_count").sum().alias("announcement_count"),
        )
        .sort("industry")
        .collect()
    )


def _index_constituents_frame(
    instruments: Sequence[StockInstrument],
    *,
    index_symbol: str,
    as_of: date,
) -> pl.DataFrame:
    rows = [
        {
            "as_of": as_of,
            "index_symbol": index_symbol,
            "symbol": inst.symbol,
            "exchange": inst.exchange,
            "industry": inst.industry,
        }
        for inst in instruments
    ]
    return pl.DataFrame(rows)
