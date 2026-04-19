from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    schema: pa.Schema
    partition_cols: list[str]
    dedup_keys: list[str]


DATASETS: dict[str, Dataset] = {
    "stock_ohlcv": Dataset(
        name="stock_ohlcv",
        schema=pa.schema(
            [
                ("date", pa.date32()),
                ("symbol", pa.string()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("volume", pa.int64()),
                ("amount", pa.float64()),
                ("adj_factor", pa.float64()),
                ("is_suspended", pa.bool_()),
                ("limit_up", pa.bool_()),
                ("limit_down", pa.bool_()),
            ]
        ),
        partition_cols=["year", "symbol"],
        dedup_keys=["date", "symbol"],
    ),
    "fund_nav": Dataset(
        name="fund_nav",
        schema=pa.schema(
            [
                ("date", pa.date32()),
                ("symbol", pa.string()),
                ("nav", pa.float64()),
                ("acc_nav", pa.float64()),
                ("daily_return", pa.float64()),
            ]
        ),
        partition_cols=["year", "symbol"],
        dedup_keys=["date", "symbol"],
    ),
    "derivative_ohlcv": Dataset(
        name="derivative_ohlcv",
        schema=pa.schema(
            [
                ("date", pa.date32()),
                ("symbol", pa.string()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("volume", pa.int64()),
                ("amount", pa.float64()),
                ("open_interest", pa.int64()),
                ("settle_price", pa.float64()),
            ]
        ),
        partition_cols=["year", "symbol"],
        dedup_keys=["date", "symbol"],
    ),
    "fundamentals": Dataset(
        name="fundamentals",
        schema=pa.schema(
            [
                ("report_date", pa.date32()),
                ("announce_date", pa.date32()),
                ("symbol", pa.string()),
                ("pe_ttm", pa.float64()),
                ("pb", pa.float64()),
                ("ps_ttm", pa.float64()),
                ("ev_ebitda", pa.float64()),
                ("roe", pa.float64()),
                ("roa", pa.float64()),
                ("gross_margin", pa.float64()),
                ("revenue", pa.float64()),
                ("net_income", pa.float64()),
                ("total_assets", pa.float64()),
                ("operating_cashflow", pa.float64()),
            ]
        ),
        partition_cols=["year", "symbol"],
        dedup_keys=["announce_date", "symbol", "report_date"],
    ),
    "index_constituents": Dataset(
        name="index_constituents",
        schema=pa.schema(
            [
                ("as_of", pa.date32()),
                ("index_symbol", pa.string()),
                ("symbol", pa.string()),
                ("exchange", pa.string()),
                ("industry", pa.string()),
            ]
        ),
        partition_cols=["index_symbol"],
        dedup_keys=["as_of", "index_symbol", "symbol"],
    ),
    "stock_profiles": Dataset(
        name="stock_profiles",
        schema=pa.schema(
            [
                ("as_of", pa.date32()),
                ("symbol", pa.string()),
                ("short_name", pa.string()),
                ("exchange", pa.string()),
                ("industry", pa.string()),
                ("board", pa.string()),
                ("listing_date", pa.date32()),
                ("total_market_cap", pa.float64()),
                ("float_market_cap", pa.float64()),
                ("total_shares", pa.float64()),
                ("float_shares", pa.float64()),
                ("is_st", pa.bool_()),
            ]
        ),
        partition_cols=["symbol"],
        dedup_keys=["as_of", "symbol"],
    ),
    "analyst_forecasts": Dataset(
        name="analyst_forecasts",
        schema=pa.schema(
            [
                ("as_of", pa.date32()),
                ("symbol", pa.string()),
                ("report_count", pa.int64()),
                ("buy", pa.int64()),
                ("overweight", pa.int64()),
                ("neutral", pa.int64()),
                ("underweight", pa.int64()),
                ("sell", pa.int64()),
                ("eps_year1", pa.float64()),
                ("eps_year2", pa.float64()),
                ("eps_year3", pa.float64()),
                ("eps_year4", pa.float64()),
            ]
        ),
        partition_cols=["symbol"],
        dedup_keys=["as_of", "symbol"],
    ),
    "announcements": Dataset(
        name="announcements",
        schema=pa.schema(
            [
                ("symbol", pa.string()),
                ("short_name", pa.string()),
                ("title", pa.string()),
                ("notice_type", pa.string()),
                ("notice_date", pa.date32()),
                ("art_code", pa.string()),
                ("url", pa.string()),
            ]
        ),
        partition_cols=["symbol"],
        dedup_keys=["symbol", "art_code"],
    ),
    "factor_scores": Dataset(
        name="factor_scores",
        schema=pa.schema(
            [
                ("date", pa.date32()),
                ("symbol", pa.string()),
                ("factor_name", pa.string()),
                ("raw_value", pa.float64()),
                ("z_score", pa.float64()),
                ("rank_pct", pa.float64()),
            ]
        ),
        partition_cols=["date_month", "factor_name"],
        dedup_keys=["date", "symbol", "factor_name"],
    ),
    "news_scores": Dataset(
        name="news_scores",
        schema=pa.schema(
            [
                ("date", pa.date32()),
                ("symbol", pa.string()),
                ("score", pa.float64()),
                ("event_type", pa.string()),
                ("source_layer", pa.string()),
            ]
        ),
        partition_cols=["date_month"],
        dedup_keys=["date", "symbol"],
    ),
}
