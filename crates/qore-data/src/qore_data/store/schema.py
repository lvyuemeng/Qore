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
    "fund_holdings": Dataset(
        name="fund_holdings",
        schema=pa.schema(
            [
                ("report_date", pa.date32()),
                ("symbol", pa.string()),
                ("stock_symbol", pa.string()),
                ("stock_name", pa.string()),
                ("shares", pa.float64()),
                ("market_value", pa.float64()),
                ("total_share_ratio", pa.float64()),
                ("float_share_ratio", pa.float64()),
            ]
        ),
        partition_cols=["symbol"],
        dedup_keys=["report_date", "symbol", "stock_symbol"],
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
                ("roe", pa.float64()),
                ("roa", pa.float64()),
                ("gross_margin", pa.float64()),
                ("net_margin", pa.float64()),
                ("eps_ttm", pa.float64()),
                ("revenue", pa.float64()),
                ("net_income", pa.float64()),
                ("total_shares", pa.float64()),
                ("float_shares", pa.float64()),
                ("equity_yoy", pa.float64()),
                ("total_asset_yoy", pa.float64()),
                ("net_profit_yoy", pa.float64()),
                ("eps_basic_yoy", pa.float64()),
                ("net_profit_parent_yoy", pa.float64()),
                ("current_ratio", pa.float64()),
                ("quick_ratio", pa.float64()),
                ("cash_ratio", pa.float64()),
                ("total_debt_yoy", pa.float64()),
                ("debts_to_assets", pa.float64()),
                ("assets_to_equity", pa.float64()),
                ("current_assets_to_total_asset", pa.float64()),
                ("non_current_assets_to_total_asset", pa.float64()),
                ("tangible_assets_to_total_asset", pa.float64()),
                ("ebit_to_interest", pa.float64()),
                ("cfo_to_revenue", pa.float64()),
                ("cfo_to_net_profit", pa.float64()),
                ("receivable_turnover", pa.float64()),
                ("receivable_turnover_days", pa.float64()),
                ("inventory_turnover", pa.float64()),
                ("inventory_turnover_days", pa.float64()),
                ("current_assets_turnover", pa.float64()),
                ("total_asset_turnover", pa.float64()),
                ("parent_profit_ratio", pa.float64()),
                ("tax_burden", pa.float64()),
                ("interest_burden", pa.float64()),
                ("ebit_margin", pa.float64()),
                ("total_liabilities", pa.float64()),
                ("total_assets", pa.float64()),
                ("operating_cashflow", pa.float64()),
                ("total_market_cap", pa.float64()),
                ("float_market_cap", pa.float64()),
                ("is_st", pa.bool_()),
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
            ]
        ),
        partition_cols=["index_symbol"],
        dedup_keys=["as_of", "index_symbol", "symbol"],
    ),
    "stock_info": Dataset(
        name="stock_info",
        schema=pa.schema(
            [
                ("symbol", pa.string()),
                ("short_name", pa.string()),
                ("exchange", pa.string()),
                ("industry", pa.string()),
                ("board", pa.string()),
                ("listing_date", pa.date32()),
            ]
        ),
        partition_cols=[],
        dedup_keys=["symbol"],
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
    "stock_audit_opinions": Dataset(
        name="stock_audit_opinions",
        schema=pa.schema(
            [
                ("symbol", pa.string()),
                ("report_date", pa.date32()),
                ("announce_date", pa.date32()),
                ("opinion", pa.string()),
                ("opinion_code", pa.string()),
                ("source_notice_type", pa.string()),
                ("title", pa.string()),
                ("art_code", pa.string()),
                ("url", pa.string()),
            ]
        ),
        partition_cols=["symbol"],
        dedup_keys=["symbol", "art_code"],
    ),
}
