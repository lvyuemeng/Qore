"""Financial data fetcher — BaoStock (primary) + Xueqiu (fallback for valuation multiples)."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

import polars as pl

from qore_data.fetcher._base import (
    _exchange_from_stock_code,
    _symbol_digits,
    _to_float,
)
from qore_data.fetcher.concurrent import BatchConfig, batch_fetch
from qore_data.fetcher.xueqiu import (
    _latest_financial_quarter,
    _xq_period,
    _xq_symbol,
    _XueqiuSession,
)

logger = logging.getLogger(__name__)


# ── BaoStock fundamentals worker (pickleable for ProcessPoolExecutor) ────


def _report_date(year: int, quarter: int) -> date:
    if quarter == 1:
        return date(year, 3, 31)
    if quarter == 2:
        return date(year, 6, 30)
    if quarter == 3:
        return date(year, 9, 30)
    return date(year, 12, 31)


def _parse_date_str(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def _pct(v: float | None) -> float | None:
    if v is None:
        return None
    return v / 100.0


def _latest_report_quarter(as_of: date) -> tuple[int, int]:
    effective = as_of - timedelta(days=45)
    m = effective.month
    qe = ((m - 1) // 3) * 3
    if qe == 0:
        return effective.year - 1, 4
    return effective.year, qe // 3 + 1


def _query_single(
    bs, method_name: str, bs_code: str, year: int, quarter: int
) -> dict[str, str]:
    rs = getattr(bs, method_name)(code=bs_code, year=year, quarter=quarter)
    if rs.error_code != "0":
        return {}
    fields = getattr(rs, "fields", None)
    while rs.next():
        row = rs.get_row_data()
        if fields and row:
            return dict(zip(fields, row, strict=False))
    return {}


def _fundamentals_worker(item: tuple[str, date]) -> pl.DataFrame:
    """Pickleable worker: (symbol, as_of) -> DataFrame."""
    symbol, as_of = item
    import baostock as bs

    from qore_data.fetcher._base import _suppress_stdout

    with _suppress_stdout():
        lg = bs.login()
    if lg.error_code != "0":
        return _empty_fundamentals()
    try:
        year, quarter = _latest_report_quarter(as_of)
        code = _symbol_digits(symbol)
        exchange = _exchange_from_stock_code(code)
        bs_code = f"{exchange.lower()}.{code}"

        profit = _query_single(bs, "query_profit_data", bs_code, year, quarter)
        operation = _query_single(bs, "query_operation_data", bs_code, year, quarter)
        growth = _query_single(bs, "query_growth_data", bs_code, year, quarter)
        balance = _query_single(bs, "query_balance_data", bs_code, year, quarter)
        cashflow = _query_single(bs, "query_cash_flow_data", bs_code, year, quarter)
        dupont = _query_single(bs, "query_dupont_data", bs_code, year, quarter)

        br = _stock_basic_map_single(bs)
        st = br.get(f"{exchange.lower()}.{code}") or br.get(code)

        row = _assemble_fundamentals_row(
            symbol,
            year,
            quarter,
            profit,
            operation,
            growth,
            balance,
            cashflow,
            dupont,
            st,
        )

        schema_fields: dict[str, Any] = {
            "report_date": pl.Date,
            "announce_date": pl.Date,
            "symbol": pl.String,
            "is_st": pl.Boolean,
        }
        schema_fields.update(dict.fromkeys(_FUNDAMENTALS_SCHEMA, pl.Float64))
        return pl.DataFrame(row, schema=schema_fields)
    finally:
        with _suppress_stdout():
            bs.logout()


def _fundamentals_chunk_worker(chunk: list[tuple[str, date]]) -> list[pl.DataFrame]:
    """Process multiple symbols in a single BaoStock login session."""
    import baostock as bs

    from qore_data.fetcher._base import _suppress_stdout

    with _suppress_stdout():
        lg = bs.login()
    if lg.error_code != "0":
        return [_empty_fundamentals()] * len(chunk)
    try:
        results: list[pl.DataFrame] = []
        for symbol, as_of in chunk:
            year, quarter = _latest_report_quarter(as_of)
            code = _symbol_digits(symbol)
            exchange = _exchange_from_stock_code(code)
            bs_code = f"{exchange.lower()}.{code}"

            profit = _query_single(bs, "query_profit_data", bs_code, year, quarter)
            operation = _query_single(
                bs, "query_operation_data", bs_code, year, quarter
            )
            growth = _query_single(bs, "query_growth_data", bs_code, year, quarter)
            balance = _query_single(bs, "query_balance_data", bs_code, year, quarter)
            cashflow = _query_single(bs, "query_cash_flow_data", bs_code, year, quarter)
            dupont = _query_single(bs, "query_dupont_data", bs_code, year, quarter)

            br = _stock_basic_map_single(bs)
            st = br.get(f"{exchange.lower()}.{code}") or br.get(code)

            row = _assemble_fundamentals_row(
                symbol,
                year,
                quarter,
                profit,
                operation,
                growth,
                balance,
                cashflow,
                dupont,
                st,
            )

            schema_fields: dict[str, Any] = {
                "report_date": pl.Date,
                "announce_date": pl.Date,
                "symbol": pl.String,
                "is_st": pl.Boolean,
            }
            schema_fields.update(dict.fromkeys(_FUNDAMENTALS_SCHEMA, pl.Float64))
            results.append(pl.DataFrame(row, schema=schema_fields))
        return results
    finally:
        with _suppress_stdout():
            bs.logout()


def _stock_basic_map_single(bs):
    rs = bs.query_stock_basic()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return {row[0]: row for row in rows}


def _assemble_fundamentals_row(
    symbol: str,
    year: int,
    quarter: int,
    profit: dict[str, str],
    operation: dict[str, str],
    growth: dict[str, str],
    balance: dict[str, str],
    cashflow: dict[str, str],
    dupont: dict[str, str],
    st_row: list[str] | None,
) -> dict[str, Any]:
    roe = _to_float(profit.get("roeAvg"))
    gp = _to_float(profit.get("gpMargin"))
    np = _to_float(profit.get("netProfit"))
    rev = _to_float(profit.get("MBRevenue"))

    rpt_date = _report_date(year, quarter)
    pub_date = _parse_date_str(profit.get("pubDate")) or rpt_date
    stat_date = _parse_date_str(profit.get("statDate")) or rpt_date

    is_st = False
    if st_row and len(st_row) > 4:
        is_st = st_row[4] == "0"

    return {
        "report_date": stat_date,
        "announce_date": pub_date,
        "symbol": symbol,
        "pe_ttm": None,
        "pb": None,
        "ps_ttm": None,
        "roe": (roe / 100.0) if roe is not None else None,
        "roa": None,
        "gross_margin": (gp / 100.0) if gp is not None else None,
        "net_margin": _pct(_to_float(profit.get("npMargin"))),
        "eps_ttm": _to_float(profit.get("epsTTM")),
        "revenue": rev,
        "net_income": np,
        "total_shares": _to_float(profit.get("totalShare")),
        "float_shares": _to_float(profit.get("liqaShare")),
        "equity_yoy": _pct(_to_float(growth.get("YOYEquity"))),
        "total_asset_yoy": _pct(_to_float(growth.get("YOYAsset"))),
        "net_profit_yoy": _pct(_to_float(growth.get("YOYNI"))),
        "eps_basic_yoy": _pct(_to_float(growth.get("YOYEPSBasic"))),
        "net_profit_parent_yoy": _pct(_to_float(growth.get("YOYPNI"))),
        "current_ratio": _to_float(balance.get("currentRatio")),
        "quick_ratio": _to_float(balance.get("quickRatio")),
        "cash_ratio": _to_float(balance.get("cashRatio")),
        "total_debt_yoy": _pct(_to_float(balance.get("YOYLiability"))),
        "debts_to_assets": _to_float(balance.get("liabilityToAsset")),
        "assets_to_equity": _to_float(balance.get("assetToEquity")),
        "current_assets_to_total_asset": _pct(_to_float(cashflow.get("CAToAsset"))),
        "non_current_assets_to_total_asset": _pct(
            _to_float(cashflow.get("NCAToAsset"))
        ),
        "tangible_assets_to_total_asset": _pct(
            _to_float(cashflow.get("tangibleAssetToAsset"))
        ),
        "ebit_to_interest": _to_float(cashflow.get("ebitToInterest")),
        "cfo_to_revenue": _pct(_to_float(cashflow.get("CFOToOR"))),
        "cfo_to_net_profit": _pct(_to_float(cashflow.get("CFOToNP"))),
        "receivable_turnover": _to_float(operation.get("NRTurnRatio")),
        "receivable_turnover_days": _to_float(operation.get("NRTurnDays")),
        "inventory_turnover": _to_float(operation.get("INVTurnRatio")),
        "inventory_turnover_days": _to_float(operation.get("INVTurnDays")),
        "current_assets_turnover": _to_float(operation.get("CATurnRatio")),
        "total_asset_turnover": _to_float(operation.get("AssetTurnRatio")),
        "parent_profit_ratio": _pct(_to_float(dupont.get("dupontPnitoni"))),
        "tax_burden": _to_float(dupont.get("dupontTaxBurden")),
        "interest_burden": _to_float(dupont.get("dupontIntburden")),
        "ebit_margin": _pct(_to_float(dupont.get("dupontEbittogr"))),
        "total_liabilities": None,
        "total_assets": None,
        "operating_cashflow": None,
        "total_market_cap": None,
        "float_market_cap": None,
        "is_st": is_st,
    }


_FUNDAMENTALS_SCHEMA: dict[str, type[pl.Float64]] = {
    "pe_ttm": pl.Float64,
    "pb": pl.Float64,
    "ps_ttm": pl.Float64,
    "roe": pl.Float64,
    "roa": pl.Float64,
    "gross_margin": pl.Float64,
    "net_margin": pl.Float64,
    "eps_ttm": pl.Float64,
    "revenue": pl.Float64,
    "net_income": pl.Float64,
    "total_shares": pl.Float64,
    "float_shares": pl.Float64,
    "equity_yoy": pl.Float64,
    "total_asset_yoy": pl.Float64,
    "net_profit_yoy": pl.Float64,
    "eps_basic_yoy": pl.Float64,
    "net_profit_parent_yoy": pl.Float64,
    "current_ratio": pl.Float64,
    "quick_ratio": pl.Float64,
    "cash_ratio": pl.Float64,
    "total_debt_yoy": pl.Float64,
    "debts_to_assets": pl.Float64,
    "assets_to_equity": pl.Float64,
    "current_assets_to_total_asset": pl.Float64,
    "non_current_assets_to_total_asset": pl.Float64,
    "tangible_assets_to_total_asset": pl.Float64,
    "ebit_to_interest": pl.Float64,
    "cfo_to_revenue": pl.Float64,
    "cfo_to_net_profit": pl.Float64,
    "receivable_turnover": pl.Float64,
    "receivable_turnover_days": pl.Float64,
    "inventory_turnover": pl.Float64,
    "inventory_turnover_days": pl.Float64,
    "current_assets_turnover": pl.Float64,
    "total_asset_turnover": pl.Float64,
    "parent_profit_ratio": pl.Float64,
    "tax_burden": pl.Float64,
    "interest_burden": pl.Float64,
    "ebit_margin": pl.Float64,
    "total_liabilities": pl.Float64,
    "total_assets": pl.Float64,
    "operating_cashflow": pl.Float64,
    "total_market_cap": pl.Float64,
    "float_market_cap": pl.Float64,
}


def _empty_fundamentals() -> pl.DataFrame:
    schema: dict[str, Any] = {
        "report_date": pl.Date,
        "announce_date": pl.Date,
        "symbol": pl.String,
        "is_st": pl.Boolean,
    }
    schema.update(dict(_FUNDAMENTALS_SCHEMA))
    return pl.DataFrame(schema=schema)


# ── BaoStock financial source ───────────────────────────────────────────


class _BaoStockFinancialSource:
    async def fundamentals(self, symbol: str, as_of: date) -> pl.DataFrame:
        result = await asyncio.to_thread(_fundamentals_worker, (symbol, as_of))
        return result

    async def batch_fundamentals(
        self, symbols: list[str], as_of: date
    ) -> list[pl.DataFrame]:
        from qore_data.fetcher.concurrent import _chunked

        items = [(s, as_of) for s in symbols]
        chunks = _chunked(items, 50)
        chunk_results = await asyncio.to_thread(
            batch_fetch, BatchConfig.process(), _fundamentals_chunk_worker, chunks
        )
        return [r for cr in chunk_results for r in cr]

    async def close(self) -> None:
        pass


# ── Xueqiu financial source (fallback, fills valuation multiples + indicators) ──


_XQ_INCOME_URL = "https://stock.xueqiu.com/v5/stock/finance/cn/income.json"
_XQ_BALANCE_URL = "https://stock.xueqiu.com/v5/stock/finance/cn/balance.json"
_XQ_CASHFLOW_URL = "https://stock.xueqiu.com/v5/stock/finance/cn/cash_flow.json"
_XQ_INDICATOR_URL = "https://stock.xueqiu.com/v5/stock/finance/cn/indicator.json"

_FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "report_date",
    "announce_date",
    "symbol",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "eps_ttm",
    "revenue",
    "net_income",
    "total_shares",
    "float_shares",
    "equity_yoy",
    "total_asset_yoy",
    "net_profit_yoy",
    "eps_basic_yoy",
    "net_profit_parent_yoy",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "total_debt_yoy",
    "debts_to_assets",
    "assets_to_equity",
    "current_assets_to_total_asset",
    "non_current_assets_to_total_asset",
    "tangible_assets_to_total_asset",
    "ebit_to_interest",
    "cfo_to_revenue",
    "cfo_to_net_profit",
    "receivable_turnover",
    "receivable_turnover_days",
    "inventory_turnover",
    "inventory_turnover_days",
    "current_assets_turnover",
    "total_asset_turnover",
    "parent_profit_ratio",
    "tax_burden",
    "interest_burden",
    "ebit_margin",
    "total_liabilities",
    "total_assets",
    "operating_cashflow",
    "total_market_cap",
    "float_market_cap",
    "is_st",
)


def _xq_val(pair: Any) -> float | None:
    """Extract value from Xueqiu [value, yoy] pair or flat number."""
    if pair is None:
        return None
    if isinstance(pair, (int, float)):
        return float(pair)
    if isinstance(pair, list) and len(pair) > 0:
        v = pair[0]
        return float(v) if v is not None else None
    return None


def _xq_yoy(pair: Any) -> float | None:
    if not isinstance(pair, list) or len(pair) < 2:
        return None
    v = pair[1]
    return float(v) if v is not None else None


class _XueqiuFinancialSource:
    def __init__(self, session: _XueqiuSession) -> None:
        self._session = session
        self._empty: pl.DataFrame | None = None

    async def fundamentals(self, symbol: str, as_of: date) -> pl.DataFrame:
        year, quarter = _latest_financial_quarter(as_of)
        period = _xq_period(year, quarter)
        xq_sym = _xq_symbol(symbol)

        try:
            inc = await self._session.get_json(
                _XQ_INCOME_URL,
                {"symbol": xq_sym, "type": period, "is_detail": "true", "count": "1"},
            )
            bal = await self._session.get_json(
                _XQ_BALANCE_URL,
                {"symbol": xq_sym, "type": period, "is_detail": "true", "count": "1"},
            )
            csh = await self._session.get_json(
                _XQ_CASHFLOW_URL,
                {"symbol": xq_sym, "type": period, "is_detail": "true", "count": "1"},
            )
            ind = await self._session.get_json(
                _XQ_INDICATOR_URL, {"symbol": xq_sym, "type": period, "count": "1"}
            )
        except Exception:
            return self._empty_frame()

        inc_list = (inc.get("data") or {}).get("list") or []
        bal_list = (bal.get("data") or {}).get("list") or []
        csh_list = (csh.get("data") or {}).get("list") or []
        ind_list = (ind.get("data") or {}).get("list") or []

        if not inc_list and not bal_list and not csh_list and not ind_list:
            return self._empty_frame()

        i = inc_list[0] if inc_list else {}
        b = bal_list[0] if bal_list else {}
        c = csh_list[0] if csh_list else {}
        d = ind_list[0] if ind_list else {}

        ts = i.get("report_date") or b.get("report_date") or 0
        rpt_date = date.fromtimestamp(ts / 1000) if ts else _report_date(year, quarter)

        row: dict[str, Any] = {
            "report_date": rpt_date,
            "announce_date": rpt_date,
            "symbol": symbol,
            "pe_ttm": _xq_val(d.get("pe_ttm")),
            "pb": _xq_val(d.get("pb_mrq")),
            "ps_ttm": _xq_val(d.get("ps_ttm")),
            "roe": _xq_val(d.get("roe_avg")),
            "roa": _xq_val(d.get("roa")),
            "gross_margin": _xq_val(i.get("gross_profit_margin"))
            or _xq_val(d.get("gross_profit_margin")),
            "net_margin": _xq_val(d.get("net_profit_margin")),
            "eps_ttm": _xq_val(i.get("eps")) or _xq_val(d.get("eps")),
            "revenue": _xq_val(i.get("total_revenue")),
            "net_income": _xq_val(i.get("net_profit")),
            "total_shares": None,
            "float_shares": None,
            "equity_yoy": _xq_yoy(i.get("total_equity"))
            or _xq_yoy(b.get("total_equity")),
            "total_asset_yoy": _xq_yoy(b.get("total_assets")),
            "net_profit_yoy": _xq_yoy(i.get("net_profit")),
            "eps_basic_yoy": _xq_yoy(i.get("eps")),
            "net_profit_parent_yoy": _xq_yoy(i.get("net_profit_parent")),
            "current_ratio": _xq_val(b.get("current_ratio")),
            "quick_ratio": _xq_val(b.get("quick_ratio")),
            "cash_ratio": None,
            "total_debt_yoy": None,
            "debts_to_assets": _xq_val(b.get("asset_liab_ratio")),
            "assets_to_equity": None,
            "current_assets_to_total_asset": None,
            "non_current_assets_to_total_asset": None,
            "tangible_assets_to_total_asset": None,
            "ebit_to_interest": None,
            "cfo_to_revenue": None,
            "cfo_to_net_profit": None,
            "receivable_turnover": None,
            "receivable_turnover_days": None,
            "inventory_turnover": None,
            "inventory_turnover_days": None,
            "current_assets_turnover": None,
            "total_asset_turnover": None,
            "parent_profit_ratio": None,
            "tax_burden": None,
            "interest_burden": None,
            "ebit_margin": None,
            "total_liabilities": _xq_val(b.get("total_liabilities")),
            "total_assets": _xq_val(b.get("total_assets")),
            "operating_cashflow": _xq_val(c.get("ncf_from_oa")),
            "total_market_cap": None,
            "float_market_cap": None,
            "is_st": False,
        }

        return pl.DataFrame(
            row,
            schema={
                col: pl.Float64
                if col not in ("report_date", "announce_date", "symbol", "is_st")
                else pl.Date
                if "date" in col
                else pl.String
                if col == "symbol"
                else pl.Boolean
                for col in _FUNDAMENTALS_COLUMNS
            },
        )

    def _empty_frame(self) -> pl.DataFrame:
        if self._empty is None:
            schema: dict[str, Any] = {
                "report_date": pl.Date,
                "announce_date": pl.Date,
                "symbol": pl.String,
                "is_st": pl.Boolean,
            }
            for c in _FUNDAMENTALS_COLUMNS:
                if c not in schema:
                    schema[c] = pl.Float64
            self._empty = pl.DataFrame(schema=schema)
        return self._empty

    async def close(self) -> None:
        pass


# ── FinancialFetcher ────────────────────────────────────────────────────


class FinancialFetcher:
    """Financial fundamentals. BaoStock (primary) -> Xueqiu indicator/income/balance/cashflow (fallback)."""

    def __init__(self) -> None:
        self._baostock = _BaoStockFinancialSource()
        self._xueqiu_session = _XueqiuSession()
        self._xueqiu = _XueqiuFinancialSource(self._xueqiu_session)

    @classmethod
    def from_settings(cls, settings) -> FinancialFetcher:
        return cls()

    async def close(self) -> None:
        await self._baostock.close()
        await self._xueqiu_session.close()

    async def fundamentals(self, symbol: str, as_of: date) -> pl.DataFrame:
        result = await self._baostock.fundamentals(symbol, as_of)
        if not result.is_empty():
            return result
        result = await self._xueqiu.fundamentals(symbol, as_of)
        return result

    async def batch_fundamentals(
        self, symbols: list[str], as_of: date
    ) -> list[pl.DataFrame]:
        results = await self._baostock.batch_fundamentals(symbols, as_of)
        remaining: list[int] = [i for i, r in enumerate(results) if r.is_empty()]
        if remaining:
            xq_results = await asyncio.gather(
                *(self._xueqiu.fundamentals(symbols[i], as_of) for i in remaining)
            )
            for i, xq_r in zip(remaining, xq_results, strict=False):
                if not xq_r.is_empty():
                    results[i] = xq_r
        return results
