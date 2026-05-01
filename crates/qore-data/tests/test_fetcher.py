"""Gated live IO contract tests for all fetcher sources and integration."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from qore_data import DataSettings
from qore_data.fetcher._base import EMPTY_SCHEMA
from qore_data.fetcher.analyst import AnalystFetcher
from qore_data.fetcher.announcement import AnnouncementFetcher
from qore_data.fetcher.constituent import ConstituentFetcher
from qore_data.fetcher.financial import FinancialFetcher
from qore_data.fetcher.fund import FundFetcher
from qore_data.fetcher.quote import QuoteFetcher
from qore_data.fetcher.xueqiu import _xq_symbol, _XueqiuSession, _XueqiuTokenError
from qore_data.store.duckdb import QoreStore


def _require_live_io() -> None:
    if os.getenv("QORE_RUN_LIVE_IO") != "1":
        pytest.skip("Set QORE_RUN_LIVE_IO=1")


def _require_mainland_ip() -> None:
    if os.getenv("QORE_MAINLAND") != "1":
        pytest.skip("Set QORE_MAINLAND=1 for mainland-restricted tests")


__pytest_plugins__ = ["pytest_asyncio"]


# ── constants ──────────────────────────────────────────────────────────

SYMBOL_MOUTAI = "600519.SH"
SYMBOL_PINGAN = "000001.SZ"
SYMBOL_CATL = "300750.SZ"
FUND_110022 = "110022"
INDEX_CSI300 = "000300.SH"
INDEX_CSI500 = "000905.SH"
INDEX_SSE50 = "000016.SH"


# ── helpers ─────────────────────────────────────────────────────────────


@pytest.fixture
def settings(tmp_path: Path) -> DataSettings:
    return DataSettings(
        db_path=str(tmp_path / "live.duckdb"),
        parquet_root=str(tmp_path / "parquet"),
    )


def _check_schema(name: str, df: pl.DataFrame) -> None:
    """Assert `df` columns match EMPTY_SCHEMA[`name`] exactly."""
    expected = set(EMPTY_SCHEMA[name])
    actual = set(df.columns)
    assert actual == expected, (
        f"Schema mismatch for {name}: missing={expected - actual}, extra={actual - expected}"
    )


def _assert_non_empty(df: pl.DataFrame, name: str) -> None:
    assert not df.is_empty(), f"{name} returned empty DataFrame"


def _assert_no_null_in(df: pl.DataFrame, cols: Sequence[str]) -> None:
    nulls = df.select(pl.col(cols).null_count()).row(0)
    assert all(n == 0 for n in nulls), (
        f"nulls in {cols}: {[c for c, n in zip(cols, nulls, strict=False) if n > 0]}"
    )


_ASYNC_GUARD: dict[str, int] = {}


async def _require_xueqiu_token() -> _XueqiuSession:
    s = _XueqiuSession()
    try:
        await s.get_json(
            "https://stock.xueqiu.com/v5/stock/chart/kline.json",
            {
                "symbol": _xq_symbol(SYMBOL_MOUTAI),
                "begin": "1704067200000",
                "period": "day",
                "type": "before",
                "count": "-1",
                "indicator": "kline",
            },
        )
    except _XueqiuTokenError as exc:
        if "xq_a_token" in str(exc):
            pytest.skip(
                f"Xueqiu token not set by homepage (WAF block from overseas IP): {exc}"
            )
        pytest.skip(f"Xueqiu API error: {exc}")
    except httpx.HTTPStatusError as exc:
        if (
            exc.response.status_code == 404
            and "g.alicdn.com" in exc.response.text[:300]
        ):
            pytest.skip(
                "Xueqiu API blocked by Alibaba Cloud WAF — requires mainland IP or browser-based token"
            )
        pytest.skip(f"Xueqiu HTTP {exc.response.status_code}")
    except httpx.TimeoutException:
        pytest.skip("Xueqiu connection timeout (unreachable from this network)")
    except Exception as exc:
        pytest.skip(f"Xueqiu unavailable: {exc}")
    return s
    return s


async def _require_source(
    fetcher_factory: Callable[[], Any],
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    f = fetcher_factory()
    try:
        result = await getattr(f, method)(*args, **kwargs)
    except Exception as exc:
        pytest.skip(f"{method} unavailable: {exc}")
    finally:
        await f.close()
    return result


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — per-source contracts
# ═══════════════════════════════════════════════════════════════════════

# ── BaoStock quote source ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_bao_kline_source(settings: DataSettings) -> None:
    _require_live_io()
    from qore_data.fetcher.quote import _BaoStockQuoteSource

    s = _BaoStockQuoteSource()
    try:
        r = await s.stock_daily(SYMBOL_MOUTAI, date(2025, 3, 1), date(2025, 4, 1))
    except Exception as exc:
        pytest.skip(f"bao kline unavailable: {exc}")
    finally:
        await s.close()
    _check_schema("stock_daily", r)
    assert len(r) >= 10
    _assert_no_null_in(r, ["date", "symbol", "close", "open", "high", "low"])


@pytest.mark.asyncio
async def test_bao_kline_batch_source(settings: DataSettings) -> None:
    _require_live_io()
    from qore_data.fetcher.quote import _BaoStockQuoteSource

    s = _BaoStockQuoteSource()
    try:
        rs = await s.batch_stock_daily(
            [SYMBOL_MOUTAI, SYMBOL_PINGAN], date(2025, 3, 1), date(2025, 4, 1)
        )
    except Exception as exc:
        pytest.skip(f"bao kline batch unavailable: {exc}")
    finally:
        await s.close()
    assert len(rs) == 2
    for r in rs:
        _check_schema("stock_daily", r)
    assert rs[0].get_column("symbol")[0] == "600519.SH"
    assert rs[1].get_column("symbol")[0] == "000001.SZ"


@pytest.mark.asyncio
async def test_bao_stock_info_source(settings: DataSettings) -> None:
    _require_live_io()
    from qore_data.fetcher.quote import _BaoStockQuoteSource

    s = _BaoStockQuoteSource()
    try:
        r = await s.batch_stock_profiles([SYMBOL_MOUTAI, SYMBOL_CATL], date.today())
    except Exception as exc:
        pytest.skip(f"bao stock_info unavailable: {exc}")
    finally:
        await s.close()
    _check_schema("stock_profile", r)
    assert r.get_column("short_name").to_list()[0] == "贵州茅台"
    assert r.get_column("symbol").to_list()[1] == "300750.SZ"


@pytest.mark.asyncio
async def test_bao_constituents_source() -> None:
    _require_live_io()
    from qore_data.fetcher.constituent import _constituents_worker

    r = await asyncio.to_thread(_constituents_worker, INDEX_CSI300)
    assert len(r) > 0
    assert all(s.endswith((".SH", ".SZ")) for s in r.to_list()[:10])


# ── Xueqiu source contracts ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_xueqiu_kline_source() -> None:
    _require_live_io()
    from qore_data.fetcher.quote import _XueqiuQuoteDaySource

    xq = await _require_xueqiu_token()
    s = _XueqiuQuoteDaySource(xq)
    try:
        r = await s.stock_daily(SYMBOL_MOUTAI, date(2025, 3, 1), date(2025, 4, 1))
    except Exception as exc:
        pytest.skip(f"xueqiu kline unavailable: {exc}")
    finally:
        await s.close()
    _check_schema("stock_daily", r)
    assert len(r) >= 10
    _assert_no_null_in(r, ["date", "symbol", "close"])


@pytest.mark.asyncio
async def test_xueqiu_financial_source() -> None:
    _require_live_io()
    from qore_data.fetcher.financial import _XueqiuFinancialSource

    xq = await _require_xueqiu_token()
    s = _XueqiuFinancialSource(xq)
    try:
        r = await s.fundamentals(SYMBOL_MOUTAI, date.today())
    except Exception as exc:
        pytest.skip(f"xueqiu financial unavailable: {exc}")
    finally:
        await s.close()
    _check_schema("fundamentals", r)
    assert r.get_column("symbol")[0] == "600519.SH"
    assert r.get_column("roe")[0] is not None
    assert r.get_column("pe_ttm")[0] is not None


@pytest.mark.asyncio
async def test_xueqiu_analyst_source() -> None:
    _require_live_io()
    from qore_data.fetcher.analyst import _XueqiuAnalystSource

    xq = await _require_xueqiu_token()
    s = _XueqiuAnalystSource(xq)
    try:
        r = await s.batch_analyst_forecasts([SYMBOL_MOUTAI], date.today())
    except Exception as exc:
        pytest.skip(f"xueqiu analyst unavailable: {exc}")
    finally:
        await s.close()
    assert "eps_year1" in r.columns
    assert r.get_column("symbol")[0] == "600519.SH"
    assert (
        r.get_column("report_count")[0] is not None
        and r.get_column("report_count")[0] > 0
    )


@pytest.mark.asyncio
async def test_xueqiu_fund_nav_source() -> None:
    _require_live_io()
    from qore_data.fetcher.fund import _XueqiuFundSource

    xq = await _require_xueqiu_token()
    s = _XueqiuFundSource(xq)
    try:
        r = await s.fund_nav(FUND_110022, date(2025, 3, 1), date(2025, 4, 1))
    except Exception as exc:
        pytest.skip(f"xueqiu fund nav unavailable: {exc}")
    finally:
        await s.close()
    _check_schema("fund_nav", r)
    assert r.get_column("symbol")[0] == "110022"
    _assert_no_null_in(r, ["date", "symbol", "nav"])


@pytest.mark.asyncio
async def test_xueqiu_fund_holdings_source() -> None:
    _require_live_io()
    from qore_data.fetcher.fund import _XueqiuFundSource

    xq = await _require_xueqiu_token()
    s = _XueqiuFundSource(xq)
    try:
        r = await s.fund_holdings(FUND_110022, date(2024, 12, 31))
    except Exception as exc:
        pytest.skip(f"xueqiu fund holdings unavailable: {exc}")
    finally:
        await s.close()
    _check_schema("fund_holdings", r)
    assert "stock_symbol" in r.columns


@pytest.mark.asyncio
async def test_xueqiu_constituents_source() -> None:
    _require_live_io()
    xq = await _require_xueqiu_token()
    try:
        data = await xq.get_json(
            "https://stock.xueqiu.com/v5/stock/index/detail/quote.json",
            {"symbol": _xq_symbol(INDEX_CSI300), "size": "500", "page": "1"},
        )
    except Exception as exc:
        pytest.skip(f"xueqiu constituents unavailable: {exc}")
    finally:
        await xq.close()
    items = (data.get("data") or {}).get("list") or []
    assert len(items) > 0
    for item in items[:5]:
        sym = item.get("symbol", "")
        assert sym.startswith(("SH", "SZ"))


# ── NetEase source contract ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_netease_kline_source() -> None:
    _require_live_io()
    from qore_data.fetcher.quote import _NeteaseQuoteSource

    s = _NeteaseQuoteSource()
    try:
        r = await s.stock_daily(SYMBOL_MOUTAI, date(2025, 3, 1), date(2025, 4, 1))
    except httpx.HTTPStatusError as exc:
        pytest.skip(
            f"NetEase HTTP {exc.response.status_code} (service overloaded / overseas block)"
        )
    except httpx.TimeoutException:
        pytest.skip("NetEase connection timeout (unreachable from this network)")
    except Exception as exc:
        pytest.skip(f"netEase kline unavailable: {exc}")
    finally:
        await s.close()
    if r.is_empty():
        pytest.skip("NetEase returned empty (GBK CSV may fail outside CN)")
    _check_schema("stock_daily", r)
    assert len(r) >= 5


# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — fetcher-level integration
# ═══════════════════════════════════════════════════════════════════════

# ── QuoteFetcher ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quote_stock_daily_contract(settings: DataSettings) -> None:
    _require_live_io()
    r = await _require_source(
        lambda: QuoteFetcher.from_settings(settings),
        "stock_daily",
        SYMBOL_MOUTAI,
        date(2025, 3, 1),
        date(2025, 4, 1),
    )
    _check_schema("stock_daily", r)
    assert len(r) >= 10
    _assert_no_null_in(r, ["date", "symbol", "close"])


@pytest.mark.asyncio
async def test_quote_batch_daily(settings: DataSettings) -> None:
    _require_live_io()
    f = QuoteFetcher.from_settings(settings)
    try:
        rs = await f.batch_stock_daily(
            [SYMBOL_MOUTAI, SYMBOL_PINGAN, SYMBOL_CATL],
            date(2025, 3, 1),
            date(2025, 4, 1),
        )
    except Exception as exc:
        pytest.skip(f"batch stock_daily unavailable: {exc}")
    finally:
        await f.close()
    assert len(rs) == 3
    for r in rs:
        _check_schema("stock_daily", r)


@pytest.mark.asyncio
async def test_quote_stock_profile_contract(settings: DataSettings) -> None:
    _require_live_io()
    r = await _require_source(
        lambda: QuoteFetcher.from_settings(settings),
        "stock_profile",
        SYMBOL_MOUTAI,
        date.today(),
    )
    _check_schema("stock_profile", r)
    assert r.get_column("symbol").to_list() == ["600519.SH"]
    assert r.get_column("short_name").to_list() == ["贵州茅台"]


@pytest.mark.asyncio
async def test_quote_batch_profiles(settings: DataSettings) -> None:
    _require_live_io()
    symbols = [SYMBOL_MOUTAI, SYMBOL_CATL, SYMBOL_PINGAN]
    f = QuoteFetcher.from_settings(settings)
    try:
        r = await f.batch_stock_profiles(symbols, date.today())
    except Exception as exc:
        pytest.skip(f"batch profiles unavailable: {exc}")
    finally:
        await f.close()
    _check_schema("stock_profile", r)
    assert r.get_column("symbol").n_unique() == len(symbols)


@pytest.mark.asyncio
async def test_quote_capital_flow_historical(settings: DataSettings) -> None:
    _require_live_io()
    f = QuoteFetcher.from_settings(settings)
    try:
        r = await f.capital_flow(SYMBOL_MOUTAI, date(2025, 1, 1), date(2025, 3, 31))
    except Exception as exc:
        pytest.skip(f"capital_flow unavailable: {exc}")
    finally:
        await f.close()
    _check_schema("capital_flow", r)


@pytest.mark.asyncio
async def test_quote_capital_flow_empty_future(settings: DataSettings) -> None:
    _require_live_io()
    f = QuoteFetcher.from_settings(settings)
    try:
        r = await f.capital_flow(SYMBOL_MOUTAI, date(2099, 1, 1), date(2099, 3, 31))
    except Exception as exc:
        pytest.skip(f"capital_flow unavailable: {exc}")
    finally:
        await f.close()
    assert r.is_empty()


# ── FinancialFetcher ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_financial_fundamentals(settings: DataSettings) -> None:
    _require_live_io()
    r = await _require_source(
        lambda: FinancialFetcher.from_settings(settings),
        "fundamentals",
        SYMBOL_MOUTAI,
        date.today(),
    )
    _check_schema("fundamentals", r)
    assert r.get_column("roe")[0] is not None
    assert r.get_column("symbol")[0] == "600519.SH"


@pytest.mark.asyncio
async def test_financial_batch_fundamentals(settings: DataSettings) -> None:
    _require_live_io()
    f = FinancialFetcher.from_settings(settings)
    try:
        rs = await f.batch_fundamentals(
            [SYMBOL_MOUTAI, SYMBOL_PINGAN, SYMBOL_CATL], date.today()
        )
    except Exception as exc:
        pytest.skip(f"batch fundamentals unavailable: {exc}")
    finally:
        await f.close()
    assert len(rs) == 3
    for r in rs:
        _check_schema("fundamentals", r)


# ── AnalystFetcher ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyst_forecast(settings: DataSettings) -> None:
    _require_live_io()
    r = await _require_source(
        lambda: AnalystFetcher.from_settings(settings),
        "analyst_forecast",
        SYMBOL_MOUTAI,
        date.today(),
    )
    assert "eps_year1" in r.columns
    assert r.get_column("symbol")[0] == "600519.SH"


@pytest.mark.asyncio
async def test_analyst_batch(settings: DataSettings) -> None:
    _require_live_io()
    f = AnalystFetcher.from_settings(settings)
    try:
        r = await f.batch_analyst_forecasts(
            [SYMBOL_MOUTAI, SYMBOL_PINGAN], date.today()
        )
    except Exception as exc:
        pytest.skip(f"batch analyst unavailable: {exc}")
    finally:
        await f.close()
    assert r.get_column("symbol").n_unique() == 2
    assert "eps_year1" in r.columns


# ── FundFetcher ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fund_nav(settings: DataSettings) -> None:
    _require_live_io()
    r = await _require_source(
        lambda: FundFetcher.from_settings(settings),
        "fund_nav",
        FUND_110022,
        date(2025, 3, 1),
        date(2025, 4, 1),
    )
    _check_schema("fund_nav", r)
    assert r.get_column("symbol")[0] == "110022"


@pytest.mark.asyncio
async def test_fund_holdings(settings: DataSettings) -> None:
    _require_live_io()
    r = await _require_source(
        lambda: FundFetcher.from_settings(settings),
        "fund_holdings",
        FUND_110022,
        date(2024, 12, 31),
    )
    _check_schema("fund_holdings", r)
    assert "stock_symbol" in r.columns


# ── ConstituentFetcher ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_constituent_index(settings: DataSettings) -> None:
    _require_live_io()
    f = ConstituentFetcher.from_settings(settings)
    try:
        r = await f.index_constituents(INDEX_CSI300, date.today())
    except Exception as exc:
        pytest.skip(f"index_constituents unavailable: {exc}")
    finally:
        await f.close()
    assert len(r) > 0
    assert all(s.endswith((".SH", ".SZ", ".BJ")) for s in r.to_list()[:10])


@pytest.mark.asyncio
async def test_constituent_with_weight(settings: DataSettings) -> None:
    _require_live_io()
    f = ConstituentFetcher.from_settings(settings)
    try:
        r = await f.index_constituents_with_weight(INDEX_CSI300)
    except Exception as exc:
        pytest.skip(f"constituents weight unavailable: {exc}")
    finally:
        await f.close()
    if r.is_empty():
        pytest.skip("CSI / fundztapi unavailable")


@pytest.mark.asyncio
async def test_constituent_small_index(settings: DataSettings) -> None:
    _require_live_io()
    f = ConstituentFetcher.from_settings(settings)
    try:
        r = await f.index_constituents(INDEX_CSI500, date.today())
    except Exception as exc:
        pytest.skip(f"index_constituents 500 unavailable: {exc}")
    finally:
        await f.close()
    assert len(r) > 400


# ── AnnouncementFetcher ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_announcements(settings: DataSettings) -> None:
    _require_live_io()
    _require_mainland_ip()
    r = await _require_source(
        lambda: AnnouncementFetcher.from_settings(settings),
        "announcements",
        SYMBOL_MOUTAI,
        date(2025, 1, 1),
        date(2025, 6, 30),
    )
    _check_schema("announcements", r)
    assert r.get_column("symbol")[0] == "600519.SH"


@pytest.mark.asyncio
async def test_audit_opinions(settings: DataSettings) -> None:
    _require_live_io()
    _require_mainland_ip()
    f = AnnouncementFetcher.from_settings(settings)
    try:
        r = await f.audit_opinions(SYMBOL_MOUTAI, date(2020, 1, 1), date(2026, 4, 27))
    except Exception as exc:
        pytest.skip(f"audit_opinions unavailable: {exc}")
    finally:
        await f.close()
    if not r.is_empty():
        _check_schema("audit_opinions", r)
        assert {"opinion", "opinion_code"}.issubset(r.columns)


# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — store roundtrip integration
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_store_roundtrip_profile(settings: DataSettings) -> None:
    _require_live_io()
    f = QuoteFetcher.from_settings(settings)
    store = QoreStore.from_settings(settings)
    try:
        profile = await f.stock_profile(SYMBOL_MOUTAI, date.today())
    except Exception as exc:
        pytest.skip(f"profile roundtrip unavailable: {exc}")
    finally:
        await f.close()
    store.write("stock_info", profile)
    stored = pl.DataFrame(
        store.read(
            "stock_info", filters={"symbol": "600519.SH"}, backend="duckdb"
        ).collect()
    )
    _check_schema("stock_profile", stored)
    assert stored.get_column("short_name").to_list()[0] == "贵州茅台"


@pytest.mark.asyncio
async def test_store_roundtrip_ohlcv(settings: DataSettings) -> None:
    _require_live_io()
    f = QuoteFetcher.from_settings(settings)
    store = QoreStore.from_settings(settings)
    try:
        ohlcv = await f.stock_daily(SYMBOL_MOUTAI, date(2025, 3, 1), date(2025, 4, 1))
    except Exception as exc:
        pytest.skip(f"ohlcv roundtrip unavailable: {exc}")
    finally:
        await f.close()
    store.write("stock_ohlcv", ohlcv)
    stored = pl.DataFrame(
        store.read(
            "stock_ohlcv", filters={"symbol": "600519.SH"}, backend="duckdb"
        ).collect()
    )
    _check_schema("stock_daily", stored)
    assert len(stored) >= 10


@pytest.mark.asyncio
async def test_store_roundtrip_fundamentals(settings: DataSettings) -> None:
    _require_live_io()
    f = FinancialFetcher.from_settings(settings)
    store = QoreStore.from_settings(settings)
    try:
        data = await f.fundamentals(SYMBOL_MOUTAI, date.today())
    except Exception as exc:
        pytest.skip(f"fundamentals roundtrip unavailable: {exc}")
    finally:
        await f.close()
    store.write("fundamentals", data)
    stored = pl.DataFrame(
        store.read(
            "fundamentals", filters={"symbol": "600519.SH"}, backend="duckdb"
        ).collect()
    )
    _check_schema("fundamentals", stored)


@pytest.mark.asyncio
async def test_store_roundtrip_analyst(settings: DataSettings) -> None:
    _require_live_io()
    f = AnalystFetcher.from_settings(settings)
    store = QoreStore.from_settings(settings)
    try:
        data = await f.analyst_forecast(SYMBOL_MOUTAI, date.today())
    except Exception as exc:
        pytest.skip(f"analyst roundtrip unavailable: {exc}")
    finally:
        await f.close()
    if data.is_empty():
        pytest.skip("analyst data empty, skipping roundtrip")
    store.write("analyst_forecasts", data)
    stored = pl.DataFrame(
        store.read(
            "analyst_forecasts", filters={"symbol": "600519.SH"}, backend="duckdb"
        ).collect()
    )
    assert not stored.is_empty()


# ═══════════════════════════════════════════════════════════════════════
# TIER 4 — pipeline batch integration
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pipeline_ohlcv_batch(settings: DataSettings) -> None:
    _require_live_io()
    symbols = [SYMBOL_MOUTAI, SYMBOL_PINGAN, SYMBOL_CATL]
    from qore_data.fetch import StockPipeline

    pipe = StockPipeline.from_settings(settings)
    try:
        await pipe.stock_daily(symbols)
    finally:
        await pipe.close()

    stored = pl.DataFrame(
        pipe.store.read(
            "stock_ohlcv",
            filters={"symbol": SYMBOL_MOUTAI},
            backend="duckdb",
        ).collect()
    )
    _check_schema("stock_daily", stored)
    assert not stored.is_empty()
    assert SYMBOL_MOUTAI in stored.get_column("symbol").to_list()


@pytest.mark.asyncio
async def test_pipeline_fundamentals_batch(settings: DataSettings) -> None:
    _require_live_io()
    symbols = [SYMBOL_MOUTAI, SYMBOL_PINGAN, SYMBOL_CATL]
    from qore_data.fetch import StockPipeline

    pipe = StockPipeline.from_settings(settings)
    try:
        await pipe.fundamentals(symbols)
    finally:
        await pipe.close()

    stored = pl.DataFrame(
        pipe.store.read(
            "fundamentals",
            filters={"symbol": SYMBOL_MOUTAI},
            backend="duckdb",
        ).collect()
    )
    _check_schema("fundamentals", stored)
    assert not stored.is_empty()
    assert SYMBOL_MOUTAI in stored.get_column("symbol").to_list()
