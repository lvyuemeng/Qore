from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_core import StockInstrument
from qore_data.source import StockSource
from qore_data.store.duckdb import QoreStore
from qore_data.universe import (
    CandidateFilter,
    CandidateSort,
    StockCandidateSpec,
    StockSelectionPipeline,
    build_stock_universe_from_index,
    snapshot_index_constituents,
    snapshot_stock_analyst_forecasts,
    snapshot_stock_announcements,
    snapshot_stock_audit_opinions,
    snapshot_stock_profiles,
    snapshot_stock_statuses,
)


class StubStockSource(StockSource):
    async def index_constituents(
        self,
        index_symbol: str,
        as_of: date,
    ) -> list[StockInstrument]:
        del index_symbol, as_of
        return [
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="300750.SZ", exchange="SZ", industry="battery"),
        ]

    async def stock_profile(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame:
        industry = "beverage" if inst.symbol == "600519.SH" else "new_energy"
        board = "MainBoard" if inst.symbol.endswith(".SH") else "ChiNext"
        return pl.DataFrame(
            {
                "as_of": [as_of],
                "symbol": [inst.symbol],
                "short_name": [inst.symbol.split(".", maxsplit=1)[0]],
                "exchange": [inst.exchange],
                "industry": [industry],
                "board": [board],
                "listing_date": [date(2020, 1, 1)],
                "total_market_cap": [100.0],
                "float_market_cap": [80.0],
                "total_shares": [10.0],
                "float_shares": [8.0],
                "is_st": [False],
            }
        )

    async def analyst_forecast(
        self,
        inst: StockInstrument,
        as_of: date,
    ) -> pl.DataFrame:
        report_count = 12 if inst.symbol == "600519.SH" else 8
        return pl.DataFrame(
            {
                "as_of": [as_of],
                "symbol": [inst.symbol],
                "report_count": [report_count],
                "buy": [4],
                "overweight": [3],
                "neutral": [1],
                "underweight": [0],
                "sell": [0],
                "eps_year1": [3.0],
                "eps_year2": [3.2],
                "eps_year3": [3.4],
                "eps_year4": [3.6],
            }
        )

    async def announcements(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        del start, end
        return pl.DataFrame(
            {
                "symbol": [inst.symbol],
                "short_name": [inst.symbol.split(".", maxsplit=1)[0]],
                "title": ["样本公告"],
                "notice_type": ["财务报告"],
                "notice_date": [date(2026, 4, 18)],
                "art_code": [f"ANN-{inst.symbol}"],
                "url": [f"https://example.test/{inst.symbol}"],
            }
        )

    async def audit_opinions(
        self,
        inst: StockInstrument,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        del start, end
        return pl.DataFrame(
            {
                "symbol": [inst.symbol],
                "report_date": [date(2025, 12, 31)],
                "announce_date": [date(2026, 4, 18)],
                "opinion": ["无保留意见"],
                "opinion_code": ["unqualified"],
                "source_notice_type": ["财务报告"],
                "title": ["2025年年度审计报告(无保留意见)"],
                "art_code": [f"AUD-{inst.symbol}"],
                "url": [f"https://example.test/audit/{inst.symbol}"],
            }
        )


@pytest.mark.asyncio
async def test_snapshot_index_constituents_writes_dataset(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    frame = await snapshot_index_constituents(
        StubStockSource(),
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 19),
    )

    assert frame.height == 2
    persisted = pl.DataFrame(store.read("index_constituents").collect())
    assert persisted.height == 2
    assert persisted.get_column("index_symbol").unique().to_list() == ["000300.SH"]


@pytest.mark.asyncio
async def test_snapshot_stock_profiles_writes_dataset(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    frame = await snapshot_stock_profiles(
        StubStockSource(),
        store,
        instruments=[
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="300750.SZ", exchange="SZ", industry="battery"),
        ],
        as_of=date(2026, 4, 19),
    )

    assert frame.height == 2
    persisted = pl.DataFrame(store.read("stock_profiles").collect())
    assert set(persisted.get_column("board").to_list()) == {"MainBoard", "ChiNext"}


@pytest.mark.asyncio
async def test_snapshot_stock_analyst_forecasts_writes_dataset(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    frame = await snapshot_stock_analyst_forecasts(
        StubStockSource(),
        store,
        instruments=[
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="300750.SZ", exchange="SZ", industry="battery"),
        ],
        as_of=date(2026, 4, 19),
    )

    assert frame.height == 2
    persisted = pl.DataFrame(store.read("analyst_forecasts").collect())
    assert sorted(persisted.get_column("report_count").to_list()) == [8, 12]


@pytest.mark.asyncio
async def test_snapshot_stock_announcements_writes_dataset(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    frame = await snapshot_stock_announcements(
        StubStockSource(),
        store,
        instruments=[
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="300750.SZ", exchange="SZ", industry="battery"),
        ],
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    assert frame.height == 2
    persisted = pl.DataFrame(store.read("announcements").collect())
    assert persisted.height == 2


@pytest.mark.asyncio
async def test_snapshot_stock_audit_opinions_writes_dataset(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    frame = await snapshot_stock_audit_opinions(
        StubStockSource(),
        store,
        instruments=[
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="300750.SZ", exchange="SZ", industry="battery"),
        ],
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    assert frame.height == 2
    persisted = pl.DataFrame(store.read("stock_audit_opinions").collect()).sort(
        "symbol"
    )
    assert persisted.get_column("opinion_code").to_list() == [
        "unqualified",
        "unqualified",
    ]


@pytest.mark.asyncio
async def test_build_stock_universe_from_index_enriches_industry(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    universe = await build_stock_universe_from_index(
        StubStockSource(),
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 19),
    )

    assert universe.get("600519.SH").industry == "beverage"
    assert universe.get("300750.SZ").industry == "new_energy"


def test_evaluate_stock_categories_aggregates_universe_views(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "index_symbol": ["000300.SH", "000300.SH"],
                "symbol": ["600519.SH", "300750.SZ"],
                "exchange": ["SH", "SZ"],
                "industry": ["food", "battery"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "short_name": ["600519", "300750"],
                "exchange": ["SH", "SZ"],
                "industry": ["beverage", "new_energy"],
                "board": ["MainBoard", "ChiNext"],
                "listing_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "total_market_cap": [100.0, 200.0],
                "float_market_cap": [80.0, 150.0],
                "total_shares": [10.0, 20.0],
                "float_shares": [8.0, 15.0],
                "is_st": [False, False],
            }
        ),
    )
    store.write(
        "analyst_forecasts",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "report_count": [12, 8],
                "buy": [4, 3],
                "overweight": [3, 2],
                "neutral": [1, 1],
                "underweight": [0, 0],
                "sell": [0, 0],
                "eps_year1": [3.0, 2.0],
                "eps_year2": [3.2, 2.2],
                "eps_year3": [3.4, 2.4],
                "eps_year4": [3.6, 2.6],
            }
        ),
    )
    store.write(
        "announcements",
        pl.DataFrame(
            {
                "symbol": ["600519.SH", "300750.SZ", "300750.SZ"],
                "short_name": ["600519", "300750", "300750"],
                "title": ["A", "B", "C"],
                "notice_type": ["财务报告", "一般事项", "业绩快报"],
                "notice_date": [
                    date(2026, 4, 10),
                    date(2026, 4, 11),
                    date(2026, 4, 12),
                ],
                "art_code": ["a", "b", "c"],
                "url": ["u1", "u2", "u3"],
            }
        ),
    )

    report = StockSelectionPipeline.from_index(
        store,
        index_symbol="000300.SH",
        as_of=as_of,
        announcement_start=date(2026, 4, 1),
        announcement_end=date(2026, 4, 30),
    ).category_report()

    assert report.height == 2
    assert set(report.columns) == {
        "industry",
        "board",
        "symbol_count",
        "avg_total_market_cap",
        "avg_report_count",
        "announcement_count",
    }


def test_snapshot_stock_statuses_derives_tradeability(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "short_name": ["*ST茅台", "宁德时代"],
                "exchange": ["SH", "SZ"],
                "industry": ["beverage", "new_energy"],
                "board": ["MainBoard", "ChiNext"],
                "listing_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "total_market_cap": [100.0, 200.0],
                "float_market_cap": [80.0, 150.0],
                "total_shares": [10.0, 20.0],
                "float_shares": [8.0, 15.0],
                "is_st": [True, False],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.1, 20.2],
                "volume": [100, 200],
                "amount": [1000.0, 2000.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, True],
                "limit_up": [False, False],
                "limit_down": [False, False],
            }
        ),
    )

    frame = snapshot_stock_statuses(store, as_of=as_of)

    assert frame.height == 2
    rows = {row["symbol"]: row for row in frame.to_dicts()}
    assert rows["600519.SH"]["price_limit_pct"] == 0.05
    assert rows["600519.SH"]["is_tradeable"] is False
    assert rows["300750.SZ"]["is_suspended"] is True


def test_build_selection_stock_universe_filters_untradeable(tmp_path: Path) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "index_symbol": ["000300.SH", "000300.SH"],
                "symbol": ["600519.SH", "300750.SZ"],
                "exchange": ["SH", "SZ"],
                "industry": ["food", "battery"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "short_name": ["*ST茅台", "宁德时代"],
                "exchange": ["SH", "SZ"],
                "industry": ["beverage", "new_energy"],
                "board": ["MainBoard", "ChiNext"],
                "listing_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "total_market_cap": [100.0, 200.0],
                "float_market_cap": [80.0, 150.0],
                "total_shares": [10.0, 20.0],
                "float_shares": [8.0, 15.0],
                "is_st": [True, False],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.1, 20.2],
                "volume": [100, 200],
                "amount": [1000.0, 2000.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, False],
                "limit_up": [False, False],
                "limit_down": [False, False],
            }
        ),
    )

    universe = StockSelectionPipeline.from_index(
        store,
        index_symbol="000300.SH",
        as_of=as_of,
    ).to_universe(StockCandidateSpec())

    assert universe.symbols() == ["300750.SZ"]


def test_build_stock_selection_frame_joins_status_metadata_and_fundamentals(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "index_symbol": ["8841431.WI", "8841431.WI"],
                "symbol": ["600519.SH", "300750.SZ"],
                "exchange": ["SH", "SZ"],
                "industry": ["food", "battery"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "short_name": ["贵州茅台", "宁德时代"],
                "exchange": ["SH", "SZ"],
                "industry": ["beverage", "new_energy"],
                "board": ["MainBoard", "ChiNext"],
                "listing_date": [date(2010, 1, 1), date(2026, 3, 1)],
                "total_market_cap": [100.0, 80.0],
                "float_market_cap": [60.0, 50.0],
                "total_shares": [10.0, 8.0],
                "float_shares": [6.0, 5.0],
                "is_st": [False, False],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.1, 20.2],
                "volume": [100, 200],
                "amount": [8000000.0, 3000000.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, False],
                "limit_up": [False, True],
                "limit_down": [False, False],
            }
        ),
    )
    store.write(
        "fundamentals",
        pl.DataFrame(
            {
                "report_date": [date(2025, 12, 31), date(2025, 12, 31)],
                "announce_date": [date(2026, 4, 10), date(2026, 4, 11)],
                "symbol": ["600519.SH", "300750.SZ"],
                "pe_ttm": [18.0, 35.0],
                "pb": [2.2, 2.8],
                "ps_ttm": [5.0, 6.0],
                "ev_ebitda": [12.0, 14.0],
                "roe": [0.18, 0.12],
                "roa": [0.09, 0.06],
                "gross_margin": [0.45, 0.31],
                "revenue": [1000.0, 800.0],
                "net_income": [300.0, 120.0],
                "total_liabilities": [800.0, 550.0],
                "total_assets": [2000.0, 1500.0],
                "operating_cashflow": [250.0, 180.0],
            }
        ),
    )
    store.write(
        "analyst_forecasts",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of],
                "symbol": ["600519.SH", "300750.SZ"],
                "report_count": [12, 8],
                "buy": [4, 3],
                "overweight": [3, 2],
                "neutral": [1, 1],
                "underweight": [0, 0],
                "sell": [0, 0],
                "eps_year1": [3.0, 2.0],
                "eps_year2": [3.2, 2.2],
                "eps_year3": [3.4, 2.4],
                "eps_year4": [3.6, 2.6],
            }
        ),
    )
    store.write(
        "announcements",
        pl.DataFrame(
            {
                "symbol": ["600519.SH", "300750.SZ", "300750.SZ"],
                "short_name": ["贵州茅台", "宁德时代", "宁德时代"],
                "title": ["A", "B", "C"],
                "notice_type": ["财务报告", "一般事项", "业绩快报"],
                "notice_date": [
                    date(2026, 4, 10),
                    date(2026, 4, 11),
                    date(2026, 4, 12),
                ],
                "art_code": ["a", "b", "c"],
                "url": ["u1", "u2", "u3"],
            }
        ),
    )

    frame = StockSelectionPipeline.from_index(
        store,
        index_symbol="8841431.WI",
        as_of=as_of,
        announcement_start=date(2026, 4, 1),
        announcement_end=date(2026, 4, 30),
    ).selection_frame()

    assert frame.height == 2
    assert frame.get_column("symbol").to_list() == ["300750.SZ", "600519.SH"]
    rows = {row["symbol"]: row for row in frame.to_dicts()}
    assert rows["300750.SZ"]["listing_days"] == 49
    assert rows["300750.SZ"]["limit_up"] is True
    assert rows["300750.SZ"]["announcement_count"] == 2
    assert rows["600519.SH"]["pe_ttm"] == 18.0
    assert rows["600519.SH"]["operating_cashflow"] == 250.0


def test_stock_selection_pipeline_keeps_category_inputs_separate(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [as_of],
                "index_symbol": ["8841431.WI"],
                "symbol": ["600519.SH"],
                "exchange": ["SH"],
                "industry": ["food"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [as_of],
                "symbol": ["600519.SH"],
                "short_name": ["贵州茅台"],
                "exchange": ["SH"],
                "industry": ["beverage"],
                "board": ["MainBoard"],
                "listing_date": [date(2010, 1, 1)],
                "total_market_cap": [100.0],
                "float_market_cap": [60.0],
                "total_shares": [10.0],
                "float_shares": [6.0],
                "is_st": [False],
            }
        ),
    )
    store.write(
        "analyst_forecasts",
        pl.DataFrame(
            {
                "as_of": [as_of],
                "symbol": ["600519.SH"],
                "report_count": [12],
                "buy": [4],
                "overweight": [3],
                "neutral": [1],
                "underweight": [0],
                "sell": [0],
                "eps_year1": [3.0],
                "eps_year2": [3.2],
                "eps_year3": [3.4],
                "eps_year4": [3.6],
            }
        ),
    )
    store.write(
        "announcements",
        pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "short_name": ["贵州茅台"],
                "title": ["A"],
                "notice_type": ["财务报告"],
                "notice_date": [date(2026, 4, 10)],
                "art_code": ["a"],
                "url": ["u1"],
            }
        ),
    )

    frame = (
        StockSelectionPipeline.from_index(
            store,
            index_symbol="8841431.WI",
            as_of=as_of,
            announcement_start=date(2026, 4, 1),
            announcement_end=date(2026, 4, 30),
        )
        .with_category_inputs()
        .collect()
    )

    assert "report_count" in frame.columns
    assert "announcement_count" in frame.columns
    assert "pe_ttm" not in frame.columns
    assert "amount" not in frame.columns


def test_stock_selection_pipeline_joins_latest_audit_opinion_state(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of, as_of],
                "index_symbol": ["8841431.WI", "8841431.WI", "8841431.WI"],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "exchange": ["SH", "SZ", "SZ"],
                "industry": ["food", "tech", "utility"],
            }
        ),
    )
    store.write(
        "stock_audit_opinions",
        pl.DataFrame(
            {
                "symbol": ["AAA.SH", "AAA.SH", "BBB.SZ"],
                "report_date": [
                    date(2024, 12, 31),
                    date(2025, 12, 31),
                    date(2025, 12, 31),
                ],
                "announce_date": [
                    date(2025, 4, 10),
                    date(2026, 4, 15),
                    date(2026, 4, 10),
                ],
                "opinion": ["无保留意见", "否定意见", "保留意见"],
                "opinion_code": ["unqualified", "adverse", "qualified"],
                "source_notice_type": ["财务报告", "财务报告", "财务报告"],
                "title": ["2024 audit", "2025 audit", "2025 audit"],
                "art_code": ["AUD-1", "AUD-2", "AUD-3"],
                "url": ["u1", "u2", "u3"],
            }
        ),
    )

    frame = (
        StockSelectionPipeline.from_index(
            store,
            index_symbol="8841431.WI",
            as_of=as_of,
        )
        .with_audit_opinion_state(max_age_days=365)
        .collect()
        .sort("symbol")
    )

    rows = {row["symbol"]: row for row in frame.iter_rows(named=True)}
    assert rows["AAA.SH"]["latest_audit_opinion_code"] == "adverse"
    assert rows["AAA.SH"]["has_adverse_audit_opinion"] is True
    assert rows["AAA.SH"]["active_audit_exclusion"] is True
    assert rows["BBB.SZ"]["latest_audit_opinion_code"] == "qualified"
    assert rows["BBB.SZ"]["has_adverse_audit_opinion"] is False
    assert rows["BBB.SZ"]["active_audit_exclusion"] is False
    assert rows["CCC.SZ"]["has_adverse_audit_opinion"] is False
    assert rows["CCC.SZ"]["active_audit_exclusion"] is False


def test_stock_candidate_spec_applies_native_polars_filters(
    tmp_path: Path,
) -> None:
    store = QoreStore(str(tmp_path / "qore.duckdb"), str(tmp_path / "raw"))
    as_of = date(2026, 4, 19)
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of, as_of],
                "index_symbol": ["8841431.WI", "8841431.WI", "8841431.WI"],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "exchange": ["SH", "SZ", "SZ"],
                "industry": ["food", "battery", "tech"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [as_of, as_of, as_of],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "short_name": ["AAA", "BBB", "CCC"],
                "exchange": ["SH", "SZ", "SZ"],
                "industry": ["food", "battery", "tech"],
                "board": ["MainBoard", "ChiNext", "ChiNext"],
                "listing_date": [date(2020, 1, 1), date(2025, 1, 1), date(2026, 4, 1)],
                "total_market_cap": [90.0, 70.0, 50.0],
                "float_market_cap": [60.0, 40.0, 30.0],
                "total_shares": [10.0, 8.0, 6.0],
                "float_shares": [6.0, 4.0, 3.0],
                "is_st": [False, False, False],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [as_of, as_of, as_of],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "open": [10.0, 20.0, 30.0],
                "high": [10.5, 20.5, 30.5],
                "low": [9.8, 19.5, 29.5],
                "close": [10.1, 20.2, 30.1],
                "volume": [100, 200, 300],
                "amount": [8000000.0, 3000000.0, 1000000.0],
                "adj_factor": [1.0, 1.0, 1.0],
                "is_suspended": [False, False, False],
                "limit_up": [False, False, True],
                "limit_down": [False, False, False],
            }
        ),
    )
    store.write(
        "fundamentals",
        pl.DataFrame(
            {
                "report_date": [
                    date(2025, 12, 31),
                    date(2025, 12, 31),
                    date(2025, 12, 31),
                ],
                "announce_date": [
                    date(2026, 4, 10),
                    date(2026, 4, 10),
                    date(2026, 4, 10),
                ],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "pe_ttm": [18.0, 25.0, 15.0],
                "pb": [2.0, 2.5, 2.0],
                "ps_ttm": [5.0, 5.5, 4.0],
                "ev_ebitda": [12.0, 13.0, 11.0],
                "roe": [0.18, 0.11, 0.16],
                "roa": [0.09, 0.06, 0.08],
                "gross_margin": [0.45, 0.31, 0.28],
                "revenue": [1000.0, 800.0, 500.0],
                "net_income": [300.0, 120.0, 100.0],
                "total_liabilities": [800.0, 550.0, 300.0],
                "total_assets": [2000.0, 1500.0, 900.0],
                "operating_cashflow": [250.0, 180.0, 90.0],
            }
        ),
    )

    selection = StockSelectionPipeline.from_index(
        store,
        index_symbol="8841431.WI",
        as_of=as_of,
    ).selection_frame()
    candidates = StockCandidateSpec(
        filters=(
            CandidateFilter("roe", "gt", 0.0, fill_null=float("-inf")),
            CandidateFilter(
                "operating_cashflow",
                "gt",
                0.0,
                fill_null=float("-inf"),
            ),
            CandidateFilter("pe_ttm", "between", (0.0, 50.0)),
            CandidateFilter("pb", "between", (0.0, 3.0)),
        ),
        sort_by=(CandidateSort("total_market_cap"),),
        top_n=20,
        min_listing_days=60,
        exclude_limit_up=True,
        exclude_limit_down=True,
    ).apply(
        selection,
    )

    assert candidates.get_column("symbol").to_list() == ["BBB.SZ", "AAA.SH"]
