from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from qore_core import StockInstrument
from qore_data.store.duckdb import QoreStore
from qore_data.universe import (
    build_stock_universe_from_index,
    evaluate_stock_categories,
    snapshot_index_constituents,
    snapshot_stock_analyst_forecasts,
    snapshot_stock_announcements,
    snapshot_stock_profiles,
)


class StubStockSource:
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
    persisted = store.read("index_constituents").collect()
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
    persisted = store.read("stock_profiles").collect()
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
    persisted = store.read("analyst_forecasts").collect()
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
    persisted = store.read("announcements").collect()
    assert persisted.height == 2


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

    report = evaluate_stock_categories(
        store,
        index_symbol="000300.SH",
        as_of=as_of,
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    assert report.height == 2
    assert set(report.columns) == {
        "industry",
        "board",
        "symbol_count",
        "avg_total_market_cap",
        "avg_report_count",
        "announcement_count",
    }
