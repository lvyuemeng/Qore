from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from qore_data import DataSettings
from qore_data.fetcher import (
    AnnouncementFetcher,
    ConstituentFetcher,
    FinancialFetcher,
    QuoteFetcher,
)
from qore_data.fetcher.analyst import AnalystFetcher
from qore_data.store.duckdb import QoreStore

logger = logging.getLogger("qore.data.fetch")


_DATE_COLUMNS: dict[str, str] = {
    "stock_ohlcv": "date",
    "fund_nav": "date",
    "fundamentals": "report_date",
    "announcements": "notice_date",
    "stock_audit_opinions": "report_date",
    "analyst_forecasts": "as_of",
    "index_constituents": "as_of",
}


@dataclass(slots=True)
class StockPipeline:
    store: QoreStore
    quote: QuoteFetcher
    financial: FinancialFetcher
    analyst: AnalystFetcher
    announcement: AnnouncementFetcher
    index: ConstituentFetcher

    @classmethod
    def from_settings(cls, settings: DataSettings) -> StockPipeline:
        store = QoreStore.from_settings(settings)
        return cls(
            store=store,
            quote=QuoteFetcher.from_settings(settings),
            financial=FinancialFetcher.from_settings(settings),
            analyst=AnalystFetcher.from_settings(settings),
            announcement=AnnouncementFetcher.from_settings(settings),
            index=ConstituentFetcher.from_settings(settings),
        )

    # ── read (provenance) ─────────────────────────────────────────────────

    def read(
        self,
        dataset: str,
        symbols: list[str] | None = None,
        dates: tuple[date, ...] | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Read a dataset with optional symbol and date-range filters.

        Auto-resolves the date column per dataset (``date``, ``report_date``, etc.).
        Returns a ``pl.LazyFrame`` — compose with Polars lazy API.
        """
        filters: dict[str, Any] = {}
        if symbols is not None:
            filters["symbol"] = symbols
        if dates is not None:
            date_col = _DATE_COLUMNS.get(dataset, "date")
            filters[date_col] = dates
        return self.store.read(
            dataset, filters=filters if filters else None, columns=columns
        )

    def read_sql(self, query: str) -> pl.LazyFrame:
        """Execute raw SQL against the DuckDB store."""
        return self.store.sql(query)

    # ── corpus lenses ────────────────────────────────────────────────────

    def market_corpus(
        self,
        symbols: list[str],
        start: date,
        end: date,
        include_fundamentals: bool = True,
    ) -> pl.LazyFrame:
        """Join ``stock_ohlcv`` with ``stock_info`` (and optionally ``fundamentals``).

        Returns a lazy frame with daily OHLCV + identity columns,
        optionally enriched with latest fundamentals per symbol.
        """
        ohlcv = self.read("stock_ohlcv", symbols=symbols)
        ohlcv = ohlcv.filter(pl.col("date").is_between(start, end))
        info = self.read("stock_info", symbols=symbols)
        lf = ohlcv.join(info, on="symbol", how="left")
        if include_fundamentals:
            fund = self.read("fundamentals", symbols=symbols)
            fund = fund.sort("report_date", descending=True).unique(
                subset=["symbol"], keep="first"
            )
            lf = lf.join(fund, on="symbol", how="left")
        return lf

    def fundamental_corpus(
        self,
        symbols: list[str],
        as_of: date | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Join ``fundamentals`` with ``analyst_forecasts`` per symbol.

        Returns the latest fundamentals row per symbol, optionally
        enriched with analyst EPS consensus. Filters to ``as_of``
        announcement date if provided.
        """
        fund = self.read("fundamentals", symbols=symbols, columns=columns)
        fund = fund.sort("announce_date", descending=True).unique(
            subset=["symbol"], keep="first"
        )
        if as_of is not None:
            fund = fund.filter(pl.col("announce_date") <= as_of)
        analyst = self.read("analyst_forecasts", symbols=symbols)
        analyst = analyst.sort("as_of", descending=True).unique(
            subset=["symbol"], keep="first"
        )
        return fund.join(analyst, on="symbol", how="left")

    # ── fetch pipeline ───────────────────────────────────────────────────

    async def resolve(self, index_symbol: str, as_of: date) -> pl.Series:
        symbols = await self.index.index_constituents(index_symbol, as_of)
        self.store.write(
            "index_constituents",
            pl.DataFrame(
                {
                    "as_of": [as_of] * len(symbols),
                    "index_symbol": [index_symbol] * len(symbols),
                    "symbol": symbols.to_list(),
                },
                schema={
                    "as_of": pl.Date,
                    "index_symbol": pl.String,
                    "symbol": pl.String,
                },
            ),
        )
        return symbols

    async def stock_profiles(
        self, symbols: list[str], as_of: date | None = None
    ) -> None:
        a = as_of or date.today()
        frame = await self.quote.batch_stock_profiles(symbols, a)
        if not frame.is_empty():
            self.store.write("stock_info", frame)

    async def stock_daily(
        self, symbols: list[str], start: date | None = None, end: date | None = None
    ) -> None:
        if not symbols:
            return
        s = start or date(2015, 1, 1)
        e = end or date.today()
        total = len(symbols)
        logger.info("fetch_quote dataset=stock_ohlcv symbols=%d start", total)
        t0 = time.monotonic()
        results = await self.quote.batch_stock_daily(symbols, s, e)
        non_empty = [r for r in results if not r.is_empty()]
        if non_empty:
            combined = pl.concat(non_empty, how="vertical_relaxed")
            self.store.write("stock_ohlcv", combined)
        elapsed = time.monotonic() - t0
        rows = sum(len(r) for r in non_empty)
        logger.info(
            "fetch_quote dataset=stock_ohlcv symbols=%d rows=%d elapsed=%.2fs",
            total,
            rows,
            elapsed,
        )

    async def fundamentals(self, symbols: list[str], as_of: date | None = None) -> None:
        if not symbols:
            return
        a = as_of or date.today()
        total = len(symbols)
        logger.info("fetch_financial dataset=fundamentals symbols=%d start", total)
        t0 = time.monotonic()
        results = await self.financial.batch_fundamentals(symbols, a)
        non_empty = [r for r in results if not r.is_empty()]
        if non_empty:
            combined = pl.concat(non_empty, how="vertical_relaxed")
            self.store.write("fundamentals", combined)
        elapsed = time.monotonic() - t0
        rows = sum(len(r) for r in non_empty)
        logger.info(
            "fetch_financial dataset=fundamentals symbols=%d rows=%d elapsed=%.2fs",
            total,
            rows,
            elapsed,
        )

    async def analyst_forecasts(
        self, symbols: list[str], as_of: date | None = None
    ) -> None:
        a = as_of or date.today()
        frame = await self.analyst.batch_analyst_forecasts(symbols, a)
        if not frame.is_empty():
            self.store.write("analyst_forecasts", frame)

    async def announcements(
        self, symbols: list[str], start: date | None = None, end: date | None = None
    ) -> None:
        s, e = start or date(2000, 1, 1), end or date.today()
        frame = await self.announcement.batch_announcements(symbols, s, e)
        if not frame.is_empty():
            self.store.write("announcements", frame)

    async def audit_opinions(
        self, symbols: list[str], start: date | None = None, end: date | None = None
    ) -> None:
        s, e = start or date(2000, 1, 1), end or date.today()
        frame = await self.announcement.batch_audit_opinions(symbols, s, e)
        if not frame.is_empty():
            self.store.write("stock_audit_opinions", frame)

    async def close(self) -> None:
        await self.quote.close()
        await self.financial.close()
        await self.analyst.close()
        await self.announcement.close()
        await self.index.close()
