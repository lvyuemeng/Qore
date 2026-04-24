from collections.abc import Sequence
from datetime import date
from functools import reduce
from typing import (
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

import polars as pl
from pyparsing import Callable

import quant_trade.provider.akshare as ak
import quant_trade.provider.baostock as bs
from quant_trade.config.arctic import ArcticAdapter, ArcticDB, Lib
from quant_trade.config.logger import log
from quant_trade.feature.process import MarginShort, Northbound, Shibor

from .process import (
    Behavioral,
    Fundamental,
    SectorGroup,
)
from .query import MacroBook, QuarterBook


class AssetLib:
    """Base class for ArcticDB-backed asset data."""

    REGION: str
    SYMBOL: str

    def __init__(self, db: ArcticDB):
        self._lib = db.get_lib(self.lib_name())

    @classmethod
    def lib_name(cls) -> str:
        return f"{cls.REGION}_{cls.SYMBOL}"

    @property
    def raw(self) -> Lib:
        """Direct access to ArcticDB library (advanced use only)."""
        return self._lib

    def _has(self, book: str) -> bool:
        return self._lib.has_symbol(book)

    def _try_read(self: Self, book: str) -> pl.DataFrame:
        log.info(f"Read data: {book}")
        try:
            return ArcticAdapter.from_read(self._lib.read(book))
        except Exception:
            log.warning(f"Read empty data of {book}")
            return pl.DataFrame()

    def _try_write(self: Self, book: str, df: pl.DataFrame) -> None:
        log.info(f"Write data: {book}")
        try:
            self._lib.write(book, ArcticAdapter.to_write(df))
        except Exception as e:
            log.error(f"Database write failed for {book}: {e}")


@runtime_checkable
class BookFetcher[B](Protocol):
    """Protocol for classes that can fetch data by BookKey."""

    def _key(self, book: B) -> str:
        raise NotImplementedError

    def _fetch(self, book: B) -> pl.DataFrame: ...
    def _process(self, book: B, raw: pl.DataFrame) -> pl.DataFrame:
        return raw


@runtime_checkable
class BatchBookFetcher[B](BookFetcher, Protocol):
    """Protocol for classes that can fetch data by BookKey in batch."""

    def _fetch_batch(self, books: Sequence[B]) -> Sequence[pl.DataFrame]: ...


class BookLib[B](BookFetcher[B], AssetLib):
    """Mixin for ArcticDB-backed data by BookKey."""

    def _fresh(self: Self, book: B) -> None:
        log.info(f"Fetching & processing data for {book}")
        raw = self._fetch(book)
        # if raw.is_empty():
        #     log.warning(f"No raw data for {book}")
        #     return
        feat = self._process(book, raw)
        self._try_write(self._key(book), feat)

    def _read(self: Self, book: B, fresh: bool = False) -> pl.DataFrame:
        key = self._key(book)
        if fresh or not self._lib.has_symbol(key):
            self._fresh(book)
        if self._lib.has_symbol(key):
            return ArcticAdapter.from_read(self._lib.read(key))
        raise KeyError(f"Data for {book} not found in {self.lib_name()}")


class BatchBookLib[B](BatchBookFetcher[B], AssetLib):
    def _batch_fresh(self: Self, books: Sequence[B]) -> None:
        batch = self._fetch_batch(books)
        for book, df in zip(books, batch, strict=False):
            # if df.is_empty():
            #     log.warning(f"No market data for {book}")
            #     continue
            feat = self._process(book, df)
            self._try_write(self._key(book), feat)

    def _batch_read(
        self: Self, books: Sequence[B], *, fresh: bool = False
    ) -> list[pl.DataFrame]:
        data: dict[B, pl.DataFrame] = {}
        to_fetch: list[B] = []

        if not fresh:
            for book in books:
                key = self._key(book)
                if (df := self._try_read(key)) is not None:
                    data[book] = df
                else:
                    to_fetch.append(book)
        else:
            to_fetch.extend(books)

        if to_fetch:
            for book, raw in zip(to_fetch, self._fetch_batch(to_fetch), strict=False):
                if raw.is_empty():
                    continue
                df = self._process(book, raw)
                if not df.is_empty():
                    data[book] = df
                    key = self._key(book)
                    self._try_write(key, df)

        return [
            data[book] for book in books if (book in data) and not data[book].is_empty()
        ]


class CNStockPool(AssetLib):
    REGION = "CN"
    SYMBOL = "stock"

    type Book = Literal["industry_code"]
    type Universe = Literal[
        "sci", "ssmi", "csi500", "csi1000", "csi2000", "csia100", "csia500_value"
    ]

    def __init__(self, db: ArcticDB):
        AssetLib.__init__(self, db)

    @staticmethod
    def _index_universe(universe: Universe, date: date) -> str:
        return f"{universe}_{date}"

    def _fetch_whole(self, book: Book) -> pl.DataFrame:
        match book:
            # case "stock_code":
            #     return ak.AkShareUniverse().stock_whole()
            case "industry_code":
                return ak.SWIndustryCls().stock_l1_industry_cls()

    def _fetch_pool(self, universe: Universe, date: date) -> pl.DataFrame:
        match universe:
            case "sci":
                return ak.AkShareUniverse().csi_index_cons("000001")
            case "ssmi":
                return ak.AkShareUniverse().csi_index_cons("000046")
            case "csia100":
                return ak.AkShareUniverse().csi_index_cons("000903")
            case "csi500":
                return ak.AkShareUniverse().csi_index_cons("000905")
            case "csi2000":
                return ak.AkShareUniverse().csi_index_cons("932000")
            case "csi1000":
                return ak.AkShareUniverse().csi_index_cons("000852")
            case "csia500_value":
                return ak.AkShareUniverse().csi_index_cons("932497")

    def read_codes(self, book: Book, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._has(book):
            self._try_write(book, self._fetch_whole(book))
        df = self._try_read(book)
        return df if df is not None else pl.DataFrame()

    def read_pool(
        self,
        universe: Universe,
        date: date,
        industry_cls: bool = False,
        fresh: bool = False,
    ) -> pl.DataFrame:
        index = self._index_universe(universe, date)
        if fresh or not self._has(index):
            self._try_write(index, self._fetch_pool(universe, date))
        df = self._try_read(index)
        if industry_cls:
            ind = self.read_codes("industry_code", fresh=fresh)
            df = df.join(ind, on="ts_code", how="inner")
            return df
        return df


class CNMarket(BookLib[str], BatchBookLib[str], AssetLib):
    REGION = "CN"
    SYMBOL = "market"

    def __init__(
        self, db: ArcticDB, source: Literal["akshare", "baostock"] = "akshare"
    ):
        AssetLib.__init__(self, db)
        self._source: Literal["akshare", "baostock"] = source

    def _key(self, book: str) -> str:
        return book

    def _fetch(self, book: str) -> pl.DataFrame:
        match self._source:
            case "akshare":
                return ak.AkShareMicro().market_ohlcv(book, "daily")
            case "baostock":
                return bs.BaoMicro().market_ohlcv(book, "daily")

    def _fetch_batch(self, books: Sequence[str]) -> list[pl.DataFrame]:
        match self._source:
            case "akshare":
                return ak.AkShareMicro().batch_market_ohlcv(books, "daily")
            case "baostock":
                return bs.BaoMicro().batch_market_ohlcv(books, "daily")

    def _process(self, book: str, raw: pl.DataFrame) -> pl.DataFrame:
        log.debug(f"raw df: {raw.head(5)}")
        return (
            Behavioral()
            .metrics(raw, idents=["ts_code", "name"])
            .sort("date")
            .unique(subset=["date"], keep="last")
        )

    def read(self, book: str, fresh: bool = False) -> pl.DataFrame:
        return self._read(book, fresh=fresh)

    def stack_read(self, books: Sequence[str], fresh: bool = False) -> pl.DataFrame:
        frames = self._batch_read(books, fresh=fresh)
        df = pl.concat(frames, how="diagonal")
        return df

    def range_read(
        self,
        books: Sequence[str],
        start: date,
        end: date,
        fresh: bool = False,
    ) -> pl.DataFrame:
        stack = self.stack_read(books=books, fresh=fresh)
        df = stack.filter(pl.col("date").is_between(start, end))
        return df


class CNFundamental(BookLib[QuarterBook], BatchBookLib[QuarterBook], AssetLib):
    REGION = "CN"
    SYMBOL = "fundamental"

    def __init__(self, db: ArcticDB):
        AssetLib.__init__(self, db)

    def _key(self, book: QuarterBook) -> str:
        return book.key

    def _fetch(self, book: QuarterBook) -> pl.DataFrame:
        return ak.AkShareMicro().quarterly_fundamentals(book.year, book.literal_quarter)

    def _fetch_batch(self, books: Sequence[QuarterBook]) -> Sequence[pl.DataFrame]:
        return [
            ak.AkShareMicro().quarterly_fundamentals(book.year, book.literal_quarter)
            for book in books
        ]

    def _process(self, book: QuarterBook, raw: pl.DataFrame) -> pl.DataFrame:
        log.debug(f"raw df: {raw.head(5)}")
        return (
            Fundamental()
            .metrics(raw)
            .sort("ts_code")
            .unique(subset=["ts_code"], keep="last")
        )

    def read(self, book: QuarterBook, fresh: bool = False) -> pl.DataFrame:
        return self._read(book, fresh=fresh)

    def update_range(self, start: date, end: date, force: bool = False):
        """
        Optimized Batch Entry Point.
        Fetches the entire range once, then distributes data to individual quarter books.
        """
        log.info(f"Batch updating {self.SYMBOL} from {start} to {end}")

        raw_frames = ak.AkShareMicro().batch_quarterly_fundamentals(start, end)
        books = list(QuarterBook.date_range(start, end))
        for book, df in zip(books, raw_frames, strict=True):
            if not df.is_empty() or force:
                processed = self._process(book, df)
                self._try_write(self._key(book), processed)

    def range_read(self, start: date, end: date, fresh: bool = False) -> pl.DataFrame:
        books = list(QuarterBook.date_range(start, end))
        if fresh:
            self.update_range(start, end)
        frames = self._batch_read(books)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")


class CNMacro(BookLib[MacroBook], AssetLib):
    REGION = "CN"
    SYMBOL = "macro"

    _INITIALIZED: ClassVar[bool] = False
    _FETCHERS: ClassVar[dict[str, Callable[[], pl.DataFrame]]] = {}
    _FEATURES: ClassVar[dict[str, Callable[[pl.DataFrame], pl.DataFrame]]] = {}

    def __init__(self, db: ArcticDB):
        AssetLib.__init__(self, db)
        self._init_components_once()

    def _init_components_once(self) -> None:
        """Lazy one-time initialization of all fetcher & feature callables"""
        if hasattr(self.__class__, "_components_initialized"):
            return

        macro = ak.AkShareMacro()
        self.__class__._FETCHERS.update(
            {
                "northbound": macro.northbound_flow,
                "marginshort": macro.market_margin_short,
                "shibor": macro.shibor,
                "index": macro.csi1000_daily_ohlcv,
                "qvix": macro.csi1000qvix_daily_ohlc,
            }
        )
        nb = Northbound()
        ms = MarginShort()
        sh = Shibor()
        beh = Behavioral()
        self.__class__._FEATURES.update(
            {
                "northbound": nb.metrics,
                "marginshort": ms.metrics,
                "shibor": sh.metrics,
                "index": beh.metrics,
                "qvix": beh.metrics,
            }
        )

        self.__class__._INITIALIZED = True

    def _key(self, book: MacroBook) -> str:
        return book.to_key()

    def _fetch(self, book: MacroBook) -> pl.DataFrame:
        try:
            return self._FETCHERS[book.macro]()
        except KeyError as e:
            raise ValueError(f"Unknown macro book fetch: {book.macro}") from e

    def _process(self, book: MacroBook, raw: pl.DataFrame) -> pl.DataFrame:
        log.debug(f"raw df: {raw.head(5)}")
        try:
            return self._FEATURES[book.macro](raw)
        except KeyError as e:
            raise ValueError(f"Unknown macro book process: {book.macro}") from e

    def read(self, book: MacroBook, *, fresh: bool = False) -> pl.DataFrame:
        df = self._read(book, fresh)
        return df

    def read_all(self, *, fresh: bool = False) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for book in MacroBook.list():
            try:
                m = self._read(book, fresh=fresh)
                frames.append(m)
            except KeyError as e:
                log.warning(f"Failed to read book '{book}': {e}")
                pass
        df = reduce(
            lambda df_1, df_2: df_1.join(df_2, on="date", how="inner", coalesce=True),
            frames,
        )
        return df


class CNIndustrySectorGroup(SectorGroup):
    INDUS_COL: Final[list[str]] = ["sw_l1_code"]

    @classmethod
    def default_fundamental_metric(cls) -> list[str]:
        """Default fundamental metrics

        ```
        # Profitability & margins (industry structure driven)
        "gross_margin",
        "operate_margin",
        "net_margin",
        # Capital efficiency (very industry dependent)
        "roe",
        "roa",
        "asset_turnover",
        # Leverage & balance sheet structure
        "debt_to_equity",
        "debt_asset_ratio",
        # Liquidity ratios (business-model specific)
        "current_ratio",
        "quick_ratio",
        # Cash flow efficiency
        "cfo_yield",
        "ttm_cfo_yield",
        # Accrual / earnings quality (sector accounting conventions)
        "accrual_ratio",
        # Growth rates (structural growth differences)
        "revenue_growth_yoy",
        "net_profit_growth_yoy",
        "asset_growth_yoy",
        "debt_growth_yoy",
        # TTM profitability
        "ttm_roe",
        ```
        """
        return [
            # Profitability & margins (industry structure driven)
            "gross_margin",
            "operate_margin",
            "net_margin",
            # Capital efficiency (very industry dependent)
            "roe",
            "roa",
            "asset_turnover",
            # Leverage & balance sheet structure
            "debt_to_equity",
            "debt_asset_ratio",
            # Liquidity ratios (business-model specific)
            "current_ratio",
            "quick_ratio",
            # Cash flow efficiency
            "cfo_yield",
            "ttm_cfo_yield",
            # Accrual / earnings quality (sector accounting conventions)
            "accrual_ratio",
            # Growth rates (structural growth differences)
            "revenue_growth_yoy",
            "net_profit_growth_yoy",
            "asset_growth_yoy",
            "debt_growth_yoy",
            # TTM profitability
            "ttm_roe",
        ]

    def __init__(
        self,
        db: ArcticDB,
        factors: list[str],
        by: list[str] | None = None,
        zsuffix: str = "z",
        min_group_size: int = 5,
        winsorize: bool = True,
        winsor_limits: tuple[float, float] = (0.01, 0.99),
    ):
        self.stock_pool = CNStockPool(db)
        self.industry_df: pl.DataFrame | None = None
        group_by = self.INDUS_COL + by if by else list(self.INDUS_COL)
        super().__init__(
            by=group_by,
            factors=factors,
            zsuffix=zsuffix,
            min_group_size=min_group_size,
            winsorize=winsorize,
            winsor_limits=winsor_limits,
        )

    def __call__(self, df: pl.DataFrame, fresh: bool = False) -> pl.DataFrame:
        self.industry_df = self.stock_pool.read_codes("industry_code", fresh)
        required = ["ts_code"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"factor_df missing required columns: {missing}")

        merged = df.join(self.industry_df, on=required, how="inner")

        original_rows = len(df)
        merged_rows = len(merged)
        if merged_rows < original_rows:
            log.info(
                f"Industry join: {merged_rows}/{original_rows} rows retained "
                f"({merged_rows / original_rows * 100:.1f}%)"
            )
        if merged_rows == 0:
            log.warning("No data after industry join. Check industry data coverage.")
            return df

        return self.normalize(merged)
