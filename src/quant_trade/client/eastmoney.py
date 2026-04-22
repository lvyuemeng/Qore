"""East Money data provider with separated concerns.

Architecture:
- Fetcher: Independent HTTP fetching (network only)
- Parser: Converts raw JSON to structured data
- Builder: Constructs final output DataFrame
- ReportPipe: Specialized pipe for quarterly financial reports
- KlinePipe: Specialized pipe for stock historical data
- EastMoney: Facade that composes all components (no inner Fetcher exposed)

Uses generic traits from trait.py for reusable patterns.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import polars as pl

from quant_trade.client.traits import (
    BaseBuilder,
    BaseFetcher,
    ColumnSpec,
)
from quant_trade.config.logger import log

from ..transform import AdjustCN, Period, Quarter, quarter_range

# =============================================================================
# Constants
# =============================================================================

EASTMONEY_FINANCE_API: Final[str] = (
    "https://datacenter-web.eastmoney.com/api/data/v1/get"
)
EASTMONEY_KLINE_API: Final[str] = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
)
QUARTER_END_MONTH_DAYS: Final[dict[int, str]] = {
    1: "03-31",
    2: "06-30",
    3: "09-30",
    4: "12-31",
}
KLINE_ADJUST_MAP: Final[dict[str, str]] = {"qfq": "1", "hfq": "2", "": "0"}
KLINE_PERIOD_MAP: Final[dict[str, str]] = {
    "daily": "101",
    "weekly": "102",
    "monthly": "103",
}
DEFAULT_HIST_START: Final[date] = date(1970, 1, 1)
DEFAULT_HIST_END: Final[date] = date(2050, 1, 1)

# =============================================================================
# Helper Functions
# =============================================================================


def _build_quarterly_date(year: int, quarter: int) -> str:
    """Build formatted date string from year and quarter.

    Args:
        year: Report year (e.g., 2024)
        quarter: Report quarter (1, 2, 3, or 4)

    Returns:
        Formatted date string (e.g., "2024-03-31")
    """
    month, day = QUARTER_END_MONTH_DAYS[quarter].split("-")
    return f"{year}-{month}-{day}"


def _build_quarterly_params(report_name: str, formatted_date: str) -> dict:
    """Build params for quarterly report API.

    Args:
        report_name: Report name (e.g., "RPT_DMSK_FN_INCOME")
        formatted_date: Formatted date string (e.g., "2024-03-31")

    Returns:
        Dictionary of API parameters
    """
    return {
        "sortColumns": "NOTICE_DATE,SECURITY_CODE",
        "sortTypes": "-1,-1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": report_name,
        "columns": "ALL",
        "filter": f"""(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")(REPORT_DATE='{formatted_date}')""",
    }


def _build_kline_params(
    symbol: str,
    period: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> tuple[str, dict]:
    """Build params for kline API.

    Args:
        symbol: Stock code (e.g., "000001")
        period: "daily", "weekly", or "monthly"
        start_date: Start date in "YYYYMMDD" format
        end_date: End date in "YYYYMMDD" format
        adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

    Returns:
        Tuple of (secid, params_dict)
    """
    market_code = 1 if symbol.startswith("6") else 0

    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": KLINE_PERIOD_MAP[period],
        "fqt": KLINE_ADJUST_MAP[adjust],
        "secid": f"{market_code}.{symbol}",
        "beg": start_date,
        "end": end_date,
    }

    return f"{market_code}.{symbol}", params


def _resolve_hist_date_bounds(
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    return start_date or DEFAULT_HIST_START, end_date or DEFAULT_HIST_END


# =============================================================================
# Concrete Implementations
# =============================================================================
#
class EastMoneyFetcher(BaseFetcher):
    """EastMoney-specific Fetcher - extends BaseFetch."""

    # User agents for rotation
    USER_AGENTS: list[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/2010",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a private event loop for this instance to handle the async bridge
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError as e:
            log.error(f"Async runtime error: {e}")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def run(self, future) -> Any:
        return self._loop.run_until_complete(future)

    async def _fetch_once(self, url: str, params: dict) -> dict:
        """Fetch initial page to get total pages."""
        response = await self.client.get(
            url,
            params=params,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _fetch_page(self, url: str, params: dict, page: int) -> dict:
        """Fetch single page."""
        params_copy = dict(params)
        params_copy["pageNumber"] = str(page)

        response = await self.client.get(
            url,
            params=params_copy,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    def sclose(self) -> None:
        """Gracefully close the underlying async connection pool."""
        if self.client:
            self._loop.run_until_complete(self.client.aclose())


class EastMoneyParser:
    """Base Parser - converts raw JSON to structured data.

    Provides:
    - Configurable data path traversal via DATA_PATH
    - Total pages extraction via get_total_pages()
    - Basic DataFrame cleaning

    Usage:
        class MyParser(BaseParser):
            DATA_PATH = ("result", "data")

            def parse(self, raw: dict) -> list[dict]:
                # Custom parsing logic
                ...
    """

    def parse(self, raw: dict) -> list[dict]: ...

    def clean(self, data: list[dict]) -> pl.DataFrame:
        """Clean and convert to Polars DataFrame.

        Args:
            data: List of data dictionaries

        Returns:
            Polars DataFrame
        """
        if not data:
            return pl.DataFrame()

        # Use infer_schema_length=None to scan all rows for proper type inference
        # This prevents issues with large numeric values causing overflow
        return pl.DataFrame(data, infer_schema_length=None)


class FundamentalParser(EastMoneyParser):
    """EastMoney-specific Parser - extends BaseParser.

    DATA_PATH configured to extract data from EastMoney API response.
    """

    def parse(self, raw: dict) -> list[dict]:
        """Parse response and extract data list.

        Args:
            raw: Raw JSON response dictionary

        Returns:
            List of data dictionaries
        """
        DATA_PATH: list[str] = ["result", "data"]
        if not raw.get("result"):
            return []

        # Navigate to data using DATA_PATH
        data = raw
        for key in DATA_PATH:
            data = data.get(key, {})
            if data is None:
                return []

        return data if isinstance(data, list) else []

    def get_total_pages(self, raw: dict) -> int:
        """Get total pages from response.

        Args:
            raw: Raw JSON response dictionary

        Returns:
            Total number of pages (default: 1)
        """
        PAGES_KEY: str = "pages"
        return raw.get("result", {}).get(PAGES_KEY, 1)


FundemantalParser = FundamentalParser


class KlineParser(EastMoneyParser):
    """Parser for stock kline (historical) data.

    DATA_PATH configured for kline API response structure.
    Data is returned as comma-separated strings that need parsing.
    """

    DATA_PATH = ("data", "klines")

    def parse(self, raw: dict) -> list[dict]:
        """Parse kline response and extract data list.

        Args:
            raw: Raw JSON response dictionary

        Returns:
            List of data dictionaries
        """
        if not raw.get("data"):
            return []

        klines = raw.get("data", {}).get("klines", [])
        if not klines:
            return []

        # Parse comma-separated strings into dictionaries
        column_names = [
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "change_pct",
            "change_amt",
            "turnover",
        ]

        data = []
        for kline in klines:
            values = kline.split(",")
            if len(values) == len(column_names):
                data.append(dict(zip(column_names, values, strict=False)))

        return data


# =============================================================================
# Builders - Use ColumnSpec pattern from trait.py
# =============================================================================


class IncomeBuilder(BaseBuilder):
    """Builder for LRB (Income Statement) data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("SECURITY_CODE", "ts_code"),
        ColumnSpec("SECURITY_NAME_ABBR", "name"),
        ColumnSpec("NOTICE_DATE", "notice_date"),
        ColumnSpec("PARENT_NETPROFIT", "net_profit", pl.Float64),
        ColumnSpec("PARENT_NETPROFIT_RATIO", "net_profit_yoy", pl.Float64),
        ColumnSpec("TOTAL_OPERATE_INCOME", "total_revenue", pl.Float64),
        ColumnSpec("TOI_RATIO", "total_revenue_yoy", pl.Float64),
        ColumnSpec("TOTAL_OPERATE_COST", "total_cost", pl.Float64),
        ColumnSpec("TOE_RATIO", "total_cost_yoy", pl.Float64),
        ColumnSpec("OPERATE_COST", "operate_cost", pl.Float64),
        ColumnSpec("OPERATE_EXPENSE", "operate_expense", pl.Float64),
        ColumnSpec("OPERATE_EXPENSE_RATIO", "operate_expense_ratio", pl.Float64),
        ColumnSpec("SALE_EXPENSE", "sale_expense", pl.Float64),
        ColumnSpec("MANAGE_EXPENSE", "manage_expense", pl.Float64),
        ColumnSpec("FINANCE_EXPENSE", "finance_expense", pl.Float64),
        ColumnSpec("OPERATE_PROFIT", "operate_profit", pl.Float64),
        ColumnSpec("TOTAL_PROFIT", "total_profit", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "ts_code",
        "name",
        "notice_date",
        "total_profit",
        "net_profit",
        "net_profit_yoy",
        "operate_profit",
        "total_revenue",
        "total_revenue_yoy",
        "total_cost",
        "total_cost_yoy",
        "operate_cost",
        "operate_expense",
        "operate_expense_ratio",
        "sale_expense",
        "manage_expense",
        "finance_expense",
    ]

    DATE_COL = "notice_date"
    DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")


class BalanceSheetBuilder(BaseBuilder):
    """Builder for Balance Sheet data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("SECURITY_CODE", "ts_code"),
        ColumnSpec("SECURITY_NAME_ABBR", "name"),
        ColumnSpec("NOTICE_DATE", "notice_date"),
        ColumnSpec("TOTAL_ASSETS", "total_asset", pl.Float64),
        ColumnSpec("TOTAL_ASSETS_RATIO", "total_asset_yoy", pl.Float64),
        ColumnSpec("MONETARYFUNDS", "cash", pl.Float64),
        ColumnSpec("ACCOUNTS_RECE", "accounts_receivable", pl.Float64),
        ColumnSpec("INVENTORY", "inventory", pl.Float64),
        ColumnSpec("TOTAL_LIABILITIES", "total_debt", pl.Float64),
        ColumnSpec("TOTAL_LIAB_RATIO", "total_debt_yoy", pl.Float64),
        ColumnSpec("ACCOUNTS_PAYABLE", "accounts_payable", pl.Float64),
        ColumnSpec("ADVANCE_RECEIVABLES", "advance_receivable", pl.Float64),
        ColumnSpec("TOTAL_EQUITY", "total_equity", pl.Float64),
        ColumnSpec("DEBT_ASSET_RATIO", "debt_asset_ratio", pl.Float64),
        ColumnSpec("CURRENT_RATIO", "current_ratio", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "ts_code",
        "name",
        "notice_date",
        "total_asset",
        "total_asset_yoy",
        "cash",
        "accounts_receivable",
        "inventory",
        "total_debt",
        "total_debt_yoy",
        "accounts_payable",
        "advance_receivable",
        "debt_asset_ratio",
        "current_ratio",
        "total_equity",
    ]

    DATE_COL = "notice_date"
    DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")


class CashFlowBuilder(BaseBuilder):
    """Builder for Cash Flow Statement data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("SECURITY_CODE", "ts_code"),
        ColumnSpec("SECURITY_NAME_ABBR", "name"),
        ColumnSpec("NOTICE_DATE", "notice_date"),
        ColumnSpec("NETCASH_OPERATE", "net_cashflow", pl.Float64),
        ColumnSpec("NETCASH_OPERATE_RATIO", "net_cashflow_yoy", pl.Float64),
        ColumnSpec("SALES_SERVICES", "cfo", pl.Float64),
        ColumnSpec("SALES_SERVICES_RATIO", "cfo_ratio", pl.Float64),
        ColumnSpec("NETCASH_INVEST", "cfi", pl.Float64),
        ColumnSpec("NETCASH_INVEST_RATIO", "cfi_ratio", pl.Float64),
        ColumnSpec("NETCASH_FINANCE", "cff", pl.Float64),
        ColumnSpec("NETCASH_FINANCE_RATIO", "cff_ratio", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "ts_code",
        "name",
        "notice_date",
        "net_cashflow",
        "net_cashflow_yoy",
        "cfo",
        "cfo_ratio",
        "cfi",
        "cfi_ratio",
        "cff",
        "cff_ratio",
    ]

    DATE_COL = "notice_date"
    DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")


class KlineBuilder(BaseBuilder):
    """Builder for stock kline (historical) data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("date", "date"),
        ColumnSpec("open", "open", pl.Float64),
        ColumnSpec("close", "close", pl.Float64),
        ColumnSpec("high", "high", pl.Float64),
        ColumnSpec("low", "low", pl.Float64),
        ColumnSpec("volume", "volume", pl.Float64),
        ColumnSpec("amount", "amount", pl.Float64),
        ColumnSpec("amplitude", "amplitude", pl.Float64),
        ColumnSpec("change_pct", "change_pct", pl.Float64),
        ColumnSpec("change_amt", "change_amt", pl.Float64),
        ColumnSpec("turnover", "turnover", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "date",
        "ts_code",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "change_pct",
        "change_amt",
        "turnover",
    ]

    DATE_COL = "date"
    DUPLICATE_COLS = ("ts_code", "date")


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class ReportConfig:
    """Configuration for quarterly financial reports.

    Attributes:
        report_name: Report name for API (e.g., "RPT_DMSK_FN_INCOME")
        parser_class: Parser class to use
        builder_class: Builder class to use
        url: API endpoint URL
    """

    report_name: str
    builder_class: type[BaseBuilder]
    url: str = EASTMONEY_FINANCE_API


# =============================================================================
# Specialized Pipes
# =============================================================================


class ReportPipe:
    """Pipe for fetching quarterly financial reports.

    Handles the common pattern of:
    1. Building params from year/quarter
    2. Fetching initial page
    3. Fetching remaining pages concurrently
    4. Parsing and building DataFrame
    """

    def __init__(self, fetcher: EastMoneyFetcher, config: ReportConfig):
        """Initialize ReportPipe.

        Args:
            fetcher: Fetcher instance
            config: Report configuration
        """
        self._fetcher = fetcher
        self._config = config

    async def _fetch_rows(self, params: dict[str, str]) -> list[dict[str, Any]]:
        parser = FundamentalParser()
        raw = await self._fetcher.fetch_once(self._config.url, params)
        if not raw:
            log.error("Failed to fetch raw data")
            return []

        data = parser.parse(raw)
        if not data:
            log.error("Failed to parse raw data")
            return []

        total_pages = parser.get_total_pages(raw)
        log.info(f"Total pages to fetch: {total_pages}")
        if total_pages == 1:
            return data

        pages_to_fetch = list(range(2, total_pages + 1))
        page_results = await self._fetcher.fetch_pages_concurrent(
            self._config.url,
            params,
            pages_to_fetch,
        )
        for page_raw in page_results:
            data.extend(parser.parse(page_raw))
        return data

    async def fetch_one(self, year: int, quarter: Quarter) -> pl.DataFrame:
        """Fetch quarterly report data.

        Args:
            year: Report year (e.g., 2024)
            quarter: Report quarter (1, 2, 3, or 4)

        Returns:
            Polars DataFrame with report data
        """
        # Build date and params
        formatted_date = _build_quarterly_date(year, quarter)
        params = _build_quarterly_params(self._config.report_name, formatted_date)

        # Create parser and builder instances
        builder = self._config.builder_class()
        data = await self._fetch_rows(params)
        if not data:
            return pl.DataFrame()

        # Build DataFrame
        df = FundamentalParser().clean(data)
        df = builder.normalize(df)
        return df


class KlinePipe:
    """Pipe for fetching stock historical data (kline).

    Handles the pattern of:
    1. Building params from symbol, period, dates
    2. Fetching data
    3. Parsing and building DataFrame
    """

    def __init__(self, fetcher: EastMoneyFetcher):
        """Initialize KlinePipe.

        Args:
            fetcher: Fetcher instance
        """
        self._fetcher = fetcher

    async def fetch_one(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: Period = "daily",
        adjust: AdjustCN | None = None,
    ) -> pl.DataFrame:
        """Fetch stock historical data (kline).

        Args:
            symbol: Stock code (e.g., "000001")
            period: "daily", "weekly", or "monthly"
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format
            adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

        Returns:
            Polars DataFrame with historical stock data
        """
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        _secid, params = _build_kline_params(
            symbol, period, start_date_str, end_date_str, adjust or ""
        )

        raw = await self._fetcher.fetch_once(EASTMONEY_KLINE_API, params)
        parser = KlineParser()
        data = parser.parse(raw)
        if not data:
            return pl.DataFrame()
        df = parser.clean(data)
        df = df.with_columns(pl.lit(symbol).alias("ts_code"))

        builder = KlineBuilder()
        df = builder.normalize(df)
        return df


class EastMoney:
    """Main interface for East Money data.

    Composes Fetcher, Parser, Builder internally.
    Fetcher is not exposed in the public interface.

    Usage:
        with EastMoney() as client:
            df = client.quarterly_income(2024, 1)
            print(df)
    """

    def __init__(
        self,
        max_retries: int = 3,
    ):
        """Initialize EastMoney client.

        Args:
            max_retries: Number of retry attempts on failure
        """
        self._fetcher = EastMoneyFetcher(max_retries=max_retries)
        self._report_configs: dict[str, ReportConfig] = {
            "income": ReportConfig(
                report_name="RPT_DMSK_FN_INCOME",
                builder_class=IncomeBuilder,
            ),
            "balance": ReportConfig(
                report_name="RPT_DMSK_FN_BALANCE",
                builder_class=BalanceSheetBuilder,
            ),
            "cashflow": ReportConfig(
                report_name="RPT_DMSK_FN_CASHFLOW",
                builder_class=CashFlowBuilder,
            ),
        }

    def _get_report_config(self, name: str) -> ReportConfig:
        return self._report_configs[name]

    def _report(
        self, config: ReportConfig, year: int, quarter: Quarter
    ) -> pl.DataFrame:
        """Internal generic fetcher to DRY up the public API."""
        log.info(f"Fetching {config.report_name} for {year} Q{quarter}")
        pipe = ReportPipe(self._fetcher, config)
        return self._fetcher.run(pipe.fetch_one(year, quarter))

    def quarterly_income(self, year: int, quarter: Quarter) -> pl.DataFrame:
        return self._report(self._get_report_config("income"), year, quarter)

    def quarterly_balance(self, year: int, quarter: Quarter) -> pl.DataFrame:
        return self._report(self._get_report_config("balance"), year, quarter)

    def quarterly_cashflow(self, year: int, quarter: Quarter) -> pl.DataFrame:
        return self._report(self._get_report_config("cashflow"), year, quarter)

    def _batch_report(
        self, config: ReportConfig, start: date, end: date
    ) -> list[pl.DataFrame]:
        log.info(f"Fetching {config.report_name} from {start} to {end}")
        pipe = ReportPipe(self._fetcher, config)
        date_range = quarter_range(start_date=start, end_date=end)
        tasks = [pipe.fetch_one(year, quarter) for year, quarter in date_range]
        return self._fetcher.run(asyncio.gather(*tasks))

    def batch_quarterly_income(self, start: date, end: date) -> list[pl.DataFrame]:
        return self._batch_report(self._get_report_config("income"), start, end)

    def batch_quarterly_balance(self, start: date, end: date) -> list[pl.DataFrame]:
        return self._batch_report(self._get_report_config("balance"), start, end)

    def batch_quarterly_cashflow(self, start: date, end: date) -> list[pl.DataFrame]:
        return self._batch_report(self._get_report_config("cashflow"), start, end)

    def batch_stock_hist(
        self,
        symbols: Sequence[str],
        period: Period = "daily",
        start_date: date | None = None,
        end_date: date | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> list[pl.DataFrame]:
        """Fetch stock historical data (kline).

        Args:
            symbol: Stock code (e.g., "000001")
            period: "daily", "weekly", or "monthly"
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format
            adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

        Returns:
            Polars DataFrame with historical stock data
        """
        log.info(f"Fetching stock historical data for {symbols}")
        pipe = KlinePipe(self._fetcher)
        start_date, end_date = _resolve_hist_date_bounds(start_date, end_date)
        tasks = [
            pipe.fetch_one(symbol, start_date, end_date, period, adjust)
            for symbol in symbols
        ]
        return self._fetcher.run(asyncio.gather(*tasks))

    def stock_hist(
        self,
        symbol: str,
        period: Period = "daily",
        start_date: date | None = None,
        end_date: date | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> pl.DataFrame:
        """Fetch stock historical data (kline).

        Args:
            symbol: Stock code (e.g., "000001")
            period: "daily", "weekly", or "monthly"
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format
            adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

        Returns:
            Polars DataFrame with historical stock data
        """
        return self.batch_stock_hist(
            [symbol],
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )[0]

    def close(self) -> None:
        """Close the fetcher."""
        self._fetcher.sclose()

    def __enter__(self) -> "EastMoney":
        return self

    def __exit__(self, *args) -> None:
        self.close()
