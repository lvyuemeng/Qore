from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

TradingSession = Literal["auction", "nav", "continuous"]


class SessionInstrument(Protocol):
    symbol: str
    session: TradingSession


@dataclass(frozen=True, slots=True)
class StockInstrument:
    symbol: str
    exchange: Literal["SH", "SZ", "BJ"]
    industry: str
    price_limit_pct: float = 0.10
    session: TradingSession = "auction"

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, object],
        *,
        symbol_key: str = "symbol",
        exchange_key: str = "exchange",
        industry_key: str = "industry",
        price_limit_key: str = "price_limit_pct",
    ) -> StockInstrument:
        exchange = _stock_exchange(row.get(exchange_key))
        price_limit = row.get(price_limit_key)
        return cls(
            symbol=str(row[symbol_key]),
            exchange=exchange,
            industry=str(row.get(industry_key) or ""),
            price_limit_pct=(
                float(price_limit)
                if isinstance(price_limit, int | float | str)
                else 0.10
            ),
        )


@dataclass(frozen=True, slots=True)
class FundInstrument:
    symbol: str
    fund_type: Literal["active", "passive_etf", "bond", "mixed", "qdii"]
    subscription_delay: int = 1
    redemption_delay: int = 2
    session: TradingSession = "nav"


@dataclass(frozen=True, slots=True)
class DerivativeInstrument:
    symbol: str
    exchange: str
    underlying: str
    derivative_type: Literal["futures", "perpetual", "option"]
    contract_size: float
    margin_rate: float
    quote_currency: str = "CNY"
    expiry: date | None = None
    session: TradingSession = "continuous"


Instrument = StockInstrument | FundInstrument | DerivativeInstrument


def _stock_exchange(value: object) -> Literal["SH", "SZ", "BJ"]:
    normalized = str(value).upper()
    if normalized == "SH":
        return "SH"
    if normalized == "SZ":
        return "SZ"
    if normalized == "BJ":
        return "BJ"
    msg = f"Unsupported stock exchange code: {value}"
    raise ValueError(msg)
