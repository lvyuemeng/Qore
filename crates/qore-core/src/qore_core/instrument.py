from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

TradingSession = Literal["auction", "nav", "continuous"]


@dataclass(frozen=True, slots=True)
class StockInstrument:
    symbol: str
    exchange: Literal["SH", "SZ", "BJ"]
    industry: str
    price_limit_pct: float = 0.10
    session: TradingSession = "auction"


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
