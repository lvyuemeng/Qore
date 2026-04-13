from datetime import date

import pytest
from qore_core import (
    FundInstrument,
    QoreConfig,
    StockInstrument,
    TradingCalendar,
    Universe,
)


def test_universe_must_be_homogeneous() -> None:
    with pytest.raises(TypeError):
        Universe(
            [
                StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
                FundInstrument(symbol="110022", fund_type="active"),
            ]
        )


def test_calendar_fill_date_stock_and_fund() -> None:
    calendar = TradingCalendar.from_config(QoreConfig())
    signal_date = date(2026, 4, 10)
    stock = StockInstrument(symbol="600519.SH", exchange="SH", industry="food")
    fund = FundInstrument(symbol="110022", fund_type="active", subscription_delay=2)
    assert calendar.fill_date(signal_date, stock) == date(2026, 4, 13)
    assert calendar.fill_date(signal_date, fund) == date(2026, 4, 14)
