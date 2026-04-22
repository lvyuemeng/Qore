from datetime import date

import pytest
from qore_core import (
    DerivativeInstrument,
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
    derivative = DerivativeInstrument(
        symbol="IF2503",
        exchange="CFFEX",
        underlying="IF",
        derivative_type="futures",
        contract_size=300.0,
        margin_rate=0.12,
    )
    assert calendar.fill_date(signal_date, stock) == date(2026, 4, 13)
    assert calendar.fill_date(signal_date, fund) == date(2026, 4, 14)
    assert calendar.fill_date(signal_date, derivative) == date(2026, 4, 10)


def test_universe_exposes_homogeneous_session() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="000858.SZ", exchange="SZ", industry="food"),
        ]
    )

    assert universe.session == "auction"
    assert universe.instrument_type is StockInstrument


def test_universe_pipe_and_filtered_keep_structure() -> None:
    universe = Universe(
        [
            StockInstrument(symbol="600519.SH", exchange="SH", industry="food"),
            StockInstrument(symbol="000858.SZ", exchange="SZ", industry="food"),
        ]
    )
    as_of = date(2026, 4, 21)
    universe.set_suspended("000858.SZ", as_of)

    filtered = universe.pipe(lambda current: current.tradeable_on(as_of))

    assert filtered.symbols() == ["600519.SH"]
    assert next(iter(universe.values())).symbol == "600519.SH"
