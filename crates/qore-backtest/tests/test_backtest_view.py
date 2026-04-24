from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from qore_backtest.engine import BacktestResult
from qore_backtest.view import BacktestView


def _result_fixture() -> BacktestResult:
    return BacktestResult(
        nav=pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15)],
                "nav": [100.0, 110.0, 99.0],
                "return": [0.0, 0.10, -0.10],
            }
        ),
        positions=pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 14)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "weight": [1.0, 1.0],
            }
        ),
        turnover=pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15)],
                "turnover": [1.0, 1.0, 0.0],
                "commission": [10.0, 10.0, 0.0],
                "risk_flag": [False, False, False],
            }
        ),
        fills=pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 14)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "status": ["filled", "filled"],
                "fill_date": [date(2026, 4, 14), date(2026, 4, 15)],
                "fill_price": [10.0, 11.0],
                "quantity": [1.0, 1.0],
                "reason": [None, None],
            }
        ),
        diagnostics=pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15)],
                "nav": [100.0, 110.0, 99.0],
            }
        ),
    )


def test_backtest_result_view_returns_backtest_view() -> None:
    result = _result_fixture()

    view = result.view()

    assert isinstance(view, BacktestView)
    assert view.nav.height == 3
    assert view.trades is not None
    assert view.trades.height == 2
    assert view.diagnostics is not None
    assert view.diagnostics.height == 3


def test_backtest_view_with_drawdown_and_window() -> None:
    view = (
        _result_fixture()
        .view()
        .with_drawdown()
        .with_benchmark(
            "CSI300",
            pl.DataFrame(
                {
                    "date": [date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15)],
                    "nav": [100.0, 102.0, 101.0],
                }
            ),
        )
    )

    assert view.drawdown is not None
    drawdown_values = view.drawdown.get_column("drawdown").to_list()
    assert drawdown_values[0] == pytest.approx(0.0)
    assert drawdown_values[1] == pytest.approx(0.0)
    assert drawdown_values[2] == pytest.approx(-0.1)
    assert "CSI300" in view.benchmarks

    windowed = view.window(start=date(2026, 4, 14), end=date(2026, 4, 15))
    assert windowed.nav.height == 2
    assert windowed.drawdown is not None
    assert windowed.drawdown.height == 2
    assert windowed.trades is not None
    assert windowed.trades.height == 1
    assert windowed.benchmarks["CSI300"].height == 2


def test_backtest_view_plot_equity_smoke() -> None:
    pytest.importorskip("matplotlib.pyplot")

    figure = _result_fixture().view().plot().equity()

    assert figure is not None
