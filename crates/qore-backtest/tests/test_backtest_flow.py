from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from qore_backtest import BacktestResult, BacktestSettings, TradingCalendar
from qore_backtest.engine import BacktestEngine
from qore_runner.sizer import EqualWeightSizer

D = date(2026, 4, 13)


def _ohlcv(symbols: list[str], dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [d for d in dates for _ in symbols],
            "symbol": symbols * len(dates),
            "open": [10.0] * len(symbols) * len(dates),
            "high": [10.5] * len(symbols) * len(dates),
            "low": [9.8] * len(symbols) * len(dates),
            "close": [10.1] * len(symbols) * len(dates),
            "volume": [100] * len(symbols) * len(dates),
            "amount": [1000.0] * len(symbols) * len(dates),
            "adj_factor": [1.0] * len(symbols) * len(dates),
            "is_suspended": [False] * len(symbols) * len(dates),
            "limit_up": [False] * len(symbols) * len(dates),
            "limit_down": [False] * len(symbols) * len(dates),
        }
    )


def _engine(signal_rows, *, cal=None, market=None, **kw) -> BacktestEngine:
    cal = cal or TradingCalendar()
    all_dates = list({r[0] for r in signal_rows})
    md = _ohlcv(["AAA.SH", "BBB.SZ"], all_dates) if market is None else market
    sig_rows = []
    for d, symbols, sigs in signal_rows:
        sig_rows.append(
            pl.DataFrame(
                {"date": [d] * len(symbols), "symbol": symbols, "signal": sigs}
            )
        )
    signals = pl.concat(sig_rows).lazy()
    return BacktestEngine(
        config=BacktestSettings(buy_delay=0, sell_delay=0, **kw),
        calendar=cal,
        signals=signals,
        market_data=md.lazy(),
        sizer=EqualWeightSizer(max_weight=0.5),
        top_k=1,
    )


# -- integration: multi-day run ───────────────────────────────────────


def test_full_run_returns_all_frames() -> None:
    engine = _engine(
        [
            (D, ["AAA.SH", "BBB.SZ"], [0.1, 0.9]),
            (D + timedelta(days=1), ["AAA.SH", "BBB.SZ"], [0.9, 0.1]),
        ],
        start=D,
        end=D + timedelta(days=1),
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert result.nav.height == 2
    assert result.positions.height >= 1
    assert result.turnover.height == 2
    assert result.fills.height >= 1
    assert result.diagnostics.height == 2


# -- selection edge cases ─────────────────────────────────────────────


def test_rank_symbols_selects_highest_signal() -> None:
    engine = _engine([(D, ["AAA.SH", "BBB.SZ"], [0.1, 0.9])])
    result = engine.run()
    selected = result.positions.get_column("symbol").to_list()
    assert "BBB.SZ" in selected


def test_empty_signals_day_skips_day() -> None:
    engine = _engine(
        [(D, ["AAA.SH"], [0.9])],
        start=D + timedelta(days=5),
        end=D + timedelta(days=5),
    )
    result = engine.run()
    assert result.nav.height == 0


# -- fill execution edge cases ─────────────────────────────────────────


def test_pending_order_on_limit_up() -> None:
    md = _ohlcv(["AAA.SH"], [D]).with_columns(pl.lit(True).alias("limit_up"))
    engine = _engine([(D, ["AAA.SH"], [1.0])], market=md)
    result = engine.run()
    pending = result.fills.filter(pl.col("status") == "pending")
    assert pending.height == 1


def test_fill_schema_has_no_reason_column() -> None:
    engine = _engine([(D, ["AAA.SH"], [0.9])])
    result = engine.run()
    assert "reason" not in result.fills.columns
    assert set(result.fills.columns) == {
        "date",
        "symbol",
        "status",
        "fill_date",
        "fill_price",
        "quantity",
    }


def test_same_signal_two_days_keeps_position_unchanged() -> None:
    engine = _engine(
        [
            (D, ["AAA.SH"], [1.0]),
            (D + timedelta(days=1), ["AAA.SH"], [1.0]),
        ],
        start=D,
        end=D + timedelta(days=1),
    )
    result = engine.run()
    day2_fills = result.fills.filter(pl.col("date") == D + timedelta(days=1))
    assert day2_fills.is_empty() or (
        day2_fills.filter(pl.col("status") == "filled").height == 0
    )


def test_signal_change_triggers_new_fill() -> None:
    engine = _engine(
        [
            (D, ["AAA.SH"], [1.0]),
            (D + timedelta(days=1), ["AAA.SH", "BBB.SZ"], [0.01, 0.99]),
        ],
        start=D,
        end=D + timedelta(days=1),
    )
    result = engine.run()
    fills = result.fills.filter(pl.col("status") == "filled")
    fill_symbols = fills.get_column("symbol").unique().to_list()
    assert "BBB.SZ" in fill_symbols


# -- metrics ──────────────────────────────────────────────────────────


def test_metrics_shape_and_keys() -> None:
    engine = _engine([(D, ["AAA.SH"], [0.9])], start=D, end=D)
    result = engine.run()
    m = result.metrics()
    assert isinstance(m, dict)
    expected = {
        "annualized_return",
        "sharpe_ratio",
        "calmar_ratio",
        "max_drawdown",
        "sortino_ratio",
        "information_ratio",
        "win_rate",
        "profit_factor",
        "avg_turnover",
        "total_commission_cost",
    }
    assert set(m) == expected


def test_metrics_all_zero_on_empty_result() -> None:
    empty = BacktestResult(
        nav=pl.DataFrame(
            schema={"date": pl.Date, "nav": pl.Float64, "return": pl.Float64}
        ),
        positions=pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.String, "weight": pl.Float64}
        ),
        turnover=pl.DataFrame(
            schema={
                "date": pl.Date,
                "turnover": pl.Float64,
                "commission": pl.Float64,
                "risk_flag": pl.Boolean,
            }
        ),
        fills=pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "status": pl.String,
                "fill_date": pl.Date,
                "fill_price": pl.Float64,
                "quantity": pl.Float64,
            }
        ),
        diagnostics=pl.DataFrame(
            schema={
                "date": pl.Date,
                "nav": pl.Float64,
                "daily_return": pl.Float64,
                "turnover": pl.Float64,
                "commission_cost": pl.Float64,
                "fill_request_count": pl.Int64,
                "filled_count": pl.Int64,
                "pending_count": pl.Int64,
                "rejected_count": pl.Int64,
                "selected_count": pl.Int64,
            }
        ),
    )
    m = empty.metrics()
    assert all(v == 0.0 for v in m.values())


def test_metrics_with_benchmark() -> None:
    engine = _engine(
        [(D, ["AAA.SH"], [0.9]), (D + timedelta(days=1), ["AAA.SH"], [0.9])],
        start=D,
        end=D + timedelta(days=1),
    )
    result = engine.run()
    benchmark = pl.Series("nav", [100.0, 101.0])
    m = result.metrics(benchmark_nav=benchmark)
    assert m["information_ratio"] != 0.0


# -- view integration ─────────────────────────────────────────────────


def test_view_chain() -> None:
    engine = _engine([(D, ["AAA.SH"], [0.9])], start=D, end=D)
    result = engine.run()
    view = result.view().with_drawdown()
    assert view.drawdown is not None
    windowed = view.window(start=D)
    assert windowed.nav.height == 1
