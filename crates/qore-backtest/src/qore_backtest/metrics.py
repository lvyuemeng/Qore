from __future__ import annotations

from math import sqrt

import polars as pl

from qore_backtest.engine import BacktestResult


def compute_metrics(
    result: BacktestResult,
    benchmark_nav: pl.Series | None = None,
) -> dict[str, float]:
    nav = result.nav
    if nav.is_empty():
        return {
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "sortino_ratio": 0.0,
            "information_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_turnover": 0.0,
            "total_commission_cost": 0.0,
        }

    returns = nav.get_column("return")
    nav_values = nav.get_column("nav")
    mean_return = float(returns.mean() or 0.0)
    std_return = float(returns.std() or 0.0)
    downside = [min(float(value), 0.0) for value in returns.to_list()]
    downside_std = float(pl.Series("downside", downside).std() or 0.0)
    annualized_return = (
        (float(nav_values[-1]) / float(nav_values[0])) - 1.0
        if len(nav_values) > 1
        else 0.0
    )
    max_drawdown = _max_drawdown(nav_values)
    sharpe = mean_return / std_return * sqrt(252.0) if std_return > 0 else 0.0
    sortino = mean_return / downside_std * sqrt(252.0) if downside_std > 0 else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    info_ratio = _information_ratio(returns, benchmark_nav)
    win_rate = _win_rate(returns)
    profit_factor = _profit_factor(returns)
    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino,
        "information_ratio": info_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_turnover": sum(result.turnovers) / len(result.turnovers)
        if result.turnovers
        else 0.0,
        "total_commission_cost": sum(result.commissions),
    }


def _max_drawdown(nav: pl.Series) -> float:
    peak = float(nav[0])
    max_dd = 0.0
    for value in nav.to_list():
        current = float(value)
        peak = max(peak, current)
        if peak > 0.0:
            max_dd = min(max_dd, current / peak - 1.0)
    return max_dd


def _information_ratio(returns: pl.Series, benchmark_nav: pl.Series | None) -> float:
    if benchmark_nav is None or len(benchmark_nav) < 2:
        return 0.0
    benchmark_returns = benchmark_nav.pct_change().fill_null(0.0)
    excess = returns - benchmark_returns.head(len(returns))
    std = float(excess.std() or 0.0)
    if std <= 0.0:
        return 0.0
    return float(excess.mean() or 0.0) / std * sqrt(252.0)


def _win_rate(returns: pl.Series) -> float:
    values = [float(value) for value in returns.to_list()]
    if not values:
        return 0.0
    wins = sum(1 for value in values if value > 0.0)
    return wins / len(values)


def _profit_factor(returns: pl.Series) -> float:
    gains = sum(float(value) for value in returns.to_list() if float(value) > 0.0)
    losses = abs(sum(float(value) for value in returns.to_list() if float(value) < 0.0))
    if losses <= 0.0:
        return 0.0 if gains <= 0.0 else float("inf")
    return gains / losses
