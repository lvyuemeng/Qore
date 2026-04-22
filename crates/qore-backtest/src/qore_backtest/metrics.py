from __future__ import annotations

from math import sqrt

import polars as pl

from qore_backtest.engine import BacktestResult

_EMPTY_METRICS = {
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


def compute_metrics(
    result: BacktestResult,
    benchmark_nav: pl.Series | None = None,
) -> dict[str, float]:
    nav = result.nav
    if nav.is_empty():
        return dict(_EMPTY_METRICS)

    returns = nav.get_column("return").cast(pl.Float64, strict=False).fill_null(0.0)
    nav_values = nav.get_column("nav").cast(pl.Float64, strict=False).fill_null(0.0)
    mean_return = _series_stat(returns, "mean")
    std_return = _series_stat(returns, "std")
    downside_std = _downside_std(returns)
    annualized_return = _annualized_return(nav_values)
    max_drawdown = _max_drawdown(nav_values)
    sharpe = mean_return / std_return * sqrt(252.0) if std_return > 0.0 else 0.0
    sortino = mean_return / downside_std * sqrt(252.0) if downside_std > 0.0 else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0.0 else 0.0
    info_ratio = _information_ratio(returns, benchmark_nav)
    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino,
        "information_ratio": info_ratio,
        "win_rate": _win_rate(returns),
        "profit_factor": _profit_factor(returns),
        "avg_turnover": _diagnostic_stat(result, "turnover", "mean"),
        "total_commission_cost": _diagnostic_stat(result, "commission_cost", "sum"),
    }


def _annualized_return(nav: pl.Series) -> float:
    if len(nav) < 2:
        return 0.0
    start = float(nav[0])
    end = float(nav[-1])
    periods = max(len(nav) - 1, 1)
    if start <= 0.0 or end <= 0.0:
        return 0.0
    return (end / start) ** (252.0 / periods) - 1.0


def _max_drawdown(nav: pl.Series) -> float:
    frame = pl.DataFrame({"nav": nav})
    drawdown = frame.select(
        (pl.col("nav") / pl.col("nav").cum_max() - 1.0).min()
    ).item()
    return float(drawdown) if isinstance(drawdown, (int, float)) else 0.0


def _downside_std(returns: pl.Series) -> float:
    downside = (
        pl.DataFrame({"returns": returns})
        .select(
            pl.when(pl.col("returns") < 0.0)
            .then(pl.col("returns"))
            .otherwise(0.0)
            .alias("downside")
        )
        .to_series()
    )
    return _series_stat(downside, "std")


def _information_ratio(returns: pl.Series, benchmark_nav: pl.Series | None) -> float:
    if benchmark_nav is None or len(benchmark_nav) < 2:
        return 0.0
    benchmark_returns = benchmark_nav.pct_change().fill_null(0.0)
    common_length = min(len(returns), len(benchmark_returns))
    if common_length == 0:
        return 0.0
    excess = returns.head(common_length) - benchmark_returns.head(common_length)
    std = _series_stat(excess, "std")
    if std <= 0.0:
        return 0.0
    return _series_stat(excess, "mean") / std * sqrt(252.0)


def _win_rate(returns: pl.Series) -> float:
    if len(returns) == 0:
        return 0.0
    value = returns.gt(0.0).mean()
    return float(value) if isinstance(value, (int, float)) else 0.0


def _profit_factor(returns: pl.Series) -> float:
    gains = returns.filter(returns > 0.0).sum()
    losses = returns.filter(returns < 0.0).sum()
    gain_value = float(gains) if isinstance(gains, (int, float)) else 0.0
    loss_value = abs(float(losses)) if isinstance(losses, (int, float)) else 0.0
    if loss_value <= 0.0:
        return 0.0 if gain_value <= 0.0 else float("inf")
    return gain_value / loss_value


def _diagnostic_stat(result: BacktestResult, column: str, stat: str) -> float:
    if result.diagnostics.is_empty() or column not in result.diagnostics.columns:
        return 0.0
    value = result.diagnostics.get_column(column)
    aggregated = value.mean() if stat == "mean" else value.sum()
    value = aggregated
    return float(value) if isinstance(value, (int, float)) else 0.0


def _series_stat(series: pl.Series, stat: str) -> float:
    value = series.mean() if stat == "mean" else series.std()
    return float(value) if isinstance(value, (int, float)) else 0.0
