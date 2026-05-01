from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import TYPE_CHECKING

import polars as pl
from qore_runner import rank_symbols
from qore_runner.sizer import PositionSizer

from qore_backtest import BacktestSettings
from qore_backtest.calendar import TradingCalendar

if TYPE_CHECKING:
    from qore_backtest.view import BacktestView


# -- result -------------------------------------------------------------------


_METRIC_NAMES = (
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
)


@dataclass(slots=True)
class BacktestResult:
    nav: pl.DataFrame
    positions: pl.DataFrame
    turnover: pl.DataFrame
    fills: pl.DataFrame
    diagnostics: pl.DataFrame

    def view(self) -> BacktestView:
        from qore_backtest.view import BacktestView

        return BacktestView(
            nav=self.nav, diagnostics=self.diagnostics, trades=self.fills
        )

    def metrics(self, benchmark_nav: pl.Series | None = None) -> dict[str, float]:
        nav = self.nav
        if nav.is_empty():
            return dict.fromkeys(_METRIC_NAMES, 0.0)
        r = nav.get_column("return").cast(pl.Float64, strict=False).fill_null(0.0)
        v = nav.get_column("nav").cast(pl.Float64, strict=False).fill_null(0.0)
        mr = float(r.mean()) or 0.0
        sr = float(r.std() or 0.0) if r.std() is not None else 0.0
        ds = (
            float(r.filter(r < 0.0).std() or 0.0)
            if r.filter(r < 0.0).std() is not None
            else 0.0
        )
        ar = _annualized_return(v)
        md = float((v / v.cum_max() - 1.0).min()) or 0.0
        sharpe = mr / sr * sqrt(252.0) if sr > 0.0 else 0.0
        sortino = mr / ds * sqrt(252.0) if ds > 0.0 else 0.0
        calmar = ar / abs(md) if md < 0.0 else 0.0
        ir = _information_ratio(r, benchmark_nav)
        wt = float(r.gt(0.0).mean())
        pf = _profit_factor(r)
        return {
            "annualized_return": ar,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "max_drawdown": md,
            "sortino_ratio": sortino,
            "information_ratio": ir,
            "win_rate": wt,
            "profit_factor": pf,
            "avg_turnover": _diag_mean(self, "turnover"),
            "total_commission_cost": _diag_sum(self, "commission_cost"),
        }


def _annualized_return(nav: pl.Series) -> float:
    if len(nav) < 2:
        return 0.0
    start, end = float(nav[0]), float(nav[-1])
    return (
        0.0
        if start <= 0.0 or end <= 0.0
        else (end / start) ** (252.0 / (len(nav) - 1)) - 1.0
    )


def _information_ratio(r: pl.Series, bm: pl.Series | None) -> float:
    if bm is None or len(bm) < 2:
        return 0.0
    br = bm.pct_change().fill_null(0.0)
    n = min(len(r), len(br))
    if n == 0:
        return 0.0
    excess = r.head(n) - br.head(n)
    s = float(excess.std()) or 0.0
    return float(excess.mean()) / s * sqrt(252.0) if s > 0.0 else 0.0


def _profit_factor(r: pl.Series) -> float:
    g = float(r.filter(r > 0.0).sum()) or 0.0
    ls = abs(float(r.filter(r < 0.0).sum())) or 0.0
    return 0.0 if ls <= 0.0 else g / ls


def _diag_mean(result: BacktestResult, col: str) -> float:
    if result.diagnostics.is_empty() or col not in result.diagnostics.columns:
        return 0.0
    return float(result.diagnostics.get_column(col).mean()) or 0.0


def _diag_sum(result: BacktestResult, col: str) -> float:
    if result.diagnostics.is_empty() or col not in result.diagnostics.columns:
        return 0.0
    return float(result.diagnostics.get_column(col).sum()) or 0.0


# -- engine -------------------------------------------------------------------


_NAV_SCHEMA = {"date": pl.Date, "nav": pl.Float64, "return": pl.Float64}
_POS_SCHEMA = {"date": pl.Date, "symbol": pl.String, "weight": pl.Float64}
_TUR_SCHEMA = {
    "date": pl.Date,
    "turnover": pl.Float64,
    "commission": pl.Float64,
    "risk_flag": pl.Boolean,
}
_FILL_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.String,
    "status": pl.String,
    "fill_date": pl.Date,
    "fill_price": pl.Float64,
    "quantity": pl.Float64,
}
_DIAG_SCHEMA = {
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


@dataclass(slots=True)
class BacktestEngine:
    config: BacktestSettings
    calendar: TradingCalendar
    signals: pl.LazyFrame
    market_data: pl.LazyFrame
    sizer: PositionSizer
    top_k: int | None = None

    def run(self) -> BacktestResult:
        nav_buf: list[pl.DataFrame] = []
        pos_buf: list[pl.DataFrame] = []
        tur_buf: list[pl.DataFrame] = []
        fill_buf: list[pl.DataFrame] = []
        diag_buf: list[pl.DataFrame] = []
        current_weights = pl.DataFrame(
            schema={"symbol": pl.String, "weight": pl.Float64}
        )
        nav_value = self.config.initial_capital

        for trading_day in self.calendar.trading_days_between(
            self.config.start, self.config.end
        ):
            day_signals = pl.DataFrame(
                self.signals.filter(pl.col("date") == trading_day)
                .drop("date")
                .collect()
            )
            if day_signals.is_empty():
                continue
            day_market = pl.DataFrame(
                self.market_data.filter(pl.col("date") == trading_day)
                .drop("date")
                .collect()
            )
            if day_market.is_empty():
                continue

            selected = self._selected_symbols(day_signals)
            selected_signals = day_signals.filter(pl.col("symbol").is_in(selected))
            weights = self.sizer.size(selected_signals)

            fill_requests = _fill_requests(weights, current_weights)
            day_fills = self._resolve_fills(fill_requests, trading_day, day_market)
            turnover = (
                float(fill_requests.select(pl.col("quantity").sum()).item()) or 0.0
            )
            commission_cost = turnover * self.config.commission * nav_value
            daily_return = _portfolio_return(day_fills, weights, day_market)
            nav_value = nav_value * (1.0 + daily_return) - commission_cost

            nav_buf.append(
                pl.DataFrame(
                    {
                        "date": [trading_day],
                        "nav": [nav_value],
                        "return": [daily_return],
                    },
                    schema=_NAV_SCHEMA,
                )
            )
            pos_buf.append(_positions_row(trading_day, weights))
            tur_buf.append(
                pl.DataFrame(
                    {
                        "date": [trading_day],
                        "turnover": [turnover],
                        "commission": [commission_cost],
                        "risk_flag": [False],
                    },
                    schema=_TUR_SCHEMA,
                )
            )
            if not day_fills.is_empty():
                fill_buf.append(day_fills)
            diag_buf.append(
                _diag_row(
                    trading_day,
                    nav_value,
                    daily_return,
                    turnover,
                    commission_cost,
                    fill_requests.height,
                    day_fills,
                    len(selected),
                )
            )
            current_weights = weights

        return BacktestResult(
            nav=pl.concat(nav_buf) if nav_buf else pl.DataFrame(schema=_NAV_SCHEMA),
            positions=pl.concat(pos_buf)
            if pos_buf
            else pl.DataFrame(schema=_POS_SCHEMA),
            turnover=pl.concat(tur_buf)
            if tur_buf
            else pl.DataFrame(schema=_TUR_SCHEMA),
            fills=pl.concat(fill_buf)
            if fill_buf
            else pl.DataFrame(schema=_FILL_SCHEMA),
            diagnostics=pl.concat(diag_buf)
            if diag_buf
            else pl.DataFrame(schema=_DIAG_SCHEMA),
        )

    def _selected_symbols(self, day_signals: pl.DataFrame) -> list[str]:
        return rank_symbols(day_signals, top_k=self.top_k)

    def _resolve_fills(
        self, requests: pl.DataFrame, trading_day: date, day_market: pl.DataFrame
    ) -> pl.DataFrame:
        if requests.is_empty():
            return pl.DataFrame(schema=_FILL_SCHEMA)
        plan = self.calendar.fill_plan(
            requests, trading_day, self.config.buy_delay, self.config.sell_delay
        )
        return _evaluate_fills(plan, day_market, self.config.slippage, trading_day)


# -- helpers ------------------------------------------------------------------


def _fill_requests(target: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
    m = target.join(current, on="symbol", how="full", coalesce=True).with_columns(
        pl.col("weight").fill_null(0.0).alias("tw"),
        pl.col("weight_right").fill_null(0.0).alias("cw"),
    )
    return pl.DataFrame(
        m.lazy()
        .with_columns((pl.col("tw") - pl.col("cw")).alias("delta"))
        .filter(pl.col("delta").abs() > 1e-12)
        .with_columns(
            pl.when(pl.col("delta") > 0)
            .then(pl.lit("buy"))
            .otherwise(pl.lit("sell"))
            .alias("direction"),
            pl.col("delta").abs().alias("quantity"),
        )
        .select("symbol", "direction", "quantity")
        .collect()
    )


def _evaluate_fills(
    plan: pl.DataFrame, market: pl.DataFrame, slippage: float, trading_day: date
) -> pl.DataFrame:
    sb, ss = 1.0 + slippage, 1.0 - slippage
    base = plan.join(market, on="symbol", how="left")
    return pl.DataFrame(
        base.lazy()
        .select(
            pl.lit(trading_day).cast(pl.Date).alias("date"),
            pl.col("symbol"),
            pl.col("fill_date"),
            pl.col("quantity"),
            pl.col("direction"),
            pl.col("open").cast(pl.Float64),
            pl.col("is_suspended").fill_null(False),
            pl.col("limit_up").fill_null(False),
            pl.col("limit_down").fill_null(False),
        )
        .with_columns(
            pl.when(pl.col("open").is_null() | pl.col("is_suspended"))
            .then(pl.lit("rejected"))
            .when((pl.col("direction") == "buy") & pl.col("limit_up"))
            .then(pl.lit("pending"))
            .when((pl.col("direction") == "sell") & pl.col("limit_down"))
            .then(pl.lit("pending"))
            .when(pl.col("open") <= 0.0)
            .then(pl.lit("rejected"))
            .otherwise(pl.lit("filled"))
            .alias("status"),
        )
        .with_columns(
            pl.when(pl.col("status") == "filled")
            .then(
                pl.when(pl.col("direction") == "buy")
                .then(pl.col("open") * sb)
                .otherwise(pl.col("open") * ss)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("fill_price"),
        )
        .select("date", "symbol", "status", "fill_date", "fill_price", "quantity")
        .collect()
    )


def _portfolio_return(
    fills: pl.DataFrame, weights: pl.DataFrame, market: pl.DataFrame
) -> float:
    eligible = weights.join(
        fills.filter(pl.col("status") == "filled").select("symbol"),
        on="symbol",
        how="semi",
    )
    if eligible.is_empty():
        return 0.0
    wr = (
        eligible.join(market, on="symbol", how="left")
        .select(
            (
                pl.col("weight").cast(pl.Float64)
                * pl.when(pl.col("open").cast(pl.Float64) > 0.0)
                .then(
                    pl.col("close").cast(pl.Float64) / pl.col("open").cast(pl.Float64)
                    - 1.0
                )
                .otherwise(0.0)
                .fill_null(0.0)
            ).sum()
        )
        .item()
    )
    return float(wr) if isinstance(wr, (int, float)) else 0.0


def _positions_row(trading_day: date, weights: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        weights.with_columns(pl.lit(trading_day).cast(pl.Date).alias("date")).select(
            "date", "symbol", "weight"
        ),
        schema=_POS_SCHEMA,
    )


def _diag_row(
    trading_day: date,
    nav: float,
    daily_return: float,
    turnover: float,
    commission_cost: float,
    request_count: int,
    fills: pl.DataFrame,
    selected: int,
) -> pl.DataFrame:
    counts = dict.fromkeys(("filled", "pending", "rejected"), 0)
    if not fills.is_empty():
        grp = fills.group_by("status").len()
        for row in grp.iter_rows():
            counts[str(row[0])] = row[1]
    return pl.DataFrame(
        {
            "date": [trading_day],
            "nav": [nav],
            "daily_return": [daily_return],
            "turnover": [turnover],
            "commission_cost": [commission_cost],
            "fill_request_count": [request_count],
            "filled_count": [counts["filled"]],
            "pending_count": [counts["pending"]],
            "rejected_count": [counts["rejected"]],
            "selected_count": [selected],
        },
        schema=_DIAG_SCHEMA,
    )
