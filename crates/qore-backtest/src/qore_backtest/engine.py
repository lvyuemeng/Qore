from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import TypeVar

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.config import BacktestConfig, QoreConfig
from qore_core.instrument import SessionInstrument, TradingSession
from qore_core.universe import Universe
from qore_data.store.duckdb import QoreStore
from qore_runner.runner import StrategyRunner

from qore_backtest.simulate import Fill, fill_delay, fill_order

TInstrument = TypeVar("TInstrument", bound=SessionInstrument)


@dataclass(slots=True)
class BacktestResult:
    nav: pl.DataFrame
    positions: list[dict[str, float]]
    turnovers: list[float]
    commissions: list[float]
    risk_flags: list[bool]
    fills: list[list[Fill]]
    diagnostics: pl.DataFrame


@dataclass(slots=True)
class BacktestEngine:
    runner: StrategyRunner
    store: QoreStore
    config: BacktestConfig
    calendar: TradingCalendar
    _market_data_cache: dict[tuple[str, date], pl.DataFrame] = field(
        default_factory=dict
    )

    @classmethod
    def from_config(
        cls,
        config: QoreConfig,
        runner: StrategyRunner,
        store: QoreStore,
        calendar: TradingCalendar,
    ) -> BacktestEngine:
        return cls(
            runner=runner,
            store=store,
            config=config.backtest,
            calendar=calendar,
        )

    def run(
        self,
        universe: Universe[TInstrument],
        start: date,
        end: date,
    ) -> BacktestResult:
        trading_days = self.calendar.trading_days_between(start, end)
        execution_metadata = self._execution_metadata(universe)
        current_weights: dict[str, float] = {}
        nav_value = self.config.initial_capital
        nav_rows: list[dict[str, object]] = []
        positions: list[dict[str, float]] = []
        turnovers: list[float] = []
        commissions: list[float] = []
        risk_flags: list[bool] = []
        fills: list[list[Fill]] = []
        diagnostics_rows: list[dict[str, object]] = []

        for trading_day in trading_days:
            factor_lf = self._factor_frame_for_day(trading_day)
            nav_series = pl.Series(
                "nav", [row["nav"] for row in nav_rows] or [nav_value]
            )
            target = self.runner.step(
                factor_lf=factor_lf,
                inputs=self._strategy_inputs_for_day(trading_day),
                universe=universe,
                date=trading_day,
                current_weights=current_weights,
                nav=nav_series,
                calendar=self.calendar,
            )
            fill_requests = self._fill_requests(target.weights, current_weights)
            day_fills = self._fills_for_requests(
                fill_requests,
                trading_day,
                universe,
                execution_metadata,
            )
            turnover = self._turnover(fill_requests)
            commission_cost = turnover * self.config.commission * nav_value
            daily_return = self._portfolio_return(
                day_fills, target.weights, trading_day, universe
            )
            nav_value = nav_value * (1.0 + daily_return) - commission_cost

            nav_rows.append(
                {"date": trading_day, "nav": nav_value, "return": daily_return}
            )
            positions.append(target.weights)
            turnovers.append(turnover)
            commissions.append(commission_cost)
            risk_flags.append(target.risk_triggered)
            fills.append(day_fills)
            diagnostics_rows.append(
                self._diagnostics_row(
                    trading_day=trading_day,
                    target=target,
                    fill_requests=fill_requests,
                    fills=day_fills,
                    turnover=turnover,
                    commission_cost=commission_cost,
                    daily_return=daily_return,
                    nav_value=nav_value,
                )
            )
            current_weights = target.weights

        return BacktestResult(
            nav=pl.DataFrame(nav_rows),
            positions=positions,
            turnovers=turnovers,
            commissions=commissions,
            risk_flags=risk_flags,
            fills=fills,
            diagnostics=pl.DataFrame(diagnostics_rows),
        )

    def _factor_frame_for_day(self, trading_day: date) -> pl.LazyFrame:
        factor_scores = self.store.read(
            "factor_scores",
            filters={"date": trading_day},
            columns=["symbol", "factor_name", "z_score"],
            backend="duckdb",
        )
        frame = pl.DataFrame(factor_scores.collect())
        if frame.is_empty():
            return pl.DataFrame({"symbol": []}, schema={"symbol": pl.String}).lazy()
        return frame.pivot(on="factor_name", index="symbol", values="z_score").lazy()

    def _strategy_inputs_for_day(self, trading_day: date) -> Mapping[str, object]:
        overlays = self._signal_overlays_for_day(trading_day)
        return {"signal_overlays": overlays} if overlays else {}

    def _signal_overlays_for_day(self, trading_day: date) -> dict[str, float]:
        news = pl.DataFrame(
            self.store.read(
                "news_scores",
                filters={"date": trading_day},
                columns=["symbol", "score"],
                backend="duckdb",
            ).collect()
        )
        if news.is_empty():
            return {}
        return _float_mapping(news, key_col="symbol", value_col="score")

    def _portfolio_return(
        self,
        fills: list[Fill],
        weights: dict[str, float],
        trading_day: date,
        universe: Universe[TInstrument],
    ) -> float:
        if not fills or not weights:
            return 0.0
        filled_symbols = [fill.symbol for fill in fills if fill.status == "filled"]
        if not filled_symbols:
            return 0.0
        session = universe.session
        if session is None:
            return 0.0
        dataset = _dataset_for_session(session)
        market = self._market_data_for_date(dataset, trading_day)
        if market.is_empty():
            return 0.0
        weighted_return = pl.DataFrame(
            pl.DataFrame(
                {"symbol": list(weights), "weight": list(weights.values())},
                schema={"symbol": pl.String, "weight": pl.Float64},
            )
            .lazy()
            .join(
                pl.DataFrame(
                    {"symbol": filled_symbols}, schema={"symbol": pl.String}
                ).lazy(),
                on="symbol",
                how="inner",
            )
            .join(market.lazy(), on="symbol", how="left")
            .with_columns(_return_expr_for_session(session).alias("day_return"))
            .select((pl.col("weight") * pl.col("day_return").fill_null(0.0)).sum())
            .collect()
        )
        weighted_value = weighted_return.item()
        return (
            float(weighted_value) if isinstance(weighted_value, (int, float)) else 0.0
        )

    def _fills_for_requests(
        self,
        requests: pl.DataFrame,
        trading_day: date,
        universe: Universe[TInstrument],
        execution_metadata: pl.DataFrame,
    ) -> list[Fill]:
        if requests.is_empty():
            return []
        execution_plan = self._execution_plan(requests, trading_day, execution_metadata)
        fills: list[Fill] = []
        for dataset, fill_date in (
            execution_plan.select("dataset", "fill_date").unique().iter_rows()
        ):
            market_slice = self._market_data_for_date(str(dataset), fill_date)
            joined = execution_plan.filter(
                (pl.col("dataset") == dataset) & (pl.col("fill_date") == fill_date)
            ).join(market_slice, on="symbol", how="left")
            for row in joined.iter_rows(named=True):
                symbol = str(row["symbol"])
                inst = universe.get(symbol)
                price_data = pl.DataFrame([row])
                fills.append(
                    fill_order(
                        inst,
                        trading_day,
                        str(row["direction"]),
                        _safe_float(row["quantity"]),
                        price_data,
                        self.config,
                        self.calendar,
                    )
                )
        return fills

    def _execution_plan(
        self,
        requests: pl.DataFrame,
        trading_day: date,
        execution_metadata: pl.DataFrame,
    ) -> pl.DataFrame:
        if requests.is_empty():
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                    "dataset": pl.String,
                    "fill_date": pl.Date,
                }
            )
        plan = pl.DataFrame(
            requests.lazy()
            .join(execution_metadata.lazy(), on="symbol", how="left")
            .with_columns(
                pl.when(pl.col("direction") == "buy")
                .then(pl.col("buy_delay"))
                .otherwise(pl.col("sell_delay"))
                .alias("fill_delay")
            )
            .collect()
        )
        fill_dates = [
            trading_day
            if int(delay) == 0
            else self.calendar.next_trading_day(trading_day, int(delay))
            for delay in plan.get_column("fill_delay").to_list()
        ]
        return plan.with_columns(pl.Series("fill_date", fill_dates)).drop("fill_delay")

    def _execution_metadata(self, universe: Universe[TInstrument]) -> pl.DataFrame:
        rows = [
            {
                "symbol": inst.symbol,
                "dataset": _dataset_for_session(inst.session),
                "buy_delay": fill_delay(inst, "buy"),
                "sell_delay": fill_delay(inst, "sell"),
            }
            for inst in universe
        ]
        if not rows:
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "dataset": pl.String,
                    "buy_delay": pl.Int64,
                    "sell_delay": pl.Int64,
                }
            )
        return pl.DataFrame(rows)

    def _fill_requests(
        self,
        target: dict[str, float],
        current: dict[str, float],
    ) -> pl.DataFrame:
        symbols = sorted(set(target) | set(current))
        if not symbols:
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "direction": pl.String,
                    "quantity": pl.Float64,
                }
            )
        return pl.DataFrame(
            pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.String})
            .lazy()
            .join(
                pl.DataFrame(
                    {"symbol": list(target), "target_weight": list(target.values())},
                    schema={"symbol": pl.String, "target_weight": pl.Float64},
                ).lazy(),
                on="symbol",
                how="left",
            )
            .join(
                pl.DataFrame(
                    {"symbol": list(current), "current_weight": list(current.values())},
                    schema={"symbol": pl.String, "current_weight": pl.Float64},
                ).lazy(),
                on="symbol",
                how="left",
            )
            .with_columns(
                pl.col("target_weight").fill_null(0.0),
                pl.col("current_weight").fill_null(0.0),
            )
            .with_columns(
                (pl.col("target_weight") - pl.col("current_weight")).alias("delta")
            )
            .filter(pl.col("delta").abs() > 1e-12)
            .with_columns(
                pl.when(pl.col("delta") > 0.0)
                .then(pl.lit("buy"))
                .otherwise(pl.lit("sell"))
                .alias("direction"),
                pl.col("delta").abs().alias("quantity"),
            )
            .select("symbol", "direction", "quantity")
            .collect()
        )

    def _market_data_for_date(self, dataset: str, trading_day: date) -> pl.DataFrame:
        cache_key = (dataset, trading_day)
        cached = self._market_data_cache.get(cache_key)
        if cached is not None:
            return cached
        frame = pl.DataFrame(
            self.store.read(
                dataset,
                filters={"date": trading_day},
                backend="duckdb",
            ).collect()
        )
        self._market_data_cache[cache_key] = frame
        return frame

    def _turnover(self, requests: pl.DataFrame) -> float:
        if requests.is_empty():
            return 0.0
        turnover = requests.select(pl.col("quantity").sum()).item()
        return float(turnover) if isinstance(turnover, (int, float)) else 0.0

    def _diagnostics_row(
        self,
        *,
        trading_day: date,
        target,
        fill_requests: pl.DataFrame,
        fills: list[Fill],
        turnover: float,
        commission_cost: float,
        daily_return: float,
        nav_value: float,
    ) -> dict[str, object]:
        status_counts = (
            pl.DataFrame(
                {"status": [fill.status for fill in fills]},
                schema={"status": pl.String},
            )
            .group_by("status")
            .len()
            if fills
            else pl.DataFrame(
                {"status": [], "len": []},
                schema={"status": pl.String, "len": pl.UInt32},
            )
        )
        counts = {
            str(status): int(count) for status, count in status_counts.iter_rows()
        }
        return {
            "date": trading_day,
            "nav": nav_value,
            "daily_return": daily_return,
            "turnover": turnover,
            "commission_cost": commission_cost,
            "fill_request_count": fill_requests.height,
            "filled_count": counts.get("filled", 0),
            "pending_count": counts.get("pending", 0),
            "rejected_count": counts.get("rejected", 0),
            "eligible_count": target.diagnostics.eligible_count,
            "signal_count": target.diagnostics.signal_count,
            "selected_count": target.diagnostics.selected_count,
            "risk_blocked": target.diagnostics.risk_blocked,
        }


def _safe_float(value: object | None) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    try:
        return float(0.0 if value is None else str(value))
    except (TypeError, ValueError):
        return 0.0


def _float_mapping(
    frame: pl.DataFrame,
    *,
    key_col: str,
    value_col: str,
) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in frame.select(key_col, value_col).iter_rows()
        if value is not None
    }


def _dataset_for_session(session: TradingSession) -> str:
    if session == "nav":
        return "fund_nav"
    return "stock_ohlcv"


def _return_expr_for_session(session: TradingSession) -> pl.Expr:
    if session == "nav":
        return pl.col("daily_return").cast(pl.Float64, strict=False).fill_null(0.0)
    open_col = pl.col("open").cast(pl.Float64, strict=False)
    close_col = pl.col("close").cast(pl.Float64, strict=False)
    return pl.when(open_col > 0.0).then(close_col / open_col - 1.0).otherwise(0.0)
