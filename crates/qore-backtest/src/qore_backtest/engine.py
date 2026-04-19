from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
from qore_core.calendar import TradingCalendar
from qore_core.config import BacktestConfig, QoreConfig
from qore_core.instrument import FundInstrument, Instrument
from qore_core.universe import Universe
from qore_data.store.duckdb import QoreStore
from qore_runner.runner import StrategyRunner

from qore_backtest.simulate import Fill, fill_order


@dataclass(slots=True)
class BacktestResult:
    nav: pl.DataFrame
    positions: list[dict[str, float]]
    turnovers: list[float]
    commissions: list[float]
    risk_flags: list[bool]
    fills: list[list[Fill]]


@dataclass(slots=True)
class BacktestEngine:
    runner: StrategyRunner
    store: QoreStore
    config: BacktestConfig
    calendar: TradingCalendar

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

    def run(self, universe: Universe, start: date, end: date) -> BacktestResult:
        trading_days = self.calendar.trading_days_between(start, end)
        current_weights: dict[str, float] = {}
        nav_value = self.config.initial_capital
        nav_rows: list[dict[str, object]] = []
        positions: list[dict[str, float]] = []
        turnovers: list[float] = []
        commissions: list[float] = []
        risk_flags: list[bool] = []
        fills: list[list[Fill]] = []

        for trading_day in trading_days:
            factor_lf = self._factor_frame_for_day(trading_day)
            news_scores = self._news_scores_for_day(trading_day)
            nav_series = pl.Series(
                "nav", [row["nav"] for row in nav_rows] or [nav_value]
            )
            target = self.runner.step(
                factor_lf=factor_lf,
                news_scores=news_scores,
                universe=universe,
                date=trading_day,
                current_weights=current_weights,
                nav=nav_series,
                calendar=self.calendar,
            )
            day_fills = self._fills_for_target(
                target.weights,
                current_weights,
                trading_day,
                universe,
            )
            turnover = self._turnover(current_weights, target.weights)
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
            current_weights = target.weights

        return BacktestResult(
            nav=pl.DataFrame(nav_rows),
            positions=positions,
            turnovers=turnovers,
            commissions=commissions,
            risk_flags=risk_flags,
            fills=fills,
        )

    def _factor_frame_for_day(self, trading_day: date) -> pl.LazyFrame:
        factor_scores = self.store.read("factor_scores", filters={"date": trading_day})
        frame = factor_scores.collect()
        if frame.is_empty():
            return pl.DataFrame({"symbol": []}, schema={"symbol": pl.String}).lazy()
        pivot = frame.pivot(on="factor_name", index="symbol", values="z_score")
        return pivot.lazy()

    def _news_scores_for_day(self, trading_day: date) -> dict[str, float]:
        news = self.store.read("news_scores", filters={"date": trading_day}).collect()
        if news.is_empty():
            return {}
        return {
            str(symbol): float(score)
            for symbol, score in zip(
                news.get_column("symbol").to_list(),
                news.get_column("score").to_list(),
                strict=False,
            )
            if score is not None
        }

    def _portfolio_return(
        self,
        fills: list[Fill],
        weights: dict[str, float],
        trading_day: date,
        universe: Universe,
    ) -> float:
        if not weights or not fills:
            return 0.0
        filled_symbols = {
            symbol
            for symbol, fill in zip(weights, fills, strict=False)
            if fill.status == "filled"
        }
        total = 0.0
        for symbol, weight in weights.items():
            if symbol not in filled_symbols:
                continue
            inst = universe.get(symbol)
            dataset = self._dataset_for_instrument(inst)
            prices = self.store.read(
                dataset, filters={"symbol": symbol, "date": trading_day}
            ).collect()
            if prices.is_empty():
                continue
            row = prices.to_dicts()[0]
            day_return = self._row_return(row, inst)
            total += weight * day_return
        return total

    def _dataset_for_instrument(self, inst: Instrument) -> str:
        match inst:
            case FundInstrument():
                return "fund_nav"
            case _:
                return "stock_ohlcv"

    def _row_return(self, row: dict[str, object], inst: Instrument) -> float:
        match inst:
            case FundInstrument():
                return _safe_float(row.get("daily_return"))
            case _:
                open_price = _safe_float(row.get("open"))
                close_price = _safe_float(row.get("close"))
                if open_price <= 0.0:
                    return 0.0
                return close_price / open_price - 1.0

    def _fills_for_target(
        self,
        target: dict[str, float],
        current: dict[str, float],
        trading_day: date,
        universe: Universe,
    ) -> list[Fill]:
        fills: list[Fill] = []
        symbols = sorted(set(target) | set(current))
        for symbol in symbols:
            target_weight = target.get(symbol, 0.0)
            current_weight = current.get(symbol, 0.0)
            delta = target_weight - current_weight
            if abs(delta) <= 1e-12:
                continue
            inst = universe.get(symbol)
            dataset = self._dataset_for_instrument(inst)
            price_data = self.store.read(dataset, filters={"symbol": symbol}).collect()
            direction = "buy" if delta > 0 else "sell"
            fills.append(
                fill_order(
                    inst,
                    trading_day,
                    direction,
                    abs(delta),
                    price_data,
                    self.config,
                    self.calendar,
                )
            )
        return fills

    def _turnover(
        self,
        current: dict[str, float],
        target: dict[str, float],
    ) -> float:
        symbols = set(current) | set(target)
        return sum(
            abs(target.get(symbol, 0.0) - current.get(symbol, 0.0))
            for symbol in symbols
        )


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
