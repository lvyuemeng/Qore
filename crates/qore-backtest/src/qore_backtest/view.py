from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl


@dataclass(frozen=True, slots=True)
class BacktestView:
    nav: pl.DataFrame
    drawdown: pl.DataFrame | None = None
    benchmarks: dict[str, pl.DataFrame] = field(default_factory=dict)
    trades: pl.DataFrame | None = None
    diagnostics: pl.DataFrame | None = None

    def with_drawdown(self) -> BacktestView:
        if self.nav.is_empty() or not {"date", "nav"}.issubset(self.nav.columns):
            empty = pl.DataFrame(schema={"date": pl.Date, "drawdown": pl.Float64})
            return BacktestView(
                nav=self.nav,
                drawdown=empty,
                benchmarks=self.benchmarks,
                trades=self.trades,
                diagnostics=self.diagnostics,
            )
        drawdown = pl.DataFrame(
            self.nav.lazy()
            .select(
                pl.col("date").cast(pl.Date, strict=False),
                (
                    pl.col("nav").cast(pl.Float64, strict=False)
                    / pl.col("nav").cast(pl.Float64, strict=False).cum_max()
                    - 1.0
                ).alias("drawdown"),
            )
            .collect()
        )
        return BacktestView(
            nav=self.nav,
            drawdown=drawdown,
            benchmarks=self.benchmarks,
            trades=self.trades,
            diagnostics=self.diagnostics,
        )

    def with_benchmark(self, name: str, benchmark_nav: pl.DataFrame) -> BacktestView:
        normalized = pl.DataFrame(
            benchmark_nav.lazy()
            .select(
                pl.col("date").cast(pl.Date, strict=False),
                pl.col("nav").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("date").is_not_null())
            .sort("date")
            .collect()
        )
        benchmarks = dict(self.benchmarks)
        benchmarks[name] = normalized
        return BacktestView(
            nav=self.nav,
            drawdown=self.drawdown,
            benchmarks=benchmarks,
            trades=self.trades,
            diagnostics=self.diagnostics,
        )

    def window(
        self, start: date | None = None, end: date | None = None
    ) -> BacktestView:
        return BacktestView(
            nav=_window_frame(self.nav, start=start, end=end),
            drawdown=_window_optional_frame(self.drawdown, start=start, end=end),
            benchmarks={
                name: _window_frame(frame, start=start, end=end)
                for name, frame in self.benchmarks.items()
            },
            trades=_window_optional_frame(self.trades, start=start, end=end),
            diagnostics=_window_optional_frame(self.diagnostics, start=start, end=end),
        )

    def plot(self) -> BacktestPlotter:
        return BacktestPlotter(self)


@dataclass(frozen=True, slots=True)
class BacktestPlotter:
    view: BacktestView

    def equity(self) -> object:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        nav = self.view.nav.sort("date")
        axis.plot(
            nav.get_column("date").to_list(),
            nav.get_column("nav").to_list(),
            label="nav",
        )
        for name, benchmark in self.view.benchmarks.items():
            series = benchmark.sort("date")
            axis.plot(
                series.get_column("date").to_list(),
                series.get_column("nav").to_list(),
                label=name,
            )
        axis.set_title("Backtest Equity")
        axis.legend()
        return figure

    def overview(self) -> object:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, sharex=True)
        nav = self.view.nav.sort("date")
        axes[0].plot(nav.get_column("date").to_list(), nav.get_column("nav").to_list())
        axes[0].set_title("Equity")
        drawdown = self.view.drawdown
        if drawdown is None:
            drawdown = self.view.with_drawdown().drawdown
        if drawdown is not None and not drawdown.is_empty():
            series = drawdown.sort("date")
            axes[1].plot(
                series.get_column("date").to_list(),
                series.get_column("drawdown").to_list(),
            )
        axes[1].set_title("Drawdown")
        return figure

    def timeseries(
        self,
        series: pl.DataFrame,
        *,
        date_col: str = "date",
        value_col: str = "value",
        title: str = "Time Series",
        ylabel: str | None = None,
        label: str | None = None,
    ) -> object:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        if (
            date_col in series.columns
            and value_col in series.columns
            and not series.is_empty()
        ):
            plotted = series.sort(date_col)
            axis.plot(
                plotted.get_column(date_col).to_list(),
                plotted.get_column(value_col).to_list(),
                label=label,
            )
            if label is not None:
                axis.legend()
        axis.set_title(title)
        if ylabel is not None:
            axis.set_ylabel(ylabel)
        return figure

    def tearsheet(self) -> object:
        return self.overview()


def _window_frame(
    frame: pl.DataFrame, *, start: date | None, end: date | None
) -> pl.DataFrame:
    if frame.is_empty() or "date" not in frame.columns:
        return frame
    predicates: list[pl.Expr] = []
    if start is not None:
        predicates.append(pl.col("date") >= pl.lit(start).cast(pl.Date))
    if end is not None:
        predicates.append(pl.col("date") <= pl.lit(end).cast(pl.Date))
    if not predicates:
        return frame
    predicate = predicates[0]
    for next_predicate in predicates[1:]:
        predicate = predicate & next_predicate
    return pl.DataFrame(frame.lazy().filter(predicate).collect())


def _window_optional_frame(
    frame: pl.DataFrame | None,
    *,
    start: date | None,
    end: date | None,
) -> pl.DataFrame | None:
    if frame is None:
        return None
    return _window_frame(frame, start=start, end=end)
