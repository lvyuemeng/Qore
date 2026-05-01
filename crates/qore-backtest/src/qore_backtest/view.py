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
    _nav_sorted: pl.DataFrame | None = None

    def with_drawdown(self) -> BacktestView:
        if self.nav.is_empty() or "nav" not in self.nav.columns:
            empty = pl.DataFrame(schema={"date": pl.Date, "drawdown": pl.Float64})
            return BacktestView(
                nav=self.nav,
                drawdown=empty,
                benchmarks=self.benchmarks,
                trades=self.trades,
                diagnostics=self.diagnostics,
            )
        sorted_nav = self.nav.sort("date")
        dd = pl.DataFrame(
            {
                "date": sorted_nav.get_column("date"),
                "drawdown": sorted_nav.get_column("nav")
                / sorted_nav.get_column("nav").cum_max()
                - 1.0,
            }
        )
        return BacktestView(
            nav=self.nav,
            drawdown=dd,
            benchmarks=self.benchmarks,
            trades=self.trades,
            diagnostics=self.diagnostics,
            _nav_sorted=sorted_nav,
        )

    def with_benchmark(self, name: str, benchmark_nav: pl.DataFrame) -> BacktestView:
        bm = pl.DataFrame(
            benchmark_nav.lazy()
            .select(pl.col("date").cast(pl.Date), pl.col("nav").cast(pl.Float64))
            .filter(pl.col("date").is_not_null())
            .sort("date")
            .collect()
        )
        bm_copy = dict(self.benchmarks)
        bm_copy[name] = bm
        return BacktestView(
            nav=self.nav,
            drawdown=self.drawdown,
            benchmarks=bm_copy,
            trades=self.trades,
            diagnostics=self.diagnostics,
            _nav_sorted=self._nav_sorted,
        )

    def window(
        self, start: date | None = None, end: date | None = None
    ) -> BacktestView:
        return BacktestView(
            nav=_window(self.nav, start, end),
            drawdown=_window(self.drawdown, start, end)
            if self.drawdown is not None
            else None,
            benchmarks={n: _window(f, start, end) for n, f in self.benchmarks.items()},
            trades=_window(self.trades, start, end)
            if self.trades is not None
            else None,
            diagnostics=_window(self.diagnostics, start, end)
            if self.diagnostics is not None
            else None,
            _nav_sorted=None,
        )

    def plot(self) -> BacktestPlotter:
        return BacktestPlotter(self)


def _window(frame: pl.DataFrame, start: date | None, end: date | None) -> pl.DataFrame:
    if frame.is_empty() or "date" not in frame.columns:
        return frame
    preds = [
        p
        for p in (
            pl.col("date") >= pl.lit(start).cast(pl.Date) if start else None,
            pl.col("date") <= pl.lit(end).cast(pl.Date) if end else None,
        )
        if p is not None
    ]
    return (
        pl.DataFrame(frame.lazy().filter(pl.all_horizontal(preds)).collect())
        if preds
        else frame
    )


@dataclass(frozen=True, slots=True)
class BacktestPlotter:
    view: BacktestView

    def _nav_sorted(self) -> pl.DataFrame:
        sorted_nav = self.view._nav_sorted
        if sorted_nav is not None:
            return sorted_nav
        return self.view.nav.sort("date")

    def equity(self) -> object:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        nav = self._nav_sorted()
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

        bv = self.view.with_drawdown()
        figure, axes = plt.subplots(2, 1, sharex=True)
        nav = bv._nav_sorted() if bv._nav_sorted is not None else bv.nav.sort("date")
        axes[0].plot(nav.get_column("date").to_list(), nav.get_column("nav").to_list())
        axes[0].set_title("Equity")
        if bv.drawdown is not None and not bv.drawdown.is_empty():
            series = bv.drawdown.sort("date")
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
