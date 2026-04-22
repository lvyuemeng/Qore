from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import date

import polars as pl
from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_core import QoreConfig, TradingCalendar
from qore_data.store.duckdb import QoreStore
from qore_data.universe import StockCandidateSpec, StockSelectionPipeline
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.workflow import fit_and_save_model_from_store
from qore_intelligence.strategy import build_ranking_strategy
from qore_runner.runner import StrategyRunner
from qore_runner.sizer import EqualWeightSizer


def run_stock_ranking_workflow(config: QoreConfig) -> BacktestResult:
    store = QoreStore.from_config(config)
    _seed_training_inputs(store)
    fit_and_save_model_from_store(
        config=config,
        model_name="stock_ranker",
        store=store,
        factor_names=["factor_a"],
        forward_returns=_example_forward_returns(),
        version="workflow",
        model=MultiHorizonRanker(horizons=[1], weights={"1d": 1.0}),
    )
    _seed_backtest_inputs(store)

    universe = StockSelectionPipeline.from_index(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
    ).to_universe(
        StockCandidateSpec(exclude_st=False, exclude_suspended=False),
        keep_suspended=True,
    )
    runner = StrategyRunner.from_config(
        config,
        build_ranking_strategy(config),
        EqualWeightSizer(top_k=1),
    )
    engine = BacktestEngine.from_config(
        config,
        runner,
        store,
        TradingCalendar.from_config(config),
    )
    return engine.run(universe, date(2026, 4, 13), date(2026, 4, 13))


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the example Qore stock ranking workflow")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a Qore YAML config file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    category_report = build_stock_category_report(config)
    result = run_stock_ranking_workflow(config)
    print(category_report)
    print(result.nav)
    print(result.positions)
    return 0


def run_example_backtest(config: QoreConfig) -> None:
    result = run_stock_ranking_workflow(config)
    print(result.nav)
    print(result.positions)


def build_stock_category_report(config: QoreConfig) -> pl.DataFrame:
    store = QoreStore.from_config(config)
    _seed_universe_inputs(store)
    return StockSelectionPipeline.from_index(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
        announcement_start=date(2026, 4, 1),
        announcement_end=date(2026, 4, 30),
    ).category_report()


def _config_from_args(args: Namespace) -> QoreConfig:
    if args.config:
        return QoreConfig.from_yaml(args.config)
    return QoreConfig()


def _seed_training_inputs(store: QoreStore) -> None:
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [
                    date(2026, 1, 1),
                    date(2026, 1, 1),
                    date(2026, 1, 2),
                    date(2026, 1, 2),
                    date(2026, 1, 3),
                    date(2026, 1, 3),
                    date(2026, 1, 4),
                    date(2026, 1, 4),
                ],
                "symbol": ["AAA.SH", "BBB.SZ"] * 4,
                "factor_name": ["factor_a"] * 8,
                "raw_value": [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.4, 0.2],
                "z_score": [0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.4, 0.2],
                "rank_pct": [0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            }
        ),
    )


def _example_forward_returns() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 3),
                date(2026, 1, 4),
                date(2026, 1, 4),
            ],
            "symbol": ["AAA.SH", "BBB.SZ"] * 4,
            "forward_return_1d": [0.01, 0.02, 0.02, 0.01, 0.03, 0.01, 0.04, 0.02],
        }
    ).lazy()


def _seed_backtest_inputs(store: QoreStore) -> None:
    _seed_universe_inputs(store)
    store.write(
        "factor_scores",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "factor_name": ["factor_a", "factor_a"],
                "raw_value": [0.1, 0.9],
                "z_score": [0.1, 0.9],
                "rank_pct": [0.5, 1.0],
            }
        ),
    )
    store.write(
        "stock_ohlcv",
        pl.DataFrame(
            {
                "date": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "open": [10.0, 10.0],
                "high": [10.5, 11.0],
                "low": [9.8, 9.9],
                "close": [10.1, 11.0],
                "volume": [100, 120],
                "amount": [1000.0, 1200.0],
                "adj_factor": [1.0, 1.0],
                "is_suspended": [False, False],
                "limit_up": [False, False],
                "limit_down": [False, False],
            }
        ),
    )


def _seed_universe_inputs(store: QoreStore) -> None:
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [date(2026, 4, 13), date(2026, 4, 13)],
                "index_symbol": ["000300.SH", "000300.SH"],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "exchange": ["SH", "SZ"],
                "industry": ["bank", "tech"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "short_name": ["AAA", "BBB"],
                "exchange": ["SH", "SZ"],
                "industry": ["bank", "tech"],
                "board": ["MainBoard", "ChiNext"],
                "listing_date": [date(2010, 1, 1), date(2015, 6, 1)],
                "total_market_cap": [1000.0, 1500.0],
                "float_market_cap": [800.0, 900.0],
                "total_shares": [100.0, 120.0],
                "float_shares": [80.0, 90.0],
                "is_st": [False, False],
            }
        ),
    )
    store.write(
        "analyst_forecasts",
        pl.DataFrame(
            {
                "as_of": [date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ"],
                "report_count": [6, 12],
                "buy": [2, 5],
                "overweight": [2, 4],
                "neutral": [1, 2],
                "underweight": [1, 1],
                "sell": [0, 0],
                "eps_year1": [1.2, 2.3],
                "eps_year2": [1.3, 2.5],
                "eps_year3": [1.4, 2.7],
                "eps_year4": [1.5, 2.9],
            }
        ),
    )
    store.write(
        "announcements",
        pl.DataFrame(
            {
                "symbol": ["AAA.SH", "BBB.SZ", "BBB.SZ"],
                "short_name": ["AAA", "BBB", "BBB"],
                "title": ["AAA公告", "BBB年报", "BBB快报"],
                "notice_type": ["一般事项", "财务报告", "业绩快报"],
                "notice_date": [
                    date(2026, 4, 10),
                    date(2026, 4, 12),
                    date(2026, 4, 15),
                ],
                "art_code": ["AAA-1", "BBB-1", "BBB-2"],
                "url": [
                    "https://example.test/AAA-1",
                    "https://example.test/BBB-1",
                    "https://example.test/BBB-2",
                ],
            }
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
