from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import yaml
from qore_backtest import BacktestSettings, TradingCalendar
from qore_backtest.engine import BacktestEngine, BacktestResult
from qore_data import DataSettings, Universe
from qore_data.store.duckdb import QoreStore
from qore_data.universe import StockCandidateSpec, StockSelectionPipeline
from qore_factor.event.alert import AlertCondition, AlertRule, build_alert_frame
from qore_factor.ohlcv.liquidity import (
    AverageAmountFactor,
    MinimumAmountFactor,
    PositionToLiquidityRatioFactor,
)
from qore_factor.pipeline import FactorPipeline
from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.model.workflow import fit_and_save_model_from_store
from qore_intelligence.strategy import build_ranking_strategy
from qore_runner import RunnerSettings
from qore_runner.sizer import EqualWeightSizer


@dataclass(frozen=True, slots=True)
class StockWorkflowAssembly:
    selection_frame: pl.DataFrame
    selected_frame: pl.DataFrame
    decision_frame: pl.DataFrame
    decision_overlay_frame: pl.DataFrame
    alert_frame: pl.DataFrame


@dataclass(frozen=True, slots=True)
class WorkflowDataConfig:
    db_path: str = "data/qore.duckdb"
    parquet_root: str = "data/raw"


@dataclass(frozen=True, slots=True)
class WorkflowIntelligenceConfig:
    model_store_root: str = "models"
    news_llm_daily_budget: int = 50
    news_llm_model: str = "claude-sonnet-4-20250514"
    news_finbert_model: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    news_score_half_life_days: int = 5


@dataclass(frozen=True, slots=True)
class WorkflowStockConfig:
    max_weight: float = 0.05


@dataclass(frozen=True, slots=True)
class WorkflowBacktestConfig:
    initial_capital: float = 10_000_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    drawdown_stop: float = 0.15


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    data: WorkflowDataConfig = WorkflowDataConfig()
    intelligence: WorkflowIntelligenceConfig = WorkflowIntelligenceConfig()
    stock: WorkflowStockConfig = WorkflowStockConfig()
    backtest: WorkflowBacktestConfig = WorkflowBacktestConfig()

    @classmethod
    def from_yaml(cls, path: str) -> WorkflowConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data_raw = raw.get("data", {})
        intelligence_raw = raw.get("intelligence", {})
        stock_raw = raw.get("stock", {})
        backtest_raw = raw.get("backtest", {})
        return cls(
            data=WorkflowDataConfig(
                db_path=str(data_raw.get("db_path", "data/qore.duckdb")),
                parquet_root=str(data_raw.get("parquet_root", "data/raw")),
            ),
            intelligence=WorkflowIntelligenceConfig(
                model_store_root=str(
                    intelligence_raw.get("model_store_root", "models")
                ),
                news_llm_daily_budget=int(
                    intelligence_raw.get("news_llm_daily_budget", 50)
                ),
                news_llm_model=str(
                    intelligence_raw.get("news_llm_model", "claude-sonnet-4-20250514")
                ),
                news_finbert_model=str(
                    intelligence_raw.get(
                        "news_finbert_model",
                        "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
                    )
                ),
                news_score_half_life_days=int(
                    intelligence_raw.get("news_score_half_life_days", 5)
                ),
            ),
            stock=WorkflowStockConfig(
                max_weight=float(stock_raw.get("max_weight", 0.05))
            ),
            backtest=WorkflowBacktestConfig(
                initial_capital=float(
                    backtest_raw.get("initial_capital", 10_000_000.0)
                ),
                commission=float(backtest_raw.get("commission", 0.0003)),
                slippage=float(backtest_raw.get("slippage", 0.0005)),
                drawdown_stop=float(backtest_raw.get("drawdown_stop", 0.15)),
            ),
        )


def run_stock_ranking_workflow(config: WorkflowConfig) -> BacktestResult:
    store = QoreStore.from_settings(_data_settings(config))
    _seed_training_inputs(store)
    fit_and_save_model_from_store(
        intelligence_settings=_intelligence_settings(config),
        model_name="stock_ranker",
        store=store,
        factor_names=["factor_a"],
        forward_returns=_example_forward_returns(),
        version="workflow",
        model=MultiHorizonRanker(horizons=[1], weights={"1d": 1.0}),
    )
    _seed_backtest_inputs(store)
    assembly = build_stock_strategy_assembly(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
        suggested_max_aum_cny=50_000_000.0,
        top_n=20,
        audit_exclusion_days=365,
        liquidity_lookback_days=2,
    )

    as_of = date(2026, 4, 13)
    universe = Universe.from_frame(
        StockCandidateSpec(
            exclude_st=False, exclude_suspended=False
        ).apply_universe_frame(assembly.selection_frame),
        symbol_col="symbol",
        tradeable_col=None,
        suspended_col="is_suspended",
        session_marker="auction",
    )
    engine = BacktestEngine.from_components(
        _backtest_settings(config),
        strategy=build_ranking_strategy(_intelligence_settings(config)),
        sizer=EqualWeightSizer(top_k=1),
        runner_settings=_runner_settings(config),
        store=store,
        calendar=TradingCalendar(),
        decision_overlays_by_day={as_of: assembly.decision_overlay_frame},
    )
    return engine.run(universe, as_of, as_of)


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


def run_example_backtest(config: WorkflowConfig) -> None:
    result = run_stock_ranking_workflow(config)
    print(result.nav)
    print(result.positions)


def build_stock_category_report(config: WorkflowConfig) -> pl.DataFrame:
    store = QoreStore.from_settings(_data_settings(config))
    _seed_universe_inputs(store)
    return StockSelectionPipeline.from_index(
        store,
        index_symbol="000300.SH",
        as_of=date(2026, 4, 13),
        announcement_start=date(2026, 4, 1),
        announcement_end=date(2026, 4, 30),
    ).category_report()


def build_stock_strategy_assembly(
    store: QoreStore,
    *,
    index_symbol: str,
    as_of: date,
    suggested_max_aum_cny: float,
    top_n: int,
    audit_exclusion_days: int,
    liquidity_lookback_days: int,
) -> StockWorkflowAssembly:
    pipeline = (
        StockSelectionPipeline.from_index(
            store,
            index_symbol=index_symbol,
            as_of=as_of,
        )
        .with_profiles()
        .with_statuses()
        .with_fundamentals()
        .with_daily_market()
        .with_audit_opinion_state(max_age_days=audit_exclusion_days)
    )
    selection_frame = pipeline.collect().with_columns(
        pl.lit(suggested_max_aum_cny / max(top_n, 1)).alias("target_position_cny")
    )
    selection_frame = selection_frame.join(
        _liquidity_metrics_frame(
            store,
            symbols=selection_frame.get_column("symbol").to_list(),
            as_of=as_of,
            lookback_days=liquidity_lookback_days,
            target_position_cny=suggested_max_aum_cny / max(top_n, 1),
        ),
        on="symbol",
        how="left",
    )
    selection_frame = selection_frame.join(
        _price_alert_inputs_frame(
            store,
            symbols=selection_frame.get_column("symbol").to_list(),
            as_of=as_of,
        ),
        on="symbol",
        how="left",
    )
    selection_frame = selection_frame.with_columns(
        pl.col("active_audit_exclusion").fill_null(False).alias("audit_excluded"),
        (
            (
                pl.col("position_to_amount_2d_ratio").is_null()
                | (pl.col("position_to_amount_2d_ratio") <= 0.10)
            )
            & (
                pl.col("min_amount_2d").is_null()
                | (pl.col("min_amount_2d") >= 10_000_000.0)
            )
        ).alias("capacity_pass"),
    ).with_columns(
        pl.when(pl.col("audit_excluded") & ~pl.col("capacity_pass"))
        .then(pl.lit("audit|capacity"))
        .when(pl.col("audit_excluded"))
        .then(pl.lit("audit"))
        .when(~pl.col("capacity_pass"))
        .then(pl.lit("capacity"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("base_exclude_reason"),
    )
    signal_frame = _decision_signal_frame(store, as_of=as_of)
    decision_frame = _build_decision_frame(
        selection_frame=selection_frame,
        signal_frame=signal_frame,
        top_n=top_n,
    )
    selected_frame = decision_frame.filter(pl.col("selected"))
    alert_frame = pl.DataFrame(
        build_alert_frame(
            selection_frame.lazy(),
            rules=(
                AlertRule(
                    name="single_day_drop",
                    conditions=(
                        AlertCondition("pct_change", "le", -0.07),
                        AlertCondition("turnover_cny", "gt", 5_000_000.0),
                    ),
                ),
                AlertRule(
                    name="adverse_audit_context",
                    conditions=(AlertCondition("active_audit_exclusion", "eq", True),),
                    action="record_alert",
                ),
            ),
            date_column="selection_date",
        ).collect()
    )
    return StockWorkflowAssembly(
        selection_frame=selection_frame,
        selected_frame=selected_frame,
        decision_frame=decision_frame,
        decision_overlay_frame=decision_frame.select(
            "symbol", "selected", "exclude_reason"
        ),
        alert_frame=alert_frame,
    )


def _decision_signal_frame(store: QoreStore, *, as_of: date) -> pl.DataFrame:
    scores = pl.DataFrame(
        store.read(
            "factor_scores",
            filters={"date": as_of},
            columns=["symbol", "z_score"],
            backend="duckdb",
        ).collect()
    )
    if scores.is_empty():
        return pl.DataFrame(schema={"symbol": pl.String, "signal": pl.Float64})
    return pl.DataFrame(
        scores.lazy()
        .group_by("symbol")
        .agg(pl.col("z_score").cast(pl.Float64, strict=False).mean().alias("signal"))
        .collect()
    )


def _build_decision_frame(
    *,
    selection_frame: pl.DataFrame,
    signal_frame: pl.DataFrame,
    top_n: int,
) -> pl.DataFrame:
    decision = pl.DataFrame(
        selection_frame.lazy()
        .join(signal_frame.lazy(), on="symbol", how="left")
        .collect()
    )
    ranked = (
        decision.filter(pl.col("base_exclude_reason").is_null())
        .filter(pl.col("signal").is_not_null() & pl.col("signal").is_finite())
        .sort(
            ["signal", "symbol"],
            descending=[True, False],
        )
    )
    if top_n > 0:
        ranked = ranked.head(top_n)
    selected_symbols = [str(symbol) for symbol in ranked.get_column("symbol").to_list()]
    return pl.DataFrame(
        decision.lazy()
        .with_columns(
            pl.col("symbol").is_in(selected_symbols).alias("selected"),
            pl.when(pl.col("symbol").is_in(selected_symbols))
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("base_exclude_reason").is_not_null())
            .then(pl.col("base_exclude_reason").cast(pl.String, strict=False))
            .when(pl.col("signal").is_null() | ~pl.col("signal").is_finite())
            .then(pl.lit("missing_signal"))
            .otherwise(pl.lit("rank_cutoff"))
            .alias("exclude_reason"),
        )
        .collect()
    )


def _liquidity_metrics_frame(
    store: QoreStore,
    *,
    symbols: list[str],
    as_of: date,
    lookback_days: int,
    target_position_cny: float,
) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                f"avg_amount_{lookback_days}d": pl.Float64,
                f"min_amount_{lookback_days}d": pl.Float64,
                f"position_to_amount_{lookback_days}d_ratio": pl.Float64,
            }
        )
    history = pl.DataFrame(
        store.read_duckdb("stock_ohlcv")
        .filter(pl.col("date") <= as_of)
        .filter(pl.col("symbol").is_in(symbols))
        .select("date", "symbol", "amount")
        .collect()
    )
    if history.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                f"avg_amount_{lookback_days}d": pl.Float64,
                f"min_amount_{lookback_days}d": pl.Float64,
                f"position_to_amount_{lookback_days}d_ratio": pl.Float64,
            }
        )
    metrics = pl.DataFrame(
        FactorPipeline()
        .add(
            AverageAmountFactor(window=lookback_days),
            MinimumAmountFactor(window=lookback_days),
            PositionToLiquidityRatioFactor(
                liquidity_column=f"avg_amount_{lookback_days}d",
            ),
        )
        .run(
            history.lazy().with_columns(
                pl.lit(target_position_cny).alias("target_position_cny")
            )
        )
        .collect()
    )
    return (
        metrics.sort(["symbol", "date"])
        .group_by("symbol")
        .tail(1)
        .select(
            "symbol",
            f"avg_amount_{lookback_days}d",
            f"min_amount_{lookback_days}d",
            f"position_to_amount_{lookback_days}d_ratio",
        )
    )


def _price_alert_inputs_frame(
    store: QoreStore,
    *,
    symbols: list[str],
    as_of: date,
) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "pct_change": pl.Float64,
                "turnover_cny": pl.Float64,
            }
        )
    history = pl.DataFrame(
        store.read_duckdb("stock_ohlcv")
        .filter(pl.col("date") <= as_of)
        .filter(pl.col("symbol").is_in(symbols))
        .select("date", "symbol", "close", "amount")
        .collect()
    )
    if history.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "pct_change": pl.Float64,
                "turnover_cny": pl.Float64,
            }
        )
    return pl.DataFrame(
        history.lazy()
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias(
                "pct_change"
            ),
            pl.col("amount").alias("turnover_cny"),
        )
        .group_by("symbol")
        .tail(1)
        .select("symbol", "pct_change", "turnover_cny")
        .collect()
    )


def _config_from_args(args: Namespace) -> WorkflowConfig:
    if args.config:
        return WorkflowConfig.from_yaml(args.config)
    return WorkflowConfig()


def _data_settings(config: WorkflowConfig) -> DataSettings:
    return DataSettings(
        db_path=config.data.db_path,
        parquet_root=config.data.parquet_root,
    )


def _intelligence_settings(config: WorkflowConfig) -> IntelligenceSettings:
    return IntelligenceSettings(
        model_store_root=config.intelligence.model_store_root,
        news_llm_daily_budget=config.intelligence.news_llm_daily_budget,
        news_llm_model=config.intelligence.news_llm_model,
        news_finbert_model=config.intelligence.news_finbert_model,
        news_score_half_life_days=config.intelligence.news_score_half_life_days,
    )


def _runner_settings(config: WorkflowConfig) -> RunnerSettings:
    return RunnerSettings(
        max_single=config.stock.max_weight,
        drawdown_stop=config.backtest.drawdown_stop,
    )


def _backtest_settings(config: WorkflowConfig) -> BacktestSettings:
    return BacktestSettings(
        initial_capital=config.backtest.initial_capital,
        commission=config.backtest.commission,
        slippage=config.backtest.slippage,
        drawdown_stop=config.backtest.drawdown_stop,
    )


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
                "date": [
                    date(2026, 4, 12),
                    date(2026, 4, 13),
                    date(2026, 4, 12),
                    date(2026, 4, 13),
                    date(2026, 4, 12),
                    date(2026, 4, 13),
                ],
                "symbol": [
                    "AAA.SH",
                    "AAA.SH",
                    "BBB.SZ",
                    "BBB.SZ",
                    "CCC.SZ",
                    "CCC.SZ",
                ],
                "open": [10.0, 10.0, 10.0, 10.0, 9.8, 9.7],
                "high": [10.2, 10.5, 10.4, 11.0, 9.9, 9.8],
                "low": [9.9, 9.8, 9.8, 9.9, 9.6, 9.4],
                "close": [10.0, 10.1, 10.3, 9.4, 9.7, 9.0],
                "volume": [100, 100, 120, 120, 80, 80],
                "amount": [
                    12_000_000.0,
                    11_000_000.0,
                    6_500_000.0,
                    6_000_000.0,
                    4_000_000.0,
                    3_000_000.0,
                ],
                "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "is_suspended": [False, False, False, False, False, False],
                "limit_up": [False, False, False, False, False, False],
                "limit_down": [False, False, False, False, False, False],
            }
        ),
    )


def _seed_universe_inputs(store: QoreStore) -> None:
    store.write(
        "index_constituents",
        pl.DataFrame(
            {
                "as_of": [date(2026, 4, 13), date(2026, 4, 13), date(2026, 4, 13)],
                "index_symbol": ["000300.SH", "000300.SH", "000300.SH"],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "exchange": ["SH", "SZ", "SZ"],
                "industry": ["bank", "tech", "utility"],
            }
        ),
    )
    store.write(
        "stock_profiles",
        pl.DataFrame(
            {
                "as_of": [date(2026, 4, 13), date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "short_name": ["AAA", "BBB", "CCC"],
                "exchange": ["SH", "SZ", "SZ"],
                "industry": ["bank", "tech", "utility"],
                "board": ["MainBoard", "ChiNext", "MainBoard"],
                "listing_date": [date(2010, 1, 1), date(2015, 6, 1), date(2012, 3, 1)],
                "total_market_cap": [1000.0, 1500.0, 800.0],
                "float_market_cap": [800.0, 900.0, 500.0],
                "total_shares": [100.0, 120.0, 90.0],
                "float_shares": [80.0, 90.0, 60.0],
                "is_st": [False, False, False],
            }
        ),
    )
    store.write(
        "analyst_forecasts",
        pl.DataFrame(
            {
                "as_of": [date(2026, 4, 13), date(2026, 4, 13), date(2026, 4, 13)],
                "symbol": ["AAA.SH", "BBB.SZ", "CCC.SZ"],
                "report_count": [6, 12, 4],
                "buy": [2, 5, 1],
                "overweight": [2, 4, 1],
                "neutral": [1, 2, 1],
                "underweight": [1, 1, 1],
                "sell": [0, 0, 0],
                "eps_year1": [1.2, 2.3, 0.8],
                "eps_year2": [1.3, 2.5, 0.9],
                "eps_year3": [1.4, 2.7, 1.0],
                "eps_year4": [1.5, 2.9, 1.1],
            }
        ),
    )
    store.write(
        "announcements",
        pl.DataFrame(
            {
                "symbol": ["AAA.SH", "BBB.SZ", "BBB.SZ", "CCC.SZ"],
                "short_name": ["AAA", "BBB", "BBB", "CCC"],
                "title": ["AAA公告", "BBB年报", "BBB快报", "CCC公告"],
                "notice_type": ["一般事项", "财务报告", "业绩快报", "一般事项"],
                "notice_date": [
                    date(2026, 4, 10),
                    date(2026, 4, 12),
                    date(2026, 4, 15),
                    date(2026, 4, 11),
                ],
                "art_code": ["AAA-1", "BBB-1", "BBB-2", "CCC-1"],
                "url": [
                    "https://example.test/AAA-1",
                    "https://example.test/BBB-1",
                    "https://example.test/BBB-2",
                    "https://example.test/CCC-1",
                ],
            }
        ),
    )
    store.write(
        "stock_audit_opinions",
        pl.DataFrame(
            {
                "symbol": ["AAA.SH", "BBB.SZ"],
                "report_date": [date(2025, 12, 31), date(2025, 12, 31)],
                "announce_date": [date(2026, 4, 10), date(2026, 4, 12)],
                "opinion": ["无保留意见", "否定意见"],
                "opinion_code": ["unqualified", "adverse"],
                "source_notice_type": ["财务报告", "财务报告"],
                "title": ["AAA审计", "BBB审计"],
                "art_code": ["AUD-AAA", "AUD-BBB"],
                "url": ["https://example.test/AUD-AAA", "https://example.test/AUD-BBB"],
            }
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
