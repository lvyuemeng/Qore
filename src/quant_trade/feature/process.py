"""Feature engineering module for quantitative trading.

Implements the feature calculation pipeline as specified in docs/table.md,
including quality metrics, leverage ratios, growth metrics, valuation ratios,
behavioral indicators, and macro context features.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs

from quant_trade.config.logger import log

from ..transform import GSIZE


class Fundamental:
    """Fundamental factor engineering for Chinese A-shares quarterly data.
    Designed to work directly on AkShare Eastmoney merged fundamentals.
    All monetary values in CNY yuan, growth rates as decimal (0.25 = 25%).
    """

    IDENT_COLS = ["ts_code", "name", "notice_date"]

    @staticmethod
    def _safe_div(
        num: str | pl.Expr,
        den: str | pl.Expr,
        alias: str,
        *,
        min_den: float = 1e-6,
        handle_neg_den: Literal["null", "abs", "keep"] = "null",
        fill_null: float | None = None,
    ) -> pl.Expr:
        n = pl.col(num) if isinstance(num, str) else num
        d = pl.col(den) if isinstance(den, str) else den

        neg_mask = d <= 0

        d_adj = d.abs() if handle_neg_den == "abs" else d
        d_safe = d_adj.clip(lower_bound=min_den)

        ratio = n / d_safe

        if handle_neg_den == "null":
            ratio = pl.when(neg_mask).then(None).otherwise(ratio)

        if fill_null is not None:
            ratio = ratio.fill_null(fill_null)

        return ratio.alias(alias)

    @staticmethod
    def quality(df: pl.DataFrame) -> pl.DataFrame:
        """Profitability, margins, efficiency"""
        log.info("Computing quality / profitability metrics")

        return df.with_columns(
            # Gross profit
            (pl.col("total_revenue") - pl.col("operate_cost")).alias("gross_profit"),
            # Margins
            Fundamental._safe_div(
                pl.col("total_revenue") - pl.col("operate_cost"),
                pl.col("total_revenue"),
                "gross_margin",
                fill_null=float("nan"),
            ),
            Fundamental._safe_div(
                "operate_profit",
                "total_revenue",
                "operate_margin",
                fill_null=float("nan"),
            ),
            Fundamental._safe_div(
                "net_profit",
                "total_revenue",
                "net_margin",
                fill_null=float("nan"),
            ),
            # Returns
            Fundamental._safe_div(
                "net_profit",
                "total_equity",
                "roe",
                handle_neg_den="null",
                fill_null=float("nan"),
            ),
            Fundamental._safe_div(
                "net_profit",
                "total_asset",
                "roa",
                handle_neg_den="null",
                fill_null=float("nan"),
            ),
            # Efficiency
            Fundamental._safe_div(
                "total_revenue",
                "total_asset",
                "asset_turnover",
                fill_null=float("nan"),
            ),
        ).select(
            *Fundamental.IDENT_COLS,
            "gross_margin",
            "operate_margin",
            "net_margin",
            "roe",
            "roa",
            "asset_turnover",
        )

    @staticmethod
    def leverage(df: pl.DataFrame) -> pl.DataFrame:
        """Leverage, liquidity, coverage ratios"""
        log.info("Computing leverage & safety metrics")
        return (
            df.with_columns(
                current_assets=pl.col("cash")
                + pl.col("accounts_receivable")
                + pl.col("inventory"),
                current_liabilities=pl.col("accounts_payable")
                + pl.col("advance_receivable"),
            )
            .with_columns(
                Fundamental._safe_div(
                    "total_debt",
                    "total_asset",
                    "debt_asset_ratio",
                    handle_neg_den="null",
                ),
                Fundamental._safe_div(
                    "total_debt",
                    "total_equity",
                    "debt_to_equity",
                    handle_neg_den="null",
                ),
                Fundamental._safe_div(
                    "current_assets",
                    "current_liabilities",
                    "current_ratio",
                    handle_neg_den="abs",
                ),
                Fundamental._safe_div(
                    pl.col("cash") + pl.col("accounts_receivable"),
                    "current_liabilities",
                    "quick_ratio",
                    handle_neg_den="abs",
                ),
                Fundamental._safe_div(
                    "operate_profit",
                    "finance_expense",
                    "interest_coverage",
                    min_den=1.0,
                ),
            )
            .select(
                *Fundamental.IDENT_COLS,
                "debt_asset_ratio",
                "debt_to_equity",
                "current_ratio",
                "quick_ratio",
                "interest_coverage",
            )
        )

    @staticmethod
    def growth(df: pl.DataFrame) -> pl.DataFrame:
        """YoY growth rates and quality signals"""
        log.info("Computing growth metrics")
        return df.with_columns(
            revenue_growth_yoy=pl.col("total_revenue_yoy") / 100.0,
            net_profit_growth_yoy=pl.col("net_profit_yoy") / 100.0,
            asset_growth_yoy=pl.col("total_asset_yoy") / 100.0,
            debt_growth_yoy=pl.col("total_debt_yoy") / 100.0,
            profit_growth_premium=(
                pl.col("net_profit_yoy") - pl.col("total_revenue_yoy")
            )
            / 100.0,
            roe_last=pl.when(pl.col("total_equity").shift(1).over("ts_code") > 0)
            .then(
                pl.col("net_profit").shift(1).over("ts_code")
                / pl.col("total_equity").shift(1).over("ts_code")
            )
            .otherwise(pl.lit(0.0)),
        ).select(
            *Fundamental.IDENT_COLS,
            "revenue_growth_yoy",
            "net_profit_growth_yoy",
            "asset_growth_yoy",
            "debt_growth_yoy",
            "profit_growth_premium",
            "roe_last",
        )

    @staticmethod
    def cashflow(df: pl.DataFrame) -> pl.DataFrame:
        """Cash flow quality, accrual, efficiency"""
        log.info("Computing cash flow metrics")
        return (
            df.with_columns(
                fcf_proxy=pl.col("cfo") + pl.col("cfi"),
                accrual=pl.col("net_profit") - pl.col("cfo"),
            )
            .with_columns(
                Fundamental._safe_div(
                    "cfo", "net_profit", "cfo_to_net_profit", fill_null=0
                ),
                Fundamental._safe_div(
                    "fcf_proxy", "net_profit", "fcf_to_net_profit", fill_null=0
                ),
                Fundamental._safe_div("cfo", "total_asset", "cfo_yield"),
                Fundamental._safe_div("accrual", "total_asset", "accrual_ratio"),
            )
            .select(
                *Fundamental.IDENT_COLS,
                "cfo_to_net_profit",
                "fcf_to_net_profit",
                "cfo_yield",
                "accrual",
                "accrual_ratio",
            )
        )

    @staticmethod
    def rolling_ttm(df: pl.DataFrame, min_periods: int = 3) -> pl.DataFrame:
        """Compute trailing twelve months (TTM) aggregates and momentum.
        Assumes df is sorted by ['ts_code', 'notice_date'].
        Uses group_by_dynamic with 4-quarter window.
        """
        log.info(f"Computing TTM rolling metrics (min_periods={min_periods})")

        if "notice_date" not in df.columns:
            raise ValueError("notice_date column missing — cannot compute TTM")

        if "report_period" in df.columns:
            df = df.sort(["ts_code", "report_period", "notice_date"]).unique(
                subset=["ts_code", "report_period"], keep="last"
            )
        else:
            df = df.sort(["ts_code", "notice_date"]).unique(
                subset=["ts_code", "notice_date"], keep="last"
            )

        df = df.sort(["ts_code", "notice_date"])

        ttm_agg = (
            df.group_by_dynamic(
                index_column="notice_date",
                every="1q",
                period="4q",
                offset="-3q",
                group_by="ts_code",
                closed="left",
                label="left",
            )
            .agg(
                ttm_revenue=pl.col("total_revenue").sum(),
                ttm_net_profit=pl.col("net_profit").sum(),
                ttm_cfo=pl.col("cfo").sum(),
                ttm_fcf=(pl.col("cfo") + pl.col("cfi")).sum(),
                latest_equity=pl.col("total_equity").last(),
                latest_assets=pl.col("total_asset").last(),
                count=pl.len(),
            )
            .filter(pl.col("count") >= min_periods)
            .sort(["ts_code", "notice_date"])
        )

        ttm_agg = ttm_agg.with_columns(
            profit_momentum=(
                ttm_agg["ttm_net_profit"] / ttm_agg["ttm_net_profit"].shift(1) - 1
            ).fill_null(0.0)
        )

        ttm_agg = ttm_agg.with_columns(
            Fundamental._safe_div(
                "ttm_net_profit", "latest_equity", "ttm_roe", handle_neg_den="null"
            ),
            Fundamental._safe_div("ttm_cfo", "latest_assets", "ttm_cfo_yield"),
        )

        return ttm_agg.select(
            "ts_code",
            pl.col("notice_date").alias("report_date"),
            "ttm_revenue",
            "ttm_net_profit",
            "ttm_cfo",
            "ttm_fcf",
            "ttm_roe",
            "ttm_cfo_yield",
            "profit_momentum",
        )

    @staticmethod
    def metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Combine all fundamental feature groups + optional TTM rolling"""
        log.info("Computing full fundamental feature set")

        # Base selection — only needed columns
        needed_raw = [
            *Fundamental.IDENT_COLS,
            "net_profit",
            "total_revenue",
            "operate_profit",
            "operate_cost",
            "total_asset",
            "total_equity",
            "total_debt",
            "cash",
            "accounts_receivable",
            "inventory",
            "accounts_payable",
            "advance_receivable",
            "finance_expense",
            "cfo",
            "cfi",
            "net_profit_yoy",
            "total_revenue_yoy",
            "total_asset_yoy",
            "total_debt_yoy",
        ]

        available = [c for c in needed_raw if c in df.columns]
        missing = set(needed_raw) - set(available)
        if missing:
            log.warning(f"Missing raw columns (will produce nulls): {missing}")

        base = df.select(available)

        q = Fundamental.quality(base)
        lev = Fundamental.leverage(base)
        g = Fundamental.growth(base)
        cf = Fundamental.cashflow(base)

        combined = (
            q.join(lev, on=Fundamental.IDENT_COLS, how="inner")
            .join(g, on=Fundamental.IDENT_COLS, how="inner")
            .join(cf, on=Fundamental.IDENT_COLS, how="inner")
        )

        return combined.sort(["ts_code", "notice_date"])


class Behavioral:
    """Feature engineering class for behavioral (market microstructure and price action) metrics.
    Designed to be robust for stocks, indices, and volatility indices (QVIX-like).

    Required core columns (OHLCV):
        date (date), ts_code (str, optional for index), open, high, low, close
    Optional: volume, amount, amplitude, pct_chg, change, turnover_rate

    Behavior when columns missing:
    - Returns frame with IDENT_COLS + expected feature columns (all null if input missing)
    - Never returns completely empty schema (avoids join crashes)
    """

    IDENT_COLS = [
        "date"
    ]  # for index/QVIX; can be extended to ["ts_code", "date"] for stocks

    @staticmethod
    def _ensure_base_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Add missing IDENT_COLS with nulls if needed."""
        missing = [c for c in Behavioral.IDENT_COLS if c not in df.columns]
        if missing:
            log.debug(f"Adding missing identifier columns: {missing}")
            for c in missing:
                dtype = pl.Date if c == "date" else pl.Utf8
                df = df.with_columns(pl.lit(None).cast(dtype).alias(c))
        return df

    @staticmethod
    def momentum(df: pl.DataFrame) -> pl.DataFrame:
        """Price momentum, moving average deviation, ROC, channel position."""
        log.info("Behavioral.momentum: computing momentum features")
        df = Behavioral._ensure_base_schema(df.sort("date"))

        if "close" not in df.columns:
            log.warning("No 'close' column → returning empty momentum features")
            return df.select(*Behavioral.IDENT_COLS)

        return (
            df.with_columns(
                ret_1=pl.col("close").shift(-1) / pl.col("close") - 1,
                ret_4=pl.col("close").shift(-4) / pl.col("close") - 1,
                ret_12=pl.col("close").shift(-12) / pl.col("close") - 1,
                ret_1m=pl.col("close").shift(-22) / pl.col("close") - 1,
                ret_1q=pl.col("close").shift(-65) / pl.col("close") - 1,
                ret_1y=pl.col("close").shift(-261) / pl.col("close") - 1,
                ma_4=pl.col("close").rolling_mean(4, min_samples=2),
                ma_12=pl.col("close").rolling_mean(12, min_samples=4),
                ma_24=pl.col("close").rolling_mean(24, min_samples=8),
            )
            .with_columns(
                ma12_dev=(pl.col("close") / pl.col("ma_12") - 1).clip(-10, 10),
                ma24_dev=(pl.col("close") / pl.col("ma_24") - 1).clip(-10, 10),
                roc_4=pl.col("close").pct_change(4),
                roc_12=pl.col("close").pct_change(12),
            )
            .with_columns(
                ch_high=pl.col("high").rolling_max(20, min_samples=5)
                if "high" in df.columns
                else None,
                ch_low=pl.col("low").rolling_min(20, min_samples=5)
                if "low" in df.columns
                else None,
            )
            .with_columns(
                ch_pos=(
                    (pl.col("close") - pl.col("ch_low"))
                    / (pl.col("ch_high") - pl.col("ch_low")).clip(lower_bound=1e-8)
                ).clip(0, 1)
                if "ch_high" in df.columns and "ch_low" in df.columns
                else None,
                above_ma12=(pl.col("close") > pl.col("ma_12")).cast(pl.Int8),
                ma_cross_up=(pl.col("ma_4") > pl.col("ma_12")).cast(pl.Int8),
            )
            .select(
                *Behavioral.IDENT_COLS,
                cs.starts_with(
                    "ret_", "ma", "roc_", "ch_pos", "above_ma12", "ma_cross_up"
                ),
            )
        )

    @staticmethod
    def volatility(df: pl.DataFrame) -> pl.DataFrame:
        """Volatility estimators (Parkinson, Garman-Klass, realized, ATR)."""
        log.info("Behavioral.volatility: computing volatility features")
        df = Behavioral._ensure_base_schema(df.sort("date"))

        if "close" not in df.columns:
            log.warning("No 'close' → returning empty volatility features")
            return df.select(*Behavioral.IDENT_COLS)

        df = df.with_columns(log_ret_1=pl.col("close").log().diff())

        exprs: list[pl.Expr] = []

        # Parkinson (high-low)
        if all(c in df.columns for c in ["high", "low"]):
            exprs.append(
                ((pl.col("high") / pl.col("low")).log() ** 2 / (4 * np.log(2)))
                .sqrt()
                .rolling_mean(20, min_samples=5)
                .alias("vola_parkinson")
            )

        # Garman-Klass (OHLC)
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            exprs.append(
                (
                    0.5 * (pl.col("high") / pl.col("low")).log() ** 2
                    - (2 * np.log(2) - 1)
                    * (pl.col("close") / pl.col("open")).log() ** 2
                )
                .sqrt()
                .rolling_mean(20, min_samples=5)
                .alias("vola_gk")
            )

        # Realized volatility
        exprs.extend(
            [
                pl.col("log_ret_1").rolling_std(20, min_samples=10).alias("vola_20"),
                pl.col("log_ret_1").rolling_std(60, min_samples=20).alias("vola_60"),
            ]
        )

        # ATR % and regime ratio
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            df = df.with_columns(
                tr=pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                )
            )
            exprs.append(
                (pl.col("tr").rolling_mean(14, min_samples=5) / pl.col("close"))
                .clip(0, 1)
                .alias("atr_pct")
            )

        df = df.with_columns(*exprs)

        if "vola_20" in df.columns and "vola_60" in df.columns:
            df = df.with_columns(
                vola_regime_ratio=(
                    pl.col("vola_20") / pl.col("vola_60").clip(lower_bound=1e-8)
                ).clip(0.1, 10)
            )

        return df.select(
            *Behavioral.IDENT_COLS, cs.starts_with("vola_", "atr_pct", "tr")
        )

    @staticmethod
    def volume(df: pl.DataFrame) -> pl.DataFrame:
        """Volume deviation, spikes, VWAP deviation."""
        log.info("Behavioral.volume: computing volume features")
        df = Behavioral._ensure_base_schema(df.sort("date"))

        if "volume" not in df.columns or df["volume"].null_count() == df.height:
            log.info("No valid volume → returning null volume features")
            return df.select(
                *Behavioral.IDENT_COLS,
                pl.lit(None).alias("vol_dev20"),
                pl.lit(None).alias("vol_dev60"),
                pl.lit(None).alias("vol_spike"),
                pl.lit(None).alias("vwap_dev_1d"),
                pl.lit(None).alias("avg_trade_price"),
            )

        exprs = [
            (pl.col("volume") / pl.col("volume").rolling_mean(20, min_samples=5) - 1)
            .clip(-5, 5)
            .alias("vol_dev20"),
            (pl.col("volume") / pl.col("volume").rolling_mean(60, min_samples=10) - 1)
            .clip(-5, 5)
            .alias("vol_dev60"),
            (pl.col("volume") > pl.col("volume").rolling_mean(20, min_samples=5) * 2)
            .cast(pl.Int8)
            .alias("vol_spike"),
        ]

        if "amount" in df.columns and "volume" in df.columns:
            exprs.extend(
                [
                    (
                        pl.col("close")
                        / (pl.col("amount") / (pl.col("volume") + 1e-8)).clip(
                            lower_bound=1e-6
                        )
                        - 1
                    )
                    .clip(-0.5, 0.5)
                    .alias("vwap_dev_1d"),
                    (pl.col("amount") / (pl.col("volume") + 1e-8))
                    .clip(lower_bound=1e-6)
                    .alias("avg_trade_price"),
                ]
            )

        return df.with_columns(exprs).select(
            *Behavioral.IDENT_COLS,
            cs.starts_with("vol_", "vwap_dev", "avg_trade_price"),
        )

    @staticmethod
    def structure(df: pl.DataFrame) -> pl.DataFrame:
        """Variance ratio, autocorrelation, gap, near-high/low position."""
        log.info("Behavioral.structure: computing microstructure features")
        df = Behavioral._ensure_base_schema(df.sort("date"))

        if "close" not in df.columns:
            log.warning("No 'close' → skipping structure features")
            return df.select(*Behavioral.IDENT_COLS)

        df = df.with_columns(log_ret_1=pl.col("close").log().diff())

        exprs = [
            # Variance ratio (2-day vs 1-day)
            (
                pl.col("log_ret_1").diff(1).rolling_var(60, min_samples=20)
                / (2 * pl.col("log_ret_1").rolling_var(60, min_samples=20) + 1e-9)
            )
            .clip(0.1, 5)
            .alias("vr_2"),
            # Autocorrelation at lag 1 and 5
            pl.rolling_corr(
                pl.col("log_ret_1"),
                pl.col("log_ret_1").shift(1),
                window_size=60,
                min_samples=20,
            ).alias("ac_1"),
            pl.rolling_corr(
                pl.col("log_ret_1"),
                pl.col("log_ret_1").shift(5),
                window_size=60,
                min_samples=20,
            ).alias("ac_5"),
        ]

        if "high" in df.columns:
            exprs.append(
                (pl.col("close") / pl.col("high").rolling_max(50, min_samples=10) - 1)
                .clip(-1, 0.5)
                .alias("near_high")
            )

        if "low" in df.columns:
            exprs.append(
                (pl.col("close") / pl.col("low").rolling_min(50, min_samples=10) - 1)
                .clip(-0.5, 1)
                .alias("near_low")
            )

        if all(c in df.columns for c in ["open", "close"]):
            exprs.append(
                (
                    (pl.col("open") - pl.col("close").shift(1))
                    / pl.col("close").shift(1).clip(lower_bound=1e-8)
                )
                .clip(-0.2, 0.2)
                .alias("gap")
            )

        return df.with_columns(exprs).select(
            *Behavioral.IDENT_COLS, cs.starts_with("vr_", "ac_", "near_", "gap")
        )

    @staticmethod
    def metrics(df: pl.DataFrame, idents: list[str] | None = None) -> pl.DataFrame:
        """Combine all behavioral feature groups."""
        log.info("Behavioral.metrics: computing full behavioral feature suite")

        if df.is_empty() or "date" not in df.columns:
            log.warning("Input empty or missing 'date' → returning empty schema")
            return pl.DataFrame(schema={"date": pl.Date})

        df = df.sort("date")
        mom = Behavioral.momentum(df)
        vola = Behavioral.volatility(df)
        vol = Behavioral.volume(df)
        stru = Behavioral.structure(df)

        combined = (
            mom.join(vola, on=Behavioral.IDENT_COLS, how="inner")
            .join(vol, on=Behavioral.IDENT_COLS, how="inner")
            .join(stru, on=Behavioral.IDENT_COLS, how="inner")
        )

        if idents:
            to_add = [
                c for c in idents if c in df.columns and c not in combined.columns
            ]
            if to_add:
                combined = combined.join(
                    df.select([*Behavioral.IDENT_COLS, *to_add]).unique(),
                    on=Behavioral.IDENT_COLS,
                    how="left",
                )

        return combined


class Northbound:
    """Northbound (沪深港通 北向资金) behavioral feature extractor.
    Input: daily northbound flow data from AkShare (net_buy, fund_inflow, cum_net_buy, date)
    Output: wide-format feature table with momentum, persistence, acceleration signals.
    All features are null-safe and work with partial/missing data.
    """

    IDENT_COLS = ["date"]

    @staticmethod
    def _safe_zscore(
        col: str,
        mean_win: int = 20,
        std_win: int = 60,
        min_periods_mean: int | None = None,
        min_periods_std: int | None = None,
    ) -> pl.Expr:
        """Rolling z-score with configurable min_periods and safe division."""
        m_win = min_periods_mean or max(3, mean_win // 3)
        s_win = min_periods_std or max(5, std_win // 3)

        mean_expr = pl.col(col).rolling_mean(mean_win, min_samples=m_win)
        std_expr = (
            pl.col(col).rolling_std(std_win, min_samples=s_win).clip(lower_bound=1e-8)
        )

        z = (pl.col(col) - mean_expr) / std_expr
        return z.clip(-10, 10).alias(f"{col}_z")

    @staticmethod
    def flow(df: pl.DataFrame) -> pl.DataFrame:
        """Short/medium-term flow strength & shock signals."""
        log.info("Northbound.flow: computing flow z-score & shock signals")

        if "net_buy" not in df.columns:
            log.warning("No 'net_buy' column → returning empty flow features")
            return df.select(*Northbound.IDENT_COLS)

        df = df.with_columns(
            Northbound._safe_zscore(
                "net_buy",
                mean_win=10,
                std_win=30,
                min_periods_mean=5,
                min_periods_std=10,
            ).alias("nb_flow_z_short"),
            Northbound._safe_zscore("net_buy", mean_win=20, std_win=60).alias(
                "nb_flow_z"
            ),
            Northbound._safe_zscore("net_buy", mean_win=60, std_win=120).alias(
                "nb_flow_z_long"
            ),
        )

        z_cols = ["nb_flow_z_short", "nb_flow_z", "nb_flow_z_long"]
        df = df.with_columns(
            # Absolute values (per window)
            pl.col("nb_flow_z_short").abs().alias("nb_flow_z_short_abs"),
            pl.col("nb_flow_z").abs().alias("nb_flow_z_abs"),
            pl.col("nb_flow_z_long").abs().alias("nb_flow_z_long_abs"),
            # Positive / shock indicators (per window)
            pl.col(z_cols).gt(0).cast(pl.Int8).name.prefix("nb_flow_pos_"),
            pl.col(z_cols).abs().gt(2).cast(pl.Int8).name.prefix("nb_flow_shock_"),
        )
        df = df.with_columns(
            # Max absolute z-score across windows
            nb_flow_abs_max=pl.max_horizontal(
                cs.contains(
                    "_abs"
                )  # selects nb_flow_z_short_abs, nb_flow_z_abs, nb_flow_z_long_abs
            ),
            # Any shock across windows
            nb_flow_shock_any=pl.max_horizontal(cs.starts_with("nb_flow_shock_")).cast(
                pl.Int8
            ),
            # Average absolute z (optional, for overall strength)
            nb_flow_abs_avg=pl.mean_horizontal(cs.contains("_abs")),
        )

        return df.select(
            *Northbound.IDENT_COLS,
            cs.starts_with(
                "nb_flow_z",
                "nb_flow_pos",
                "nb_flow_shock",
                "nb_flow_abs",
                "nb_flow_conviction",
            ),
        )

    @staticmethod
    def persistence(df: pl.DataFrame) -> pl.DataFrame:
        """Trend strength, conviction, and cumulative pressure."""
        log.info("Northbound.persistence: computing persistence & trend signals")

        if "net_buy" not in df.columns:
            return df.select(*Northbound.IDENT_COLS)

        return df.with_columns(
            # Rolling sign-weighted conviction (stronger when consistent direction)
            nb_flow_conviction=(
                pl.col("net_buy").sign()
                * pl.col("net_buy").abs().rolling_mean(20, min_samples=5)
            ),
            # Long-term trend strength
            nb_flow_trend=pl.col("net_buy").rolling_mean(60, min_samples=15),
            # Cumulative z-score direction (persistent buying/selling pressure)
            nb_cum_z_trend=Northbound._safe_zscore(
                "net_buy", mean_win=60, std_win=120
            ).rolling_mean(20, min_samples=5),
        ).select(
            *Northbound.IDENT_COLS,
            cs.starts_with("nb_flow_conviction", "nb_flow_trend", "nb_cum_z_trend"),
        )

    @staticmethod
    def accel(df: pl.DataFrame) -> pl.DataFrame:
        """Acceleration / deceleration of flow (second difference)."""
        log.info("Northbound.accel: computing flow acceleration")

        if "net_buy" not in df.columns:
            log.warning("No 'net_buy' → returning empty accel features")
            return df.select(*Northbound.IDENT_COLS)

        df = df.with_columns(
            nb_flow_delta=pl.col("net_buy").diff(),
            nb_flow_accel=pl.col("net_buy").diff().diff(),
        )

        df = df.with_columns(
            nb_flow_accel_clipped=pl.col("nb_flow_accel").clip(-1e6, 1e6)
        )
        df = df.with_columns(
            nb_flow_accel_up=(pl.col("nb_flow_accel_clipped") > 0).cast(pl.Int8),
            nb_flow_accel_down=(pl.col("nb_flow_accel_clipped") < 0).cast(pl.Int8),
            # Shock: accel > 95th percentile of historical accel (absolute)
            nb_flow_shock_accel=(
                pl.col("nb_flow_accel_clipped").abs()
                > pl.col("nb_flow_accel_clipped")
                .abs()
                .rolling_quantile(0.95, window_size=60, min_samples=20)
            ).cast(pl.Int8),
        )

        return df.select(
            *Northbound.IDENT_COLS,
            cs.starts_with("nb_flow_delta", "nb_flow_accel", "nb_flow_shock_accel"),
        )

    @staticmethod
    def metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Combine all Northbound feature groups."""
        log.info("Northbound.metrics: computing full northbound feature suite")

        if df.is_empty() or "date" not in df.columns:
            log.warning("Input empty or missing 'date' → returning minimal schema")
            return pl.DataFrame(schema={"date": pl.Date})

        df = df.sort("date")

        f = Northbound.flow(df)
        p = Northbound.persistence(df)
        a = Northbound.accel(df)

        combined = f.join(p, on=Northbound.IDENT_COLS, how="inner").join(
            a, on=Northbound.IDENT_COLS, how="inner"
        )
        return combined.with_columns(
            # Overall shock: ANY shock from flow or accel windows
            nb_shock_any=pl.any_horizontal(
                cs.starts_with("nb_flow_shock"), cs.starts_with("nb_flow_accel")
            ).cast(pl.Int8),
            # Optional: overall positive flow strength
            nb_flow_pos_any=pl.any_horizontal(cs.starts_with("nb_flow_pos")).cast(
                pl.Int8
            ),
        ).sort("date")


class MarginShort:
    """Feature extractor for A-share margin & short-selling data (SH+SZ combined).
    All operations use consistent clipping and null-safe logic.
    """

    IDENT_COLS = ["date"]

    @staticmethod
    def _safe_ratio(
        num_col: str,
        den_col: str,
        alias: str,
        min_den: float = 1e-6,
        clip_lower: float = 0.0,
        clip_upper: float | None = None,
    ) -> pl.Expr:
        """Safe division with configurable bounds."""
        num = pl.col(num_col)
        den = pl.col(den_col).clip(lower_bound=min_den)
        ratio = num / den
        if clip_upper is not None:
            ratio = ratio.clip(lower_bound=clip_lower, upper_bound=clip_upper)
        else:
            ratio = ratio.clip(lower_bound=clip_lower)
        return ratio.alias(alias)

    @staticmethod
    def level(df: pl.DataFrame) -> pl.DataFrame:
        log.info("MarginShort.level: computing margin level & deviation")
        if "total_margin_balance" not in df.columns:
            log.warning("No 'total_margin_balance' → empty level features")
            return df.select(*MarginShort.IDENT_COLS)

        return (
            df.with_columns(
                margin_ma20=pl.col("total_margin_balance").rolling_mean(
                    20, min_samples=5
                ),
                margin_ma60=pl.col("total_margin_balance").rolling_mean(
                    60, min_samples=15
                ),
                margin_ma120=pl.col("total_margin_balance").rolling_mean(
                    120, min_samples=30
                ),
            )
            .with_columns(
                margin_dev60=(
                    pl.col("total_margin_balance") / pl.col("margin_ma60").clip(1e-6)
                    - 1
                ).clip(-0.8, 2.0),
                margin_dev120=(
                    pl.col("total_margin_balance") / pl.col("margin_ma120").clip(1e-6)
                    - 1
                ).clip(-0.8, 2.0),
                leverage_dev=(
                    pl.col("total_margin_balance") / pl.col("margin_ma120").clip(1e-6)
                    - 1
                ).clip(-0.8, 2.0),
            )
            .select(
                *MarginShort.IDENT_COLS,
                "total_margin_balance",
                cs.starts_with("margin_ma"),
                cs.starts_with("margin_dev"),
                "leverage_dev",
            )
        )

    @staticmethod
    def impulse(df: pl.DataFrame) -> pl.DataFrame:
        log.info("MarginShort.impulse: computing margin flow impulse")
        if "total_margin_balance" not in df.columns:
            return df.select(*MarginShort.IDENT_COLS)

        return (
            df.with_columns(
                margin_delta=pl.col("total_margin_balance").diff(),
            )
            .with_columns(
                # Use relative change — much safer than absolute diff
                margin_delta_pct=pl.col("total_margin_balance")
                .pct_change()
                .clip(-1.0, 1.0),
                margin_accel_pct=pl.col("total_margin_balance")
                .pct_change()
                .diff()
                .clip(-1.0, 1.0),
            )
            .with_columns(
                margin_impulse5=pl.col("margin_delta_pct").rolling_mean(
                    5, min_samples=3
                ),
                margin_impulse20=pl.col("margin_delta_pct").rolling_mean(
                    20, min_samples=8
                ),
                margin_impulse_up=(pl.col("margin_delta_pct") > 0).cast(pl.Int8),
            )
            .select(
                *MarginShort.IDENT_COLS,
                cs.starts_with("margin_delta", "margin_impulse"),
            )
        )

    @staticmethod
    def stress(df: pl.DataFrame) -> pl.DataFrame:
        log.info("MarginShort.stress: computing short pressure & leverage stress")
        required = [
            "short_sell_volume",
            "margin_buy_amount",
            "short_balance",
            "total_margin_balance",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            log.warning(f"Missing columns for stress: {missing}")
            return df.select(*MarginShort.IDENT_COLS)

        df = df.with_columns(
            short_long_ratio=MarginShort._safe_ratio(
                "short_sell_volume",
                "margin_buy_amount",
                "short_long_ratio",
                clip_lower=0.0,
                clip_upper=10.0,
            ),
            short_leverage_ratio=MarginShort._safe_ratio(
                "short_balance",
                "total_margin_balance",
                "short_leverage_ratio",
                clip_lower=0.0,
                clip_upper=2.0,  # raised upper limit
            ),
            short_ma20=pl.col("short_sell_volume").rolling_mean(20, min_samples=5),
            short_ma60=pl.col("short_sell_volume").rolling_mean(60, min_samples=15),
        ).with_columns(
            short_pressure_20d=(
                pl.col("short_sell_volume") > pl.col("short_ma20") * 1.5
            ).cast(pl.Int8),
            short_pressure_60d=(
                pl.col("short_sell_volume") > pl.col("short_ma60") * 1.5
            ).cast(pl.Int8),
            short_stress_60d=(
                pl.col("short_leverage_ratio")
                > pl.col("short_leverage_ratio").rolling_mean(60, min_samples=15)
            ).cast(pl.Int8),
        )

        pressure_cols = cs.starts_with("short_pressure_")
        stress_cols = cs.starts_with("short_stress_")

        if not df.select(pressure_cols.append(stress_cols)).columns:
            log.warning("No stress/pressure indicators created → short_stress_any = 0")
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("short_stress_any"))
        else:
            df = df.with_columns(
                short_stress_any=pl.any_horizontal(pressure_cols, stress_cols).cast(
                    pl.Int8
                )
            )

        return df.select(*MarginShort.IDENT_COLS, cs.starts_with("short_", "margin_"))

    @staticmethod
    def metrics(df: pl.DataFrame) -> pl.DataFrame:
        log.info("MarginShort.metrics: computing full margin/short feature suite")

        if df.is_empty() or "date" not in df.columns:
            log.warning("Input empty or missing 'date' → returning minimal schema")
            return pl.DataFrame(schema={"date": pl.Date})

        df = df.sort("date")

        lev = MarginShort.level(df)
        imp = MarginShort.impulse(df)
        stress = MarginShort.stress(df)

        combined = lev.join(imp, on=MarginShort.IDENT_COLS, how="inner").join(
            stress, on=MarginShort.IDENT_COLS, how="inner"
        )

        return combined.with_columns(
            overall_leverage_stress=pl.any_horizontal(
                cs.contains("leverage_dev").gt(0.5), cs.starts_with("short_stress_")
            ).cast(pl.Int8),
            margin_flow_pressure=pl.any_horizontal(
                cs.contains("margin_impulse").gt(0), cs.starts_with("short_pressure_")
            ).cast(pl.Int8),
        ).sort("date")


class Shibor:
    """SHIBOR funding rate feature extractor.
    Focuses on liquidity shocks, curve shape, and stress signals.
    Input columns (from ak.macro_china_shibor_all):
        date, ON_rate, 1W_rate, 1M_rate, 3M_rate, 1Y_rate (and optional change columns)
    All features are null-safe and robust to partial/missing tenors.
    """

    IDENT_COLS = ["date"]

    @staticmethod
    def _safe_zscore(
        col: str,
        win: int = 60,
        min_samples: int | None = None,
    ) -> pl.Expr:
        """Rolling z-score with safe std and clipping."""
        m_periods = min_samples or max(5, win // 4)
        mean = pl.col(col).rolling_mean(win, min_samples=m_periods)
        std = pl.col(col).rolling_std(win, min_samples=m_periods).clip(lower_bound=1e-8)
        return ((pl.col(col) - mean) / std).clip(-10, 10)

    @staticmethod
    def shock(df: pl.DataFrame) -> pl.DataFrame:
        """Liquidity shock & funding pressure signals (focus on overnight)."""
        log.info("Shibor.shock: computing funding shock & liquidity pressure")

        if "ON_rate" not in df.columns:
            log.warning("No 'ON_rate' column → returning empty shock features")
            return df.select(*Shibor.IDENT_COLS)

        df = df.with_columns(
            on_change=pl.col("ON_rate").diff(),
        )
        df = df.with_columns(
            on_change_ma20=pl.col("on_change").rolling_mean(20, min_samples=5),
            on_change_ma60=pl.col("on_change").rolling_mean(60, min_samples=15),
        )
        df = df.with_columns(
            shibor_on_shock_short=pl.col("on_change") - pl.col("on_change_ma20"),
            shibor_on_shock_long=pl.col("on_change") - pl.col("on_change_ma60"),
        )
        df = df.with_columns(
            shibor_on_shock_z_short=Shibor._safe_zscore(
                "shibor_on_shock_short", win=60, min_samples=15
            ),
            shibor_on_shock_z_long=Shibor._safe_zscore(
                "shibor_on_shock_long", win=120, min_samples=30
            ),
        )
        df = df.with_columns(
            shibor_liquidity_tighten=(pl.col("shibor_on_shock_z_short") > 1.5).cast(
                pl.Int8
            ),
            shibor_funding_stress=(pl.col("shibor_on_shock_z_short") > 2.0).cast(
                pl.Int8
            ),
        )

        return df.select(
            *Shibor.IDENT_COLS,
            cs.starts_with("shibor_on_shock", "shibor_liquidity_", "shibor_funding_"),
        )

    @staticmethod
    def curve(df: pl.DataFrame) -> pl.DataFrame:
        log.info("Shibor.curve: computing yield curve structure & slope signals")

        tenors = ["ON_rate", "1W_rate", "1M_rate", "3M_rate", "1Y_rate"]
        available = [t for t in tenors if t in df.columns]

        if len(available) < 2:
            log.warning(f"Too few tenors ({available}) → empty curve features")
            return df.select(*Shibor.IDENT_COLS)

        exprs = []

        if "ON_rate" in available:
            for t in ["3M_rate", "1Y_rate"]:
                if t in available:
                    exprs.append(
                        (pl.col(t) - pl.col("ON_rate"))
                        .clip(-5, 5)
                        .alias(f"shibor_spread_{t.split('_')[0].lower()}_on")
                    )

        if "1Y_rate" in available and "1M_rate" in available:
            exprs.append(
                (pl.col("1Y_rate") - pl.col("1M_rate"))
                .clip(-5, 5)
                .alias("shibor_slope_long_short")
            )

        if "ON_rate" in available and "3M_rate" in available:
            exprs.append(
                (pl.col("ON_rate") > pl.col("3M_rate"))
                .cast(pl.Int8)
                .alias("shibor_inverted_short")
            )

            # momentum (no derived-column dependency)
            exprs.append(
                (pl.col("3M_rate") - pl.col("ON_rate"))
                .diff()
                .rolling_mean(5, min_samples=3)
                .clip(-2, 2)
                .alias("shibor_spread_3m_mom")
            )

        if len(available) >= 3:
            exprs.append(
                pl.concat_list([pl.col(t) for t in available])
                .list.std()
                .clip(0, 10)
                .alias("shibor_curve_dispersion")
            )

        return df.with_columns(exprs).select(
            *Shibor.IDENT_COLS,
            cs.starts_with(
                "shibor_spread_",
                "shibor_slope_",
                "shibor_inverted",
                "shibor_curve_",
            ),
        )

    @staticmethod
    def metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Combine all SHIBOR feature groups."""
        log.info("Shibor.metrics: computing full SHIBOR feature suite")

        if df.is_empty() or "date" not in df.columns:
            log.warning("Input empty or missing 'date' → returning minimal schema")
            return pl.DataFrame(schema={"date": pl.Date})

        df = df.sort("date")

        shock_df = Shibor.shock(df)
        curve_df = Shibor.curve(df)

        combined = shock_df.join(curve_df, on=Shibor.IDENT_COLS, how="inner")
        return combined.with_columns(
            shibor_stress_any=pl.any_horizontal(
                cs.contains("shibor_funding_stress"), cs.contains("shibor_inverted")
            ).cast(pl.Int8),
            shibor_pressure=pl.mean_horizontal(
                cs.contains("shibor_on_shock_z"), cs.contains("shibor_spread_")
            )
            .clip(-5, 5)
            .alias("shibor_pressure"),
        ).sort("date")


@dataclass(frozen=True)
class CrossSectionFlow:
    by: list[str]
    namespace: str = "cs"
    sep: str = "_"
    q_limits: tuple[float, float] = (0.01, 0.99)
    # Formalized internal prefix to avoid collisions with user columns
    _TMP_PREFIX: str = "__tmp_cs_internal__"

    def col_name(self, factor: str, suffix: str, is_tmp: bool = False) -> str:
        prefix = self._TMP_PREFIX if is_tmp else f"{self.namespace}{self.sep}"
        return f"{prefix}{factor}{self.sep}{suffix}"

    def apply(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        factors: list[str],
        winsorize: bool = True,
        zsuffix: str = "z",
    ) -> pl.DataFrame:
        # 1. Build Intermediate Expressions
        # We only include quantile expressions if winsorize is actually requested
        stats_exprs = []
        for f in factors:
            stats_exprs.extend(
                [
                    pl.col(f)
                    .mean()
                    .over(self.by)
                    .alias(self.col_name(f, "mean", True)),
                    pl.col(f).std().over(self.by).alias(self.col_name(f, "std", True)),
                ]
            )

            if winsorize:
                lo, hi = self.q_limits
                stats_exprs.extend(
                    [
                        pl.col(f)
                        .quantile(lo)
                        .over(self.by)
                        .alias(self.col_name(f, "qlo", True)),
                        pl.col(f)
                        .quantile(hi)
                        .over(self.by)
                        .alias(self.col_name(f, "qhi", True)),
                    ]
                )

        # 2. Build Final Transformation Expressions
        trans_exprs = []
        for f in factors:
            # References to internals
            mean = pl.col(self.col_name(f, "mean", True))
            std = pl.col(self.col_name(f, "std", True)).clip(lower_bound=1e-9)

            if winsorize:
                win_expr = pl.col(f).clip(
                    pl.col(self.col_name(f, "qlo", True)),
                    pl.col(self.col_name(f, "qhi", True)),
                )
                trans_exprs.append(win_expr.alias(self.col_name(f, "win")))
                z_input = win_expr
            else:
                z_input = pl.col(f)

            z_score = ((z_input - mean) / std).alias(self.col_name(f, zsuffix))
            trans_exprs.append(z_score)

        # 3. Execution Pipeline
        # We convert to lazy if it's not already to allow Polars to optimize the graph
        is_lazy = isinstance(df, pl.LazyFrame)
        lf = df.lazy() if not is_lazy else df

        processed = (
            lf.with_columns(stats_exprs)
            .with_columns(trans_exprs)
            .drop(pl.selectors.starts_with(self._TMP_PREFIX))
        )
        return processed.collect()


@dataclass
class SectorGroup:
    """Applies cross-sectional normalization (winsor + z-score) per group."""

    by: list[str]
    factors: list[str]

    winsor_limits: tuple[float, float] = field(default=(0.01, 0.99))
    winsorize: bool = field(default=False)
    min_group_size: int = field(default=5)
    zsuffix: str = "z"

    def __post_init__(self):
        self.flow = CrossSectionFlow(by=self.by, q_limits=self.winsor_limits)

    def zfactors(self) -> list[str]:
        return [self.flow.col_name(f, self.zsuffix, is_tmp=False) for f in self.factors]

    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = set(self.by).difference(df.columns)
        if missing:
            log.warning(f"Missing group keys {missing}, skip normalization")
            return df

        factors = [f for f in self.factors if f in df.columns]
        if not factors:
            log.info("No valid factors to normalize")
            return df

        # ---- drop small groups ----
        if self.min_group_size > 1:
            df = (
                df.with_columns(pl.len().over(self.by).alias(GSIZE))
                .filter(pl.col(GSIZE) >= self.min_group_size)
                .drop(GSIZE)
            )

        flow = self.flow
        processed = flow.apply(
            df, factors, winsorize=self.winsorize, zsuffix=self.zsuffix
        )
        return processed
