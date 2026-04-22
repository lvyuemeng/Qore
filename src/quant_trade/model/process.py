from collections.abc import Generator
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final, Protocol

import polars as pl
from scipy import stats

from quant_trade.config.logger import debug_null_profile, log
from quant_trade.transform import GSIZE, RANK

LABEL_PREFIX: Final[str] = "label_"
HORZION_DAY:Final[int] = 21
EMBARGO_DAY:Final[int] = 5

@dataclass
class PurgedTimeSplit:
    """
    Time series split with purging to prevent lookahead bias.

    Purging removes training samples whose label period overlaps with
    test period to prevent information leakage.

    Diagram:
        [TRAIN]---[PURGE]---[TEST]---[FUTURE]
                  ↑        ↑
                embargo    horizon
    """

    horizon_days: int = HORZION_DAY
    embargo_days: int = EMBARGO_DAY

    def __post_init__(self):
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {self.horizon_days}")
        if self.embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {self.embargo_days}")

    def split(
        self,
        df: pl.DataFrame,
        split_date: date,
        date_col: str | pl.Expr = "date",
        label_end_col: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data at a specific date with purging.

        Args:
            df: DataFrame with time series data
            date_col: Column containing observation dates
            split_date: Date to split train/test
            label_end_col: Optional column with label end date for purging

        Returns:
            Tuple of (train_df, test_df)
        """
        # Convert inputs
        if isinstance(date_col, str):
            date_expr = pl.col(date_col)
        else:
            date_expr = date_col

        purge_start = split_date - timedelta(days=self.horizon_days)
        test_start = split_date + timedelta(days=self.embargo_days)
        log.debug(f"purge start: {purge_start}")
        log.debug(f"test start: {test_start}")
        test_df = df.filter(date_expr >= test_start)

        if label_end_col:
            train_df = df.filter(
                (pl.col(label_end_col) < purge_start)  # Label ends before purge
                & (date_expr < purge_start)  # Observation before purge
            )
        else:
            train_df = df.filter(date_expr < purge_start)

        if len(train_df) == 0:
            raise ValueError(f"No training data before {purge_start}")
        if len(test_df) == 0:
            raise ValueError(f"No test data after {test_start}")

        log.info(f"Purging: Removed(embargo) data from {purge_start} to {split_date}")

        return train_df, test_df


@dataclass
class PurgedKFold:
    """
    K-Fold cross-validation with purging for time series data.

    Each fold has:
    1. Training period (all data before test start minus purge window)
    2. Purge window (horizon + embargo before test start)
    3. Test period

    Prevents information leakage from future to past.
    """

    n_splits: int = 5
    horizon_days: int = HORZION_DAY
    embargo_days: int = EMBARGO_DAY
    min_train_size: int = 0

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.horizon_days < 0:
            raise ValueError(f"horizon_days must be >= 0, got {self.horizon_days}")
        if self.embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {self.embargo_days}")

    def split(
        self,
        df: pl.DataFrame,
        date_col: str | pl.Expr = "date",
        label_end_col: str | None = None,
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate train/test splits with purging.

        Args:
            df: DataFrame with time series data
            date_col: Column containing observation dates
            label_end_col: Optional column with label end date for purging

        Yields:
            Tuples of (train_df, test_df) for each fold
        """
        if isinstance(date_col, str):
            date_expr = pl.col(date_col)
        else:
            date_expr = date_col

        dates = df.select(date_expr).unique().sort(date_expr).to_series().to_list()
        n_dates = len(dates)

        if n_dates < self.n_splits * 2:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for {self.n_splits} splits"
            )

        test_size = n_dates // (self.n_splits + 1)
        log.debug(f"Purged k fold test size {test_size}")
        for fold in range(1, self.n_splits + 1):
            start_idx = fold * test_size
            end_idx = (
                n_dates - 1 if fold == self.n_splits else start_idx + test_size - 1
            )
            test_start = dates[start_idx]
            log.debug(f"Purged k fold test start {fold}: {test_start}")
            test_end = dates[end_idx]
            log.debug(f"Purged k fold test end {fold}: {test_end}")
            purge_cutoff = test_start - timedelta(
                days=self.horizon_days + self.embargo_days
            )

            log.debug(
                f"Fold {fold}: test from {test_start} to {test_end}, purge cutoff {purge_cutoff}"
            )
            test_df = df.filter((date_expr >= test_start) & (date_expr <= test_end))

            if label_end_col is not None:
                train_df = df.filter(pl.col(label_end_col) < purge_cutoff)
            else:
                train_df = df.filter(date_expr < purge_cutoff)

            if len(train_df) < self.min_train_size:
                log.warning(
                    f"Fold {fold + 1}: only {len(train_df)} training samples, skipped"
                )
                continue

            if len(test_df) == 0:
                log.warning(f"Fold {fold + 1}: empty test set, skipped")
                continue

            yield train_df, test_df


@dataclass
class WalkForwardValidation:
    """
    Walk-forward validation with expanding/rolling window and purging.

    Common in financial time series to simulate live trading.
    """

    n_windows: int = 10
    window_size: int = 252  # ~1 trading year
    step_size: int = 63  # ~1 trading quarter
    horizon_days: int = HORZION_DAY
    embargo_days: int = EMBARGO_DAY
    min_train_size: int = 0
    expanding: bool = True

    def __post_init__(self):
        if self.n_windows < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.window_size}")
        if total_gap := self.horizon_days + self.embargo_days >= self.window_size:
            raise ValueError(
                f"horizon_days + embargo_days ({total_gap}), must be less than"
                f"window size ({self.window_size})"
            )
        if total_gap >= self.step_size:
            raise ValueError(
                f"horizon_days + embargo_days ({total_gap}), must be less than"
                f"step size ({self.step_size})"
            )

    def split(
        self,
        df: pl.DataFrame,
        date_col: str | pl.Expr = "date",
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate expanding/rolling walk-forward splits.
        """
        if isinstance(date_col, str):
            date_expr = pl.col(date_col)
        else:
            date_expr = date_col

        dates = (
            df.select(date_expr)
            .unique()
            .sort(date_expr)
            .get_column(date_expr.meta.output_name())
            .to_list()
        )
        n_dates = len(dates)

        for i in range(1, self.n_windows + 1):
            test_start_idx = i * self.step_size
            test_end_idx = min(test_start_idx + self.window_size, n_dates - 1)

            if test_end_idx <= test_start_idx:
                break

            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            purge_cutoff = test_start - timedelta(
                days=self.horizon_days + self.embargo_days
            )
            log.debug(
                f"Window {i}: test from {test_start} to {test_end}, purge cutoff {purge_cutoff}"
            )
            if self.expanding:
                # Expanding window: all data before purge cutoff
                train_df = df.filter(date_expr < purge_cutoff)
                log.debug(f"Window {i} expanding: train < {purge_cutoff}")
            else:
                # Rolling window: fixed size before purge cutoff
                train_start_idx = max(0, test_start_idx - self.step_size)
                train_start = dates[train_start_idx]
                train_df = df.filter(
                    (date_expr >= train_start) & (date_expr < purge_cutoff)
                )
                log.debug(
                    f"Window {i} rolling: train from {train_start} to {purge_cutoff}"
                )

            test_df = df.filter((date_expr >= test_start) & (date_expr <= test_end))

            if len(train_df) < self.min_train_size:
                log.warning(
                    f"Window {i}: only {len(train_df)} training samples, skipped"
                )
                continue
            yield train_df, test_df


class LabelBuilder(Protocol):
    factor:str

    def label(self, df: pl.DataFrame) -> pl.DataFrame: ...

    @property
    def label_name(self) -> str: ...
    @property
    def rank_by_name(self) -> str: ...

@dataclass
class Gaussian:
    by:list[str]
    winsor_limits: tuple[float, float] | None = (0.01, 0.99)
    alpha:float = 0.5

    def __call__(self,df:pl.DataFrame,factor:str,alias:str) -> pl.DataFrame:
        if factor not in df.columns:
            raise ValueError(f"Factor '{factor}' not found in DataFrame")

        x = pl.col(factor)
        by = self.by
        if self.winsor_limits is not None:
            lo, hi = self.winsor_limits
            x = x.clip(
                x.quantile(lo).over(by),
                x.quantile(hi).over(by),
            )

        rank = x.rank("average").over(by)
        n = pl.len().over(by)

        u = ((rank - self.alpha) / (n + 1 - 2 * self.alpha)).clip(1e-12, 1 - 1e-12)

        label = (
            pl.when(n >= 3)
            .then(u.map_batches(stats.norm.ppf))
            .alias(alias)
        )

        return df.with_columns(label).filter(pl.col(alias).is_not_null())


@dataclass
class GaussianLabelBuilder:
    """
    Builds Gaussian-transformed labels from returns for ranking models.

    This applies rank-Gaussian transformation (inverse normal transformation)
    which preserves ordering while creating normally-distributed targets.

    Formula:
    1. winsorize returns within groups
    2. rank winsorized returns: rank = r.rank().over("date")
    3. uniform transform: u = (rank - 0.5) / n
    4. Gaussian transform: y = Φ⁻¹(u) where Φ is standard normal CDF
    """

    factor: str
    rank_by: str = "date"
    by: list[str] | None = None
    winsor_limits: tuple[float, float] | None = (0.01, 0.99)
    alpha: float = 0.5
    label_prefix: str = LABEL_PREFIX

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        log.debug(f"preparing df sanity: {debug_null_profile(df)}")
        by = self.by + [self.rank_by] if self.by else [self.rank_by]
        return Gaussian(
                by=by,
                winsor_limits=self.winsor_limits,
                alpha=self.alpha
            )(df,self.factor,self.label_name)

    @property
    def label_name(self) -> str:
        return f"{self.label_prefix}{self.factor}"

    @property
    def rank_by_name(self) -> str:
        return self.rank_by


@dataclass
class DiscreteLabelBuilder:
    """Discrete labels via binning (for LambdaRank ranking)."""

    factor: str
    rank_by: str = "date"
    by: list[str] | None = None
    num_bins: int = 5  # e.g., 4 bins: 0, 1, 2, 3 (relevance grades)
    winsor_limits: tuple[float, float] | None = (0.01, 0.99)
    alpha: float = 0.5
    label_prefix: str = LABEL_PREFIX

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """Bin factor into discrete relevance grades (0 to num_bins-1)."""
        log.debug(f"preparing df sanity: {debug_null_profile(df)}")
        by = self.by + [self.rank_by] if self.by else [self.rank_by]
        temp_col = f"_{self.label_name}"
        df = Gaussian(
                by=by,
                winsor_limits=self.winsor_limits,
                alpha=self.alpha
            )(df,self.factor,temp_col)
        return self._discretize(df,temp_col)

    def _discretize(self, df: pl.DataFrame,col:str) -> pl.DataFrame:
        return df.with_columns(
                [
                    pl.col(col)
                    .rank("average")
                    .over(self.rank_by)
                    .alias(RANK),

                    pl.len()
                    .over(self.rank_by)
                    .alias(GSIZE),
                ]
            ).with_columns(

                    (pl.col(RANK) - 1)
                    .mul(self.num_bins)
                    .floordiv(pl.col(GSIZE))
                    .cast(pl.UInt8)
                    .alias(self.label_name)

            ).drop(RANK,GSIZE)

    @property
    def label_name(self) -> str:
        return f"{self.label_prefix}{self.factor}"

    @property
    def rank_by_name(self) -> str:
        return self.rank_by


@dataclass
class IdentityLabelBuilder:
    """Identity transformation - no change to labels (for regression)."""

    factor: str
    rank_by: str = "date"
    label_prefix: str = LABEL_PREFIX

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(self.factor).alias(self.label_name))

    @property
    def label_name(self) -> str:
        return f"{self.label_prefix}{self.factor}"

    @property
    def rank_by_name(self) -> str:
        return self.rank_by


@dataclass
class BinaryLabelBuilder:
    """Binary labels via thresholding (for binary classification)."""

    factor: str
    threshold: float = 0.0
    rank_by: str = "date"
    label_prefix: str = LABEL_PREFIX

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (pl.col(self.factor) > self.threshold).cast(pl.Int64).alias(self.label_name)
        )

    @property
    def label_name(self) -> str:
        return f"{self.label_prefix}{self.factor}"

    @property
    def rank_by_name(self) -> str:
        return self.rank_by

