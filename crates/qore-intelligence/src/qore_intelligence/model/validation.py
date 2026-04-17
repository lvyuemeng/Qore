from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass(slots=True)
class PurgedTimeSplit:
    horizon_days: int = 21
    embargo_days: int = 5

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            msg = f"horizon_days must be >= 1, got {self.horizon_days}"
            raise ValueError(msg)
        if self.embargo_days < 0:
            msg = f"embargo_days must be >= 0, got {self.embargo_days}"
            raise ValueError(msg)

    def split(
        self,
        df: pl.DataFrame,
        split_date: date,
        date_col: str = "date",
        label_end_col: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        purge_start = split_date - timedelta(days=self.horizon_days)
        test_start = split_date + timedelta(days=self.embargo_days)
        test_df = df.filter(pl.col(date_col) >= test_start)
        if label_end_col is not None:
            train_df = df.filter(
                (pl.col(label_end_col) < purge_start) & (pl.col(date_col) < purge_start)
            )
        else:
            train_df = df.filter(pl.col(date_col) < purge_start)
        if train_df.is_empty() or test_df.is_empty():
            msg = "PurgedTimeSplit produced an empty train or test set."
            raise ValueError(msg)
        return train_df, test_df


@dataclass(slots=True)
class PurgedKFold:
    n_splits: int = 5
    horizon_days: int = 21
    embargo_days: int = 5
    min_train_size: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            msg = f"n_splits must be >= 2, got {self.n_splits}"
            raise ValueError(msg)
        if self.horizon_days < 0:
            msg = f"horizon_days must be >= 0, got {self.horizon_days}"
            raise ValueError(msg)
        if self.embargo_days < 0:
            msg = f"embargo_days must be >= 0, got {self.embargo_days}"
            raise ValueError(msg)

    def split(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
        label_end_col: str | None = None,
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
        dates = (
            df.select(date_col).unique().sort(date_col).get_column(date_col).to_list()
        )
        n_dates = len(dates)
        if n_dates < self.n_splits * 2:
            msg = f"Not enough unique dates ({n_dates}) for {self.n_splits} splits"
            raise ValueError(msg)

        test_size = n_dates // (self.n_splits + 1)
        for fold in range(1, self.n_splits + 1):
            start_idx = fold * test_size
            end_idx = (
                n_dates - 1 if fold == self.n_splits else start_idx + test_size - 1
            )
            test_start = dates[start_idx]
            test_end = dates[end_idx]
            purge_cutoff = test_start - timedelta(
                days=self.horizon_days + self.embargo_days
            )

            test_df = df.filter(
                (pl.col(date_col) >= test_start) & (pl.col(date_col) <= test_end)
            )
            if label_end_col is not None:
                train_df = df.filter(pl.col(label_end_col) < purge_cutoff)
            else:
                train_df = df.filter(pl.col(date_col) < purge_cutoff)

            if len(train_df) < self.min_train_size or test_df.is_empty():
                continue
            yield train_df, test_df


@dataclass(slots=True)
class WalkForwardValidation:
    n_windows: int = 5
    window_size: int = 252
    step_size: int = 63
    horizon_days: int = 21
    embargo_days: int = 5
    expanding: bool = True

    def __post_init__(self) -> None:
        if self.n_windows < 1:
            msg = f"n_windows must be >= 1, got {self.n_windows}"
            raise ValueError(msg)
        total_gap = self.horizon_days + self.embargo_days
        if total_gap >= self.window_size:
            msg = (
                f"horizon_days + embargo_days ({total_gap}) must be less than "
                f"window_size ({self.window_size})"
            )
            raise ValueError(msg)
        if total_gap >= self.step_size:
            msg = (
                f"horizon_days + embargo_days ({total_gap}) must be less than "
                f"step_size ({self.step_size})"
            )
            raise ValueError(msg)

    def split(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
        dates = (
            df.select(date_col).unique().sort(date_col).get_column(date_col).to_list()
        )
        for window_index in range(1, self.n_windows + 1):
            test_start_idx = window_index * self.step_size
            test_end_idx = min(test_start_idx + self.window_size, len(dates) - 1)
            if test_start_idx >= len(dates) or test_end_idx <= test_start_idx:
                break
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]
            purge_cutoff = test_start - timedelta(
                days=self.horizon_days + self.embargo_days
            )
            if self.expanding:
                train_df = df.filter(pl.col(date_col) < purge_cutoff)
            else:
                train_start_idx = max(0, test_start_idx - self.step_size)
                train_start = dates[train_start_idx]
                train_df = df.filter(
                    (pl.col(date_col) >= train_start)
                    & (pl.col(date_col) < purge_cutoff)
                )
            test_df = df.filter(
                (pl.col(date_col) >= test_start) & (pl.col(date_col) <= test_end)
            )
            if not train_df.is_empty() and not test_df.is_empty():
                yield train_df, test_df
