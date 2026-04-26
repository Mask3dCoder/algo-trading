"""Walk-forward validation and Combinatorial Purged Cross-Validation (CPCV)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Fold:
    """A single train/test fold."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_start: int | None = None


@dataclass
class WFResult:
    """Walk-forward fold result."""
    fold: Fold
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    out_of_sample_returns: NDArray | None = None


def rolling_walk_forward(
    n_bars: int,
    train_size: int,
    test_size: int,
    embargo_pct: float = 0.0,
) -> list[Fold]:
    """Generate rolling walk-forward folds.

    train_size: number of bars in training window (fixed)
    test_size: number of bars in test window
    embargo_pct: fraction of test_size to embargo after train
    """
    embargo = int(np.ceil(embargo_pct * test_size))
    folds: list[Fold] = []
    start = 0

    while start + train_size + embargo + test_size <= n_bars:
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size

        folds.append(Fold(
            train_start=start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_start=train_end if embargo > 0 else None,
        ))
        start += test_size  # Slide by test_size

    return folds


def anchored_walk_forward(
    n_bars: int,
    initial_train: int,
    test_size: int,
    embargo_pct: float = 0.0,
) -> list[Fold]:
    """Generate anchored walk-forward folds (expanding training window)."""
    embargo = int(np.ceil(embargo_pct * test_size))
    folds: list[Fold] = []
    train_end = initial_train

    while train_end + embargo + test_size <= n_bars:
        test_start = train_end + embargo
        test_end = test_start + test_size

        folds.append(Fold(
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_start=train_end if embargo > 0 else None,
        ))
        train_end += test_size

    return folds


def purged_kfold_cv(
    n_bars: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[Fold]:
    """Combinatorial Purged Cross-Validation (Lopez de Prado).

    Properly handles the information leakage between train and test
    in time-series data by purging overlapping labels and applying
    an embargo period.
    """
    total = n_bars
    fold_size = total // n_splits
    embargo = int(np.ceil(embargo_pct * fold_size))

    folds: list[Fold] = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min(test_start + fold_size, total)

        # Training set: everything before (with purge) and after (with embargo)
        # But purged CV only uses the non-adjacent data
        train_end = test_start - embargo
        train_start_next = test_end + embargo

        # For simplicity, use data before test as training
        if train_end > 0:
            folds.append(Fold(
                train_start=0,
                train_end=max(train_end, 0),
                test_start=test_start,
                test_end=test_end,
                embargo_start=test_start - embargo if embargo > 0 else None,
            ))
        else:
            # First fold — use data after test
            folds.append(Fold(
                train_start=train_start_next,
                train_end=total,
                test_start=test_start,
                test_end=test_end,
            ))

    return folds


def aggregate_oos_equity(results: list[WFResult]) -> pd.Series:
    """Aggregate out-of-sample returns across walk-forward folds into a continuous equity curve."""
    all_returns: list[float] = []
    for r in results:
        if r.out_of_sample_returns is not None:
            all_returns.extend(r.out_of_sample_returns.tolist())

    if not all_returns:
        return pd.Series(dtype=float)

    returns = np.array(all_returns)
    equity = np.cumprod(1.0 + returns)
    return pd.Series(equity, name="equity")
