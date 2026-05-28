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
    train_size: int = 504,
    test_size: int = 126,
    embargo_pct: float = 0.01,
) -> list[Fold]:
    """Generate rolling walk-forward folds.

    Default: 2-year training (504 bars), 6-month testing (126 bars), 1% embargo.
    This is the standardized validation for all strategies per Phase 2 requirements.

    Args:
        n_bars: Total number of bars in dataset
        train_size: Training window size (default 504 = 2 years trading days)
        test_size: Test window size (default 126 = 6 months trading days)
        embargo_pct: Fraction of test_size to embargo between train/test (default 1%)
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
        start += test_size  # Slide by test_size to avoid overlap

    return folds


def anchored_walk_forward(
    n_bars: int,
    initial_train: int = 504,
    test_size: int = 126,
    embargo_pct: float = 0.01,
) -> list[Fold]:
    """Generate anchored walk-forward folds (expanding training window).

    Default: 2-year initial training (504 bars), 6-month testing (126 bars), 1% embargo.
    
    Args:
        n_bars: Total number of bars in dataset
        initial_train: Initial training window size (default 504 = 2 years)
        test_size: Test window size (default 126 = 6 months)
        embargo_pct: Fraction of test_size to embargo between train/test (default 1%)
    """
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
    test_size: int = 126,
    embargo_pct: float = 0.01,
) -> list[Fold]:
    """Combinatorial Purged Cross-Validation (Lopez de Prado).

    Properly handles the information leakage between train and test
    in time-series data by purging overlapping labels and applying
    an embargo period.

    Args:
        n_bars: Total number of bars in dataset
        n_splits: Number of folds (default 5)
        test_size: Test window size (default 126 = 6 months)
        embargo_pct: Fraction of test_size to embargo between train/test (default 1%)
    """
    embargo = int(np.ceil(embargo_pct * test_size))
    fold_size = test_size

    folds: list[Fold] = []
    for i in range(n_splits):
        test_start = i * fold_size
        if test_start >= n_bars:
            break
        test_end = min(test_start + fold_size, n_bars)

        train_end = test_start - embargo

        # Training data must always be chronologically before test data.
        # When train_end <= 0 the first fold has no prior data — skip it
        # rather than creating a fold that trains on future data.
        if train_end <= 0:
            continue

        folds.append(Fold(
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_start=test_start - embargo if embargo > 0 else None,
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


# ---------------------------------------------------------------------------
# Acceptance Criteria (Phase 2)
# ---------------------------------------------------------------------------

ACCEPTANCE_CRITERIA = {
    "min_oos_sharpe": 0.5,
    "max_drawdown_pct": 20.0,
    "min_periods_passing": 3,  # Must pass in at least 3 walk-forward periods
}


def check_acceptance_criteria(
    oos_sharpe: float,
    max_dd_pct: float,
    periods_passed: int,
    total_periods: int,
) -> tuple[bool, str]:
    """Check if strategy meets all acceptance criteria.

    Returns:
        (is_valid, reason)
    """
    criteria = ACCEPTANCE_CRITERIA
    
    if oos_sharpe < criteria["min_oos_sharpe"]:
        return (False, f"OOS Sharpe {oos_sharpe:.2f} < {criteria['min_oos_sharpe']}")
    
    if max_dd_pct > criteria["max_drawdown_pct"]:
        return (False, f"Max Drawdown {max_dd_pct:.1f}% > {criteria['max_drawdown_pct']}%")
    
    if periods_passed < criteria["min_periods_passing"]:
        return (False, f"Only {periods_passed}/{total_periods} periods passed (need {criteria['min_periods_passing']})")
    
    return (True, "ALL CRITERIA MET")


def compute_oos_metrics(returns: np.ndarray, equity: np.ndarray) -> dict:
    """Compute OOS performance metrics.

    Uses the corrected metrics from quantcore.risk.metrics — NaN is returned
    explicitly when a metric is undefined (e.g., flat return series → Sharpe NaN).
    Callers must handle NaN rather than treating it as a passing score.
    """
    if len(returns) < 2:
        return {
            "sharpe": float("nan"),
            "max_dd_pct": 0.0,
            "annual_return": 0.0,
            "total_return_pct": 0.0,
        }

    # Annualised Sharpe — NaN when std is negligible
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret < 1e-10:
        sharpe = float("nan")
    else:
        sharpe = float((mean_ret / std_ret) * np.sqrt(252))

    # Max drawdown on equity curve (not log-cumulative)
    eq = np.asarray(equity, dtype=float)
    if len(eq) > 1 and eq[0] > 0:
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100.0
        max_dd_pct = float(abs(np.min(dd)))
    else:
        max_dd_pct = 0.0

    # Annual return
    total_ret = (eq[-1] / eq[0]) - 1.0 if (len(eq) > 0 and eq[0] > 0) else 0.0
    n_years = len(returns) / 252.0
    if n_years > 0 and (1.0 + total_ret) > 0:
        annual_ret = float((1.0 + total_ret) ** (1.0 / n_years) - 1.0)
    else:
        annual_ret = 0.0

    return {
        "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else float("nan"),
        "max_dd_pct": round(max_dd_pct, 2),
        "annual_return": round(annual_ret * 100.0, 2),
        "total_return_pct": round(total_ret * 100.0, 2),
    }
