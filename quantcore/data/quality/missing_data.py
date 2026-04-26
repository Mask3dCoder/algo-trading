"""Missing data handling: gap detection, forward-fill, interpolation, EM imputation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ImputationReport:
    method: str
    n_filled: int
    filled_indices: list[int]
    max_gap_filled: int


def detect_gaps(series: pd.Series, freq: str = "B") -> list[pd.DatetimeIndex]:
    """Identify missing business days in a time-indexed series."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return []
    full_range = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    missing_dates = full_range.difference(series.index)
    return [missing_dates] if len(missing_dates) > 0 else []


def forward_fill(series: pd.Series, max_gap: int = 5) -> pd.Series:
    """Forward-fill missing values with a maximum gap limit.

    Only fills gaps of <= max_gap consecutive NaNs.
    """
    result = series.copy()

    # Find runs of NaNs
    is_nan = result.isna()
    groups = (is_nan != is_nan.shift()).cumsum()

    for group_id in is_nan[is_nan].groupby(groups).groups:
        indices = is_nan[is_nan].groupby(groups).get_group(group_id).index
        if len(indices) <= max_gap:
            # Fill with last valid value before the gap
            first_idx_pos = result.index.get_loc(indices[0])
            if first_idx_pos > 0:
                fill_value = result.iloc[first_idx_pos - 1]
                result.loc[indices] = fill_value

    return result


def interpolate_missing(series: pd.Series, method: str = "cubic") -> pd.Series:
    """Interpolate missing values using scipy interpolation methods."""
    return series.interpolate(method=method)


def em_impute(matrix: NDArray, max_iter: int = 100, tol: float = 1e-6) -> NDArray:
    """EM algorithm for multivariate missing data imputation.

    Iteratively estimates the mean and covariance, then fills missing values
    with their conditional expectation given the observed values.
    """
    X = np.asarray(matrix, dtype=float).copy()
    n, p = X.shape
    mask = np.isnan(X)

    if not np.any(mask):
        return X

    # Initialize with column means
    col_means = np.nanmean(X, axis=0)
    for j in range(p):
        X[mask[:, j], j] = col_means[j]

    mu = np.mean(X, axis=0)
    C = np.cov(X, rowvar=False) + 1e-6 * np.eye(p)

    for iteration in range(max_iter):
        X_old = X.copy()

        # E-step: fill missing with conditional expectation
        for i in range(n):
            obs_idx = np.where(~mask[i])[0]
            mis_idx = np.where(mask[i])[0]

            if len(mis_idx) == 0 or len(obs_idx) == 0:
                continue

            C_obs = C[np.ix_(obs_idx, obs_idx)]
            C_mis_obs = C[np.ix_(mis_idx, obs_idx)]

            try:
                C_obs_inv = np.linalg.inv(C_obs)
                X[i, mis_idx] = mu[mis_idx] + C_mis_obs @ C_obs_inv @ (X[i, obs_idx] - mu[obs_idx])
            except np.linalg.LinAlgError:
                pass

        # M-step: update mean and covariance
        mu = np.mean(X, axis=0)
        C = np.cov(X, rowvar=False) + 1e-6 * np.eye(p)

        # Check convergence
        change = np.max(np.abs(X - X_old))
        if change < tol:
            break

    return X


def impute(
    series: pd.Series,
    method: str = "ffill",
    max_gap: int = 5,
) -> tuple[pd.Series, ImputationReport]:
    """Apply imputation method and return report of what was filled."""
    n_before = series.isna().sum()
    filled_indices = list(np.where(series.isna())[0])

    if method == "ffill":
        result = forward_fill(series, max_gap)
    elif method in ("linear", "cubic", "spline"):
        result = interpolate_missing(series, method=method)
    else:
        result = forward_fill(series, max_gap)

    n_after = result.isna().sum()
    n_filled = int(n_before - n_after)

    return result, ImputationReport(
        method=method,
        n_filled=n_filled,
        filled_indices=filled_indices[:n_filled],
        max_gap_filled=max_gap,
    )
