"""Outlier detection: z-score, Grubbs, IQR, return-based methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t


@dataclass
class OutlierReport:
    """Combined outlier detection report."""
    zscore_mask: NDArray
    grubbs_indices: list[int]
    iqr_mask: NDArray
    return_mask: NDArray
    combined_mask: NDArray  # Union of all methods


def zscore_outliers(series: NDArray, threshold: float = 4.0) -> NDArray:
    """Flag points > threshold standard deviations from mean.

    Returns boolean mask.
    """
    x = np.asarray(series, dtype=float)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return np.zeros(len(x), dtype=bool)
    z = np.abs((x - mean) / std)
    return z > threshold


def grubbs_test(series: NDArray, alpha: float = 0.05) -> list[int]:
    """Iterative Grubbs test for single outliers.

    Returns list of indices flagged as outliers.
    """
    x = np.asarray(series, dtype=float).copy()
    n = len(x)
    outliers = []

    for _ in range(min(10, n // 3)):  # Max iterations
        n_current = len(x)
        if n_current < 3:
            break

        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            break

        # Find point with max absolute deviation
        deviations = np.abs(x - mean)
        max_idx = np.argmax(deviations)
        G = deviations[max_idx] / std

        # Critical value
        t_crit = student_t.ppf(1 - alpha / (2 * n_current), n_current - 2)
        G_crit = (n_current - 1) / np.sqrt(n_current) * np.sqrt(t_crit**2 / (n_current - 2 + t_crit**2))

        if G_crit < G:
            outliers.append(max_idx)
            x = np.delete(x, max_idx)
        else:
            break

    return outliers


def iqr_outliers(series: NDArray, factor: float = 1.5) -> NDArray:
    """Flag points outside Q1 - factor*IQR and Q3 + factor*IQR.

    Returns boolean mask.
    """
    x = np.asarray(series, dtype=float)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (x < lower) | (x > upper)


def return_outliers(prices: NDArray, max_sigma: float = 10.0) -> NDArray:
    """Flag daily returns exceeding max_sigma standard deviations.

    Uses median + MAD (median absolute deviation) as robust estimators
    to avoid self-contamination where extreme returns inflate the standard
    deviation and hide themselves from detection.

    Returns mask aligned with the price series (first value is always False).
    """
    prices = np.asarray(prices, dtype=float)
    returns = np.diff(np.log(prices))
    returns = np.insert(returns, 0, 0.0)

    valid = returns[1:]
    if len(valid) < 2:
        return np.zeros(len(prices), dtype=bool)

    # Robust scale: MAD * 1.4826 normalises to std for normally-distributed data
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    if mad == 0:
        return np.zeros(len(prices), dtype=bool)
    std_robust = mad * 1.4826

    mask = np.abs(returns - median) / std_robust > max_sigma
    mask[0] = False  # First return is artificial
    return mask


def detect_all(series: NDArray, threshold: float = 4.0, grubbs_alpha: float = 0.05, iqr_factor: float = 1.5) -> OutlierReport:
    """Run all outlier detection methods and return combined report."""
    series = np.asarray(series, dtype=float)

    z_mask = zscore_outliers(series, threshold)
    g_indices = grubbs_test(series, grubbs_alpha)
    iqr_mask = iqr_outliers(series, iqr_factor)

    # For return-based: assume series is prices
    try:
        r_mask = return_outliers(series)
    except Exception:
        r_mask = np.zeros(len(series), dtype=bool)

    # Mark Grubbs outliers in mask
    g_mask = np.zeros(len(series), dtype=bool)
    for idx in g_indices:
        if idx < len(g_mask):
            g_mask[idx] = True

    combined = z_mask | g_mask | iqr_mask | r_mask

    return OutlierReport(
        zscore_mask=z_mask,
        grubbs_indices=g_indices,
        iqr_mask=iqr_mask,
        return_mask=r_mask,
        combined_mask=combined,
    )
