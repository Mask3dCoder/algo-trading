"""Stale quote detection and cross-source data validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def detect_stale(
    prices: NDArray,
    timestamps: NDArray | None = None,
    max_unchanged: int = 5,
    is_trading_hours: NDArray | None = None,
) -> NDArray:
    """Detect stale quotes: consecutive identical prices during market hours.

    Returns boolean mask where True indicates stale.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    stale = np.zeros(n, dtype=bool)

    unchanged_count = 0
    for i in range(1, n):
        if prices[i] == prices[i - 1]:
            unchanged_count += 1
        else:
            unchanged_count = 0

        # Only flag as stale during trading hours
        if unchanged_count >= max_unchanged:
            if is_trading_hours is not None:
                if is_trading_hours[i]:
                    stale[i] = True
            else:
                stale[i] = True

    return stale


@dataclass
class DiscrepancyReport:
    """Report of cross-source price discrepancies."""
    n_compared: int
    n_discrepancies: int
    max_discrepancy_pct: float
    mean_discrepancy_pct: float
    discrepancy_indices: list[int]


def cross_validate(
    source_a: NDArray,
    source_b: NDArray,
    threshold: float = 0.01,
) -> DiscrepancyReport:
    """Compare two price series and flag discrepancies > threshold.

    threshold: maximum acceptable relative difference (1% = 0.01)
    """
    a = np.asarray(source_a, dtype=float)
    b = np.asarray(source_b, dtype=float)

    n = min(len(a), len(b))
    rel_diff = np.abs(a[:n] - b[:n]) / np.maximum(np.abs(a[:n]), 1e-10)
    mask = rel_diff > threshold

    return DiscrepancyReport(
        n_compared=n,
        n_discrepancies=int(np.sum(mask)),
        max_discrepancy_pct=float(np.max(rel_diff)) if n > 0 else 0.0,
        mean_discrepancy_pct=float(np.mean(rel_diff)) if n > 0 else 0.0,
        discrepancy_indices=list(np.where(mask)[0]),
    )


def majority_vote(sources: dict[str, NDArray]) -> NDArray:
    """Take median of 3+ sources as the "truth" value.

    Returns array of median prices.
    """
    if len(sources) < 3:
        # Use mean with 2 sources
        arrays = [np.asarray(v, dtype=float) for v in sources.values()]
        return np.mean(arrays, axis=0)

    # Align all sources to the same length
    min_len = min(len(v) for v in sources.values())
    aligned = np.column_stack([np.asarray(v, dtype=float)[:min_len] for v in sources.values()])

    return np.median(aligned, axis=1)
