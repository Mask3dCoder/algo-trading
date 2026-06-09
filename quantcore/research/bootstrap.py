"""Block bootstrap confidence intervals for the Sharpe ratio.

Uses a stationary block bootstrap to preserve serial correlation in
daily returns while re-sampling.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    returns: NDArray[np.floating],
    block_size: int | None = None,
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a block-bootstrap confidence interval for the Sharpe ratio.

    Parameters
    ----------
    returns :
        Daily (or higher-frequency) return series.
    block_size :
        Block size for the bootstrap. If *None*, an optimal value is
        derived from ``max(5, int(n ** (1/3)))``.
    n_boot :
        Number of bootstrap replications (default 2,000).
    ci :
        Confidence level (default 0.90 for a 90% CI).
    seed :
        RNG seed for full reproducibility.

    Returns
    -------
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)
    if n == 0:
        return 0.0, 0.0
    if n < 10:
        sr = _sharpe(r)
        return sr, sr

    if block_size is None:
        block_size = max(5, int(n ** (1 / 3)))

    rng = np.random.default_rng(seed)
    boot_sharpes: list[float] = []

    for _ in range(n_boot):
        # Circular block bootstrap (stationary version)
        idx = 0
        resampled: list[float] = []
        while len(resampled) < n:
            # Geometric distribution for block lengths (Liu & Singh, 1992)
            block_len = rng.geometric(1.0 / block_size)
            start = rng.integers(0, n)
            for j in range(block_len):
                resampled.append(float(r[(start + j) % n]))
        resampled_arr = np.array(resampled[:n])
        boot_sharpes.append(_sharpe(resampled_arr))

    boot_arr = np.array(boot_sharpes)
    alpha = 0.5 * (1.0 - ci)
    lower = float(np.quantile(boot_arr, alpha))
    upper = float(np.quantile(boot_arr, 1.0 - alpha))
    return lower, upper


def _sharpe(returns: NDArray[np.floating]) -> float:
    """Annualised Sharpe ratio (daily returns)."""
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return mean / std * np.sqrt(252)
