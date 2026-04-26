"""ML evaluation: IC, Rank IC, decay curves, turnover, OOS degradation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------


def information_coefficient(
    predictions: NDArray,
    actuals: NDArray,
    method: str = "pearson",
) -> float:
    """Compute IC between predictions and actual returns.

    Args:
        predictions: model predictions.
        actuals: actual forward returns.
        method: "pearson" or "spearman" (rank IC).

    Returns:
        IC value (typically -0.1 to +0.1 for financial signals).
    """
    valid = ~(np.isnan(predictions) | np.isnan(actuals))
    p = predictions[valid]
    a = actuals[valid]

    if len(p) < 5:
        return 0.0

    if method == "spearman":
        p = _rank(p)
        a = _rank(a)

    ic = np.corrcoef(p, a)[0, 1]
    return float(ic) if not np.isnan(ic) else 0.0


def _rank(x: NDArray) -> NDArray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def rolling_ic(
    predictions: NDArray,
    actuals: NDArray,
    window: int = 252,
) -> NDArray:
    """Rolling IC time series."""
    n = len(predictions)
    result = np.full(n, np.nan, dtype=float)

    for i in range(window - 1, n):
        p_window = predictions[i - window + 1:i + 1]
        a_window = actuals[i - window + 1:i + 1]
        valid = ~(np.isnan(p_window) | np.isnan(a_window))
        if np.sum(valid) >= 20:
            ic = np.corrcoef(p_window[valid], a_window[valid])[0, 1]
            result[i] = ic if not np.isnan(ic) else 0.0

    return result


# ---------------------------------------------------------------------------
# IC Decay (Signal Autocorrelation)
# ---------------------------------------------------------------------------


def ic_decay(
    signal: NDArray,
    forward_returns: NDArray,
    max_lag: int = 20,
) -> NDArray:
    """IC as a function of holding period.

    Computes IC between signal and k-day forward returns for k=1..max_lag.
    Decaying IC suggests signal has short-lived alpha.
    """
    ics = np.zeros(max_lag, dtype=float)

    for k in range(1, max_lag + 1):
        # k-day forward returns
        fr = np.full(len(forward_returns), np.nan, dtype=float)
        for i in range(len(forward_returns) - k):
            if not np.isnan(forward_returns[i]) and not np.isnan(forward_returns[i + k]):
                fr[i] = np.prod(1 + forward_returns[i:i + k]) - 1.0

        ics[k - 1] = information_coefficient(signal, fr)

    return ics


# ---------------------------------------------------------------------------
# Turnover Analysis
# ---------------------------------------------------------------------------


def signal_turnover(
    signal: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Rolling turnover: fraction of positions that change direction.

    High turnover = frequent signal changes = higher transaction costs.
    """
    n = len(signal)
    changes = np.zeros(n, dtype=float)

    for i in range(1, n):
        if not np.isnan(signal[i]) and not np.isnan(signal[i - 1]):
            if signal[i] * signal[i - 1] < 0:  # Sign change
                changes[i] = 1.0
            elif np.abs(signal[i] - signal[i - 1]) > 0.5:
                changes[i] = 0.5

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        result[i] = np.mean(changes[i - lookback + 1:i + 1])

    return result


# ---------------------------------------------------------------------------
# OOS Degradation
# ---------------------------------------------------------------------------


def oos_degradation(
    ic_in_sample: float,
    ic_out_of_sample: float,
) -> dict:
    """Compute OOS IC degradation ratio and assessment.

    degradation = IS_IC / OOS_IC
    Ratio > 2 = likely overfitted.
    Ratio between 1 and 2 = expected degradation.
    """
    if abs(ic_out_of_sample) < 1e-10:
        ratio = float("inf")
    else:
        ratio = abs(ic_in_sample) / abs(ic_out_of_sample)

    return {
        "ic_is": ic_in_sample,
        "ic_oos": ic_out_of_sample,
        "degradation_ratio": ratio,
        "is_overfitted": ratio > 2.0 and ic_in_sample > 0,
        "keep_pct": min(1.0 / ratio, 1.0) if ratio > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Quintile Portfolio Analysis
# ---------------------------------------------------------------------------


def quintile_analysis(
    signal: NDArray,
    forward_returns: NDArray,
    n_quantiles: int = 5,
) -> dict:
    """Analyze returns by signal quintile.

    Monotonic spread from Q1 to Q5 indicates signal quality.
    """
    valid = ~(np.isnan(signal) | np.isnan(forward_returns))
    s = signal[valid]
    r = forward_returns[valid]

    if len(s) < n_quantiles * 3:
        return {"quintiles": [], "spread": 0.0, "is_monotonic": False}

    order = np.argsort(s)
    sorted_returns = r[order]
    quantile_size = len(sorted_returns) // n_quantiles

    quintile_means = []
    for q in range(n_quantiles):
        start = q * quantile_size
        end = start + quantile_size if q < n_quantiles - 1 else len(sorted_returns)
        quintile_means.append(float(np.mean(sorted_returns[start:end])))

    # Check monotonicity
    diffs = np.diff(quintile_means)
    is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    # Top minus bottom spread (annualized)
    spread = quintile_means[-1] - quintile_means[0]

    return {
        "quintiles": quintile_means,
        "spread": spread,
        "is_monotonic": is_monotonic,
        "spread_annualized": spread * 252,
    }
