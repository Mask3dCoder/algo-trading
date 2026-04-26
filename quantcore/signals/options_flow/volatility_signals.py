"""Options volatility signals: IVR, IVP, VRP, skew, term structure."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Implied Volatility Rank (IVR)
# ---------------------------------------------------------------------------


def iv_rank(
    current_iv: float,
    historical_iv: NDArray,
) -> float:
    """Implied Volatility Rank: percentile rank of current IV in historical range.

    0% = IV at lowest point in history (cheap vol).
    100% = IV at highest point (expensive vol).
    """
    valid = historical_iv[~np.isnan(historical_iv)]
    if len(valid) < 1:
        return 50.0
    return float(np.mean(valid <= current_iv) * 100.0)


def iv_rank_series(
    iv_series: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Rolling IV Rank series.

    Returns values in [0, 100]. Below 25 = cheap vol, above 75 = expensive.
    """
    n = len(iv_series)
    result = np.full(n, np.nan, dtype=float)

    for i in range(lookback - 1, n):
        window = iv_series[i - lookback + 1:i + 1]
        if not np.isnan(iv_series[i]):
            result[i] = iv_rank(iv_series[i], window)

    return result


# ---------------------------------------------------------------------------
# Implied Volatility Percentile (IVP)
# ---------------------------------------------------------------------------


def iv_percentile(
    current_iv: float,
    historical_iv: NDArray,
) -> float:
    """Implied Volatility Percentile: percentage of days IV was below current.

    Similar to IV Rank but uses percentage-below rather than rank in range.
    """
    valid = historical_iv[~np.isnan(historical_iv)]
    if len(valid) < 1:
        return 50.0
    return float(np.mean(valid < current_iv) * 100.0)


def iv_percentile_series(
    iv_series: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Rolling IV Percentile series."""
    n = len(iv_series)
    result = np.full(n, np.nan, dtype=float)

    for i in range(lookback - 1, n):
        window = iv_series[i - lookback + 1:i + 1]
        if not np.isnan(iv_series[i]):
            result[i] = iv_percentile(iv_series[i], window)

    return result


# ---------------------------------------------------------------------------
# Volatility Risk Premium (VRP)
# ---------------------------------------------------------------------------


def vol_risk_premium(
    implied_vol: NDArray,
    realized_vol: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Volatility Risk Premium: IV - RV (positive = premium, sell vol).

    Z-scored against rolling history. Positive signal = sell vol opportunity.
    """
    n = len(implied_vol)
    vrp = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isnan(implied_vol[i]) and not np.isnan(realized_vol[i]):
            vrp[i] = implied_vol[i] - realized_vol[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = vrp[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(vrp[i]):
            result[i] = np.clip((vrp[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# IV Skew Signal
# ---------------------------------------------------------------------------


def iv_skew_signal(
    iv_25delta_put: NDArray,
    iv_atm: NDArray,
    iv_25delta_call: NDArray | None = None,
    lookback: int = 252,
) -> NDArray:
    """IV skew: difference between OTM put IV and ATM IV.

    Higher skew = more demand for downside protection = fear.
    Z-scored against rolling history.

    If iv_25delta_call provided, computes (put_iv - call_iv) as full skew.
    """
    n = len(iv_25delta_put)
    skew = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if np.isnan(iv_25delta_put[i]) or np.isnan(iv_atm[i]):
            continue
        if iv_25delta_call is not None:
            if np.isnan(iv_25delta_call[i]):
                continue
            skew[i] = iv_25delta_put[i] - iv_25delta_call[i]
        else:
            skew[i] = iv_25delta_put[i] - iv_atm[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = skew[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(skew[i]):
            result[i] = np.clip((skew[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# IV Term Structure Signal
# ---------------------------------------------------------------------------


def iv_term_structure_signal(
    iv_short: NDArray,
    iv_long: NDArray,
    lookback: int = 252,
) -> NDArray:
    """IV term structure: short-dated IV vs long-dated IV.

    Contango: iv_long > iv_short (normal, negative signal — sell long-dated).
    Backwardation: iv_short > iv_long (fear, positive signal — buy long-dated).
    """
    n = len(iv_short)
    slope = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isnan(iv_short[i]) and not np.isnan(iv_long[i]) and iv_long[i] > 0:
            slope[i] = (iv_short[i] - iv_long[i]) / iv_long[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = slope[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(slope[i]):
            result[i] = np.clip((slope[i] - m) / s / 3.0, -1.0, 1.0)

    return result
