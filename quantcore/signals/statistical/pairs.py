"""Pairs trading signals: cointegration scanner, Kalman hedge ratio, z-score, cost-adjusted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantcore.math_models.statistics.cointegration import engle_granger

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class CointPair:
    """A cointegrated pair result."""

    asset_y: int
    asset_x: int
    hedge_ratio: float
    half_life: float
    t_stat: float
    p_value: float
    spread_mean: float
    spread_std: float


# ---------------------------------------------------------------------------
# Cointegration Scanner
# ---------------------------------------------------------------------------


def scan_cointegrated_pairs(
    prices: NDArray,
    significance: float = 0.05,
    max_pairs: int = 50,
) -> list[CointPair]:
    """Scan all N*(N-1)/2 pairs for cointegration.

    Args:
        prices: (n_periods, n_assets) price matrix.
        significance: p-value threshold for Engle-Granger test.
        max_pairs: maximum number of pairs to return (sorted by t-stat).

    Returns:
        List of CointPair, sorted by absolute t-stat (strongest first).
    """
    n_periods, n_assets = prices.shape
    pairs: list[CointPair] = []

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            y = prices[:, i]
            x = prices[:, j]

            # Skip if too many NaN
            valid = ~(np.isnan(y) | np.isnan(x))
            if np.sum(valid) < 60:
                continue

            try:
                result = engle_granger(y[valid], x[valid])
            except Exception:
                continue

            if result.p_value > significance:
                continue

            # Estimate half-life from spread
            spread = y[valid] - result.hedge_ratio * x[valid]
            hl = _estimate_half_life(spread)

            pairs.append(CointPair(
                asset_y=i,
                asset_x=j,
                hedge_ratio=result.hedge_ratio,
                half_life=hl,
                t_stat=result.t_stat,
                p_value=result.p_value,
                spread_mean=np.mean(spread),
                spread_std=np.std(spread, ddof=1),
            ))

    # Sort by strongest cointegration (lowest p-value / highest |t|)
    pairs.sort(key=lambda p: abs(p.t_stat), reverse=True)
    return pairs[:max_pairs]


def _estimate_half_life(spread: NDArray) -> float:
    """Estimate OU half-life from spread via regression."""
    lag = spread[:-1]
    diff = np.diff(spread)

    if len(lag) < 10:
        return float("inf")

    x_mean = np.mean(lag)
    y_mean = np.mean(diff)
    ss_xy = np.sum((lag - x_mean) * (diff - y_mean))
    ss_xx = np.sum((lag - x_mean) ** 2)

    if ss_xx == 0:
        return float("inf")

    b = ss_xy / ss_xx
    theta = -b

    if theta > 0:
        return np.log(2.0) / theta
    return float("inf")


# ---------------------------------------------------------------------------
# Kalman Filter Hedge Ratio
# ---------------------------------------------------------------------------


def kalman_hedge_ratio(
    y: NDArray,
    x: NDArray,
    obs_noise: float = 1e-4,
    state_noise: float = 1e-6,
) -> tuple[NDArray, NDArray]:
    """Online Kalman Filter estimate of hedge ratio.

    State: [hedge_ratio, intercept]
    Observation: y_t = [x_t, 1] @ [beta, alpha] + noise

    Returns:
        (ratio_series, spread_series) — both same length as input.
    """
    n = len(y)
    ratio = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)

    if n < 2:
        return ratio, spread

    # Simple 1D Kalman filter for hedge ratio
    # State: beta (hedge ratio)
    beta_hat = 0.0
    P = 1.0  # State variance
    Q = state_noise  # Process noise
    R = obs_noise  # Observation noise

    for t in range(n):
        if np.isnan(y[t]) or np.isnan(x[t]) or x[t] == 0:
            continue

        # Predict
        beta_pred = beta_hat
        P_pred = P + Q

        # Update
        innovation = y[t] - beta_pred * x[t]
        S = x[t] ** 2 * P_pred + R
        if S > 0:
            K = P_pred * x[t] / S
            beta_hat = beta_pred + K * innovation
            P = (1.0 - K * x[t]) * P_pred

        ratio[t] = beta_hat
        spread[t] = y[t] - beta_hat * x[t]

    return ratio, spread


# ---------------------------------------------------------------------------
# Spread Z-Score
# ---------------------------------------------------------------------------


def spread_zscore(
    spread: NDArray,
    lookback: int = 60,
) -> NDArray:
    """Rolling z-score of spread.

    Positive = spread above mean (y expensive relative to x).
    Negative = spread below mean (y cheap relative to x).
    """
    n = len(spread)
    result = np.full(n, np.nan, dtype=float)

    if n < lookback:
        return result

    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(spread[i]):
            result[i] = (spread[i] - m) / s

    return result


# ---------------------------------------------------------------------------
# Pairs Signal
# ---------------------------------------------------------------------------


def pairs_signal(
    zscore: NDArray,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
) -> NDArray:
    """Generate pairs trading signal from z-score.

    Logic:
    - z > entry → short spread (signal = -1)
    - z < -entry → long spread (signal = +1)
    - |z| < exit → close position (signal = 0)
    - |z| > stop → stop loss (signal = 0)

    Returns signal in [-1, +1].
    """
    n = len(zscore)
    signal = np.zeros(n, dtype=float)
    position = 0.0

    for i in range(n):
        z = zscore[i]
        if np.isnan(z):
            signal[i] = position
            continue

        # Stop loss
        if abs(z) > stop:
            position = 0.0
        # Entry
        elif z > entry and position <= 0:
            position = -1.0
        elif z < -entry and position >= 0:
            position = 1.0
        # Exit
        elif abs(z) < exit:
            position = 0.0

        signal[i] = position

    return signal


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------


def halflife_size(
    half_life: float,
    base_size: float = 1.0,
    target_hl: float = 10.0,
    max_multiplier: float = 3.0,
) -> float:
    """Scale position size by mean-reversion speed.

    Shorter half-life → stronger mean reversion → larger position.
    Longer half-life → weaker → smaller position.
    """
    if half_life <= 0 or np.isinf(half_life):
        return base_size * 0.1  # Tiny position if no mean reversion

    multiplier = min(target_hl / half_life, max_multiplier)
    return base_size * multiplier


# ---------------------------------------------------------------------------
# Cost-Adjusted Threshold
# ---------------------------------------------------------------------------


def cost_adjusted_threshold(
    entry_z: float = 2.0,
    spread_cost_bps: float = 10.0,
    spread_std: float = 1.0,
    price: float = 100.0,
) -> float:
    """Widen entry threshold to account for transaction costs.

    The idea: don't enter a trade unless the expected profit
    exceeds the cost of the round-trip.
    """
    # Convert cost in bps to z-score terms
    cost_dollars = 2.0 * price * spread_cost_bps / 10000.0  # Round-trip
    cost_z = cost_dollars / spread_std if spread_std > 0 else 0.0
    return entry_z + cost_z
