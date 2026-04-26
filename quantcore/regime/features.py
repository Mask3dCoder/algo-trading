"""Regime detection features: vol, correlation, trend, liquidity, breadth, credit."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def realized_volatility(
    returns: NDArray[np.floating],
    window: int = 21,
    annualize: bool = True,
) -> float:
    """Rolling realized volatility."""
    r = np.asarray(returns, dtype=float).flatten()
    if len(r) < max(window, 2):
        return float(np.std(r))
    r_window = r[-window:]
    vol = float(np.std(r_window, ddof=1))
    return vol * np.sqrt(252) if annualize else vol


def rolling_correlation(
    returns1: NDArray[np.floating],
    returns2: NDArray[np.floating],
    window: int = 63,
) -> float:
    """Rolling cross-asset correlation."""
    r1 = np.asarray(returns1, dtype=float).flatten()[-window:]
    r2 = np.asarray(returns2, dtype=float).flatten()[-window:]
    n = min(len(r1), len(r2))
    if n < 2:
        return 0.0
    return float(np.corrcoef(r1[:n], r2[:n])[0, 1])


def adx(
    high: NDArray[np.floating],
    low: NDArray[np.floating],
    close: NDArray[np.floating],
    period: int = 14,
) -> float:
    """Average Directional Index — trend strength indicator.

    ADX > 25: trending. ADX < 20: non-trending (mean-reverting).
    """
    h = np.asarray(high, dtype=float).flatten()
    l = np.asarray(low, dtype=float).flatten()
    c = np.asarray(close, dtype=float).flatten()

    n = min(len(h), len(l), len(c))
    if n <= period:
        return 20.0  # Neutral

    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )

    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Directional movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Smoothed directional indicators
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        sm_plus = np.sum(plus_dm[i - period + 1:i + 1])
        sm_minus = np.sum(minus_dm[i - period + 1:i + 1])
        sm_atr = np.sum(atr[i - period + 1:i + 1])
        if sm_atr > 0:
            plus_di[i] = 100.0 * sm_plus / sm_atr
            minus_di[i] = 100.0 * sm_minus / sm_atr

    # ADX
    dx = np.zeros(n)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom

    adx_vals = np.zeros(n)
    adx_vals[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, n):
        adx_vals[i] = (adx_vals[i - 1] * (period - 1) + dx[i]) / period

    return float(adx_vals[-1]) if n > 2 * period else 20.0


def hurst_exponent(
    returns: NDArray[np.floating],
    max_lag: int = 100,
) -> float:
    """Hurst exponent (R/S method).

    H > 0.5: trending (persistent)
    H ≈ 0.5: random walk
    H < 0.5: mean-reverting (anti-persistent)
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)
    if n < max_lag * 2:
        max_lag = max(10, n // 4)

    series = np.cumsum(r - np.mean(r))
    lags = np.arange(10, min(max_lag, n // 2))
    rs = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        n_segments = n // lag
        if n_segments < 2:
            continue
        segments = series[:n_segments * lag].reshape(n_segments, lag)
        # Range / std for each segment
        r_vals = np.max(segments, axis=1) - np.min(segments, axis=1)
        s_vals = np.std(segments, axis=1, ddof=1)
        valid = s_vals > 0
        if np.any(valid):
            rs[i] = np.mean(r_vals[valid] / s_vals[valid])

    valid_rs = rs > 0
    if np.sum(valid_rs) < 3:
        return 0.5

    # Log-log regression: log(RS) = H * log(lag) + c
    log_lags = np.log(lags[valid_rs])
    log_rs = np.log(rs[valid_rs])
    H, _ = np.polyfit(log_lags, log_rs, 1)

    return float(H)


def bid_ask_spread(
    bids: NDArray[np.floating],
    asks: NDArray[np.floating],
    prices: NDArray[np.floating] | None = None,
) -> float:
    """Average bid-ask spread as % of price."""
    b = np.asarray(bids, dtype=float).flatten()
    a = np.asarray(asks, dtype=float).flatten()
    n = min(len(b), len(a))

    if n == 0:
        return 0.0

    if prices is not None:
        p = np.asarray(prices, dtype=float).flatten()[:n]
        mid = p
    else:
        mid = (b[:n] + a[:n]) / 2.0

    spreads = (a[:n] - b[:n]) / np.maximum(mid, 1e-10)
    return float(np.mean(spreads))


def turnover_ratio(
    volume: NDArray[np.floating],
    shares_outstanding: float = 1.0,
    window: int = 21,
) -> float:
    """Rolling turnover ratio (volume / shares outstanding)."""
    v = np.asarray(volume, dtype=float).flatten()
    if len(v) < window:
        window = len(v)
    avg_vol = float(np.mean(v[-window:]))
    if shares_outstanding <= 0:
        return 0.0
    return avg_vol / shares_outstanding


def advance_decline_ratio(
    advancing: int,
    declining: int,
) -> float:
    """Advance/Decline ratio."""
    if declining == 0:
        return float("inf") if advancing > 0 else 1.0
    return advancing / declining


def credit_spread(
    high_yield_yield: float,
    investment_grade_yield: float,
) -> float:
    """Credit spread: HY yield - IG yield (or similar)."""
    return high_yield_yield - investment_grade_yield


def yield_curve_slope(
    long_rate: float,
    short_rate: float,
) -> float:
    """Yield curve slope: 10Y - 2Y."""
    return long_rate - short_rate


def yield_curve_curvature(
    rate_2y: float,
    rate_5y: float,
    rate_10y: float,
) -> float:
    """Yield curve curvature: 2×5Y - 2Y - 10Y."""
    return 2.0 * rate_5y - rate_2y - rate_10y


def compute_all_features(
    returns: NDArray[np.floating],
    high: NDArray[np.floating] | None = None,
    low: NDArray[np.floating] | None = None,
    close: NDArray[np.floating] | None = None,
    volume: NDArray[np.floating] | None = None,
    benchmark_returns: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """Compute all regime features as a dict for HMM input."""
    r = np.asarray(returns, dtype=float).flatten()

    features: dict[str, float] = {}
    features["realized_vol"] = realized_volatility(r)
    features["hurst"] = hurst_exponent(r)

    if high is not None and low is not None and close is not None:
        features["adx"] = adx(high, low, close)

    if benchmark_returns is not None:
        bm = np.asarray(benchmark_returns, dtype=float).flatten()
        features["market_correlation"] = rolling_correlation(r, bm)

    if volume is not None:
        features["turnover"] = turnover_ratio(volume)

    return features
