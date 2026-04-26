"""Mean reversion signals: Bollinger Bands, RSI, Stochastic, CCI, OU half-life signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcore.signals.technical.trend import sma, ema

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger_bands(
    series: NDArray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Bollinger Bands.

    Returns (upper, mid, lower, pct_b, bandwidth).
    pct_b = (close - lower) / (upper - lower). 0=at lower, 1=at upper.
    bandwidth = (upper - lower) / mid.
    """
    mid = sma(series, period)
    s = pd.Series(series)
    rolling_std = s.rolling(period, min_periods=period).std(ddof=1).values
    upper = mid + num_std * rolling_std
    lower = mid - num_std * rolling_std
    band_width = upper - lower
    pct_b = np.where(band_width > 0, (series - lower) / band_width, 0.5)
    bandwidth = np.where(mid > 0, band_width / mid, 0.0)
    return upper, mid, lower, pct_b, bandwidth


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


@njit
def _rsi_core(series: np.ndarray, period: int) -> np.ndarray:
    """RSI inner Wilders smoothing loop (numba-accelerated)."""
    n = len(series)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result

    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)

    return result


def rsi(series: NDArray, period: int = 14) -> NDArray:
    """Relative Strength Index (0-100)."""
    return _rsi_core(np.asarray(series, dtype=np.float64), period)


def rsi_signal(series: NDArray, period: int = 14) -> NDArray:
    """RSI normalized to [-1, +1] signal.

    -1 = oversold (RSI near 0), +1 = overbought (RSI near 100).
    Centered at 50 → signal = 0.
    """
    rsi_vals = rsi(series, period)
    return (rsi_vals - 50.0) / 50.0


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------


def stochastic(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[NDArray, NDArray]:
    """Stochastic oscillator. Returns (%K, %D)."""
    n = len(close)
    k_val = np.full(n, np.nan, dtype=float)

    hh = pd.Series(high).rolling(k_period, min_periods=k_period).max().values
    ll = pd.Series(low).rolling(k_period, min_periods=k_period).min().values
    denom = hh - ll
    mask = denom != 0
    k_val[mask] = 100.0 * (close[mask] - ll[mask]) / denom[mask]
    k_val[~mask] = 50.0

    d_val = sma(k_val, d_period)
    return k_val, d_val


def stochastic_signal(high: NDArray, low: NDArray, close: NDArray, k_period: int = 14, d_period: int = 3) -> NDArray:
    """Stochastic signal normalized to [-1, +1]."""
    k_val, _ = stochastic(high, low, close, k_period, d_period)
    return (k_val - 50.0) / 50.0


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------


def williams_r(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
    """Williams %R (range: -100 to 0)."""
    n = len(close)
    result = np.full(n, np.nan, dtype=float)

    hh = pd.Series(high).rolling(period, min_periods=period).max().values
    ll = pd.Series(low).rolling(period, min_periods=period).min().values
    denom = hh - ll
    mask = denom != 0
    result[mask] = -100.0 * (hh[mask] - close[mask]) / denom[mask]
    result[~mask] = -50.0
    return result


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------


def cci(high: NDArray, low: NDArray, close: NDArray, period: int = 20) -> NDArray:
    """Commodity Channel Index."""
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    tp = (high + low + close) / 3.0

    s = pd.Series(tp)
    tp_sma = s.rolling(period, min_periods=period).mean()
    mean_dev = (s - tp_sma).abs().rolling(period, min_periods=period).mean()

    tp_sma_v = tp_sma.values
    mean_dev_v = mean_dev.values
    mask = mean_dev_v > 0
    result[mask] = (tp[mask] - tp_sma_v[mask]) / (0.015 * mean_dev_v[mask])
    result[~mask] = 0.0
    return result


# ---------------------------------------------------------------------------
# MA Z-score
# ---------------------------------------------------------------------------


def ma_zscore(series: NDArray, period: int = 20) -> NDArray:
    """Z-score of distance from moving average: (price - SMA) / rolling_std."""
    ma = sma(series, period)
    s = pd.Series(series)
    rolling_std = s.rolling(period, min_periods=period).std(ddof=1).values
    result = np.full(len(series), np.nan, dtype=float)
    mask = rolling_std > 0
    result[mask] = (series[mask] - ma[mask]) / rolling_std[mask]
    return np.clip(result / 3.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# OU Half-life Signal
# ---------------------------------------------------------------------------


def ou_halflife_signal(series: NDArray, lookback: int = 252) -> tuple[NDArray, NDArray]:
    """Fit Ornstein-Uhlenbeck process on rolling windows.

    Returns (halflife, zscore_signal).
    Signal: z-score of current deviation from OU mean, scaled by half-life.
    """
    n = len(series)
    halflife = np.full(n, np.nan, dtype=float)
    signal = np.full(n, np.nan, dtype=float)

    for i in range(lookback, n):
        window = series[i - lookback:i]
        # Fit OU via regression: X_t - X_{t-1} = a + b * X_{t-1}
        y = np.diff(window)
        x = window[:-1]
        if len(x) < 10:
            continue

        # OLS
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0:
            continue

        b = ss_xy / ss_xx
        a = y_mean - b * x_mean

        theta = -b
        if theta > 0:
            hl = np.log(2.0) / theta
            halflife[i] = min(hl, lookback)  # Cap at lookback

            # OU equilibrium
            ou_mean = -a / theta if theta > 0 else np.mean(window)
            ou_std = np.std(window - ou_mean, ddof=1) if theta > 0 else np.std(window, ddof=1)

            if ou_std > 0:
                z = (series[i] - ou_mean) / ou_std
                # Scale signal by half-life: shorter half-life = stronger mean reversion
                scale = min(252.0 / hl, 5.0) if hl > 0 else 1.0
                signal[i] = np.clip(z * scale / 5.0, -1.0, 1.0)

    return halflife, signal


# ---------------------------------------------------------------------------
# Warm-up: pre-compile Numba JIT functions at import time
# ---------------------------------------------------------------------------

def _warmup() -> None:
    """Pre-compile all njit functions so first real call is fast."""
    try:
        from numba import njit as _njit
    except ImportError:
        return
    tiny = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    _rsi_core(tiny, 2)

_warmup()
