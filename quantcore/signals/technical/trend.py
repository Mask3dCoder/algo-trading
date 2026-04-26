"""Trend signals: MA family, MACD, ADX, Supertrend, breakout detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import cast

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Numba-accelerated inner loops (shared utilities)
# ---------------------------------------------------------------------------


@njit
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range computation (vectorizable with numba)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return tr


@njit
def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing: result[i] = result[i-1] * (period-1)/period + values[i]."""
    n = len(values)
    result = np.full(n, np.nan)
    if n <= period:
        return result
    result[period] = np.mean(values[1:period + 1])
    for i in range(period + 1, n):
        result[i] = result[i - 1] - result[i - 1] / period + values[i]
    return result


@njit
def _ema_core(series: np.ndarray, period: int) -> np.ndarray:
    """EMA inner loop (numba-accelerated)."""
    n = len(series)
    result = np.full(n, np.nan)
    if period <= 0 or n < period:
        return result
    k = 2.0 / (period + 1.0)
    result[period - 1] = np.mean(series[:period])
    for i in range(period, n):
        result[i] = series[i] * k + result[i - 1] * (1.0 - k)
    return result


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------


def sma(series: NDArray, period: int) -> NDArray:
    """Simple moving average."""
    result = np.full_like(series, np.nan, dtype=float)
    if period <= 0 or len(series) < period:
        return result
    cumsum = np.cumsum(series)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return result


def ema(series: NDArray, period: int) -> NDArray:
    """Exponential moving average."""
    return _ema_core(np.asarray(series, dtype=np.float64), period)


def dema(series: NDArray, period: int) -> NDArray:
    """Double EMA: 2*EMA - EMA(EMA)."""
    e1 = ema(series, period)
    e2 = ema(cast(NDArray, np.where(np.isnan(e1), np.nan, e1)), period)
    return 2.0 * e1 - e2


def tema(series: NDArray, period: int) -> NDArray:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = ema(series, period)
    e2 = ema(cast(NDArray, np.where(np.isnan(e1), np.nan, e1)), period)
    e3 = ema(cast(NDArray, np.where(np.isnan(e2), np.nan, e2)), period)
    return 3.0 * e1 - 3.0 * e2 + e3


def hull_ma(series: NDArray, period: int) -> NDArray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    wma_half = _wma(series, half)
    wma_full = _wma(series, period)
    diff = 2.0 * wma_half - wma_full
    return _wma(cast(NDArray, np.where(np.isnan(diff), np.nan, diff)), sqrt_period)


@njit
def _kama_core(series: np.ndarray, er_period: int, fast_sc: float, slow_sc: float) -> np.ndarray:
    """Kaufman AMA inner loop (numba-accelerated)."""
    n = len(series)
    result = np.full(n, np.nan)
    if n < er_period + 1:
        return result
    result[er_period] = series[er_period]
    for i in range(er_period + 1, n):
        direction = abs(series[i] - series[i - er_period])
        volatility = np.sum(np.abs(np.diff(series[i - er_period:i + 1])))
        er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (series[i] - result[i - 1])
    return result


def kaufman_ama(
    series: NDArray,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> NDArray:
    """Kaufman Adaptive Moving Average."""
    return _kama_core(
        np.asarray(series, dtype=np.float64),
        er_period,
        2.0 / (fast_period + 1.0),
        2.0 / (slow_period + 1.0),
    )


def _wma(series: NDArray, period: int) -> NDArray:
    """Weighted moving average (linear weights)."""
    result = np.full_like(series, np.nan, dtype=float)
    if period <= 0 or len(series) < period:
        return result
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    convolved = np.convolve(series, weights[::-1], mode="valid")
    result[period - 1:] = convolved
    return result


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    series: NDArray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[NDArray, NDArray, NDArray]:
    """MACD: (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(cast(NDArray, np.where(np.isnan(macd_line), np.nan, macd_line)), signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


@njit
def _adx_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple:
    """ADX inner computation (numba-accelerated)."""
    n = len(high)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx_val = np.full(n, np.nan)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0

    atr_s = _wilder_smooth(tr, period)
    plus_dm_s = _wilder_smooth(plus_dm, period)
    minus_dm_s = _wilder_smooth(minus_dm, period)

    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100.0 * plus_dm_s[i] / atr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / atr_s[i]

    dx = np.zeros(n)
    for i in range(period, n):
        sum_di = plus_di[i] + minus_di[i]
        dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / sum_di if sum_di > 0 else 0.0

    adx_val[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, n):
        adx_val[i] = (adx_val[i - 1] * (period - 1) + dx[i]) / period

    return adx_val, plus_di, minus_di


def adx(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    period: int = 14,
) -> tuple[NDArray, NDArray, NDArray]:
    """Average Directional Index with +DI and -DI."""
    return _adx_core(
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
        period,
    )


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------


@njit
def _supertrend_core(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    atr_val: np.ndarray, period: int, multiplier: float,
) -> tuple:
    """Supertrend inner stateful loop (numba-accelerated)."""
    n = len(close)
    st = np.full(n, np.nan)
    direction = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)

    for i in range(period, n):
        if np.isnan(atr_val[i]):
            continue
        hl2 = (high[i] + low[i]) / 2.0
        upper_band[i] = hl2 + multiplier * atr_val[i]
        lower_band[i] = hl2 - multiplier * atr_val[i]

        if i == period:
            st[i] = upper_band[i] if close[i] <= upper_band[i] else lower_band[i]
            direction[i] = -1.0 if close[i] <= upper_band[i] else 1.0
        else:
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = max(lower_band[i], lower_band[i - 1])

            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = min(upper_band[i], upper_band[i - 1])

            if direction[i - 1] == 1.0:
                if close[i] < lower_band[i]:
                    direction[i] = -1.0
                    st[i] = upper_band[i]
                else:
                    direction[i] = 1.0
                    st[i] = lower_band[i]
            else:
                if close[i] > upper_band[i]:
                    direction[i] = 1.0
                    st[i] = lower_band[i]
                else:
                    direction[i] = -1.0
                    st[i] = upper_band[i]

    return st, direction


def supertrend(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[NDArray, NDArray]:
    """Supertrend indicator. Returns (supertrend_line, direction: +1=up, -1=down)."""
    atr_val = _atr_raw(high, low, close, period)
    return _supertrend_core(
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
        np.asarray(atr_val, dtype=np.float64),
        period, multiplier,
    )


@njit
def _atr_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR calculation with Wilder smoothing (numba-accelerated)."""
    n = len(close)
    result = np.full(n, np.nan)
    tr = _true_range(high, low, close)
    if n < period:
        return result
    result[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def _atr_raw(high: NDArray, low: NDArray, close: NDArray, period: int) -> NDArray:
    """ATR calculation (Wilder smoothing)."""
    return _atr_core(np.asarray(high, dtype=np.float64),
                     np.asarray(low, dtype=np.float64),
                     np.asarray(close, dtype=np.float64), period)


# ---------------------------------------------------------------------------
# Linear Regression Signal
# ---------------------------------------------------------------------------


def linear_regression_signal(series: NDArray, period: int = 20) -> tuple[NDArray, NDArray]:
    """Rolling linear regression: slope and R-squared as trend strength.

    Returns (slope_zscore, r_squared).
    """
    n = len(series)
    slope = np.full(n, np.nan, dtype=float)
    r_sq = np.full(n, np.nan, dtype=float)

    for i in range(period - 1, n):
        y = series[i - period + 1:i + 1]
        x = np.arange(period, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_yy = np.sum((y - y_mean) ** 2)

        if ss_xx > 0:
            slope[i] = ss_xy / ss_xx
            if ss_yy > 0:
                r_sq[i] = (ss_xy ** 2) / (ss_xx * ss_yy)

    # Z-score normalize slope
    valid = ~np.isnan(slope)
    if np.any(valid):
        mean_s = np.nanmean(slope)
        std_s = np.nanstd(slope)
        if std_s > 0:
            slope_z = np.where(valid, (slope - mean_s) / std_s, np.nan)
            slope_z = np.clip(slope_z, -3.0, 3.0) / 3.0  # Normalize to [-1, 1]
        else:
            slope_z = np.zeros_like(slope)
    else:
        slope_z = slope

    return slope_z, r_sq


# ---------------------------------------------------------------------------
# Breakout Detection
# ---------------------------------------------------------------------------


def breakout_signal(high: NDArray, low: NDArray, close: NDArray, period: int = 20) -> NDArray:
    """Donchian channel breakout with z-score normalized signal.

    +1 = new high breakout, -1 = new low breakout, 0 = inside channel.
    """
    n = len(close)
    signal = np.zeros(n, dtype=float)

    channel_high = pd.Series(high).rolling(period, min_periods=period).max().values
    channel_low = pd.Series(low).rolling(period, min_periods=period).min().values
    channel_range = channel_high - channel_low

    up_break = close > channel_high
    down_break = close < channel_low

    signal[up_break] = 1.0
    nonzero_range = channel_range > 0
    up_scale = up_break & nonzero_range
    signal[up_scale] = np.minimum((close[up_scale] - channel_high[up_scale]) / channel_range[up_scale] * 5.0, 1.0)
    down_scale = down_break & nonzero_range
    signal[down_scale] = np.maximum((close[down_scale] - channel_low[down_scale]) / channel_range[down_scale] * 5.0, -1.0)

    return signal


# ---------------------------------------------------------------------------
# Convenience: Z-score normalizer
# ---------------------------------------------------------------------------


def zscore_normalize(signal: NDArray, lookback: int = 252) -> NDArray:
    """Rolling z-score normalization to [-1, +1] range."""
    s = pd.Series(signal)
    rolling_mean = s.rolling(lookback, min_periods=10).mean()
    rolling_std = s.rolling(lookback, min_periods=10).std(ddof=1)
    result = np.full(len(signal), np.nan, dtype=float)
    mask = rolling_std.values > 0
    result[mask] = (s.values[mask] - rolling_mean.values[mask]) / rolling_std.values[mask]
    return np.clip(result / 3.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Warm-up: pre-compile Numba JIT functions with tiny arrays at import time
# ---------------------------------------------------------------------------

def _warmup() -> None:
    """Pre-compile all njit functions so first real call is fast."""
    if not _NUMBA_AVAILABLE:
        return
    tiny = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    _true_range(tiny, tiny, tiny)
    _wilder_smooth(tiny, 2)
    _ema_core(tiny, 2)
    _atr_core(tiny, tiny, tiny, 2)
    _adx_core(tiny, tiny, tiny, 2)
    _kama_core(tiny, 2, 0.5, 0.1)
    _supertrend_core(tiny, tiny, tiny, np.full(5, 0.1), 2, 1.0)

_warmup()
