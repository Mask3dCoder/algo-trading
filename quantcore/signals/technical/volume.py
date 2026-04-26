"""Volume signals: VWAP deviation, OBV, VPT, A/D line, CMF, MFI, unusual volume, OBV divergence."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.signals.technical.trend import sma

# ---------------------------------------------------------------------------
# VWAP Deviation
# ---------------------------------------------------------------------------


def vwap_deviation(
    close: NDArray,
    volume: NDArray,
    period: int = 20,
) -> NDArray:
    """Z-score of price deviation from VWAP.

    Positive = price above VWAP (expensive), negative = below (cheap).
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < period:
        return result

    cum_vol = np.cumsum(volume)
    cum_pv = np.cumsum(close * volume)

    for i in range(period - 1, n):
        v_start = 0 if i == period - 1 else cum_vol[i - period]
        pv_start = 0 if i == period - 1 else cum_pv[i - period]
        total_vol = cum_vol[i] - v_start
        if total_vol == 0:
            continue
        vwap_val = (cum_pv[i] - pv_start) / total_vol

        # Rolling std of (close - vwap) for z-score
        deviations = np.zeros(i - period + 2)
        for j in range(period - 1, i + 1):
            v_s = 0 if j == period - 1 else cum_vol[j - period]
            pv_s = 0 if j == period - 1 else cum_pv[j - period]
            tv = cum_vol[j] - v_s
            if tv > 0:
                deviations[j - period + 1] = close[j] - (cum_pv[j] - pv_s) / tv

        std = np.std(deviations, ddof=1)
        result[i] = (close[i] - vwap_val) / std if std > 0 else 0.0

    return np.clip(result / 3.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# On-Balance Volume (OBV)
# ---------------------------------------------------------------------------


def obv(close: NDArray, volume: NDArray) -> NDArray:
    """On-Balance Volume. Cumulative volume signed by price change direction."""
    n = len(close)
    result = np.zeros(n, dtype=float)
    if n < 2:
        return result

    for i in range(1, n):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]

    return result


# ---------------------------------------------------------------------------
# Volume-Price Trend (VPT)
# ---------------------------------------------------------------------------


def volume_price_trend(close: NDArray, volume: NDArray) -> NDArray:
    """Volume-Price Trend: cumulative volume * percent price change."""
    n = len(close)
    result = np.zeros(n, dtype=float)
    if n < 2:
        return result

    for i in range(1, n):
        pct_change = (close[i] - close[i - 1]) / close[i - 1] if close[i - 1] != 0 else 0.0
        result[i] = result[i - 1] + volume[i] * pct_change

    return result


# ---------------------------------------------------------------------------
# Accumulation/Distribution Line
# ---------------------------------------------------------------------------


def ad_line(high: NDArray, low: NDArray, close: NDArray, volume: NDArray) -> NDArray:
    """Accumulation/Distribution line.

    CLV = ((close - low) - (high - close)) / (high - low)
    AD = cumsum(CLV * volume)
    """
    n = len(close)
    result = np.zeros(n, dtype=float)
    if n < 1:
        return result

    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range > 0:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
        else:
            clv = 0.0
        result[i] = (clv * volume[i]) + (result[i - 1] if i > 0 else 0.0)

    return result


# ---------------------------------------------------------------------------
# Chaikin Money Flow (CMF)
# ---------------------------------------------------------------------------


def chaikin_money_flow(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    volume: NDArray,
    period: int = 21,
) -> NDArray:
    """Chaikin Money Flow: sum(CLV * volume) / sum(volume) over period.

    Returns values in [-1, +1]. Positive = accumulation, negative = distribution.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < period:
        return result

    clv = np.zeros(n, dtype=float)
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range > 0:
            clv[i] = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range

    clv_vol = clv * volume

    cum_vol = np.cumsum(volume)
    cum_clv_vol = np.cumsum(clv_vol)

    for i in range(period - 1, n):
        v_start = 0 if i == period - 1 else cum_vol[i - period]
        cv_start = 0 if i == period - 1 else cum_clv_vol[i - period]
        total_vol = cum_vol[i] - v_start
        total_clv_vol = cum_clv_vol[i] - cv_start
        result[i] = total_clv_vol / total_vol if total_vol > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Money Flow Index (MFI)
# ---------------------------------------------------------------------------


def mfi(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    volume: NDArray,
    period: int = 14,
) -> NDArray:
    """Money Flow Index (0-100). Volume-weighted RSI.

    Uses typical price and raw money flow.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return result

    tp = (high + low + close) / 3.0
    mf = tp * volume  # raw money flow

    for i in range(period, n):
        pos_mf = 0.0
        neg_mf = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos_mf += mf[j]
            elif tp[j] < tp[j - 1]:
                neg_mf += mf[j]
            else:
                # Tie: split half
                pos_mf += mf[j] * 0.5
                neg_mf += mf[j] * 0.5

        if neg_mf == 0:
            result[i] = 100.0
        else:
            mfr = pos_mf / neg_mf
            result[i] = 100.0 - 100.0 / (1.0 + mfr)

    return result


def mfi_signal(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    volume: NDArray,
    period: int = 14,
) -> NDArray:
    """MFI normalized to [-1, +1] signal."""
    mfi_vals = mfi(high, low, close, volume, period)
    return (mfi_vals - 50.0) / 50.0


# ---------------------------------------------------------------------------
# Unusual Volume Detection
# ---------------------------------------------------------------------------


def unusual_volume(
    volume: NDArray,
    lookback: int = 20,
    threshold: float = 2.0,
) -> NDArray:
    """Unusual volume: z-score of today's volume vs rolling average.

    Returns z-score values. Values above threshold indicate unusual volume.
    """
    n = len(volume)
    result = np.full(n, np.nan, dtype=float)
    if n < lookback:
        return result

    avg = sma(volume, lookback)

    for i in range(lookback - 1, n):
        window = volume[i - lookback + 1:i + 1]
        std = np.std(window, ddof=1)
        result[i] = (volume[i] - avg[i]) / std if std > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# OBV Divergence
# ---------------------------------------------------------------------------


def obv_divergence(
    close: NDArray,
    volume: NDArray,
    lookback: int = 20,
) -> NDArray:
    """Detect OBV divergence from price.

    Signal interpretation:
    +1 = bullish divergence (price down, OBV up → potential reversal up)
    -1 = bearish divergence (price up, OBV down → potential reversal down)
    0 = no divergence
    """
    n = len(close)
    signal = np.zeros(n, dtype=float)
    if n < lookback:
        return signal

    obv_vals = obv(close, volume)

    for i in range(lookback - 1, n):
        price_start = close[i - lookback + 1]
        price_end = close[i]
        obv_start = obv_vals[i - lookback + 1]
        obv_end = obv_vals[i]

        price_up = price_end > price_start
        price_down = price_end < price_start
        obv_up = obv_end > obv_start
        obv_down = obv_end < obv_start

        # Bearish divergence: price up but OBV down
        if price_up and obv_down:
            # Scale by magnitude of divergence
            price_pct = (price_end - price_start) / price_start if price_start > 0 else 0.0
            obv_pct = (obv_end - obv_start) / abs(obv_start) if obv_start != 0 else 0.0
            div_strength = min(abs(price_pct - obv_pct) * 10, 1.0)
            signal[i] = -div_strength

        # Bullish divergence: price down but OBV up
        elif price_down and obv_up:
            price_pct = (price_start - price_end) / price_start if price_start > 0 else 0.0
            obv_pct = (obv_end - obv_start) / abs(obv_start) if obv_start != 0 else 0.0
            div_strength = min(abs(price_pct + obv_pct) * 10, 1.0)
            signal[i] = div_strength

    return signal


# ---------------------------------------------------------------------------
# Ease of Movement
# ---------------------------------------------------------------------------


def ease_of_movement(
    high: NDArray,
    low: NDArray,
    volume: NDArray,
    period: int = 14,
) -> NDArray:
    """Ease of Movement: how easily price moves given volume.

    Positive = price rising on low volume (easy movement).
    Negative = price falling on low volume.
    Large absolute values with high volume = significant move.
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=float)
    if n < 2:
        return result

    emv = np.zeros(n, dtype=float)
    for i in range(1, n):
        dm = ((high[i] + low[i]) / 2.0) - ((high[i - 1] + low[i - 1]) / 2.0)
        br = (volume[i] / 10000.0) / (high[i] - low[i]) if (high[i] - low[i]) > 0 else 0.0
        emv[i] = dm / br if br > 0 else 0.0

    # Smooth with SMA
    result = sma(emv, period)
    # Normalize to [-1, +1] using rolling z-score
    z = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        window = result[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(result[i]):
            z[i] = np.clip((result[i] - m) / s / 3.0, -1.0, 1.0)

    return z
