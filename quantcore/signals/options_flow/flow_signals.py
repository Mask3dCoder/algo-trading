"""Options flow signals: unusual options activity, sweeps, put/call ratio."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.signals.technical.trend import sma

# ---------------------------------------------------------------------------
# Unusual Options Activity (UOA)
# ---------------------------------------------------------------------------


def unusual_options_activity(
    volume: NDArray,
    open_interest: NDArray,
    avg_volume_lookback: int = 20,
    voi_threshold: float = 3.0,
) -> NDArray:
    """Detect unusual options activity via volume vs open interest ratio.

    VOI = volume / open_interest. When this ratio is unusually high
    relative to historical, it signals informed flow.

    Returns z-scored VOI signal.
    """
    n = len(volume)
    voi = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isnan(volume[i]) and not np.isnan(open_interest[i]) and open_interest[i] > 0:
            voi[i] = volume[i] / open_interest[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(avg_volume_lookback - 1, n):
        window = voi[i - avg_volume_lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(voi[i]):
            result[i] = np.clip((voi[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Volume-Weighted Delta (simplified)
# ---------------------------------------------------------------------------


def volume_weighted_delta(
    call_volume: NDArray,
    put_volume: NDArray,
    lookback: int = 20,
) -> NDArray:
    """Volume-weighted delta: net call vs put volume, normalized.

    Positive = more call volume (bullish flow).
    Negative = more put volume (bearish flow).
    """
    n = len(call_volume)
    total = call_volume + put_volume
    delta_pct = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if total[i] > 0 and not np.isnan(call_volume[i]) and not np.isnan(put_volume[i]):
            delta_pct[i] = (call_volume[i] - put_volume[i]) / total[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = delta_pct[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(delta_pct[i]):
            result[i] = np.clip((delta_pct[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Put/Call Ratio Signal
# ---------------------------------------------------------------------------


def put_call_ratio(
    put_volume: NDArray,
    call_volume: NDArray,
) -> NDArray:
    """Raw put/call ratio. Higher = more bearish sentiment."""
    n = len(put_volume)
    result = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not np.isnan(put_volume[i]) and not np.isnan(call_volume[i]) and call_volume[i] > 0:
            result[i] = put_volume[i] / call_volume[i]
    return result


def put_call_ratio_signal(
    put_volume: NDArray,
    call_volume: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Put/call ratio z-scored against rolling history.

    Contrarian interpretation: very high P/C (extreme fear) can be bullish.
    """
    pcr = put_call_ratio(put_volume, call_volume)
    n = len(pcr)
    result = np.full(n, np.nan, dtype=float)

    for i in range(lookback - 1, n):
        window = pcr[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(pcr[i]):
            # Contrarian: high PCR (fear) = negative signal → invert
            z = -(pcr[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Sweep Detection
# ---------------------------------------------------------------------------


def detect_sweeps(
    trade_size: NDArray,
    avg_trade_size: NDArray | None = None,
    size_threshold: float = 5.0,
    lookback: int = 20,
) -> NDArray:
    """Detect sweep orders: trades with unusually large size.

    A sweep = single order routed across multiple exchanges simultaneously.

    Args:
        trade_size: individual trade sizes.
        avg_trade_size: rolling average trade size (computed if None).
        size_threshold: number of standard deviations above average.
        lookback: rolling lookback for average.

    Returns:
        Binary mask: True where trade is a sweep.
    """
    n = len(trade_size)
    result = np.zeros(n, dtype=bool)

    if avg_trade_size is None:
        avg_trade_size = sma(trade_size, lookback)

    for i in range(n):
        if np.isnan(trade_size[i]) or np.isnan(avg_trade_size[i]):
            continue
        if avg_trade_size[i] > 0:
            ratio = trade_size[i] / avg_trade_size[i]
            result[i] = ratio >= size_threshold

    return result


# ---------------------------------------------------------------------------
# Options Flow Composite Signal
# ---------------------------------------------------------------------------


def options_flow_composite(
    voi_signal: NDArray,
    delta_signal: NDArray,
    pcr_signal: NDArray,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> NDArray:
    """Combine options flow signals into a composite.

    Weighted average of VOI, volume delta, and P/C ratio signals.
    """
    n = len(voi_signal)
    result = np.full(n, np.nan, dtype=float)

    for i in range(n):
        vals = [voi_signal[i], delta_signal[i], pcr_signal[i]]
        ws = list(weights)
        valid = ~np.isnan(vals)

        if not np.any(valid):
            continue

        vals_v = [v for v, ok in zip(vals, valid) if ok]
        ws_v = [w for w, ok in zip(ws, valid) if ok]
        total_w = sum(ws_v)

        if total_w > 0:
            result[i] = np.clip(sum(v * w for v, w in zip(vals_v, ws_v)) / total_w, -1.0, 1.0)

    return result
