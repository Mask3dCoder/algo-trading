"""Momentum signals: cross-sectional, time-series, dual, idiosyncratic, 52w-high, vol-scaled."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.core.constants import TRADING_DAYS_PER_YEAR
from quantcore.signals.technical.trend import sma


# ---------------------------------------------------------------------------
# Cross-Sectional Momentum
# ---------------------------------------------------------------------------


def cross_sectional_momentum(
    returns_matrix: NDArray,
    lookback: int = 252,
    skip: int = 21,
) -> NDArray:
    """Cross-sectional momentum: percentile rank by past return.

    For each asset, compute the (lookback - skip) day return, then
    rank cross-sectionally. Skip most recent `skip` days to avoid
    short-term reversal (Jegadeesh & Titman 1993).

    Args:
        returns_matrix: (n_periods, n_assets) matrix of returns.
        lookback: total lookback window (default 252 = 12 months).
        skip: recent days to skip (default 21 = 1 month).

    Returns:
        (n_periods, n_assets) matrix of cross-sectional ranks in [-1, +1].
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan, dtype=float)

    for t in range(lookback, n_periods):
        # Cumulative return from (t - lookback) to (t - skip)
        start_idx = t - lookback
        end_idx = t - skip
        if end_idx <= start_idx:
            continue

        cum_returns = np.ones(n_assets)
        for i in range(start_idx, end_idx):
            cum_returns *= (1.0 + returns_matrix[i, :])
        cum_returns -= 1.0  # Convert to simple return

        # Rank and normalize to [-1, +1]
        valid = ~np.isnan(cum_returns)
        if np.sum(valid) < 2:
            continue

        ranks = np.full(n_assets, np.nan, dtype=float)
        sorted_indices = np.argsort(cum_returns[valid])
        n_valid = np.sum(valid)
        for rank, idx in enumerate(sorted_indices):
            original_idx = np.where(valid)[0][idx]
            ranks[original_idx] = 2.0 * rank / (n_valid - 1) - 1.0

        result[t, :] = ranks

    return result


# ---------------------------------------------------------------------------
# Time-Series Momentum
# ---------------------------------------------------------------------------


def time_series_momentum(
    returns: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Time-series momentum: sign of cumulative return over lookback.

    +1 if cumulative return > 0 (trend up), -1 if < 0 (trend down).
    Scaled by strength of the trend.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=float)
    if n < lookback:
        return result

    for i in range(lookback, n):
        window = returns[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < lookback // 2:
            continue
        cum_return = np.prod(1.0 + valid) - 1.0
        # Scale magnitude by z-score of cumulative return
        # Use rolling history
        if i >= 2 * lookback:
            past_cum = np.zeros(i - 2 * lookback + 1)
            for j in range(2 * lookback, i + 1):
                w = returns[j - lookback:j]
                v = w[~np.isnan(w)]
                if len(v) >= lookback // 2:
                    past_cum[j - 2 * lookback] = np.prod(1.0 + v) - 1.0
            m = np.mean(past_cum[past_cum != 0]) if np.any(past_cum != 0) else 0.0
            s = np.std(past_cum[past_cum != 0]) if np.any(past_cum != 0) else 1.0
            if s > 0:
                result[i] = np.clip(cum_return / (3.0 * s), -1.0, 1.0)
            else:
                result[i] = np.sign(cum_return)
        else:
            result[i] = np.sign(cum_return) * min(abs(cum_return), 1.0)

    return result


# ---------------------------------------------------------------------------
# Dual Momentum
# ---------------------------------------------------------------------------


def dual_momentum(
    returns: NDArray,
    benchmark_returns: NDArray,
    lookback: int = 252,
) -> NDArray:
    """Dual momentum: combines absolute (time-series) and relative (vs benchmark).

    +1 = both absolute and relative momentum positive (go long).
    -1 = absolute momentum negative (go to cash/short).
    Intermediate values based on relative strength.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=float)
    if n < lookback:
        return result

    for i in range(lookback, n):
        # Absolute momentum
        window_r = returns[i - lookback:i]
        window_b = benchmark_returns[i - lookback:i]
        valid_r = window_r[~np.isnan(window_r)]
        valid_b = window_b[~np.isnan(window_b)]

        if len(valid_r) < lookback // 2 or len(valid_b) < lookback // 2:
            continue

        cum_asset = np.prod(1.0 + valid_r) - 1.0
        cum_bench = np.prod(1.0 + valid_b) - 1.0

        # Absolute: is the asset up?
        abs_signal = np.sign(cum_asset)

        # Relative: is asset beating benchmark?
        rel_excess = cum_asset - cum_bench
        if abs(rel_excess) > 0:
            rel_signal = np.clip(rel_excess * 5.0, -1.0, 1.0)  # Scale
        else:
            rel_signal = 0.0

        # Combine: absolute gates, relative modulates
        if abs_signal <= 0:
            result[i] = -1.0  # Absolute negative → defensive
        else:
            result[i] = max(rel_signal, 0.0)  # Only take positive relative

    return result


# ---------------------------------------------------------------------------
# Idiosyncratic Momentum
# ---------------------------------------------------------------------------


def idiosyncratic_momentum(
    returns: NDArray,
    factor_returns: NDArray,
    lookback: int = 252,
    skip: int = 21,
) -> NDArray:
    """Idiosyncratic (residual) momentum: momentum of alpha after factor adjustment.

    Regresses returns on factor returns, then computes momentum of residuals.
    Avoids factor-driven moves that may reverse.
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=float)

    if factor_returns.ndim == 1:
        factor_returns = factor_returns.reshape(-1, 1)

    n_factors = factor_returns.shape[1]
    if n < lookback + skip:
        return result

    for i in range(lookback + skip, n):
        # Estimate factor loadings from lookback window (excluding skip)
        est_end = i - skip
        est_start = est_end - lookback
        if est_start < 0:
            continue

        y = returns[est_start:est_end]
        X = factor_returns[est_start:est_end]

        # Remove NaN rows
        valid = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        y_clean = y[valid]
        X_clean = X[valid]

        if len(y_clean) < 30:
            continue

        # OLS: y = X @ beta + epsilon
        # Add intercept
        X_aug = np.column_stack([np.ones(len(X_clean)), X_clean])

        try:
            beta = np.linalg.lstsq(X_aug, y_clean, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # Compute residuals over the momentum measurement window
        mom_start = est_start
        mom_end = est_end
        y_mom = returns[mom_start:mom_end]
        X_mom = factor_returns[mom_start:mom_end]

        residuals = np.full(mom_end - mom_start, np.nan)
        for j in range(len(y_mom)):
            if not np.isnan(y_mom[j]) and not np.any(np.isnan(X_mom[j])):
                x_aug = np.concatenate([[1.0], X_mom[j]])
                predicted = x_aug @ beta
                residuals[j] = y_mom[j] - predicted

        valid_res = residuals[~np.isnan(residuals)]
        if len(valid_res) < 20:
            continue

        residual_cum = np.sum(valid_res)
        # Z-score against rolling residual momentum distribution
        std_res = np.std(valid_res, ddof=1)
        if std_res > 0:
            result[i] = np.clip(residual_cum / (std_res * np.sqrt(len(valid_res)) * 3.0), -1.0, 1.0)
        else:
            result[i] = np.sign(residual_cum)

    return result


# ---------------------------------------------------------------------------
# 52-Week High Proximity
# ---------------------------------------------------------------------------


def high_52w_proximity(
    close: NDArray,
    period: int = 252,
) -> NDArray:
    """52-week high proximity: close / rolling max close.

    1.0 = at 52w high (potential breakout).
    Lower values = further from high (weakness).

    Converted to signal: proximity mapped to [-1, +1].
    """
    n = len(close)
    proximity = np.full(n, np.nan, dtype=float)

    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < period // 2:
            continue
        high_val = np.max(valid)
        if high_val > 0:
            proximity[i] = close[i] / high_val

    # Z-score normalize the proximity
    result = np.full(n, np.nan, dtype=float)
    lookback = 252
    for i in range(lookback - 1, n):
        window = proximity[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(proximity[i]):
            z = (proximity[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Volatility-Scaled Momentum
# ---------------------------------------------------------------------------


def vol_scaled_momentum(
    returns: NDArray,
    vol_lookback: int = 63,
    mom_lookback: int = 252,
) -> NDArray:
    """Volatility-scaled momentum: cumulative return / realized vol.

    Higher signal = strong trend with low vol (high Sharpe-like momentum).
    """
    n = len(returns)
    result = np.full(n, np.nan, dtype=float)
    if n < max(vol_lookback, mom_lookback):
        return result

    for i in range(mom_lookback, n):
        mom_window = returns[i - mom_lookback:i]
        vol_window = returns[i - vol_lookback:i]

        valid_mom = mom_window[~np.isnan(mom_window)]
        valid_vol = vol_window[~np.isnan(vol_window)]

        if len(valid_mom) < mom_lookback // 2 or len(valid_vol) < vol_lookback // 2:
            continue

        cum_return = np.prod(1.0 + valid_mom) - 1.0
        ann_vol = np.std(valid_vol, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

        if ann_vol > 0:
            # Return per unit of vol
            scaled = cum_return / ann_vol
            # Normalize: compare to rolling history
            if i >= 2 * mom_lookback:
                past_scaled = []
                for j in range(mom_lookback, i, mom_lookback // 4):
                    mw = returns[j - mom_lookback:j]
                    vw = returns[j - vol_lookback:j]
                    mv = mw[~np.isnan(mw)]
                    vv = vw[~np.isnan(vw)]
                    if len(mv) >= mom_lookback // 2 and len(vv) >= vol_lookback // 2:
                        cr = np.prod(1.0 + mv) - 1.0
                        av = np.std(vv, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                        if av > 0:
                            past_scaled.append(cr / av)

                if len(past_scaled) >= 10:
                    m = np.mean(past_scaled)
                    s = np.std(past_scaled, ddof=1)
                    if s > 0:
                        result[i] = np.clip((scaled - m) / s / 3.0, -1.0, 1.0)
                    else:
                        result[i] = np.sign(scaled)
            else:
                result[i] = np.sign(scaled) * min(abs(scaled) / 2.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Rate of Change (ROC)
# ---------------------------------------------------------------------------


def roc(
    close: NDArray,
    period: int = 21,
) -> NDArray:
    """Rate of Change: (close - close[n]) / close[n] * 100.

    Positive = price increasing, negative = decreasing.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return result

    for i in range(period, n):
        if close[i - period] > 0:
            result[i] = (close[i] - close[i - period]) / close[i - period] * 100.0

    return result


def roc_signal(
    close: NDArray,
    period: int = 21,
    lookback: int = 252,
) -> NDArray:
    """ROC normalized to [-1, +1] via z-score."""
    raw = roc(close, period)
    n = len(close)
    result = np.full(n, np.nan, dtype=float)

    for i in range(lookback - 1, n):
        window = raw[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(raw[i]):
            result[i] = np.clip((raw[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Chande Momentum Oscillator (CMO)
# ---------------------------------------------------------------------------


def cmo(
    close: NDArray,
    period: int = 14,
) -> NDArray:
    """Chande Momentum Oscillator.

    CMO = 100 * (Su - Sd) / (Su + Sd)
    where Su = sum of up changes, Sd = sum of down changes.
    Range: -100 to +100.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return result

    for i in range(period, n):
        su = 0.0
        sd = 0.0
        for j in range(i - period + 1, i + 1):
            delta = close[j] - close[j - 1]
            if delta > 0:
                su += delta
            elif delta < 0:
                sd += abs(delta)

        denom = su + sd
        result[i] = 100.0 * (su - sd) / denom if denom > 0 else 0.0

    return result


def cmo_signal(close: NDArray, period: int = 14) -> NDArray:
    """CMO normalized to [-1, +1]."""
    cmo_vals = cmo(close, period)
    return cmo_vals / 100.0
