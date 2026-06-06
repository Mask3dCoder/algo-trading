"""Volatility signals: ATR, historical vol, vol regime HMM, VIX term structure, RV-IV spread, vol-of-vol."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcore.core.constants import TRADING_DAYS_PER_YEAR
from quantcore.signals.technical.trend import _atr_core

# ---------------------------------------------------------------------------
# Average True Range (ATR)
# ---------------------------------------------------------------------------


def atr(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    period: int = 14,
) -> NDArray:
    """Average True Range (Wilder smoothing)."""
    return _atr_core(np.asarray(high, dtype=np.float64),
                     np.asarray(low, dtype=np.float64),
                     np.asarray(close, dtype=np.float64), period)


def atr_normalized(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    period: int = 14,
    lookback: int = 252,
) -> NDArray:
    """ATR as percentage of price, z-scored against rolling history.

    High values = unusually volatile = potential regime change.
    """
    raw_atr = atr(high, low, close, period)
    n = len(close)
    result = np.full(n, np.nan, dtype=float)

    atr_pct = raw_atr / close * 100.0  # ATR as % of close

    for i in range(lookback - 1, n):
        window = atr_pct[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(atr_pct[i]):
            result[i] = np.clip((atr_pct[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Historical Volatility
# ---------------------------------------------------------------------------


def historical_vol(
    close: NDArray,
    windows: list[int] | None = None,
) -> dict[int, NDArray]:
    """Historical (realized) volatility at multiple windows.

    Returns dict mapping window → annualized volatility series.
    Default windows: [5, 10, 21, 63, 252].
    """
    if windows is None:
        windows = [5, 10, 21, 63, 252]

    n = len(close)
    log_returns = np.full(n, np.nan, dtype=float)
    for i in range(1, n):
        if close[i - 1] > 0 and close[i] > 0:
            log_returns[i] = np.log(close[i] / close[i - 1])

    result: dict[int, NDArray] = {}
    for w in windows:
        vol = np.full(n, np.nan, dtype=float)
        for i in range(w, n):
            window = log_returns[i - w + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                vol[i] = np.std(valid, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        result[w] = vol

    return result


def historical_vol_signal(
    close: NDArray,
    window: int = 21,
    lookback: int = 252,
) -> NDArray:
    """Historical volatility z-scored against its own history.

    +1 = unusually high vol, -1 = unusually low vol.
    """
    hv_dict = historical_vol(close, windows=[window])
    hv = hv_dict[window]
    n = len(close)
    result = np.full(n, np.nan, dtype=float)

    for i in range(lookback - 1, n):
        hv_window = hv[i - lookback + 1:i + 1]
        valid = hv_window[~np.isnan(hv_window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(hv[i]):
            result[i] = np.clip((hv[i] - m) / s / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Volatility Regime Detection (HMM)
# ---------------------------------------------------------------------------


def _forward_backward(
    obs: NDArray,
    means: NDArray,
    variances: NDArray,
    trans: NDArray,
    pi: NDArray,
) -> tuple[NDArray, NDArray, float]:
    """Forward-backward algorithm for Gaussian HMM.

    Returns (gamma, xi, log_likelihood).
    gamma: posterior state probabilities (T x K)
    xi: posterior transition probabilities (T-1 x K x K)
    """
    T = len(obs)
    K = len(means)

    # Emission probabilities
    def _log_gaussian(x: float, mu: float, var: float) -> float:
        if var <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var

    # Forward pass (log-space for numerical stability)
    log_alpha = np.full((T, K), -np.inf)
    log_trans = np.log(trans + 1e-300)
    log_pi = np.log(pi + 1e-300)

    for k in range(K):
        log_alpha[0, k] = log_pi[k] + _log_gaussian(obs[0], means[k], variances[k])

    for t in range(1, T):
        for j in range(K):
            log_emission = _log_gaussian(obs[t], means[j], variances[j])
            max_prev = np.max(log_alpha[t - 1, :] + log_trans[:, j])
            log_alpha[t, j] = max_prev + log_emission + np.log(
                np.sum(np.exp(log_alpha[t - 1, :] + log_trans[:, j] - max_prev))
            )

    # Stabilized log-sum-exp
    ll_max = np.max(log_alpha[T - 1, :])
    log_likelihood = ll_max + np.log(np.sum(np.exp(log_alpha[T - 1, :] - ll_max)))
    log_beta = np.full((T, K), -np.inf)

    for k in range(K):
        log_beta[T - 1, k] = 0.0

    for t in range(T - 2, -1, -1):
        for i in range(K):
            terms = np.full(K, -np.inf)
            for j in range(K):
                log_emission = _log_gaussian(obs[t + 1], means[j], variances[j])
                terms[j] = log_trans[i, j] + log_emission + log_beta[t + 1, j]
            max_term = np.max(terms)
            if max_term > -np.inf:
                log_beta[t, i] = max_term + np.log(np.sum(np.exp(terms - max_term)))

    # Gamma: posterior state probabilities
    log_gamma = log_alpha + log_beta
    gamma = np.zeros((T, K))
    for t in range(T):
        max_g = np.max(log_gamma[t, :])
        if max_g > -np.inf:
            gamma[t, :] = np.exp(log_gamma[t, :] - max_g)
            gamma[t, :] /= np.sum(gamma[t, :])
        else:
            gamma[t, :] = 1.0 / K

    return gamma, log_alpha, log_beta, log_likelihood


def _baum_welch(
    obs: NDArray,
    n_states: int = 3,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Baum-Welch (EM) for Gaussian HMM.

    Returns (means, variances, transition_matrix, initial_probs).
    """
    T = len(obs)
    K = n_states

    # Initialize with quantile-based clustering
    sorted_obs = np.sort(obs)
    quantiles = [sorted_obs[int(T * (i + 0.5) / K)] for i in range(K)]
    means = np.array(quantiles)
    variances = np.full(K, np.var(obs), dtype=float)
    trans = np.full((K, K), 1.0 / K)
    np.fill_diagonal(trans, 0.6)
    for i in range(K):
        trans[i, :] /= trans[i, :].sum()
    pi = np.full(K, 1.0 / K)

    prev_ll = -np.inf

    for _ in range(max_iter):
        gamma, log_alpha, log_beta, ll = _forward_backward(obs, means, variances, trans, pi)

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # M-step
        pi = gamma[0, :] / np.sum(gamma[0, :])

        for k in range(K):
            weight_sum = np.sum(gamma[:, k])
            if weight_sum > 1e-10:
                means[k] = np.sum(gamma[:, k] * obs) / weight_sum
                variances[k] = np.sum(gamma[:, k] * (obs - means[k]) ** 2) / weight_sum
                variances[k] = max(variances[k], 1e-10)

        # Transition matrix from xi (approximated via gamma)
        log_trans = np.log(trans + 1e-300)
        for i in range(K):
            for j in range(K):
                numer = 0.0
                denom = 0.0
                for t in range(T - 1):
                    log_emission_j = -0.5 * np.log(2 * np.pi * variances[j]) - \
                        0.5 * (obs[t + 1] - means[j]) ** 2 / variances[j]
                    xi_t = (log_alpha[t, i] + log_trans[i, j] + log_emission_j + log_beta[t + 1, j])
                    numer += np.exp(xi_t - ll) if xi_t - ll > -700 else 0.0
                    denom += gamma[t, i]

                if denom > 1e-10:
                    trans[i, j] = numer / denom

            row_sum = np.sum(trans[i, :])
            if row_sum > 1e-12 and not np.isnan(row_sum):
                trans[i, :] /= row_sum

    return means, variances, trans, pi


def vol_regime(
    close: NDArray,
    n_states: int = 3,
    lookback: int = 252,
    min_obs: int = 100,
) -> NDArray:
    """Volatility regime classification using Gaussian HMM.

    States labeled 0 (low vol) to n_states-1 (high vol).
    Returns regime labels (0-indexed integer array).
    States are sorted by mean volatility so 0=low, 1=medium, 2=high.
    """
    n = len(close)
    result = np.full(n, -1, dtype=int)

    # Compute rolling 21-day realized vol
    hv_dict = historical_vol(close, windows=[21])
    hv = hv_dict[21]

    valid_mask = ~np.isnan(hv)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < min_obs:
        return result

    obs = hv[valid_mask]

    try:
        means, variances, trans, pi = _baum_welch(obs, n_states=n_states, max_iter=50)
    except Exception:
        return result

    # Sort states by mean (low → high)
    state_order = np.argsort(means)

    # Viterbi decoding simplified: just use posterior decoding (gamma argmax)
    gamma, _, _, _ = _forward_backward(obs, means, variances, trans, pi)
    state_seq = np.argmax(gamma, axis=1)

    # Remap states to sorted order
    remap = {old: new for new, old in enumerate(state_order)}
    for i, idx in enumerate(valid_indices):
        result[idx] = remap.get(state_seq[i], 0)

    return result


def vol_regime_signal(
    close: NDArray,
    n_states: int = 3,
    lookback: int = 252,
) -> NDArray:
    """Vol regime as continuous signal: -1 (low vol) to +1 (high vol)."""
    regimes = vol_regime(close, n_states, lookback)
    n = len(close)
    result = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if regimes[i] >= 0:
            result[i] = 2.0 * regimes[i] / (n_states - 1) - 1.0

    return result


# ---------------------------------------------------------------------------
# VIX Term Structure Signal
# ---------------------------------------------------------------------------


def vix_term_structure_signal(
    vix_spot: NDArray,
    vix_future: NDArray,
    lookback: int = 252,
) -> NDArray:
    """VIX term structure signal.

    Basis = VIX_future - VIX_spot.
    Positive (contango) = complacent → slight negative signal.
    Negative (backwardation) = fear → positive signal (buy opportunity).

    Returns z-score normalized signal.
    """
    n = len(vix_spot)
    basis = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isnan(vix_spot[i]) and not np.isnan(vix_future[i]) and vix_spot[i] > 0:
            basis[i] = (vix_future[i] - vix_spot[i]) / vix_spot[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = basis[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(basis[i]):
            # Invert: backwardation (negative basis) = bullish signal
            z = -(basis[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Realized vs Implied Vol Spread
# ---------------------------------------------------------------------------


def rv_iv_spread(
    realized_vol: NDArray,
    implied_vol: NDArray,
    lookback: int = 252,
) -> NDArray:
    """RV-IV spread: realized vol minus implied vol.

    Positive = realized > implied → vol was cheap (sell vol signal).
    Negative = realized < implied → vol was expensive (buy vol signal).

    Note: for trading, we actually want to use current IV vs predicted RV,
    but this spread gives a historical signal. We z-score and invert
    so positive signal = buy vol opportunity (IV cheap).
    """
    n = len(realized_vol)
    spread = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isnan(realized_vol[i]) and not np.isnan(implied_vol[i]):
            spread[i] = implied_vol[i] - realized_vol[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(spread[i]):
            # Positive spread (IV > RV) = vol is expensive → sell vol = +1
            z = (spread[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Volatility of Volatility
# ---------------------------------------------------------------------------


def vol_of_vol(
    close: NDArray,
    vol_window: int = 21,
    vol_of_vol_window: int = 63,
) -> NDArray:
    """Volatility of volatility: rolling std of realized vol.

    High vol-of-vol = uncertain vol regime = risk-off signal.
    """
    n = len(close)
    hv_dict = historical_vol(close, windows=[vol_window])
    hv = hv_dict[vol_window]

    result = np.full(n, np.nan, dtype=float)

    for i in range(vol_of_vol_window, n):
        window = hv[i - vol_of_vol_window + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        result[i] = np.std(valid, ddof=1)

    # Z-score normalize
    signal = np.full(n, np.nan, dtype=float)
    lookback = 252
    for i in range(lookback - 1, n):
        window = result[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(result[i]):
            signal[i] = np.clip((result[i] - m) / s / 3.0, -1.0, 1.0)

    return signal


# ---------------------------------------------------------------------------
# Parkinson Volatility (using H/L)
# ---------------------------------------------------------------------------


def parkinson_vol(
    high: NDArray,
    low: NDArray,
    period: int = 21,
) -> NDArray:
    """Parkinson volatility estimator using high-low range.

    More efficient than close-to-close for intraday data.
    sigma = sqrt(1/(4n*ln2) * sum(ln(H/L)^2)) * sqrt(252)
    """
    n = len(high)
    result = np.full(n, np.nan, dtype=float)
    if n < period:
        return result

    log_hl_sq = np.zeros(n, dtype=float)
    for i in range(n):
        if high[i] > 0 and low[i] > 0:
            log_hl_sq[i] = (np.log(high[i] / low[i])) ** 2

    factor = 1.0 / (4.0 * np.log(2.0))

    for i in range(period - 1, n):
        window = log_hl_sq[i - period + 1:i + 1]
        result[i] = np.sqrt(factor * np.sum(window) / period) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return result


# ---------------------------------------------------------------------------
# Garman-Klass Volatility
# ---------------------------------------------------------------------------


def garman_klass_vol(
    high: NDArray,
    low: NDArray,
    open_: NDArray,
    close: NDArray,
    period: int = 21,
) -> NDArray:
    """Garman-Klass volatility estimator using OHLC.

    More efficient than Parkinson by using open-close information.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < period:
        return result

    for i in range(period - 1, n):
        gk_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if high[j] > 0 and low[j] > 0 and open_[j] > 0 and close[j] > 0:
                hl_term = 0.5 * (np.log(high[j] / low[j])) ** 2
                co_term = (2 * np.log(2) - 1) * (np.log(close[j] / open_[j])) ** 2
                gk_sum += hl_term - co_term
        gk_sum = max(gk_sum, 0.0)
        result[i] = np.sqrt(gk_sum / period) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return result


# ---------------------------------------------------------------------------
# Volatility Risk Premium (VRP) Signals
# ---------------------------------------------------------------------------


def vrp_raw(implied_vol: NDArray, realized_vol: NDArray) -> NDArray:
    """Raw volatility risk premium: (IV - RV) / RV.

    Positive = IV > RV (vol is expensive, sell vol opportunity).
    Guards against RV=0 by clamping to a minimum.

    Args:
        implied_vol: Implied volatility series (e.g., VIX/100).
        realized_vol: Realized volatility series (e.g., Garman-Klass).

    Returns:
        VRP ratio series.
    """
    iv = np.asarray(implied_vol, dtype=np.float64)
    rv = np.asarray(realized_vol, dtype=np.float64)
    rv_safe = np.maximum(rv, 1e-10)
    return (iv - rv) / rv_safe


def ts_regime_multiplier(
    ts_slope: NDArray,
    lookback: int = 252,
    steepness: float = 10.0,
    midpoint: float = 0.0,
) -> NDArray:
    """Term-structure regime multiplier via sigmoid of z-scored slope.

    Maps contango (positive slope) → values > 1 (amplify short vol).
    Maps backwardation (negative slope) → values < 1 (dampen short vol).
    Output range: [0.5, 1.5].

    Args:
        ts_slope: Term structure slope series (back - front) / front.
        lookback: Rolling window for z-scoring.
        steepness: Sigmoid steepness (higher = sharper transition).
        midpoint: Sigmoid midpoint in z-score space.

    Returns:
        Multiplier series in [0.5, 1.5].
    """
    slope = np.asarray(ts_slope, dtype=np.float64)
    n = len(slope)
    result = np.full(n, 1.0, dtype=float)

    for i in range(lookback - 1, n):
        window = slope[i - lookback + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma < 1e-10:
            continue
        z = (slope[i] - mu) / sigma if not np.isnan(slope[i]) else 0.0
        # Sigmoid: 1 / (1 + exp(-steepness * (z - midpoint)))
        sig = 1.0 / (1.0 + np.exp(-steepness * (z - midpoint)))
        # Map [0, 1] → [0.5, 1.5]
        result[i] = 0.5 + sig

    return result


def vvix_conviction(vvix: NDArray, lookback: int = 252) -> NDArray:
    """VVIX-based conviction signal.

    High VVIX = uncertainty in vol market → low conviction (smaller position).
    Low VVIX = complacency → high conviction.

    Output range: [0, 1].  1 = full conviction, 0 = no conviction.

    Args:
        vvix: VVIX index series.
        lookback: Rolling window for z-scoring.

    Returns:
        Conviction series in [0, 1].
    """
    vvix_arr = np.asarray(vvix, dtype=np.float64)
    n = len(vvix_arr)
    result = np.full(n, 0.5, dtype=float)

    for i in range(lookback - 1, n):
        window = vvix_arr[i - lookback + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma < 1e-10:
            continue
        if np.isnan(vvix_arr[i]):
            continue
        z = (vvix_arr[i] - mu) / sigma
        # High VVIX → low conviction
        conviction = np.clip(1.0 - z / 2.0, 0.0, 1.0)
        result[i] = conviction

    return result


def vrp_composite_signal(
    implied_vol: NDArray,
    realized_vol: NDArray,
    ts_slope: NDArray,
    vvix: NDArray,
    lookback: int = 252,
    ewm_span: int = 5,
) -> NDArray:
    """Composite VRP signal: vrp_raw × ts_regime_multiplier × vvix_conviction.

    EWM-smoothed for noise reduction.  Positive = sell vol opportunity.

    Args:
        implied_vol: Implied volatility series.
        realized_vol: Realized volatility series.
        ts_slope: Term structure slope series.
        vvix: VVIX index series.
        lookback: Rolling window for z-scoring.
        ewm_span: Exponential weighted moving average span.

    Returns:
        Composite signal series.
    """
    raw = vrp_raw(implied_vol, realized_vol)
    ts_mult = ts_regime_multiplier(ts_slope, lookback=lookback)
    conviction = vvix_conviction(vvix, lookback=lookback)

    composite = raw * ts_mult * conviction

    # EWM smoothing via pandas
    s = pd.Series(composite)
    smoothed = s.ewm(span=ewm_span, min_periods=1).mean().values

    return smoothed
