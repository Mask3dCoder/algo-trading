"""Signal combiner: equal-weight, IC-weight, stacking, dynamic/regime-based combination."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Equal-Weight Combination
# ---------------------------------------------------------------------------


def equal_weight_combine(signals: list[NDArray]) -> NDArray:
    """Combine signals with equal weighting.

    Simply averages all available signals at each point.

    Args:
        signals: list of signal arrays, all same length, values in [-1, +1].

    Returns:
        Combined signal in [-1, +1].
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    result = np.full(n, np.nan, dtype=float)

    for i in range(n):
        vals = []
        for sig in signals:
            if i < len(sig) and not np.isnan(sig[i]):
                vals.append(sig[i])

        if vals:
            result[i] = np.clip(np.mean(vals), -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# IC-Weight Combination
# ---------------------------------------------------------------------------


def ic_weight_combine(
    signals: list[NDArray],
    forward_returns: NDArray,
    lookback: int = 252,
    ic_floor: float = 0.0,
) -> NDArray:
    """Combine signals weighted by their rolling Information Coefficient.

    Signals with higher absolute IC get more weight.
    Negative-IC signals get their sign flipped (they're still informative).

    Args:
        signals: list of signal arrays.
        forward_returns: forward returns for IC calculation.
        lookback: rolling IC lookback window.
        ic_floor: minimum |IC| to include a signal.

    Returns:
        Combined signal in [-1, +1].
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    n_signals = len(signals)
    result = np.full(n, np.nan, dtype=float)

    # Compute rolling IC for each signal
    ic_weights = np.zeros((n, n_signals), dtype=float)

    for i in range(lookback - 1, n):
        window_start = i - lookback + 1
        fr_window = forward_returns[window_start:i + 1]

        weights = []
        for s_idx, sig in enumerate(signals):
            sig_window = sig[window_start:i + 1]
            valid = ~(np.isnan(sig_window) | np.isnan(fr_window))

            if np.sum(valid) < 20:
                weights.append(0.0)
                continue

            corr = np.corrcoef(sig_window[valid], fr_window[valid])[0, 1]
            ic = corr if not np.isnan(corr) else 0.0

            # Use absolute IC, keep sign for potential flip
            if abs(ic) >= ic_floor:
                weights.append(ic)
            else:
                weights.append(0.0)

        abs_weights = [abs(w) for w in weights]
        total_w = sum(abs_weights)

        if total_w > 0:
            combined = 0.0
            total_abs_w = 0.0
            for s_idx, sig in enumerate(signals):
                if i < len(sig) and not np.isnan(sig[i]) and abs_weights[s_idx] > 0:
                    # If IC is negative, flip the signal
                    sign = 1.0 if weights[s_idx] >= 0 else -1.0
                    w = abs_weights[s_idx] / total_w
                    combined += sign * sig[i] * w
                    total_abs_w += w

            if total_abs_w > 0:
                result[i] = np.clip(combined / total_abs_w, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Stacking Combiner (simplified)
# ---------------------------------------------------------------------------


def stacking_combine(
    signals: list[NDArray],
    forward_returns: NDArray,
    train_ratio: float = 0.7,
    lookback: int = 252,
) -> NDArray:
    """Stacking combination: train linear meta-learner on past data.

    Uses rolling OLS to learn optimal linear combination weights.

    Args:
        signals: list of signal arrays.
        forward_returns: forward returns for training.
        train_ratio: fraction of lookback used for training.
        lookback: rolling window size.

    Returns:
        Combined signal in [-1, +1].
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    n_signals = len(signals)
    result = np.full(n, np.nan, dtype=float)

    for i in range(lookback - 1, n):
        train_size = int(lookback * train_ratio)
        train_start = i - lookback + 1
        train_end = train_start + train_size

        # Build training matrix
        Y = forward_returns[train_start:train_end]
        X = np.column_stack([
            sig[train_start:train_end] for sig in signals
        ])

        # Add intercept
        X_aug = np.column_stack([np.ones(len(Y)), X])

        valid = ~(np.isnan(Y) | np.any(np.isnan(X_aug), axis=1))
        if np.sum(valid) < 20:
            continue

        Y_clean = Y[valid]
        X_clean = X_aug[valid]

        try:
            beta = np.linalg.lstsq(X_clean, Y_clean, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # Predict
        current_signals = [sig[i] if i < len(sig) and not np.isnan(sig[i]) else np.nan for sig in signals]

        if not any(np.isnan(s) for s in current_signals):
            x_pred = np.concatenate([[1.0], current_signals])
            pred = x_pred @ beta
            result[i] = np.clip(pred * 100.0, -1.0, 1.0)  # Scale to [-1, 1]

    return result


# ---------------------------------------------------------------------------
# Dynamic / Regime-Based Combination
# ---------------------------------------------------------------------------


def regime_weight_combine(
    signals: list[NDArray],
    regime: NDArray,
    regime_weights: dict[int, list[float]],
) -> NDArray:
    """Combine signals with regime-dependent weights.

    Args:
        signals: list of signal arrays.
        regime: integer regime labels (0, 1, 2, ...).
        regime_weights: dict mapping regime_id → list of weights.
            e.g. {0: [0.6, 0.2, 0.2], 1: [0.1, 0.7, 0.2]}
            Regime 0 might favor trend, regime 1 might favor mean-reversion.

    Returns:
        Combined signal in [-1, +1].
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    result = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i >= len(regime) or np.isnan(regime[i]):
            continue

        reg = int(regime[i])
        if reg not in regime_weights:
            continue

        weights = regime_weights[reg]
        combined = 0.0
        total_w = 0.0

        for s_idx, sig in enumerate(signals):
            if s_idx < len(weights) and i < len(sig) and not np.isnan(sig[i]):
                combined += sig[i] * weights[s_idx]
                total_w += weights[s_idx]

        if total_w > 0:
            result[i] = np.clip(combined / total_w, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Sign-Agreement Combiner
# ---------------------------------------------------------------------------


def sign_agreement_combine(signals: list[NDArray]) -> NDArray:
    """Combine signals based on sign agreement (majority vote).

    Signal = fraction of signals that agree, with direction
    determined by the majority.

    Returns values in [-1, +1]. Higher magnitude = more agreement.
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    result = np.full(n, np.nan, dtype=float)

    for i in range(n):
        vals = []
        for sig in signals:
            if i < len(sig) and not np.isnan(sig[i]):
                vals.append(sig[i])

        if not vals:
            continue

        # Count positive and negative
        n_pos = sum(1 for v in vals if v > 0)
        n_neg = sum(1 for v in vals if v < 0)
        n_total = len(vals)

        # Direction by majority
        if n_pos > n_neg:
            result[i] = n_pos / n_total
        elif n_neg > n_pos:
            result[i] = -n_neg / n_total
        else:
            result[i] = 0.0

    return result


# ---------------------------------------------------------------------------
# Adaptive Combiner
# ---------------------------------------------------------------------------


def adaptive_combine(
    signals: list[NDArray],
    forward_returns: NDArray,
    lookback: int = 126,
    smoothing: float = 0.1,
) -> NDArray:
    """Adaptive signal combination with exponential weight updates.

    Weights are updated based on recent IC performance, with
    exponential smoothing to prevent whiplash.

    Args:
        signals: list of signal arrays.
        forward_returns: forward returns for IC calculation.
        lookback: IC evaluation window.
        smoothing: exponential smoothing factor (0-1, lower = smoother).

    Returns:
        Combined signal in [-1, +1].
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    n_signals = len(signals)
    result = np.full(n, np.nan, dtype=float)

    # Initialize weights equally
    weights = np.ones(n_signals) / n_signals

    for i in range(lookback - 1, n):
        window_start = i - lookback + 1
        fr_window = forward_returns[window_start:i + 1]

        # Update weights based on recent IC
        ics = np.zeros(n_signals)
        for s_idx, sig in enumerate(signals):
            sig_window = sig[window_start:i + 1]
            valid = ~(np.isnan(sig_window) | np.isnan(fr_window))

            if np.sum(valid) >= 20:
                corr = np.corrcoef(sig_window[valid], fr_window[valid])[0, 1]
                ics[s_idx] = abs(corr) if not np.isnan(corr) else 0.0

        # Target weights proportional to |IC|
        target = ics.copy()
        total = np.sum(target)
        if total > 0:
            target /= total
        else:
            target = np.ones(n_signals) / n_signals

        # Exponential smoothing
        alpha = 1.0 - smoothing
        weights = alpha * weights + smoothing * target
        total_w = np.sum(weights)
        if total_w > 0:
            weights /= total_w

        # Combine current signals
        combined = 0.0
        for s_idx, sig in enumerate(signals):
            if i < len(sig) and not np.isnan(sig[i]):
                combined += sig[i] * weights[s_idx]

        result[i] = np.clip(combined, -1.0, 1.0)

    return result
