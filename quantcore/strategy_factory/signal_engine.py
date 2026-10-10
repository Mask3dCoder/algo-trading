"""Signal engine — registry of signal functions and formula parser.

Each signal function takes a data dict (same format as Strategy.on_data)
and returns a 2D numpy array of shape (n_periods, n_assets) with signal values.

The formula parser converts strings like "momentum(12)", "zscore(20)" into
function calls with arguments.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Signal function registry
# ---------------------------------------------------------------------------

_SIGNAL_REGISTRY: dict[str, Callable[..., NDArray[np.floating]]] = {}


def register_signal(name: str) -> Callable:
    """Decorator to register a signal function by name."""

    def decorator(fn: Callable) -> Callable:
        _SIGNAL_REGISTRY[name] = fn
        return fn

    return decorator


def get_signal_fn(name: str) -> Callable:
    if name not in _SIGNAL_REGISTRY:
        available = sorted(_SIGNAL_REGISTRY.keys())
        raise KeyError(f"Signal '{name}' not found. Available: {available}")
    return _SIGNAL_REGISTRY[name]


def list_signals() -> list[str]:
    return sorted(_SIGNAL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------

_FORMULA_RE = re.compile(r"^(\w+)\(([^)]*)\)$")
_FORMULA_NO_ARGS_RE = re.compile(r"^(\w+)$")


def parse_formula(formula: str) -> tuple[str, list[Any]]:
    """Parse a formula string into (function_name, args).

    Examples:
        "momentum(12)" -> ("momentum", [12])
        "zscore(20)" -> ("zscore", [20])
        "moving_average_cross(50,200)" -> ("moving_average_cross", [50, 200])
        "cointegration_spread" -> ("cointegration_spread", [])
        "factor_loading(HML)" -> ("factor_loading", ["HML"])
    """
    formula = formula.strip()

    m = _FORMULA_RE.match(formula)
    if m:
        name = m.group(1)
        args_str = m.group(2).strip()
        if not args_str:
            return name, []
        args = []
        for arg in args_str.split(","):
            arg = arg.strip().strip("'\"")
            # Try numeric conversion
            try:
                args.append(int(arg))
            except ValueError:
                try:
                    args.append(float(arg))
                except ValueError:
                    args.append(arg)
        return name, args

    m = _FORMULA_NO_ARGS_RE.match(formula)
    if m:
        return m.group(1), []

    raise ValueError(f"Cannot parse formula: '{formula}'")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rolling_mean(arr: NDArray, window: int) -> NDArray:
    """Causal rolling mean along axis 0. NaN for first window-1 rows.

    Properly handles NaN values (ignores them in the window average)
    and preserves input dimensionality (1D in → 1D out).
    Vectorized via pd.Series.rolling for performance.
    """
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr.reshape(-1, 1)
    n, ncols = arr.shape
    out = np.full((n, ncols), np.nan, dtype=float)
    for col in range(ncols):
        s = pd.Series(arr[:, col])
        out[:, col] = s.rolling(window=window, min_periods=1).mean().values
    if is_1d:
        out = out.ravel()
    return out


def _rolling_std(arr: NDArray, window: int) -> NDArray:
    """Causal rolling std along axis 0. NaN-safe, preserves dimensionality.
    Vectorized via pd.Series.rolling for performance.
    """
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr.reshape(-1, 1)
    n, ncols = arr.shape
    out = np.full((n, ncols), np.nan, dtype=float)
    for col in range(ncols):
        s = pd.Series(arr[:, col])
        out[:, col] = s.rolling(window=window, min_periods=2).std().values
    if is_1d:
        out = out.ravel()
    return out


def _returns(close: NDArray, period: int = 1) -> NDArray:
    """Compute returns: close[t] / close[t-period] - 1."""
    out = np.full_like(close, np.nan, dtype=float)
    out[period:] = close[period:] / np.maximum(close[:-period], 1e-10) - 1.0
    return out


def _safe_div(num: NDArray, den: NDArray) -> NDArray:
    """Division with zero-protection."""
    return np.where(np.abs(den) > 1e-10, num / den, 0.0)


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

@register_signal("momentum")
def momentum(data: dict, window: int = 12, skip: int = 1) -> NDArray:
    """Time-series momentum: return over [t-window-skip, t-skip].

    For each bar t >= total: signal = (close[t-skip] - close[t-window-skip]) / close[t-window-skip]

    Args:
        window: Lookback period in bars.
        skip: Recent bars to skip (avoid microstructure noise).
    """
    close = data["close"]
    total = window + skip
    n = close.shape[0]
    out = np.full_like(close, np.nan, dtype=float)
    if n <= total:
        return out
    # Both slices have shape (N - total, n_assets)
    recent = close[total - skip:n - skip]       # close[t-skip] for t in [total, N-1]
    lagged = close[total - window - skip:n - window - skip]  # close[t-window-skip]
    out[total:] = _safe_div(recent - lagged, np.maximum(lagged, 1e-10))
    return out


@register_signal("trend")
def trend(data: dict, window: int = 200) -> NDArray:
    """Trend signal: price relative to SMA. >1 = uptrend, <1 = downtrend."""
    close = data["close"]
    sma = _rolling_mean(close, window)
    return _safe_div(close, np.maximum(sma, 1e-10))


@register_signal("zscore")
def zscore(data: dict, window: int = 20) -> NDArray:
    """Rolling z-score of price."""
    close = data["close"]
    mean = _rolling_mean(close, window)
    std = _rolling_std(close, window)
    return _safe_div(close - mean, np.maximum(std, 1e-10))


@register_signal("rsi")
def rsi(data: dict, window: int = 14) -> NDArray:
    """Relative Strength Index, scaled to [-1, +1]."""
    close = data["close"]
    diff = np.diff(close, axis=0)
    diff = np.concatenate([np.zeros((1, close.shape[1])), diff], axis=0)

    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)

    avg_gain = _rolling_mean(gain, window)
    avg_loss = _rolling_mean(loss, window)

    rs = _safe_div(avg_gain, np.maximum(avg_loss, 1e-10))
    rsi_val = 100.0 - 100.0 / (1.0 + rs)
    # Scale to [-1, +1]
    return (rsi_val - 50.0) / 50.0


@register_signal("moving_average_cross")
def moving_average_cross(data: dict, fast: int = 50, slow: int = 200) -> NDArray:
    """Moving average crossover signal.

    Returns +1 when fast > slow, -1 when fast < slow, scaled by gap magnitude.
    """
    close = data["close"]
    fast_ma = _rolling_mean(close, fast)
    slow_ma = _rolling_mean(close, slow)
    gap = _safe_div(fast_ma - slow_ma, np.maximum(slow_ma, 1e-10))
    return np.sign(gap) * np.minimum(np.abs(gap) * 100, 1.0)


@register_signal("bollinger_zscore")
def bollinger_zscore(data: dict, window: int = 20, num_std: float = 2.0) -> NDArray:
    """Bollinger Band z-score: (price - SMA) / (num_std * rolling_std)."""
    close = data["close"]
    sma = _rolling_mean(close, window)
    std = _rolling_std(close, window)
    return _safe_div(close - sma, np.maximum(num_std * std, 1e-10))


@register_signal("volatility")
def volatility(data: dict, window: int = 20) -> NDArray:
    """Rolling annualized volatility of returns."""
    close = data["close"]
    rets = _returns(close, 1)
    vol = _rolling_std(rets, window) * np.sqrt(252)
    return vol


@register_signal("breakout")
def breakout(data: dict, window: int = 20) -> NDArray:
    """Breakout signal: price position within rolling high-low range.

    Returns value in [-1, +1]: +1 = at rolling high, -1 = at rolling low.
    """
    close = data["close"]
    n = close.shape[0]
    out = np.zeros_like(close, dtype=float)
    for t in range(window, n):
        window_data = close[t - window:t]
        rolling_high = np.nanmax(window_data, axis=0)
        rolling_low = np.nanmin(window_data, axis=0)
        rng = rolling_high - rolling_low
        out[t] = np.where(rng > 1e-10, 2.0 * (close[t] - rolling_low) / rng - 1.0, 0.0)
    return out


@register_signal("cointegration_spread")
def cointegration_spread(data: dict, **kwargs: Any) -> NDArray:
    """Cointegration spread for pairs trading.

    Loads pairs from CSV file (pairs_file parameter) or data["pairs"].
    Uses stored hedge ratios and intercepts for proper spread computation:
        spread = log(B) - hedge_ratio * log(A) - intercept
    Normalizes via rolling z-score for entry/exit signals.
    """
    from pathlib import Path

    close = data["close"]
    symbols = data.get("symbols", [])
    pairs = data.get("pairs", kwargs.get("pairs", []))
    hedge_window = kwargs.get("hedge_window", 60)

    # Load pairs from CSV if not provided in data dict
    if not pairs:
        pairs_file = kwargs.get("pairs_file", "data/pairs_found.csv")
        if pairs_file and Path(pairs_file).exists():
            import csv as _csv

            loaded: list[dict[str, Any]] = []
            with open(pairs_file, newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    loaded.append(row)
            if loaded:
                pairs = loaded  # list of dicts with stock_a, stock_b, hedge_ratio, intercept

    if not pairs:
        # Fallback: use first two assets as a pair
        if close.shape[1] >= 2:
            pairs = [(0, 1)]
        else:
            return np.zeros_like(close, dtype=float)

    out = np.zeros_like(close, dtype=float)
    for pair in pairs:
        # Handle CSV-loaded pairs (dict with hedge_ratio/intercept)
        if isinstance(pair, dict):
            sym_a = pair.get("stock_a", "")
            sym_b = pair.get("stock_b", "")
            if not symbols:
                continue
            try:
                a = symbols.index(sym_a)
                b = symbols.index(sym_b)
            except ValueError:
                continue
            hedge_ratio = float(pair.get("hedge_ratio", 1.0))
            intercept = float(pair.get("intercept", 0.0))
            # Log-price cointegration spread
            log_a = np.log(np.maximum(close[:, a], 1e-10))
            log_b = np.log(np.maximum(close[:, b], 1e-10))
            spread = log_b - hedge_ratio * log_a - intercept
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = pair
            if isinstance(a, str):
                try:
                    a = symbols.index(a)
                    b = symbols.index(b)
                except ValueError:
                    continue
            spread = close[:, a] - close[:, b]
        else:
            continue

        spread_mean = _rolling_mean(spread, hedge_window)
        spread_std = _rolling_std(spread, hedge_window)
        z = _safe_div(spread - spread_mean, np.maximum(spread_std, 1e-10))
        out[:, a] = -z  # short the expensive leg
        out[:, b] = z   # long the cheap leg
    return out


@register_signal("factor_loading")
def factor_loading(data: dict, factor: str = "HML") -> NDArray:
    """Factor loading signal.

    Expects data["factors"] to be a DataFrame or dict with factor returns.
    Returns the cross-sectional factor score for each asset.
    """
    close = data["close"]
    factors = data.get("factors")
    if factors is None:
        # Fallback: use momentum as proxy for factor loading
        return momentum(data, window=12, skip=1)

    if hasattr(factors, "get"):
        factor_ret = factors.get(factor)
    else:
        factor_ret = None

    if factor_ret is None:
        return momentum(data, window=12, skip=1)

    # Compute rolling factor exposure via regression residual
    rets = _returns(close, 1)
    out = np.zeros_like(close, dtype=float)
    window = 60
    for t in range(window, close.shape[0]):
        y = rets[t - window:t]
        f = factor_ret[t - window:t] if hasattr(factor_ret, '__len__') else factor_ret
        if isinstance(f, (int, float)):
            out[t] = rets[t] - f
        else:
            f_arr = np.asarray(f[:window], dtype=float)
            if len(f_arr) == window:
                beta = np.cov(y.flatten(), f_arr)[0, 1] / max(np.var(f_arr), 1e-10)
                out[t] = y[-1] - beta * f_arr[-1]
    return out


@register_signal("sentiment")
def sentiment(data: dict, window: int = 5) -> NDArray:
    """Sentiment signal placeholder.

    Expects data["sentiment"] to be a sentiment score array.
    Falls back to short-term momentum if no sentiment data available.
    """
    sent = data.get("sentiment")
    if sent is not None:
        return _rolling_mean(np.asarray(sent, dtype=float), window)
    # Fallback: short-term reversal (negative momentum)
    return -momentum(data, window=5, skip=0)


@register_signal("fred_regime_score")
def fred_regime_score(data: dict, signal_name: str = "yield_curve_regime") -> NDArray:
    """FRED regime score for regime filtering.

    Expects data["fred_signals"] to be a dict of signal name -> array.
    Returns a 1D array broadcast to all assets.
    """
    fred = data.get("fred_signals", {})
    if signal_name in fred:
        raw = np.asarray(fred[signal_name], dtype=float)
    else:
        raw = np.zeros(data["close"].shape[0], dtype=float)
    # Broadcast to all assets
    return np.tile(raw.reshape(-1, 1), (1, data["close"].shape[1]))


@register_signal("bollinger_band_width")
def bollinger_band_width(data: dict, window: int = 20, num_std: float = 2.0, min_width: float = 0.02) -> NDArray:
    """Bollinger Band width as a fraction of the middle band.

    BB width = (upper_band - lower_band) / middle_band.
    Returns 1.0 when bands are wide enough (width >= min_width), 0.0 otherwise.
    Used as a filter for mean reversion strategies — only trade when bands are wide.

    Args:
        window: Rolling window for SMA and std.
        num_std: Number of standard deviations for the bands.
        min_width: Minimum width threshold as fraction of middle band.
    """
    close = data["close"]
    sma = _rolling_mean(close, window)
    std = _rolling_std(close, window)
    width = _safe_div(2.0 * num_std * std, np.maximum(sma, 1e-10))
    return np.where(width >= min_width, 1.0, 0.0)


@register_signal("carry")
def carry(data: dict, window: int = 21) -> NDArray:
    """Carry signal: yield proxy using price momentum over short window.

    For ETFs, approximates carry via short-term return stability.
    """
    close = data["close"]
    rets = _returns(close, 1)
    # Positive carry = consistently positive returns
    return _rolling_mean(rets, window) * 252  # annualize


@register_signal("relative_strength")
def relative_strength(data: dict, window: int = 20) -> NDArray:
    """Cross-sectional relative strength: each asset's return vs universe mean."""
    close = data["close"]
    rets = _returns(close, window)
    universe_mean = np.nanmean(rets, axis=1, keepdims=True)
    return rets - universe_mean


# ---------------------------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------------------------


class SignalEngine:
    """Computes signals from formula strings.

    Usage:
        engine = SignalEngine()
        signal_array = engine.compute("momentum(12)", data)
    """

    def compute(self, formula: str, data: dict) -> NDArray:
        """Parse formula, call signal function, return signal array."""
        name, args = parse_formula(formula)
        fn = get_signal_fn(name)
        return fn(data, *args)

    def compute_raw(self, formula: str, data: dict) -> NDArray:
        """Compute raw signal without any normalization."""
        return self.compute(formula, data)

    @staticmethod
    def rank_cross_sectional(
        signals: NDArray,
        method: str = "zscore_long_short",
        threshold: float = 1.0,
    ) -> NDArray:
        """Apply cross-sectional ranking to signal values.

        Args:
            signals: 2D array (n_periods, n_assets) of raw signal values.
            method: Ranking method. Options:
                - "zscore_long_short(threshold)": Long if z > threshold, short if z < -threshold
                - "top_minus_bottom_quintile": Long top 20%, short bottom 20%
                - "top_minus_bottom_decile": Long top 10%, short bottom 10%
                - "tercile": Long top tercile, short bottom tercile
            threshold: Threshold for zscore_long_short method.

        Returns:
            Array of weights in [-1, +1] for each asset.
        """
        # Parse method with optional args
        m = re.match(r"^(\w+)\(([^)]*)\)$", method.strip())
        if m:
            method_name = m.group(1)
            try:
                threshold = float(m.group(2))
            except ValueError:
                pass
        else:
            method_name = method.strip()

        weights = np.zeros_like(signals, dtype=float)

        for t in range(signals.shape[0]):
            row = signals[t]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                continue

            valid_vals = row[valid]
            valid_idx = np.where(valid)[0]

            if method_name == "zscore_long_short":
                mu = np.mean(valid_vals)
                sigma = np.std(valid_vals)
                if sigma > 1e-10:
                    z = (valid_vals - mu) / sigma
                    for i, idx in enumerate(valid_idx):
                        if z[i] > threshold:
                            weights[t, idx] = 1.0
                        elif z[i] < -threshold:
                            weights[t, idx] = -1.0

            elif method_name in ("top_minus_bottom_quintile", "quintile"):
                n = len(valid_vals)
                sorted_idx = valid_idx[np.argsort(valid_vals)]
                n_group = max(1, n // 5)
                for idx in sorted_idx[-n_group:]:
                    weights[t, idx] = 1.0
                for idx in sorted_idx[:n_group]:
                    weights[t, idx] = -1.0

            elif method_name in ("top_minus_bottom_decile", "decile"):
                n = len(valid_vals)
                sorted_idx = valid_idx[np.argsort(valid_vals)]
                n_group = max(1, n // 10)
                for idx in sorted_idx[-n_group:]:
                    weights[t, idx] = 1.0
                for idx in sorted_idx[:n_group]:
                    weights[t, idx] = -1.0

            elif method_name == "tercile":
                n = len(valid_vals)
                sorted_idx = valid_idx[np.argsort(valid_vals)]
                n_group = max(1, n // 3)
                for idx in sorted_idx[-n_group:]:
                    weights[t, idx] = 1.0
                for idx in sorted_idx[:n_group]:
                    weights[t, idx] = -1.0

            elif method_name == "long_only_top":
                n = len(valid_vals)
                sorted_idx = valid_idx[np.argsort(valid_vals)]
                n_group = max(1, n // 5)
                for idx in sorted_idx[-n_group:]:
                    weights[t, idx] = 1.0

            else:
                # Default: simple sign-based
                for i, idx in enumerate(valid_idx):
                    if valid_vals[i] > 0:
                        weights[t, idx] = 1.0
                    elif valid_vals[i] < 0:
                        weights[t, idx] = -1.0

        return weights
