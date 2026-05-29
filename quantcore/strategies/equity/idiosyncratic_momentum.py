"""Idiosyncratic (factor-neutral) equity momentum.

Blitz, Pang & van Vliet (2011): regress each stock's 12-month return
against the market factor (SPY), take the residual as the momentum
signal.  This avoids buying high-beta / selling low-beta and isolates
the stock-specific component of momentum.

Signal construction:
1. 12-month cumulative return per stock (skip most recent month).
2. OLS regression: stock_ret = alpha + beta * SPY_ret + epsilon.
3. Cross-sectional z-score of residual epsilon.
4. Long top decile, short bottom decile.
5. Inverse-volatility weighting within each leg (dollar-neutral).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


# ---------------------------------------------------------------------------
# SPY symbol used as the market factor proxy.
# ---------------------------------------------------------------------------
_SPY_ALIASES = {"SPY", "spy", "SPDR S&P 500 ETF", "SPY.US"}


def _find_spy_index(symbols: list[str]) -> int | None:
    """Return the column index of SPY in the asset matrix, or None."""
    for i, sym in enumerate(symbols):
        if sym in _SPY_ALIASES:
            return i
    return None


def _ols_residual(
    y: NDArray, X: NDArray,
) -> float:
    """Return the last residual from OLS of y on [1, X].

    Both *y* and *X* are 1-D arrays of the same length.  Uses
    ``np.linalg.lstsq`` so there is no sklearn dependency.

    The regression is causal: all inputs are strictly historical.
    """
    n = len(y)
    if n < 10:
        return np.nan
    # Design matrix: intercept + single factor
    A = np.column_stack([np.ones(n), X])
    result = np.linalg.lstsq(A, y, rcond=None)
    beta = result[0]                    # (2,) — alpha, beta
    residuals = y - A @ beta
    return float(residuals[-1])         # most-recent residual


def _inverse_vol_weights(
    indices: NDArray, vols: NDArray, target_gross: float,
) -> NDArray:
    """Inverse-volatility weights summing to *target_gross* in abs value.

    *indices*: asset column indices for this leg.
    *vols*:    per-asset volatility (length = n_assets).
    *target_gross*:  total absolute weight budget (e.g. 0.5).
    """
    leg_vols = np.array([max(vols[i], 1e-10) for i in indices])
    inv_vol = 1.0 / leg_vols
    inv_vol /= inv_vol.sum()  # normalise to sum-1
    return inv_vol * target_gross


# ======================================================================
# Strategy
# ======================================================================

@register("equity_idiosyncratic_momentum")
class IdiosyncraticMomentumStrategy(Strategy):
    """Factor-neutral momentum: rank by residual return after removing
    market beta.

    Config params:
        lookback_days:   return lookback in trading days (default 252)
        skip_days:       skip most-recent N days to avoid reversal (default 21)
        min_history:     minimum bars required before generating signals (default 252)
        vol_lookback:    lookback for volatility estimation (default 63)
        rebalance_freq:  rebalance every N trading days (default 21)
        n_long_pct:      fraction of universe for long leg (default 0.10)
        n_short_pct:     fraction of universe for short leg (default 0.10)
        min_price:       minimum stock price filter (default 5.0)
        min_volume:      minimum average daily dollar volume filter (default 5e6)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        p = config.params
        self.lookback_days: int = int(p.get("lookback_days", 252))
        self.skip_days: int = int(p.get("skip_days", 21))
        self.min_history: int = int(p.get("min_history", 252))
        self.vol_lookback: int = int(p.get("vol_lookback", 63))
        self.rebalance_freq: int = int(p.get("rebalance_freq", 21))
        self.n_long_pct: float = float(p.get("n_long_pct", 0.10))
        self.n_short_pct: float = float(p.get("n_short_pct", 0.10))
        self.min_price: float = float(p.get("min_price", 5.0))
        self.min_volume: float = float(p.get("min_volume", 5e6))

        self._days_since_rebalance: int = 0
        self._current_weights: NDArray | None = None

    # ------------------------------------------------------------------
    # on_data — vectorised batch path (used by backtest engine)
    # ------------------------------------------------------------------
    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        volume = data.get("volume")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols: list[str] = list(
            data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        )

        # --- rebalance gate ------------------------------------------------
        self._days_since_rebalance += 1
        if (
            self._days_since_rebalance < self.rebalance_freq
            and self._current_weights is not None
        ):
            return _signals_from_weights(
                self._current_weights, symbols, self.config.max_position_notional,
            )

        # --- history gate --------------------------------------------------
        required = self.lookback_days + self.skip_days + 1
        if n_periods < required or n_periods < self.min_history:
            return []

        # --- universe filters ----------------------------------------------
        valid = np.ones(n_assets, dtype=bool)
        valid &= close[-1] >= self.min_price
        if volume is not None and volume.ndim == 2 and volume.shape[0] >= 21:
            avg_dollar_vol = np.mean(volume[-21:] * close[-21:], axis=0)
            valid &= avg_dollar_vol >= self.min_volume

        n_valid = int(np.sum(valid))
        if n_valid < 4:
            return []

        # --- compute 12-month cumulative returns (skip recent day) ---------
        end_idx = -self.skip_days - 1          # t - skip_days - 1
        start_idx = -self.lookback_days - 1    # t - lookback - 1
        # Guard against out-of-bounds if array is shorter than expected
        end_idx_abs = n_periods + end_idx
        start_idx_abs = n_periods + start_idx
        if start_idx_abs < 0 or end_idx_abs < 1:
            return []

        ret_stock = close[end_idx] / np.maximum(close[start_idx], 1e-10) - 1.0

        # --- market return (SPY) -------------------------------------------
        spy_idx = _find_spy_index(symbols)
        use_residual = spy_idx is not None

        if use_residual:
            spy_ret = float(ret_stock[spy_idx])  # SPY uses same formula
            # Expanding-window OLS regression for each stock
            # Build the full return series for the regression window
            # daily returns from start_idx to end_idx (inclusive of end_idx)
            close_window = close[start_idx:end_idx + 1]  # (T, n_assets)
            # Daily returns within that window
            daily_ret_window = close_window[1:] / np.maximum(close_window[:-1], 1e-10) - 1.0  # (T-1, n_assets)
            spy_daily = daily_ret_window[:, spy_idx]  # (T-1,)

            residuals = np.full(n_assets, np.nan)
            for i in range(n_assets):
                if not valid[i] or i == spy_idx:
                    continue
                stock_daily = daily_ret_window[:, i]
                residuals[i] = _ols_residual(stock_daily, spy_daily)

            # Fallback: if all residuals are NaN, use raw momentum
            if np.all(np.isnan(residuals[valid])):
                signal_values = ret_stock.copy()
            else:
                signal_values = residuals.copy()
        else:
            # No SPY available — fall back to raw momentum
            signal_values = ret_stock.copy()

        signal_values[~valid] = np.nan

        # --- cross-sectional z-score ---------------------------------------
        valid_vals = signal_values[valid]
        mu = np.nanmean(valid_vals)
        sigma = np.nanstd(valid_vals)
        if sigma < 1e-10:
            return []
        z_scores = np.full(n_assets, np.nan)
        z_scores[valid] = (signal_values[valid] - mu) / sigma

        # --- select long / short legs --------------------------------------
        valid_indices = np.where(valid)[0]
        sorted_idx = valid_indices[
            np.argsort(z_scores[valid_indices])
        ]  # ascending: worst first

        n_long = max(1, int(n_valid * self.n_long_pct))
        n_short = max(1, int(n_valid * self.n_short_pct))

        short_leg = sorted_idx[:n_short]
        long_leg = sorted_idx[-n_long:]

        # Remove overlap
        overlap = set(long_leg) & set(short_leg)
        if overlap:
            short_leg = np.array([i for i in short_leg if i not in overlap])
        if len(long_leg) == 0 or len(short_leg) == 0:
            return []

        # --- inverse-volatility weighting ----------------------------------
        if n_periods >= self.vol_lookback:
            daily_ret_full = close[1:] / np.maximum(close[:-1], 1e-10) - 1.0
            vols = np.nanstd(daily_ret_full[-self.vol_lookback:], axis=0)
        else:
            vols = np.ones(n_assets)

        long_w = _inverse_vol_weights(long_leg, vols, target_gross=0.5)
        short_w = _inverse_vol_weights(short_leg, vols, target_gross=0.5)

        weights = np.zeros(n_assets)
        for idx, w in zip(long_leg, long_w):
            weights[idx] = w
        for idx, w in zip(short_leg, short_w):
            weights[idx] = -w

        self._current_weights = weights
        self._days_since_rebalance = 0

        return _signals_from_weights(
            weights, symbols, self.config.max_position_notional,
        )


# ======================================================================
# Helpers
# ======================================================================

def _signals_from_weights(
    weights: NDArray,
    symbols: list[str],
    max_notional: float,
) -> list[TradeSignal]:
    """Convert a weight vector to a list of ``TradeSignal`` objects."""
    signals: list[TradeSignal] = []
    for i in range(len(weights)):
        if abs(weights[i]) < 1e-8:
            continue
        signals.append(
            TradeSignal(
                symbol=symbols[i],
                direction=1 if weights[i] > 0 else -1,
                strength=min(abs(weights[i]) * 2.0, 1.0),
                size=abs(weights[i]) * float(max_notional),
            )
        )
    return signals
