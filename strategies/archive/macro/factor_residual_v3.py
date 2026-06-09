"""Factor Residual Strategy V3 - Trade ONLY non-factor residuals.

KEY INSIGHT: Factor assets (SPY=MK, DBC=INFL, etc.) have near-zero residuals.
Solution: Remove them from the tradable universe. Only trade assets with meaningful residuals.

Pipeline:
  1. Rolling ridge regression → residuals for ALL assets
  2. Compute zscore(residual) as signal
  3. REMOVE factor assets from trading (set weight=0)
  4. Simple z-score weighting on remaining assets
  5. Vol scaling for risk control
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


# Factor definitions - assets to EXCLUDE from trading
FACTOR_ASSETS = {"SPY", "TLT", "DBC", "GLD"}


@register("macro_factor_residual_v3")
class FactorResidualStrategyV3(Strategy):
    """Trade ONLY non-factor residuals.
    
    Factor assets have near-zero residuals (R² >99%).
    Only trade non-factor assets where residuals have meaningful variation.
    
    Config params:
        rebalance_freq: days between rebalances (default 5)
        reg_window: rolling window for factor regression (default 252)
        ridge_alpha: ridge regularization strength (default 1e-6)
        residual_window: window for z-scoring residuals (default 63)
        vol_lookback: realized vol estimation window (default 20)
        min_weight: minimum abs weight to emit signal (default 0.01)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.reg_window = int(config.params.get("reg_window", 252))
        self.ridge_alpha = float(config.params.get("ridge_alpha", 1e-6))
        self.residual_window = int(config.params.get("residual_window", 63))
        self.vol_lookback = int(config.params.get("vol_lookback", 20))
        self.min_weight = float(config.params.get("min_weight", 0.01))

        self._days_since_rebalance = 0
        self._factor_map: dict[str, tuple[int, int | None, float]] = {}
        self._last_betas: NDArray | None = None

    def _init_factor_map(self, symbols: list[str]) -> None:
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        self._factor_map = {}
        # Store factor definitions
        factor_defs = {
            "MKT": ("SPY", None, 1.0),
            "RATES": ("TLT", "SPY", 1.0),
            "INFL": ("DBC", None, 1.0),
            "SAFE": ("GLD", "TLT", 0.5),
        }
        for fname, (sym1, sym2, scale) in factor_defs.items():
            idx1 = sym_to_idx.get(sym1, -1)
            idx2 = sym_to_idx.get(sym2) if sym2 else None
            if idx1 >= 0:
                self._factor_map[fname] = (idx1, idx2, scale)

    def _build_factors(self, rets: NDArray) -> NDArray:
        n_periods = rets.shape[0]
        factors = np.zeros((n_periods, len(self._factor_map)))
        for j, (fname, (idx1, idx2, scale)) in enumerate(
            sorted(self._factor_map.items())
        ):
            if idx2 is not None:
                factors[:, j] = (rets[:, idx1] - rets[:, idx2]) * scale
            else:
                factors[:, j] = rets[:, idx1] * scale
        return factors

    def _compute_residuals(
        self, rets: NDArray, factors: NDArray,
    ) -> NDArray:
        n_periods, n_assets = rets.shape
        n_factors = factors.shape[1]
        residuals = np.full((n_periods, n_assets), np.nan)

        if n_periods < self.reg_window + 1:
            return residuals

        reg = self.reg_window
        alpha = self.ridge_alpha
        I_reg = alpha * np.eye(n_factors)

        for t in range(reg, n_periods):
            X_train = factors[t - reg:t]
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0, ddof=1)
            X_std_safe = np.maximum(X_std, 1e-10)
            X_train_std = (X_train - X_mean) / X_std_safe
            X_t_std = (factors[t] - X_mean) / X_std_safe

            try:
                XtX_inv = np.linalg.inv(X_train_std.T @ X_train_std + I_reg)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(X_train_std.T @ X_train_std)

            for i in range(n_assets):
                y_train = rets[t - reg:t, i]
                if np.std(y_train) < 1e-10:
                    residuals[t, i] = 0.0
                    continue
                # BUGFIX: Add intercept by demeaning y_train and adding mean back to pred
                y_mean = np.mean(y_train)
                y_train_dm = y_train - y_mean
                beta = XtX_inv @ (X_train_std.T @ y_train_dm)
                y_pred = float(X_t_std @ beta) + y_mean
                residuals[t, i] = float(rets[t, i]) - y_pred

        return residuals

    def _compute_signals(
        self, residuals: NDArray, symbols: list[str],
    ) -> NDArray:
        """Compute signal = zscore(residual) for NON-factor assets only."""
        n_periods, n_assets = residuals.shape
        latest = n_periods - 1
        window = min(self.residual_window, n_periods - 1)

        signals = np.zeros(n_assets)

        for i in range(n_assets):
            # SKIP factor assets - they have near-zero residuals
            if i < len(symbols) and symbols[i] in FACTOR_ASSETS:
                signals[i] = 0.0
                continue

            resid_series = residuals[:latest + 1, i]
            valid = resid_series[~np.isnan(resid_series)]

            if len(valid) < 5:
                signals[i] = 0.0
                continue

            window_data = valid[-window:] if len(valid) >= window else valid
            std = np.std(window_data)
            if std < 1e-10:
                signals[i] = 0.0
                continue

            # BUGFIX: Use trailing average of residuals (smoothing), not single-day noise
            smooth_window = min(5, len(window_data))
            current = np.mean(window_data[-smooth_window:])

            # Signal: residual MOMENTUM (positive residual → long)
            signals[i] = current / std

        return signals

    def _compute_weights(
        self, signals: NDArray, rets: NDArray, symbols: list[str],
    ) -> NDArray:
        """Z-score weighting, ONLY on non-factor assets."""
        n_assets = len(signals)

        if n_assets == 0:
            return np.zeros(0)

        # BUGFIX: Exclude factor assets from z-score computation entirely
        non_factor_idx = [
            i for i in range(n_assets)
            if i < len(symbols) and symbols[i] not in FACTOR_ASSETS
        ]
        if len(non_factor_idx) < 2:
            return np.zeros(n_assets)

        non_factor_signals = np.array([signals[i] for i in non_factor_idx])

        sig_mean = np.mean(non_factor_signals)
        sig_std = np.std(non_factor_signals, ddof=1)
        if sig_std < 1e-10:
            return np.zeros(n_assets)

        weights = np.zeros(n_assets)
        for j, i in enumerate(non_factor_idx):
            weights[i] = (non_factor_signals[j] - sig_mean) / sig_std

        # Vol scaling
        if rets.shape[0] >= self.vol_lookback + 2:
            vol = np.std(rets[-self.vol_lookback:], axis=0, ddof=1) * np.sqrt(252)
            vol = np.maximum(vol, 1e-10)
            weights = weights / vol

        # BUGFIX: Re-center ONLY non-factor assets (market neutral)
        non_factor_weights = np.array([weights[i] for i in non_factor_idx])
        re_center = np.mean(non_factor_weights)
        for i in non_factor_idx:
            weights[i] -= re_center

        # Normalize
        max_abs = np.max(np.abs(weights))
        if max_abs > 1e-10:
            weights = weights / max_abs

        return weights

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        if not self._factor_map:
            self._init_factor_map(symbols)

        if n_periods < self.reg_window + 1:
            return []

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        daily_rets = (
            (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
        )

        factors = self._build_factors(daily_rets)
        residuals = self._compute_residuals(daily_rets, factors)
        signals = self._compute_signals(residuals, symbols)
        weights = self._compute_weights(signals, daily_rets, symbols)

        trade_signals: list[TradeSignal] = []
        for i in range(n_assets):
            w = float(weights[i])
            if abs(w) < self.min_weight:
                continue

            trade_signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if w > 0 else -1,
                strength=abs(w),
                size=abs(w) * float(self.config.max_position_notional),
                metadata={
                    "weight": round(w, 4),
                    "signal": round(float(signals[i]), 4),
                    "residual": round(float(residuals[-1, i]), 6)
                    if not np.isnan(residuals[-1, i]) else 0.0,
                },
            ))

        self._days_since_rebalance = 0
        return trade_signals
