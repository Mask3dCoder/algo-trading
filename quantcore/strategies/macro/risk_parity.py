"""Risk parity strategy: equal risk contribution, vol-targeted, regime overlay.

Allocates capital so each asset contributes equally to portfolio risk.
Vol-targets each asset and applies regime overlay (reduce risk in
high-vol regimes, increase in low-vol regimes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("macro_risk_parity")
class RiskParityStrategy(Strategy):
    """Equal risk contribution portfolio: vol-targeted + regime overlay.

    Config params:
        vol_target: annualized vol target per asset (default 0.10)
        lookback: lookback for covariance estimation (default 252)
        rebalance_freq: rebalance frequency in days (default 21)
        regime_overlay: enable regime-based scaling (default True)
        high_vol_threshold: above this percentile → reduce risk (default 75)
        low_vol_threshold: below this percentile → increase risk (default 25)
        max_leverage: max portfolio leverage (default 1.0)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.vol_target = float(config.params.get("vol_target", 0.10))
        self.lookback = int(config.params.get("lookback", 252))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.regime_overlay = bool(config.params.get("regime_overlay", True))
        self.high_vol_threshold = float(config.params.get("high_vol_threshold", 75))
        self.low_vol_threshold = float(config.params.get("low_vol_threshold", 25))
        self.max_leverage = float(config.params.get("max_leverage", 1.0))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def _estimate_cov(self, returns: NDArray) -> NDArray:
        """Estimate covariance with shrinkage (Ledoit-Wolf-like)."""
        n, p = returns.shape
        if n < 2:
            return np.eye(p) * 0.01

        S = np.cov(returns.T)
        if S.ndim == 0:
            return np.eye(p) * 0.01

        # Shrink toward diagonal
        diag = np.diag(np.diag(S))
        shrinkage = 0.2  # Fixed shrinkage intensity
        return (1 - shrinkage) * S + shrinkage * diag

    def _risk_parity_weights(self, cov: NDArray) -> NDArray:
        """Compute equal risk contribution weights iteratively."""
        n = cov.shape[0]
        if n == 0:
            return np.array([])

        # Initialize with inverse-vol weights
        vols = np.sqrt(np.diag(cov))
        vols[vols < 1e-10] = 1e-10
        w = 1.0 / vols
        w = w / np.sum(w)

        # Iterative ERC algorithm
        for _ in range(100):
            port_var = w @ cov @ w
            if port_var < 1e-20:
                break

            # Marginal risk contribution
            mrc = cov @ w
            # Risk contribution
            rc = w * mrc
            total_risk = np.sum(rc)

            if total_risk < 1e-20:
                break

            # Target: equal risk contribution
            target_rc = total_risk / n

            # Update weights proportional to target_rc / current_rc
            rc_ratio = np.where(rc > 1e-15, target_rc / rc, 1.0)
            w_new = w * rc_ratio**0.5  # Dampened update
            w_new = np.maximum(w_new, 1e-10)
            w_new = w_new / np.sum(w_new)

            if np.max(np.abs(w_new - w)) < 1e-8:
                w = w_new
                break
            w = w_new

        return w

    def _regime_scale(self, returns: NDArray) -> float:
        """Compute regime overlay scaling factor."""
        if not self.regime_overlay or len(returns) < 60:
            return 1.0

        # Rolling realized vol
        window = 63
        rolling_vols = []
        for i in range(window, len(returns)):
            chunk = returns[i - window:i]
            vol = np.std(np.sum(chunk, axis=1)) * np.sqrt(252)
            rolling_vols.append(vol)

        if not rolling_vols:
            return 1.0

        current_vol = rolling_vols[-1]
        float(np.percentile(rolling_vols, 100))  # Rank within history
        if len(rolling_vols) < 10:
            return 1.0

        # Percentile of current vol
        below = np.sum(np.array(rolling_vols) <= current_vol)
        pct = below / len(rolling_vols) * 100

        if pct > self.high_vol_threshold:
            # High vol regime: scale down
            scale = 0.5 + 0.5 * (100 - pct) / (100 - self.high_vol_threshold)
        elif pct < self.low_vol_threshold:
            # Low vol regime: scale up
            scale = 1.0 + 0.5 * (self.low_vol_threshold - pct) / self.low_vol_threshold
        else:
            scale = 1.0

        return np.clip(scale, 0.3, 1.5)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        if n_periods < self.lookback + 1:
            return []

        signals: list[TradeSignal] = []
        self._days_since_rebalance += 1

        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return signals

        # Compute returns
        returns = np.diff(close[-self.lookback:], axis=0) / close[-self.lookback:-1]

        # Estimate covariance
        cov = self._estimate_cov(returns)

        # Compute risk parity weights
        weights = self._risk_parity_weights(cov)

        # Vol-target scaling
        asset_vols = np.sqrt(np.diag(cov)) * np.sqrt(252)
        asset_vols[asset_vols < 1e-10] = 1e-10
        vol_scales = self.vol_target / asset_vols
        weights = weights * vol_scales

        # Regime overlay
        regime_scale = self._regime_scale(returns)
        weights = weights * regime_scale

        # Leverage constraint
        total_weight = np.sum(np.abs(weights))
        if total_weight > self.max_leverage:
            weights = weights * self.max_leverage / total_weight

        self._current_weights = weights
        self._days_since_rebalance = 0

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        for i in range(n_assets):
            if abs(weights[i]) > 1e-6:
                direction = 1 if weights[i] > 0 else -1
                signals.append(TradeSignal(
                    symbol=symbols[i],
                    direction=direction,
                    strength=min(abs(weights[i]) / (self.max_leverage / n_assets), 1.0),
                    size=abs(weights[i]) * float(self.config.max_position_notional),
                    metadata={"weight": float(weights[i]), "vol": float(asset_vols[i])},
                ))

        return signals
