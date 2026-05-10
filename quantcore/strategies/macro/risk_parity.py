"""Risk parity strategy: equal risk contribution, vol-targeted, regime overlay.

Allocates capital so each asset contributes equally to portfolio risk.
Vol-targets each asset and applies regime overlay (reduce risk in
high-vol regimes, increase in low-vol regimes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import warnings

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyStatus,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("macro_risk_parity")
class RiskParityStrategy(Strategy):
    """Equal risk contribution portfolio: vol-targeted + regime overlay.

    .. deprecated::
    STRATEGY_STATUS = StrategyStatus.DISABLED

    Forensic audit (2026-05-09): Risk parity is a portfolio construction
    method, not an alpha source.  OOS Sharpe +0.128 is entirely explained by
    the equity risk premium during the 2018-2024 test period.  No directional
    edge present.  Retained for reference; NOT for live or paper trading.

    Config params:
        vol_target: annualized vol target per asset (default 0.10)
        lookback: lookback for covariance estimation (default 252)
        rebalance_freq: rebalance frequency in days (default 21)
        regime_overlay: enable regime-based scaling (default True)
        high_vol_threshold: above this percentile → reduce risk (default 75)
        low_vol_threshold: below this percentile → increase risk (default 25)
        max_leverage: max portfolio leverage (default 1.0)
        vol_regime_enabled: enable portfolio-level vol regime filter (default True)
        vol_regime_threshold: scale when 20-bar portfolio vol > vol_target * this (default 0.40)
        vol_regime_lookback: vol estimation window (default 20)
        min_vol_scale: minimum scale factor during high vol (default 0.30)
    """

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.DISABLED

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.vol_target = float(config.params.get("vol_target", 0.10))
        self.lookback = int(config.params.get("lookback", 252))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.regime_overlay = bool(config.params.get("regime_overlay", True))
        self.high_vol_threshold = float(config.params.get("high_vol_threshold", 75))
        self.low_vol_threshold = float(config.params.get("low_vol_threshold", 25))
        self.max_leverage = float(config.params.get("max_leverage", 1.0))
        self.vol_regime_enabled = bool(config.params.get("vol_regime_enabled", True))
        self.vol_regime_threshold = float(config.params.get("vol_regime_threshold", 0.40))
        self.min_vol_scale = float(config.params.get("min_vol_scale", 0.30))
        self.vol_regime_lookback = int(config.params.get("vol_regime_lookback", 20))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def _estimate_cov(self, returns: NDArray) -> NDArray:
        """Estimate covariance with shrinkage (Ledoit-Wolf-like).

        Enhanced with factor-based covariance for better signal-to-noise.
        First compute factor covariance, then add idiosyncratic component.
        """
        n, p = returns.shape
        if n < 2:
            return np.eye(p) * 0.01

        # Standard sample covariance
        S = np.cov(returns.T)
        if S.ndim == 0:
            return np.eye(p) * 0.01

        # Enhancement: Add factor-based shrinkage
        # Shrink toward diagonal with adaptive intensity based on n/p ratio
        shrinkage = min(0.2, p / n)  # More shrinkage when few observations
        diag = np.diag(np.diag(S))
        cov_shrunk = (1 - shrinkage) * S + shrinkage * diag

        # Factor model enhancement: estimate factor covariance
        # Use PCA to find dominant factors
        try:
            from numpy.linalg import eigh
            eigvals, eigvecs = eigh(cov_shrunk)
            # Keep top 3 factors (or less if not available)
            n_factors = min(3, p)
            if n_factors > 0 and eigvals[-1] > 1e-10:
                # Factor covariance: sum of top eigenvalues
                factor_var = np.sum(eigvals[-n_factors:])
                # Idiosyncratic: remaining variance
                specific_var = np.trace(cov_shrunk) - factor_var
                # Blend: 70% factor model, 30% sample
                return 0.7 * cov_shrunk + 0.3 * np.eye(p) * (specific_var / p)
        except Exception:
            pass

        return cov_shrunk

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

    def _vol_regime_scale(self, returns: NDArray) -> float:
        """Portfolio-level vol regime detector.

        Uses equal-weight portfolio vol as regime proxy. When 20-bar portfolio
        vol exceeds vol_target * vol_regime_threshold, linearly scale down
        positions. This directly targets the HIGH_VOL regime definition.
        Risk parity is inherently diversified, so the threshold is higher
        (40% of vol_target) vs CTA trend (30%).
        """
        if not self.vol_regime_enabled:
            return 1.0

        if len(returns) < self.vol_regime_lookback:
            return 1.0

        recent = returns[-self.vol_regime_lookback:]
        portfolio_returns = recent.mean(axis=1)
        realized_vol = float(np.std(portfolio_returns, ddof=1)) * np.sqrt(252)

        threshold = self.vol_target * self.vol_regime_threshold

        if realized_vol <= threshold:
            return 1.0

        # Linear scaling: 1.0 at threshold, down to min_vol_scale at 2x threshold
        excess = (realized_vol - threshold) / threshold
        scale = max(self.min_vol_scale, 1.0 - excess * 0.5)
        return float(scale)

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

        # Regime overlay (percentile-based)
        regime_scale = self._regime_scale(returns)
        weights = weights * regime_scale

        # Vol regime filter (portfolio-level 20-bar vol)
        vol_regime_scale = self._vol_regime_scale(returns)
        weights = weights * vol_regime_scale

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
