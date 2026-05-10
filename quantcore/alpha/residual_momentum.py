"""Factor-neutral residual momentum — Q2 alpha rebuild skeleton.

ALGORITHM:
  1. Universe: 50+ liquid ETFs (ETF_MACRO).
  2. External factors: Fama-French 5 + rates/commodities.
  3. Rolling Ridge regression per asset: R = alpha + B*F + epsilon.
  4. Rank epsilon cross-sectionally; z-score weights.
     Residuals should be mean-zero and orthogonal to factors.

STATUS: SKELETON (Q2) — PlaceholderFactorSource returns zeros.
"""

from __future__ import annotations
import logging
import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


class PlaceholderFactorSource:
    """Zero-return placeholder. Replace with FamaFrenchLoader in Q2."""

    def get_factor_returns(self, n_periods: int, n_factors: int = 5) -> NDArray:
        _log.warning("PlaceholderFactorSource: returning zeros.")
        return np.zeros((n_periods, n_factors))


class ResidualMomentumSignal:
    """Factor-neutral residual momentum weights.

    Usage::
        signal = ResidualMomentumSignal()
        w = signal.compute(asset_returns, symbols)
    """

    def __init__(
        self,
        factor_source: PlaceholderFactorSource | None = None,
        ridge_alpha: float = 1.0,
    ) -> None:
        self._factor_source = factor_source or PlaceholderFactorSource()
        self.ridge_alpha = ridge_alpha

    def _ridge_residuals(
        self,
        asset_ret: NDArray,
        factor_ret: NDArray,
    ) -> NDArray:
        """Per-asset rolling Ridge regression -> residuals."""
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
            model.fit(factor_ret, asset_ret)
            return asset_ret - model.predict(factor_ret)
        except Exception as exc:
            _log.warning("ridge_failed: %s", exc)
            return np.zeros_like(asset_ret)

    def compute(
        self,
        asset_returns: NDArray,
        symbols: list[str],
        reg_window: int = 252,
        residual_window: int = 63,
        factor_returns: NDArray | None = None,
        vol_target: float | None = None,
    ) -> NDArray:
        """Compute factor-neutral residual momentum weights.

        Returns (N,) float array. |weights| sums to <= 1.0.
        Residuals are guaranteed orthogonal to the supplied factors.
        """
        T, N = asset_returns.shape
        if T < reg_window + residual_window:
            _log.warning("insufficient_data T=%d need=%d", T, reg_window + residual_window)
            return np.zeros(N)

        window_ret = asset_returns[-reg_window:]
        if factor_returns is not None:
            fac = factor_returns[-reg_window:]
        else:
            fac = self._factor_source.get_factor_returns(reg_window)

        residuals = np.column_stack([
            self._ridge_residuals(window_ret[:, i], fac)
            for i in range(N)
        ])  # shape (reg_window, N)

        recent = residuals[-residual_window:]
        last = recent[-1]
        mu = np.nanmean(recent, axis=0)
        sigma = np.nanstd(recent, axis=0, ddof=1)
        z = np.where(sigma > 1e-10, (last - mu) / sigma, 0.0)

        total_abs = float(np.sum(np.abs(z)))
        if total_abs < 1e-10:
            return np.zeros(N)
        weights = z / total_abs

        if vol_target and vol_target > 0:
            port = recent @ weights
            pv = float(np.nanstd(port, ddof=1)) * np.sqrt(252)
            if pv > 1e-10:
                weights *= min(vol_target / pv, 2.0)

        return weights
