"""ML-based residual signal predictor.

Uses Ridge regression to predict FUTURE residuals from PAST residuals.
This is a signal IMPROVEMENT, not a new strategy.

Features per asset:
1. Past residuals (1-10 day lags)
2. Residual volatility (10, 21, 63 day)
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.linear_model import Ridge

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, using numpy fallback")


class ResidualPredictor:
    """Predict next-period residual using Ridge regression.

    Features: past residuals (lags 1..n) + residual vol (10/21/63d).
    Target: next period's residual.
    One model per asset, fitted independently.
    """

    def __init__(self, n_lags: int = 10, alpha: float = 1.0):
        self.n_lags = n_lags
        self.alpha = alpha
        self.models: list[Ridge | None] = []
        self._n_features: int = n_lags + 3  # lags + 3 vol windows
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_asset_features(self, resid_col: NDArray) -> NDArray:
        """Build feature matrix for a single asset from its residual series.

        resid_col: (n_periods,) — may contain NaN at the start (regression warm-up).
        Returns: (n_periods, n_features) — zeros where insufficient history.
        """
        n = len(resid_col)
        X = np.zeros((n, self._n_features), dtype=np.float64)

        # Lag features (cols 0 .. n_lags-1)
        for lag in range(1, self.n_lags + 1):
            for t in range(lag, n):
                val = resid_col[t - lag]
                if not np.isnan(val):
                    X[t, lag - 1] = val

        # Residual volatility (cols n_lags .. n_lags+2): windows 10, 21, 63
        for j, window in enumerate([10, 21, 63]):
            for t in range(window, n):
                segment = resid_col[max(0, t - window + 1):t + 1]
                valid = segment[~np.isnan(segment)]
                if len(valid) >= max(3, window // 2):
                    X[t, self.n_lags + j] = np.std(valid, ddof=1)

        return X

    def _build_features(self, residuals: NDArray) -> NDArray:
        """Build feature matrix for all assets (horizontal stack per asset)."""
        n_periods, n_assets = residuals.shape
        blocks = []
        for i in range(n_assets):
            blocks.append(self._build_asset_features(residuals[:, i]))
        return np.hstack(blocks)  # (n_periods, n_assets * n_features)

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(self, residuals: NDArray) -> None:
        """Fit one Ridge model per asset to predict residual[t+1] from features[t]."""
        if not SKLEARN_AVAILABLE:
            self.is_fitted = False
            return

        n_periods, n_assets = residuals.shape
        X = self._build_features(residuals)  # (n_periods, n_assets * n_features)

        self.models = []
        for i in range(n_assets):
            y = residuals[1:, i]  # forward-shifted target
            # Select the feature columns for this asset
            start_col = i * self._n_features
            end_col = start_col + self._n_features
            X_asset = X[:-1, start_col:end_col]  # align with y (drop last row)

            # Build combined mask: both X and y must be non-NaN
            mask = ~np.isnan(y)
            mask &= ~np.any(np.isnan(X_asset), axis=1)

            if np.sum(mask) < 50:
                self.models.append(None)
                continue

            model = Ridge(alpha=self.alpha, fit_intercept=True)
            try:
                model.fit(X_asset[mask], y[mask])
                self.models.append(model)
            except Exception:
                self.models.append(None)

        self.is_fitted = len(self.models) > 0

    def predict(self, residuals: NDArray) -> NDArray:
        """Predict next residual for each asset from the latest feature row."""
        if not self.is_fitted:
            return (
                residuals[-1, :]
                if residuals.shape[0] > 0
                else np.zeros(residuals.shape[1])
            )

        n_assets = residuals.shape[1]
        predictions = np.zeros(n_assets, dtype=np.float64)

        X = self._build_features(residuals)
        X_last = X[-1, :]  # (n_assets * n_features,)

        for i in range(min(n_assets, len(self.models))):
            model = self.models[i]
            if model is None:
                predictions[i] = residuals[-1, i]
                continue
            start = i * self._n_features
            end = start + self._n_features
            feat = X_last[start:end].reshape(1, -1)
            if np.any(np.isnan(feat)):
                predictions[i] = residuals[-1, i]
                continue
            try:
                predictions[i] = float(model.predict(feat)[0])
            except Exception:
                predictions[i] = residuals[-1, i]

        return predictions
