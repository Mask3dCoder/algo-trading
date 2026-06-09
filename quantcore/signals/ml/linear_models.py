"""Linear ML models: LASSO, Ridge, Elastic Net, quantile regression for signal generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantcore.signals.ml.preprocessing import PreprocessingPipeline, TimeSeriesSplit

# ---------------------------------------------------------------------------
# OLS Utilities
# ---------------------------------------------------------------------------


def _ols_fit(X: NDArray, y: NDArray, alpha: float = 0.0, l1_ratio: float = 0.0) -> NDArray:
    """Fit regularized linear regression via coordinate descent.

    alpha: total regularization strength.
    l1_ratio: 0=Ridge, 1=LASSO, between=ElasticNet.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1 - l1_ratio) if (l1_ratio < 1.0) else 0.0

    # Standardize X internally for coordinate descent
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=1)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_centered = y - y_mean

    if l1_penalty == 0 and l2_penalty == 0:
        # Pure OLS
        beta = np.linalg.lstsq(X_norm, y_centered, rcond=None)[0]
    elif l1_penalty > 0:
        # Coordinate descent for LASSO/ElasticNet
        beta = _coordinate_descent(X_norm, y_centered, l1_penalty, l2_penalty, max_iter=1000, tol=1e-6)
    else:
        # Ridge: (X'X + lambda*I)^-1 X'y
        XtX = X_norm.T @ X_norm
        XtY = X_norm.T @ y_centered
        beta = np.linalg.solve(XtX + l2_penalty * np.eye(p), XtY)

    # Convert back to original scale
    beta_original = beta / X_std
    intercept = y_mean - X_mean @ beta_original

    return np.concatenate([[intercept], beta_original])


def _coordinate_descent(
    X: NDArray,
    y: NDArray,
    l1: float,
    l2: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> NDArray:
    """Coordinate descent for ElasticNet."""
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    residual = y.copy()

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Remove effect of feature j
            residual += X[:, j] * beta[j]

            # Compute partial derivative
            rho = X[:, j] @ residual / n
            z = (X[:, j] ** 2).sum() / n

            # Soft thresholding
            if z > 0:
                if rho > l1:
                    beta[j] = (rho - l1) / (z + l2)
                elif rho < -l1:
                    beta[j] = (rho + l1) / (z + l2)
                else:
                    beta[j] = 0.0

            # Update residual
            residual -= X[:, j] * beta[j]

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


# ---------------------------------------------------------------------------
# Signal Model Base
# ---------------------------------------------------------------------------


@dataclass
class LinearSignalResult:
    """Result from a linear model fit."""

    coefficients: NDArray
    intercept: float
    ic_train: float
    ic_test: float
    n_features_used: int
    selected_features: list[int] | None = None


class LassoSignal:
    """LASSO regression for signal generation.

    Uses time-series CV to select optimal alpha.
    """

    def __init__(
        self,
        alpha_range: list[float] | None = None,
        cv_splits: int = 5,
        max_features: int | None = None,
    ) -> None:
        if alpha_range is None:
            alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0]
        self.alpha_range = alpha_range
        self.cv_splits = cv_splits
        self.max_features = max_features
        self.beta_: NDArray | None = None
        self.best_alpha_: float = 0.0
        self._pipeline: PreprocessingPipeline | None = None

    def fit(
        self,
        features: NDArray,
        target: NDArray,
    ) -> LinearSignalResult:
        """Fit LASSO with CV to select alpha."""
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        self._pipeline = pipeline  # Store fitted pipeline for prediction

        splitter = TimeSeriesSplit(n_splits=self.cv_splits)
        folds = splitter.split(len(X))

        best_ic = -np.inf
        best_alpha = self.alpha_range[0]

        for alpha in self.alpha_range:
            ics = []
            for train_idx, test_idx in folds:
                X_train, y_train = X[train_idx], target[train_idx]
                X_test, y_test = X[test_idx], target[test_idx]

                valid_train = ~(np.any(np.isnan(X_train), axis=1) | np.isnan(y_train))
                valid_test = ~(np.any(np.isnan(X_test), axis=1) | np.isnan(y_test))

                if np.sum(valid_train) < 20 or np.sum(valid_test) < 5:
                    continue

                beta = _ols_fit(X_train[valid_train], y_train[valid_train], alpha=alpha, l1_ratio=1.0)

                pred = X_test[valid_test] @ beta[1:] + beta[0]
                ic = np.corrcoef(pred, y_test[valid_test])[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)

            if ics:
                mean_ic = np.mean(ics)
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_alpha = alpha

        # Final fit with best alpha
        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        beta = _ols_fit(X[valid], target[valid], alpha=best_alpha, l1_ratio=1.0)

        self.beta_ = beta
        self.best_alpha_ = best_alpha

        # Count non-zero features
        feature_coefs = beta[1:]
        selected = list(np.where(np.abs(feature_coefs) > 1e-8)[0])
        if self.max_features and len(selected) > self.max_features:
            # Keep top features by absolute coefficient
            abs_coefs = np.abs(feature_coefs[selected])
            top_k = np.argsort(abs_coefs)[::-1][:self.max_features]
            selected = [selected[i] for i in top_k]

        pred_train = X[valid] @ beta[1:] + beta[0]
        ic_train = float(np.corrcoef(pred_train, target[valid])[0, 1])
        if np.isnan(ic_train):
            ic_train = 0.0

        return LinearSignalResult(
            coefficients=beta,
            intercept=float(beta[0]),
            ic_train=ic_train,
            ic_test=best_ic,
            n_features_used=len(selected),
            selected_features=selected,
        )

    def predict(self, features: NDArray) -> NDArray:
        """Generate signal predictions."""
        if self.beta_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self._pipeline.transform(features)
        return X @ self.beta_[1:] + self.beta_[0]


class RidgeSignal:
    """Ridge regression for signal generation."""

    def __init__(
        self,
        alpha_range: list[float] | None = None,
        cv_splits: int = 5,
    ) -> None:
        if alpha_range is None:
            alpha_range = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.alpha_range = alpha_range
        self.cv_splits = cv_splits
        self.beta_: NDArray | None = None
        self.best_alpha_: float = 0.0
        self._pipeline: PreprocessingPipeline | None = None

    def fit(self, features: NDArray, target: NDArray) -> LinearSignalResult:
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        self._pipeline = pipeline
        splitter = TimeSeriesSplit(n_splits=self.cv_splits)
        folds = splitter.split(len(X))

        best_ic = -np.inf
        best_alpha = self.alpha_range[0]

        for alpha in self.alpha_range:
            ics = []
            for train_idx, test_idx in folds:
                X_train, y_train = X[train_idx], target[train_idx]
                X_test, y_test = X[test_idx], target[test_idx]
                valid_train = ~(np.any(np.isnan(X_train), axis=1) | np.isnan(y_train))
                valid_test = ~(np.any(np.isnan(X_test), axis=1) | np.isnan(y_test))
                if np.sum(valid_train) < 20 or np.sum(valid_test) < 5:
                    continue
                beta = _ols_fit(X_train[valid_train], y_train[valid_train], alpha=alpha, l1_ratio=0.0)
                pred = X_test[valid_test] @ beta[1:] + beta[0]
                ic = np.corrcoef(pred, y_test[valid_test])[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
            if ics:
                mean_ic = np.mean(ics)
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_alpha = alpha

        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        beta = _ols_fit(X[valid], target[valid], alpha=best_alpha, l1_ratio=0.0)
        self.beta_ = beta
        self.best_alpha_ = best_alpha

        pred_train = X[valid] @ beta[1:] + beta[0]
        ic_train = float(np.corrcoef(pred_train, target[valid])[0, 1])
        if np.isnan(ic_train):
            ic_train = 0.0

        return LinearSignalResult(
            coefficients=beta,
            intercept=float(beta[0]),
            ic_train=ic_train,
            ic_test=best_ic,
            n_features_used=int(np.sum(np.abs(beta[1:]) > 1e-8)),
        )

    def predict(self, features: NDArray) -> NDArray:
        if self.beta_ is None:
            raise RuntimeError("Model not fitted.")
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self._pipeline.transform(features)
        return X @ self.beta_[1:] + self.beta_[0]


class ElasticNetSignal:
    """Elastic Net regression (L1 + L2)."""

    def __init__(
        self,
        alpha_range: list[float] | None = None,
        l1_ratio: float = 0.5,
        cv_splits: int = 5,
    ) -> None:
        if alpha_range is None:
            alpha_range = [0.01, 0.1, 1.0]
        self.alpha_range = alpha_range
        self.l1_ratio = l1_ratio
        self.cv_splits = cv_splits
        self.beta_: NDArray | None = None
        self.best_alpha_: float = 0.0
        self._pipeline: PreprocessingPipeline | None = None

    def fit(self, features: NDArray, target: NDArray) -> LinearSignalResult:
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        self._pipeline = pipeline
        splitter = TimeSeriesSplit(n_splits=self.cv_splits)
        folds = splitter.split(len(X))

        best_ic = -np.inf
        best_alpha = self.alpha_range[0]

        for alpha in self.alpha_range:
            ics = []
            for train_idx, test_idx in folds:
                X_train, y_train = X[train_idx], target[train_idx]
                X_test, y_test = X[test_idx], target[test_idx]
                valid_train = ~(np.any(np.isnan(X_train), axis=1) | np.isnan(y_train))
                valid_test = ~(np.any(np.isnan(X_test), axis=1) | np.isnan(y_test))
                if np.sum(valid_train) < 20 or np.sum(valid_test) < 5:
                    continue
                beta = _ols_fit(X_train[valid_train], y_train[valid_train],
                               alpha=alpha, l1_ratio=self.l1_ratio)
                pred = X_test[valid_test] @ beta[1:] + beta[0]
                ic = np.corrcoef(pred, y_test[valid_test])[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
            if ics:
                mean_ic = np.mean(ics)
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_alpha = alpha

        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        beta = _ols_fit(X[valid], target[valid], alpha=best_alpha, l1_ratio=self.l1_ratio)
        self.beta_ = beta
        self.best_alpha_ = best_alpha

        pred_train = X[valid] @ beta[1:] + beta[0]
        ic_train = float(np.corrcoef(pred_train, target[valid])[0, 1])
        if np.isnan(ic_train):
            ic_train = 0.0

        return LinearSignalResult(
            coefficients=beta,
            intercept=float(beta[0]),
            ic_train=ic_train,
            ic_test=best_ic,
            n_features_used=int(np.sum(np.abs(beta[1:]) > 1e-8)),
        )

    def predict(self, features: NDArray) -> NDArray:
        if self.beta_ is None:
            raise RuntimeError("Model not fitted.")
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self._pipeline.transform(features)
        return X @ self.beta_[1:] + self.beta_[0]


class QuantileRegression:
    """Quantile regression via iterative least squares (IZACKE)."""

    def __init__(self, quantile: float = 0.05, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.quantile = quantile
        self.max_iter = max_iter
        self.tol = tol
        self.beta_: NDArray | None = None
        self._pipeline: PreprocessingPipeline | None = None

    def fit(self, features: NDArray, target: NDArray) -> NDArray:
        """Fit quantile regression using iterative reweighted least squares."""
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        self._pipeline = pipeline
        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        X_clean = X[valid]
        y_clean = target[valid]

        # Initialize with OLS
        beta = _ols_fit(X_clean, y_clean, alpha=0.0)
        q = self.quantile

        for _ in range(self.max_iter):
            pred = X_clean @ beta[1:] + beta[0]
            residuals = y_clean - pred

            # Weights: high weight for residuals with wrong sign
            weights = np.where(residuals >= 0, q, 1.0 - q)
            weights = np.maximum(weights, 1e-6)

            # Weighted least squares
            W = np.diag(weights)
            X_aug = np.column_stack([np.ones(len(X_clean)), X_clean])
            try:
                beta_new = np.linalg.solve(X_aug.T @ W @ X_aug, X_aug.T @ W @ y_clean)
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        self.beta_ = beta
        return beta

    def predict(self, features: NDArray) -> NDArray:
        if self.beta_ is None:
            raise RuntimeError("Model not fitted.")
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self._pipeline.transform(features)
        return X @ self.beta_[1:] + self.beta_[0]
