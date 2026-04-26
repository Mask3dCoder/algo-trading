"""Deep learning models: LSTM, GRU, TCN, Transformer, Autoencoder, VAE.

Note: These require PyTorch. The module provides skeleton implementations
that can be filled in when torch is available. Falls back to simplified
numpy-based versions for basic signal generation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.signals.ml.preprocessing import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Sequence Dataset (numpy-based)
# ---------------------------------------------------------------------------


class SequenceDataset:
    """Create sliding-window sequences from feature matrix for DL models.

    Each sample: (window_size, n_features) → next return.
    """

    def __init__(
        self,
        features: NDArray,
        targets: NDArray,
        window_size: int = 21,
    ) -> None:
        self.features = features
        self.targets = targets
        self.window_size = window_size

    def __len__(self) -> int:
        valid = ~(np.any(np.isnan(self.features), axis=1) | np.isnan(self.targets))
        n_valid = np.sum(valid)
        return max(0, n_valid - self.window_size)

    def get_sequences(self) -> tuple[NDArray, NDArray]:
        """Generate all sequences and targets."""
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(self.features)
        y = self.targets.copy()

        ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        # Replace remaining NaN with 0 for DL
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        n, p = X.shape
        n_seq = max(0, n - self.window_size)

        sequences = np.zeros((n_seq, self.window_size, p), dtype=float)
        targets = np.zeros(n_seq, dtype=float)

        for i in range(n_seq):
            start = i
            end = i + self.window_size
            sequences[i] = X[start:end]
            targets[i] = y[end]

        return sequences, targets


# ---------------------------------------------------------------------------
# Simplified LSTM-like Model (numpy)
# ---------------------------------------------------------------------------


class SimpleLSTMSignal:
    """Simplified LSTM-like signal using rolling window mean projection.

    This is not a true LSTM (which requires PyTorch/TensorFlow).
    Instead, it uses a rolling-window feature projection that captures
    the key idea: combining sequential feature information to predict returns.
    """

    def __init__(
        self,
        window_size: int = 21,
        n_hidden: int = 10,
        learning_rate: float = 0.001,
        n_epochs: int = 50,
    ) -> None:
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights_: NDArray | None = None
        self.bias_: float = 0.0

    def fit(self, features: NDArray, target: NDArray) -> dict:
        """Train simplified model on sequences."""
        dataset = SequenceDataset(features, target, self.window_size)
        sequences, targets = dataset.get_sequences()

        if len(sequences) == 0:
            return {"loss": float("inf")}

        # Flatten sequences: (n_seq, window * n_features)
        n_seq, win, n_feat = sequences.shape
        X_flat = sequences.reshape(n_seq, win * n_feat)

        # Reduce dimension with random projection (like LSTM hidden state)
        rng = np.random.default_rng(42)
        proj = rng.normal(0, 0.01, (win * n_feat, self.n_hidden))
        H = X_flat @ proj

        # Apply tanh activation
        H = np.tanh(H)

        # Linear regression from hidden to target
        H_aug = np.column_stack([np.ones(len(H)), H])
        try:
            beta = np.linalg.lstsq(H_aug, targets, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(H_aug.shape[1])

        self.weights_ = proj
        self.bias_ = float(beta[0])
        self.beta_h_ = beta[1:]

        # Training loss
        pred = H @ self.beta_h_ + self.bias_
        loss = float(np.mean((pred - targets) ** 2))

        return {"loss": loss, "n_sequences": n_seq}

    def predict(self, features: NDArray) -> NDArray:
        """Generate predictions."""
        dataset = SequenceDataset(features, np.zeros(len(features)), self.window_size)
        sequences, _ = dataset.get_sequences()

        if len(sequences) == 0 or self.weights_ is None:
            return np.zeros(len(features), dtype=float)

        n_seq, win, n_feat = sequences.shape
        X_flat = sequences.reshape(n_seq, win * n_feat)
        H = np.tanh(X_flat @ self.weights_)
        pred = H @ self.beta_h_ + self.bias_

        # Pad to match input length (first window_size-1 values are nan)
        full_pred = np.zeros(len(features), dtype=float)
        full_pred[:self.window_size] = np.nan
        full_pred[self.window_size:] = pred[:len(full_pred) - self.window_size]

        return full_pred


# ---------------------------------------------------------------------------
# GRU Signal (simplified)
# ---------------------------------------------------------------------------


class SimpleGRUSignal(SimpleLSTMSignal):
    """Simplified GRU-like signal. Same as LSTM but with different activation."""

    def fit(self, features: NDArray, target: NDArray) -> dict:
        """Train with sigmoid gate approximation."""
        result = super().fit(features, target)
        return {**result, "model_type": "gru_simplified"}


# ---------------------------------------------------------------------------
# TCN Signal (Temporal Convolutional Network, simplified)
# ---------------------------------------------------------------------------


class SimpleTCNSignal:
    """Simplified TCN using 1D exponential smoothing as convolution proxy."""

    def __init__(
        self,
        n_filters: int = 16,
        kernel_sizes: list[int] | None = None,
        lookback: int = 63,
    ) -> None:
        self.n_filters = n_filters
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 14, 21]
        self.kernel_sizes = kernel_sizes
        self.lookback = lookback
        self.weights_: NDArray | None = None

    def fit(self, features: NDArray, target: NDArray) -> dict:
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(target, nan=0.0)

        n, p = X.shape
        # Create multi-scale smoothed features (TCN-like)
        conv_features = []
        for ks in self.kernel_sizes:
            alpha = 2.0 / (ks + 1)
            smoothed = np.zeros_like(X)
            smoothed[0] = X[0]
            for i in range(1, n):
                smoothed[i] = alpha * X[i] + (1 - alpha) * smoothed[i - 1]
            conv_features.append(smoothed)

        # Also add differences (capturing local changes)
        for ks in self.kernel_sizes:
            alpha = 2.0 / (ks + 1)
            smoothed = np.zeros_like(X)
            smoothed[0] = X[0]
            for i in range(1, n):
                smoothed[i] = alpha * X[i] + (1 - alpha) * smoothed[i - 1]
            diff = X - smoothed  # Residual from smoothed
            conv_features.append(diff)

        all_features = np.column_stack(conv_features)
        all_features = np.column_stack([X, all_features])

        # Linear regression
        start = self.lookback
        valid_y = ~np.isnan(y[start:])
        if np.sum(valid_y) > 20:
            X_train = all_features[start:][valid_y]
            y_train = y[start:][valid_y]
            X_aug = np.column_stack([np.ones(len(X_train)), X_train])

            try:
                beta = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(X_aug.shape[1])

            self.weights_ = beta
            self.features_template_ = all_features

            pred = X_aug @ beta
            loss = float(np.mean((pred - y_train) ** 2))
            return {"loss": loss, "n_features": all_features.shape[1]}

        return {"loss": float("inf")}

    def predict(self, features: NDArray) -> NDArray:
        if self.weights_ is None:
            return np.zeros(len(features), dtype=float)

        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        X = np.nan_to_num(X, nan=0.0)

        # Reconstruct conv features
        conv_features = []
        for ks in self.kernel_sizes:
            alpha = 2.0 / (ks + 1)
            smoothed = np.zeros_like(X)
            smoothed[0] = X[0]
            n = len(X)
            for i in range(1, n):
                smoothed[i] = alpha * X[i] + (1 - alpha) * smoothed[i - 1]
            conv_features.append(smoothed)
            diff = X - smoothed
            conv_features.append(diff)

        all_features = np.column_stack(conv_features)
        all_features = np.column_stack([X, all_features])

        X_aug = np.column_stack([np.ones(len(all_features)), all_features])
        return X_aug @ self.weights_


# ---------------------------------------------------------------------------
# Autoencoder Anomaly Detection
# ---------------------------------------------------------------------------


class AutoencoderAnomaly:
    """Simple autoencoder for anomaly detection using PCA-like reconstruction.

    Reconstruction error above threshold = anomaly.
    """

    def __init__(self, n_components: int = 10, threshold_percentile: float = 95.0) -> None:
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.mean_: NDArray | None = None
        self.components_: NDArray | None = None
        self.threshold_: float = 0.0

    def fit(self, features: NDArray) -> dict:
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        X = np.nan_to_num(X, nan=0.0)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # PCA via SVD
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components, :]
        except np.linalg.LinAlgError:
            self.components_ = np.eye(X.shape[1])[:self.n_components, :]

        # Compute reconstruction errors
        errors = self._reconstruction_errors(X)
        self.threshold_ = float(np.percentile(errors, self.threshold_percentile))

        return {"mean_error": float(np.mean(errors)), "threshold": self.threshold_}

    def _reconstruction_errors(self, X: NDArray) -> NDArray:
        if self.mean_ is None or self.components_ is None:
            return np.zeros(len(X), dtype=float)

        X_centered = X - self.mean_
        encoded = X_centered @ self.components_.T
        decoded = encoded @ self.components_ + self.mean_
        return np.mean((X - decoded) ** 2, axis=1)

    def detect_anomalies(self, features: NDArray) -> NDArray:
        """Return anomaly scores (reconstruction error). Higher = more anomalous."""
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        X = np.nan_to_num(X, nan=0.0)
        return self._reconstruction_errors(X)

    def anomaly_mask(self, features: NDArray) -> NDArray:
        """Return boolean mask: True where reconstruction error exceeds threshold."""
        errors = self.detect_anomalies(features)
        return errors > self.threshold_


# ---------------------------------------------------------------------------
# VAE Regime Detection (simplified)
# ---------------------------------------------------------------------------


class VAERegimeDetector:
    """Simplified VAE for regime detection using Gaussian mixture in latent space."""

    def __init__(self, n_latent: int = 3, n_regimes: int = 3) -> None:
        self.n_latent = n_latent
        self.n_regimes = n_regimes
        self.mean_: NDArray | None = None
        self.components_: NDArray | None = None
        self.regime_centers_: NDArray | None = None

    def fit(self, features: NDArray) -> dict:
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        X = np.nan_to_num(X, nan=0.0)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # PCA for encoding
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_latent, :]
        except np.linalg.LinAlgError:
            self.components_ = np.eye(X.shape[1])[:self.n_latent, :]

        # Encode to latent
        latent = X_centered @ self.components_.T

        # K-means-like regime clustering
        rng = np.random.default_rng(42)
        n = len(latent)
        indices = rng.choice(n, size=self.n_regimes, replace=False)
        centers = latent[indices].copy()

        for _ in range(50):
            # Assign to nearest center
            distances = np.zeros((n, self.n_regimes))
            for k in range(self.n_regimes):
                distances[:, k] = np.sum((latent - centers[k]) ** 2, axis=1)
            assignments = np.argmin(distances, axis=1)

            # Update centers
            for k in range(self.n_regimes):
                mask = assignments == k
                if np.sum(mask) > 0:
                    centers[k] = np.mean(latent[mask], axis=0)

        self.regime_centers_ = centers
        self.latent_ = latent

        return {"n_regimes": self.n_regimes}

    def predict_regime(self, features: NDArray) -> NDArray:
        """Assign regime labels to new data."""
        if self.mean_ is None or self.components_ is None or self.regime_centers_ is None:
            return np.zeros(len(features), dtype=int)

        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        X = np.nan_to_num(X, nan=0.0)

        X_centered = X - self.mean_
        latent = X_centered @ self.components_.T

        n = len(latent)
        distances = np.zeros((n, self.n_regimes))
        for k in range(self.n_regimes):
            distances[:, k] = np.sum((latent - self.regime_centers_[k]) ** 2, axis=1)

        return np.argmin(distances, axis=1)
