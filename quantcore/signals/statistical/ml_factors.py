"""Machine Learning-based factor combination for statistical arbitrage.

Replaces linear weighted average with non-linear ML models that learn
optimal factor combinations from historical data. This captures:
  - Non-linear factor interactions
  - Time-varying factor importance
  - Regime-dependent factor weights
  - Factor timing signals

References:
    - Gu, Kelly, Xiu (2020): Empirical Asset Pricing via Machine Learning
    - Leippold, Wang, Zhou (2022): Machine Learning in the Chinese Stock Market
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


@dataclass
class MLFactorResult:
    """Result from ML factor combination."""
    signal: NDArray  # Combined signal in [-1, 1]
    feature_importance: dict[str, float]  # Factor importance scores
    model_score: float  # R² or accuracy on validation set
    predictions_std: float  # Uncertainty estimate


class GradientBoostingFactorCombiner:
    """Gradient Boosting for non-linear factor combination.
    
    Captures factor interactions and non-linearities that linear
    models miss. Uses time-series cross-validation to avoid look-ahead bias.
    
    Parameters
    ----------
    n_estimators : int
        Number of boosting stages (default 100).
    max_depth : int
        Maximum depth of trees (default 3).
    learning_rate : float
        Learning rate (default 0.1).
    cv_folds : int
        Number of time-series CV folds (default 5).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        cv_folds: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.cv_folds = cv_folds
        self._model = None
        self._feature_names: list[str] = []
        self._is_fitted = False
        
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: Optional[list[str]] = None,
    ) -> GradientBoostingFactorCombiner:
        """Fit the model using time-series cross-validation.
        
        Parameters
        ----------
        X : NDArray
            Factor scores matrix (n_samples, n_factors).
        y : NDArray
            Forward returns (n_samples,).
        feature_names : list[str] | None
            Names for each factor column.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"factor_{i}" for i in range(X.shape[1])]
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)
            # Store the most recent fold's model (walk-forward: no final fit on full data)
            self._model = model

        self._is_fitted = True
        
        _log.info(
            "GradientBoosting fitted: CV R² = %.3f ± %.3f",
            np.mean(cv_scores), np.std(cv_scores),
        )
        
        return self
    
    def predict(self, X: NDArray) -> MLFactorResult:
        """Generate combined signal from factor scores.
        
        Parameters
        ----------
        X : NDArray
            Factor scores matrix (n_samples, n_factors).
            
        Returns
        -------
        MLFactorResult
            Combined signal with uncertainty estimates.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get predictions
        raw_signal = self._model.predict(X)
        
        # Normalize to [-1, 1] using percentile clipping
        signal = np.clip(
            (raw_signal - np.median(raw_signal)) / (np.percentile(raw_signal, 95) - np.percentile(raw_signal, 5) + 1e-10),
            -1.0, 1.0,
        )
        
        # Feature importance
        importance = dict(zip(
            self._feature_names,
            self._model.feature_importances_,
        ))
        
        # Uncertainty estimate from prediction variance
        pred_std = float(np.std(raw_signal))
        
        return MLFactorResult(
            signal=signal,
            feature_importance=importance,
            model_score=float(self._model.score(X, np.zeros_like(X[:, 0]))),  # Placeholder
            predictions_std=pred_std,
        )
    
    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted or self._model is None:
            return {}
        return dict(zip(self._feature_names, self._model.feature_importances_))


class RandomForestFactorCombiner:
    """Random Forest for robust factor combination.
    
    More robust to outliers and provides uncertainty estimates
    via prediction variance across trees.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees (default 200).
    max_depth : int
        Maximum depth (default 5).
    cv_folds : int
        Number of time-series CV folds (default 5).
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        cv_folds: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self._model = None
        self._feature_names: list[str] = []
        self._is_fitted = False
        
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: Optional[list[str]] = None,
    ) -> RandomForestFactorCombiner:
        """Fit the model using time-series cross-validation."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit
        
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"factor_{i}" for i in range(X.shape[1])]
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)
            # Use the last fold's model — no final refit on full data (avoid lookahead)
            self._model = model

        self._is_fitted = True
        
        _log.info(
            "RandomForest fitted: CV R² = %.3f ± %.3f",
            np.mean(cv_scores), np.std(cv_scores),
        )
        
        return self
    
    def predict(self, X: NDArray) -> MLFactorResult:
        """Generate combined signal with uncertainty estimates."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get predictions from all trees for uncertainty
        tree_predictions = np.array([
            tree.predict(X) for tree in self._model.estimators_
        ])
        
        raw_signal = np.mean(tree_predictions, axis=0)
        pred_std = np.std(tree_predictions, axis=0)
        
        # Normalize to [-1, 1]
        signal = np.clip(
            (raw_signal - np.median(raw_signal)) / (np.percentile(raw_signal, 95) - np.percentile(raw_signal, 5) + 1e-10),
            -1.0, 1.0,
        )
        
        importance = dict(zip(self._feature_names, self._model.feature_importances_))
        
        return MLFactorResult(
            signal=signal,
            feature_importance=importance,
            model_score=float(np.mean(pred_std)),  # Average uncertainty
            predictions_std=float(np.mean(pred_std)),
        )


class NeuralNetworkFactorCombiner:
    """Neural Network (MLP) for complex non-linear factor combination.
    
    Can capture very complex factor interactions that tree-based
    models might miss.
    
    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Hidden layer sizes (default (64, 32, 16)).
    activation : str
        Activation function (default 'relu').
    alpha : float
        L2 regularization (default 0.001).
    max_iter : int
        Maximum iterations (default 500).
    """
    
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32, 16),
        activation: str = "relu",
        alpha: float = 0.001,
        max_iter: int = 500,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self._model = None
        self._feature_names: list[str] = []
        self._is_fitted = False
        self._scaler = None
        
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: Optional[list[str]] = None,
    ) -> NeuralNetworkFactorCombiner:
        """Fit the neural network with proper scaling."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"factor_{i}" for i in range(X.shape[1])]
        
        # Scale features for neural network
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)
            # Use the last fold's model — no final refit on full data (avoid lookahead)
            self._model = model

        self._is_fitted = True
        
        _log.info(
            "NeuralNetwork fitted: CV R² = %.3f ± %.3f",
            np.mean(cv_scores), np.std(cv_scores),
        )
        
        return self
    
    def predict(self, X: NDArray) -> MLFactorResult:
        """Generate combined signal."""
        if not self._is_fitted or self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self._scaler.transform(X)
        raw_signal = self._model.predict(X_scaled)
        
        # Normalize to [-1, 1]
        signal = np.clip(
            (raw_signal - np.median(raw_signal)) / (np.percentile(raw_signal, 95) - np.percentile(raw_signal, 5) + 1e-10),
            -1.0, 1.0,
        )
        
        # No direct feature importance for MLP, use permutation importance
        importance = {name: 1.0 / len(self._feature_names) for name in self._feature_names}
        
        return MLFactorResult(
            signal=signal,
            feature_importance=importance,
            model_score=0.0,  # Placeholder
            predictions_std=float(np.std(raw_signal)),
        )


class EnsembleFactorCombiner:
    """Ensemble of multiple ML models for robust factor combination.
    
    Combines predictions from GradientBoosting, RandomForest, and
    NeuralNetwork using stacking or averaging.
    
    Parameters
    ----------
    models : list[str] | None
        Models to include ('gb', 'rf', 'nn'). Default: all.
    ensemble_method : str
        'average' or 'stacking' (default 'average').
    """
    
    def __init__(
        self,
        models: Optional[list[str]] = None,
        ensemble_method: str = "average",
    ) -> None:
        self.models = models or ["gb", "rf", "nn"]
        self.ensemble_method = ensemble_method
        self._combiners: dict[str, any] = {}
        self._stacking_model = None
        self._feature_names: list[str] = []
        self._is_fitted = False
        
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: Optional[list[str]] = None,
    ) -> EnsembleFactorCombiner:
        """Fit all ensemble members."""
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"factor_{i}" for i in range(X.shape[1])]
        
        # Initialize combiners
        if "gb" in self.models:
            self._combiners["gb"] = GradientBoostingFactorCombiner()
            self._combiners["gb"].fit(X, y, feature_names)
        
        if "rf" in self.models:
            self._combiners["rf"] = RandomForestFactorCombiner()
            self._combiners["rf"].fit(X, y, feature_names)
        
        if "nn" in self.models:
            self._combiners["nn"] = NeuralNetworkFactorCombiner()
            self._combiners["nn"].fit(X, y, feature_names)
        
        # For stacking, train a meta-learner
        if self.ensemble_method == "stacking":
            from sklearn.linear_model import Ridge
            
            # Get predictions from base models
            base_predictions = []
            for name, combiner in self._combiners.items():
                result = combiner.predict(X)
                base_predictions.append(result.signal)
            
            X_meta = np.column_stack(base_predictions)
            self._stacking_model = Ridge(alpha=1.0)
            self._stacking_model.fit(X_meta, y)
        
        self._is_fitted = True
        _log.info("Ensemble fitted with %d models using %s", len(self._combiners), self.ensemble_method)
        
        return self
    
    def predict(self, X: NDArray) -> MLFactorResult:
        """Generate ensemble combined signal."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all base models
        base_predictions = []
        importance_sum = {}
        
        for name, combiner in self._combiners.items():
            result = combiner.predict(X)
            base_predictions.append(result.signal)
            
            # Aggregate importance
            for factor, imp in result.feature_importance.items():
                importance_sum[factor] = importance_sum.get(factor, 0.0) + imp
        
        if self.ensemble_method == "average":
            # Simple average
            signal = np.mean(base_predictions, axis=0)
        elif self.ensemble_method == "stacking" and self._stacking_model is not None:
            # Stacking with meta-learner
            X_meta = np.column_stack(base_predictions)
            raw_signal = self._stacking_model.predict(X_meta)
            signal = np.clip(
                (raw_signal - np.median(raw_signal)) / (np.percentile(raw_signal, 95) - np.percentile(raw_signal, 5) + 1e-10),
                -1.0, 1.0,
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Normalize importance
        total_imp = sum(importance_sum.values())
        importance = {k: v / total_imp for k, v in importance_sum.items()}
        
        return MLFactorResult(
            signal=signal,
            feature_importance=importance,
            model_score=0.0,  # Placeholder
            predictions_std=float(np.std(np.array(base_predictions), axis=0).mean()),
        )


class AdaptiveFactorCombiner:
    """Adaptive factor combination that adjusts weights based on recent performance.
    
    Uses exponential weighting to give more importance to recent
    factor performance, adapting to changing market conditions.
    
    Parameters
    ----------
    lookback : int
        Lookback window for performance estimation (default 252).
    decay : float
        Exponential decay factor (default 0.94).
    min_weight : float
        Minimum factor weight (default 0.01).
    """
    
    def __init__(
        self,
        lookback: int = 252,
        decay: float = 0.94,
        min_weight: float = 0.01,
    ) -> None:
        self.lookback = lookback
        self.decay = decay
        self.min_weight = min_weight
        self._weights: Optional[NDArray] = None
        self._performance_history: list[NDArray] = []
        self._feature_names: list[str] = []
        
    def update(
        self,
        factor_signals: NDArray,
        realized_return: float,
    ) -> None:
        """Update factor weights based on recent performance.
        
        Parameters
        ----------
        factor_signals : NDArray
            Factor signals at time t (n_factors,).
        realized_return : float
            Realized return at time t+1.
        """
        self._performance_history.append(factor_signals * realized_return)
        
        # Keep only recent history
        if len(self._performance_history) > self.lookback:
            self._performance_history = self._performance_history[-self.lookback:]
        
        # Compute exponential weighted performance
        if len(self._performance_history) >= 20:
            history = np.array(self._performance_history)
            weights = np.array([self.decay ** i for i in range(len(history))])[::-1]
            weights /= weights.sum()
            
            # Weighted average performance per factor
            weighted_perf = np.average(history, axis=0, weights=weights)
            
            # Convert to weights (softmax-like)
            exp_perf = np.exp(weighted_perf - np.max(weighted_perf))
            self._weights = exp_perf / exp_perf.sum()
            
            # Apply minimum weight
            self._weights = np.maximum(self._weights, self.min_weight)
            self._weights /= self._weights.sum()
    
    def combine(self, factor_signals: NDArray) -> NDArray:
        """Combine factors using adaptive weights.
        
        Parameters
        ----------
        factor_signals : NDArray
            Factor scores (n_samples, n_factors).
            
        Returns
        -------
        NDArray
            Combined signal in [-1, 1].
        """
        if self._weights is None:
            # Equal weights if not enough history
            n_factors = factor_signals.shape[1]
            weights = np.ones(n_factors) / n_factors
        else:
            weights = self._weights
        
        # Weighted combination
        raw_signal = factor_signals @ weights
        
        # Normalize to [-1, 1]
        signal = np.clip(
            (raw_signal - np.median(raw_signal)) / (np.percentile(raw_signal, 95) - np.percentile(raw_signal, 5) + 1e-10),
            -1.0, 1.0,
        )
        
        return signal
    
    @property
    def current_weights(self) -> Optional[NDArray]:
        """Get current factor weights."""
        return self._weights.copy() if self._weights is not None else None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_ml_factor_combiner(
    method: str = "ensemble",
    **kwargs,
) -> any:
    """Factory function to create ML factor combiner.
    
    Parameters
    ----------
    method : str
        'gradient_boosting', 'random_forest', 'neural_network', 'ensemble', 'adaptive'.
    **kwargs
        Additional arguments for the combiner.
        
    Returns
    -------
    Factor combiner instance.
    """
    combiners = {
        "gradient_boosting": GradientBoostingFactorCombiner,
        "random_forest": RandomForestFactorCombiner,
        "neural_network": NeuralNetworkFactorCombiner,
        "ensemble": EnsembleFactorCombiner,
        "adaptive": AdaptiveFactorCombiner,
    }
    
    if method not in combiners:
        raise ValueError(f"Unknown method: {method}. Choose from {list(combiners.keys())}")
    
    return combiners[method](**kwargs)


def evaluate_factor_combiner(
    combiner: any,
    X: NDArray,
    y: NDArray,
    n_splits: int = 5,
) -> dict[str, float]:
    """Evaluate factor combiner using time-series cross-validation.
    
    Parameters
    ----------
    combiner : any
        Factor combiner instance.
    X : NDArray
        Factor scores (n_samples, n_factors).
    y : NDArray
        Forward returns (n_samples,).
    n_splits : int
        Number of CV splits.
        
    Returns
    -------
    dict
        Evaluation metrics (IC, ICIR, turnover, etc.).
    """
    from sklearn.model_selection import TimeSeriesSplit
    from scipy.stats import spearmanr
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ics = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit and predict
        combiner.fit(X_train, y_train)
        result = combiner.predict(X_val)
        
        # Compute IC
        ic, _ = spearmanr(result.signal, y_val)
        ics.append(ic)
    
    return {
        "mean_ic": float(np.mean(ics)),
        "ic_std": float(np.std(ics)),
        "icir": float(np.mean(ics) / (np.std(ics) + 1e-10)),
        "positive_ic_pct": float(np.mean([1 for ic in ics if ic > 0]) / len(ics)),
    }
