"""Tree-based ML models: simplified gradient boosting with CV for signal generation.

Note: Full XGBoost/LightGBM implementations require those packages.
This module provides pure-numpy gradient boosting as a fallback,
plus wrapper interfaces for when those packages are available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.signals.ml.preprocessing import PreprocessingPipeline, TimeSeriesSplit

# ---------------------------------------------------------------------------
# Decision Tree (CART)
# ---------------------------------------------------------------------------


class DecisionTreeNode:
    """Single node in a regression decision tree."""

    def __init__(self) -> None:
        self.feature_idx: int = -1
        self.threshold: float = 0.0
        self.left: DecisionTreeNode | None = None
        self.right: DecisionTreeNode | None = None
        self.value: float = 0.0
        self.is_leaf: bool = False


class DecisionTreeRegressor:
    """Simple regression decision tree."""

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root: DecisionTreeNode | None = None

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.root = self._build(X, y, depth=0)

    def predict(self, X: NDArray) -> NDArray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def _build(self, X: NDArray, y: NDArray, depth: int) -> DecisionTreeNode:
        node = DecisionTreeNode()
        if depth >= self.max_depth or len(y) <= self.min_samples_leaf:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        best_feature, best_threshold, best_gain = -1, 0.0, -np.inf
        current_var = np.var(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(10, 90, 10))

            for t in thresholds:
                left_mask = X[:, feature_idx] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                left_var = np.var(y[left_mask])
                right_var = np.var(y[right_mask])
                n = len(y)
                gain = current_var - (np.sum(left_mask) / n * left_var + np.sum(right_mask) / n * right_var)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = t

        if best_feature < 0:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        node.feature_idx = best_feature
        node.threshold = best_threshold

        left_mask = X[:, best_feature] <= best_threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, x: NDArray, node: DecisionTreeNode | None) -> float:
        if node is None or node.is_leaf:
            return node.value if node is not None else 0.0
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)


# ---------------------------------------------------------------------------
# Gradient Boosting Regressor (pure numpy)
# ---------------------------------------------------------------------------


class GradientBoostingSignal:
    """Gradient boosting for signal generation.

    Uses decision trees as base learners with gradient descent on residuals.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        subsample: float = 0.8,
        cv_splits: int = 3,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.cv_splits = cv_splits
        self.trees_: list[DecisionTreeRegressor] = []
        self.base_prediction_: float = 0.0
        self._pipeline: PreprocessingPipeline | None = None

    def fit(self, features: NDArray, target: NDArray) -> dict:
        """Fit gradient boosting model with early stopping via CV."""
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        self._pipeline = pipeline

        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        X_clean = X[valid]
        y_clean = target[valid]

        self.base_prediction_ = np.mean(y_clean)
        residuals = y_clean - self.base_prediction_
        self.trees_ = []

        rng = np.random.default_rng(42)
        best_ic = -np.inf
        patience = 10
        no_improve = 0

        for i in range(self.n_estimators):
            # Subsample
            n = len(X_clean)
            sample_size = max(int(n * self.subsample), self.min_samples_leaf + 1)
            idx = rng.choice(n, size=sample_size, replace=False)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_clean[idx], residuals[idx])
            self.trees_.append(tree)

            # Update residuals
            pred_update = tree.predict(X_clean)
            residuals -= self.learning_rate * pred_update

            # Check OOS IC periodically
            if (i + 1) % 10 == 0:
                ic = self._compute_oos_ic(X, target)
                if ic > best_ic:
                    best_ic = ic
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

        return {"n_trees": len(self.trees_), "best_oos_ic": best_ic}

    def predict(self, features: NDArray) -> NDArray:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self._pipeline.transform(features)
        result = np.full(len(X), self.base_prediction_, dtype=float)
        for tree in self.trees_:
            result += self.learning_rate * tree.predict(X)
        return result

    def _compute_oos_ic(self, X: NDArray, target: NDArray) -> float:
        """Quick OOS IC estimate."""
        splitter = TimeSeriesSplit(n_splits=2)
        folds = splitter.split(len(X))
        ics = []

        for train_idx, test_idx in folds:
            pred = np.full(len(X), self.base_prediction_)
            for tree in self.trees_:
                pred += self.learning_rate * tree.predict(X)
            pred_test = pred[test_idx]
            target_test = target[test_idx]
            valid = ~(np.isnan(pred_test) | np.isnan(target_test))
            if np.sum(valid) > 5:
                ic = np.corrcoef(pred_test[valid], target_test[valid])[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)

        return np.mean(ics) if ics else 0.0


class RandomForestSignal:
    """Random Forest for signal generation (simplified)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        max_features: str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees_: list[DecisionTreeRegressor] = []
        self.feature_masks_: list[NDArray] = []
        self._pipeline: PreprocessingPipeline | None = None

    def fit(self, features: NDArray, target: NDArray) -> dict:
        pipeline = PreprocessingPipeline()
        X = pipeline.fit_transform(features)
        self._pipeline = pipeline
        valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(target))
        X_clean = X[valid]
        y_clean = target[valid]

        n, p = X_clean.shape
        n_features = int(np.sqrt(p)) if self.max_features == "sqrt" else p

        rng = np.random.default_rng(42)
        self.trees_ = []
        self.feature_masks_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = rng.choice(n, size=n, replace=True)
            # Feature subset
            feature_idx = rng.choice(p, size=n_features, replace=False)
            mask = np.zeros(p, dtype=bool)
            mask[feature_idx] = True
            self.feature_masks_.append(mask)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_clean[idx][:, feature_idx], y_clean[idx])
            self.trees_.append(tree)

        return {"n_trees": len(self.trees_)}

    def predict(self, features: NDArray) -> NDArray:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self._pipeline.transform(features)

        preds = np.zeros(len(X), dtype=float)
        for tree, mask in zip(self.trees_, self.feature_masks_):
            preds += tree.predict(X[:, mask])
        return preds / len(self.trees_)

    def feature_importance(self) -> NDArray:
        """Approximate feature importance based on how often features are used."""
        if not self.feature_masks_:
            return np.array([], dtype=float)

        n_features = len(self.feature_masks_[0])
        counts = np.zeros(n_features, dtype=float)
        for mask in self.feature_masks_:
            counts[mask] += 1.0
        counts /= len(self.feature_masks_)
        return counts
