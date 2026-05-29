"""ML preprocessing: winsorize, z-score, impute, feature selection, time-series CV splitter."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------


def winsorize(
    features: NDArray,
    limits: float = 3.0,
    axis: int = 0,
) -> NDArray:
    """Clip features at ±limits standard deviations cross-sectionally.

    Args:
        features: (n_samples, n_features) array.
        limits: number of standard deviations for clipping.
        axis: axis along which to compute stats (0=per feature, 1=per sample).

    Returns:
        Winsorized features array.
    """
    result = features.copy()
    if axis == 0:
        for j in range(features.shape[1]):
            col = features[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 5:
                continue
            m = np.mean(valid)
            s = np.std(valid, ddof=1)
            if s > 0:
                low = m - limits * s
                high = m + limits * s
                mask = ~np.isnan(result[:, j])
                result[mask, j] = np.clip(result[mask, j], low, high)
    elif axis == 1:
        for i in range(features.shape[0]):
            row = features[i, :]
            valid = row[~np.isnan(row)]
            if len(valid) < 5:
                continue
            m = np.mean(valid)
            s = np.std(valid, ddof=1)
            if s > 0:
                low = m - limits * s
                high = m + limits * s
                mask = ~np.isnan(result[i, :])
                result[i, mask] = np.clip(result[i, mask], low, high)

    return result


# ---------------------------------------------------------------------------
# Cross-Sectional Z-Score
# ---------------------------------------------------------------------------


def cross_sectional_zscore(
    features: NDArray,
) -> NDArray:
    """Z-score normalize each row (cross-section) independently.

    Args:
        features: (n_samples, n_features) array.

    Returns:
        Z-scored features. Each row has mean≈0, std≈1.
    """
    result = features.copy()
    n_samples, n_features = features.shape

    for i in range(n_samples):
        row = features[i, :]
        valid = ~np.isnan(row)
        if np.sum(valid) < 5:
            continue
        m = np.mean(row[valid])
        s = np.std(row[valid], ddof=1)
        if s > 0:
            result[i, valid] = (row[valid] - m) / s

    return result


# ---------------------------------------------------------------------------
# Impute Missing
# ---------------------------------------------------------------------------


def impute_median(
    features: NDArray,
) -> NDArray:
    """Impute NaN values with column median."""
    result = features.copy()
    n_features = features.shape[1]

    for j in range(n_features):
        col = features[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            median_val = np.median(valid)
            nan_mask = np.isnan(result[:, j])
            result[nan_mask, j] = median_val

    return result


def impute_knn(
    features: NDArray,
    k: int = 5,
) -> NDArray:
    """Simple KNN imputation for missing values.

    For each NaN, find k nearest rows (by non-NaN columns) and average.
    Falls back to median if not enough neighbors.
    """
    result = features.copy()
    n_samples, n_features = features.shape

    for i in range(n_samples):
        nan_cols = np.where(np.isnan(features[i, :]))[0]
        if len(nan_cols) == 0:
            continue

        for j in nan_cols:
            # Find rows that have value at column j
            valid_rows = ~np.isnan(features[:, j])
            if np.sum(valid_rows) == 0:
                continue

            # Compute distance using shared non-NaN features
            distances = np.full(n_samples, np.inf)
            for m in range(n_samples):
                if m == i or np.isnan(features[m, j]):
                    continue
                shared = ~(np.isnan(features[i, :]) | np.isnan(features[m, :]))
                if np.sum(shared) > 0:
                    distances[m] = np.sum((features[i, shared] - features[m, shared]) ** 2)

            # Take k nearest
            nearest = np.argsort(distances)[:k]
            nearest = nearest[distances[nearest] < np.inf]
            if len(nearest) > 0:
                result[i, j] = np.mean(features[nearest, j])
            else:
                # Fallback to median
                valid = features[valid_rows, j]
                result[i, j] = np.median(valid)

    return result


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------


def select_features_mutual_info(
    features: NDArray,
    target: NDArray,
    n_select: int = 100,
) -> list[int]:
    """Select top features by mutual information with target.

    Uses histogram-based MI estimation.

    Returns list of selected feature indices.
    """
    n_samples, n_features = features.shape

    mi_scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        f_col = features[:, j]
        valid = ~(np.isnan(f_col) | np.isnan(target))
        if np.sum(valid) < 30:
            continue

        x = f_col[valid]
        y = target[valid]

        # Histogram-based MI
        n_bins = min(30, int(np.sqrt(len(x))))
        try:
            hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
            hist_x = np.histogram(x, bins=n_bins)[0]
            hist_y = np.histogram(y, bins=n_bins)[0]

            pxy = hist_xy / hist_xy.sum()
            px = hist_x / hist_x.sum()
            py = hist_y / hist_y.sum()

            mi = 0.0
            for ix in range(n_bins):
                for iy in range(n_bins):
                    if pxy[ix, iy] > 0 and px[ix] > 0 and py[iy] > 0:
                        mi += pxy[ix, iy] * np.log(pxy[ix, iy] / (px[ix] * py[iy]))

            mi_scores[j] = max(mi, 0.0)
        except Exception:
            mi_scores[j] = 0.0

    # Return top n_select feature indices
    return list(np.argsort(mi_scores)[::-1][:n_select])


# ---------------------------------------------------------------------------
# Time-Series Cross-Validation Splitter
# ---------------------------------------------------------------------------


class TimeSeriesSplit:
    """Proper time-series cross-validation with embargo gap.

    Ensures no future data leaks into training. Each fold:
    [----train----][embargo][--test--]

    Args:
        n_splits: number of CV folds.
        embargo_pct: fraction of train size to skip between train and test.
        test_size: number of samples in each test set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        test_size: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.test_size = test_size

    def split(
        self,
        n_samples: int,
    ) -> list[tuple[NDArray, NDArray]]:
        """Generate train/test index pairs.

        Returns list of (train_indices, test_indices).
        """
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        folds: list[tuple[NDArray, NDArray]] = []
        step = (n_samples - test_size) // self.n_splits

        for fold in range(self.n_splits):
            train_end = step * (fold + 1)
            embargo = max(1, int(train_end * self.embargo_pct))
            test_start = train_end + embargo
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            # Verify no overlap
            assert train_end <= test_start, f"Train/test overlap in fold {fold}"

            folds.append((train_idx, test_idx))

        return folds


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """Chain of preprocessing steps applied in order.

    Supports separate fit/transform to prevent data leakage:
      - fit(X_train): learn statistics from training data
      - transform(X): apply learned statistics to new data
      - fit_transform(X): convenience for training data only
    """

    def __init__(
        self,
        winsorize_limits: float = 3.0,
        impute_method: str = "median",
        zscore: bool = True,
    ) -> None:
        self.winsorize_limits = winsorize_limits
        self.impute_method = impute_method
        self.zscore = zscore
        # Fitted statistics (set by fit())
        self._fitted = False
        self._winsor_lo: NDArray | None = None
        self._winsor_hi: NDArray | None = None
        self._impute_values: NDArray | None = None
        self._zscore_mean: NDArray | None = None
        self._zscore_std: NDArray | None = None

    def fit(self, features: NDArray) -> "PreprocessingPipeline":
        """Learn preprocessing statistics from training data."""
        result = features.copy().astype(float)

        # 1. Winsorize — compute bounds from training data
        self._winsor_lo = np.nanpercentile(result, self.winsorize_limits, axis=0)
        self._winsor_hi = np.nanpercentile(result, 100.0 - self.winsorize_limits, axis=0)

        # 2. Impute — compute medians from training data
        if self.impute_method == "median":
            self._impute_values = np.nanmedian(result, axis=0)
        else:
            self._impute_values = np.nanmedian(result, axis=0)

        # 3. Z-score — compute mean/std from training data
        if self.zscore:
            self._zscore_mean = np.nanmean(result, axis=0)
            self._zscore_std = np.nanstd(result, axis=0, ddof=1)
            self._zscore_std = np.where(self._zscore_std < 1e-10, 1.0, self._zscore_std)

        self._fitted = True
        return self

    def transform(self, features: NDArray) -> NDArray:
        """Apply fitted preprocessing statistics to new data."""
        result = features.copy().astype(float)

        # 1. Winsorize using fitted bounds
        if self._winsor_lo is not None and self._winsor_hi is not None:
            result = np.clip(result, self._winsor_lo, self._winsor_hi)

        # 2. Impute using fitted values
        if self._impute_values is not None:
            mask = np.isnan(result)
            result[mask] = np.broadcast_to(self._impute_values, result.shape)[mask]

        # 3. Z-score using fitted mean/std
        if self.zscore and self._zscore_mean is not None and self._zscore_std is not None:
            result = (result - self._zscore_mean) / self._zscore_std

        # 4. Replace remaining NaN with 0
        result = np.nan_to_num(result, nan=0.0)

        return result

    def fit_transform(self, features: NDArray) -> NDArray:
        """Fit on training data and transform it. Use ONLY for training data."""
        self.fit(features)
        return self.transform(features)
