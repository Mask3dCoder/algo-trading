"""Hierarchical Risk Parity (HRP) portfolio construction.

Implements López de Prado's HRP algorithm from
"Building Diversified Portfolios that Outperform Out of Sample" (2016).

Algorithm:
  1. Compute correlation-based distance matrix.
  2. Hierarchical clustering (single linkage).
  3. Quasi-diagonalization (seriation).
  4. Recursive bisection with inverse-variance allocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def compute_hrp_weights(returns: pd.DataFrame) -> dict[str, float]:
    """Compute Hierarchical Risk Parity portfolio weights.

    Args:
        returns: DataFrame where each column is a strategy's daily return stream.

    Returns:
        dict mapping strategy name to weight (sums to 1.0).
    """
    if returns.empty or returns.shape[1] == 0:
        return {}

    # Drop columns with insufficient data
    valid = returns.loc[:, returns.std() > 1e-12]
    if valid.shape[1] == 0:
        n = returns.shape[1]
        return {col: 1.0 / n for col in returns.columns}

    names = list(valid.columns)
    r = valid.values
    n_assets = r.shape[1]

    if n_assets == 1:
        w = {names[0]: 1.0}
        # Fill zero for excluded columns
        for col in returns.columns:
            if col not in w:
                w[col] = 0.0
        return w

    # 1. Correlation and distance matrices
    corr = np.corrcoef(r.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    # Clip to valid range
    corr = np.clip(corr, -1.0, 1.0)

    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    # 2. Hierarchical clustering (single linkage)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # 3. Quasi-diagonalization
    sorted_idx = _get_quasi_diag(link, n_assets)

    # 4. Recursive bisection on the sorted covariance
    cov = np.cov(r.T)
    if cov.ndim == 1:
        cov = cov.reshape(1, 1)

    weights_sorted = _recursive_bisection(cov, sorted_idx)
    weights_sorted = weights_sorted / weights_sorted.sum()

    # Map back to original order
    w_array = np.zeros(n_assets)
    for pos, orig_idx in enumerate(sorted_idx):
        w_array[orig_idx] = weights_sorted[pos]

    # Build dict, including zero-weight columns
    result: dict[str, float] = {}
    valid_set = set(names)
    for col in returns.columns:
        if col in valid_set:
            idx = names.index(col)
            result[col] = float(w_array[idx])
        else:
            result[col] = 0.0

    return result


def _get_quasi_diag(link: np.ndarray, n: int) -> list[int]:
    """Get quasi-diagonal order from linkage matrix.

    Walks the dendrogram leaves in order.
    """
    link_int = link.astype(int)
    sorted_idx: list[int] = []

    # Start from the root (last row) and recurse
    _walk_tree(link_int, link_int.shape[0], n, sorted_idx)

    # Add any unreached singletons
    reached = set(sorted_idx)
    for i in range(n):
        if i not in reached:
            sorted_idx.append(i)

    return sorted_idx


def _walk_tree(link: np.ndarray, node: int, n: int, out: list[int]) -> None:
    """Recursively walk the dendrogram to produce leaf order."""
    row = node - n  # row index in linkage matrix
    if row < 0:
        # This is a leaf node
        out.append(node)
        return

    left = int(link[row, 0])
    right = int(link[row, 1])

    # Order: left subtree then right subtree (preserves cluster structure)
    _walk_tree(link, left, n, out)
    _walk_tree(link, right, n, out)


def _recursive_bisection(cov: np.ndarray, sorted_idx: list[int]) -> np.ndarray:
    """Recursive bisection: allocate risk equally across cluster hierarchy."""
    n = len(sorted_idx)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones(1)

    # Reorder covariance by sorted indices
    cov_sorted = cov[np.ix_(sorted_idx, sorted_idx)]

    weights = np.ones(n)
    _bisect(cov_sorted, weights, list(range(n)))
    return weights


def _bisect(cov: np.ndarray, weights: np.ndarray, indices: list[int]) -> None:
    """Recursively bisect and allocate weights inversely to cluster variance."""
    k = len(indices)
    if k <= 1:
        return

    mid = k // 2
    left_idx = indices[:mid]
    right_idx = indices[mid:]

    # Sub-covariance matrices
    cov_left = cov[np.ix_(left_idx, left_idx)]
    cov_right = cov[np.ix_(right_idx, right_idx)]

    # Inverse-variance allocation within each sub-cluster
    w_left = _inv_var_weights(cov_left)
    w_right = _inv_var_weights(cov_right)

    # Cluster variance
    var_left = float(w_left @ cov_left @ w_left) if len(w_left) > 0 else 1.0
    var_right = float(w_right @ cov_right @ w_right) if len(w_right) > 0 else 1.0

    var_left = max(var_left, 1e-16)
    var_right = max(var_right, 1e-16)

    # Alpha: allocate inversely to variance
    alpha_left = (1.0 / var_left) / (1.0 / var_left + 1.0 / var_right)

    # Apply to weights
    for i in left_idx:
        weights[i] *= alpha_left
    for i in right_idx:
        weights[i] *= (1.0 - alpha_left)

    # Recurse
    _bisect(cov, weights, left_idx)
    _bisect(cov, weights, right_idx)


def _inv_var_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance weights for a covariance matrix."""
    diag = np.diag(cov)
    ivar = 1.0 / np.maximum(diag, 1e-10)
    total = ivar.sum()
    if total > 0:
        return ivar / total
    return np.ones(len(diag)) / max(len(diag), 1)
