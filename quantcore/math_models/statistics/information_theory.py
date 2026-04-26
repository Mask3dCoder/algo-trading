"""Information theory measures: entropy, KL divergence, mutual information, transfer entropy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_entropy


def shannon_entropy(series: NDArray, bins: int = 50) -> float:
    """Shannon entropy of a continuous random variable (discretized).

    H(X) = -sum(p(x) * log(p(x))) in nats.
    For N(0,1): H ≈ 1.42 nats (via differential entropy formula).
    """
    x = np.asarray(series, dtype=float)
    counts, _ = np.histogram(x, bins=bins, density=True)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def differential_entropy_normal(sigma: float = 1.0) -> float:
    """Analytical differential entropy of N(0, sigma): 0.5*ln(2*pi*e*sigma^2)."""
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)


def kl_divergence(p: NDArray, q: NDArray, bins: int = 50) -> float:
    """KL divergence D(P||Q) between two distributions estimated from samples.

    Uses histogram-based discretization.
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    # Shared bin edges
    all_data = np.concatenate([p_arr, q_arr])
    edges = np.linspace(np.min(all_data), np.max(all_data), bins + 1)

    p_counts, _ = np.histogram(p_arr, bins=edges, density=True)
    q_counts, _ = np.histogram(q_arr, bins=edges, density=True)

    p_probs = p_counts / np.sum(p_counts)
    q_probs = q_counts / np.sum(q_counts)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_probs = p_probs + eps
    q_probs = q_probs + eps

    return float(np.sum(p_probs * np.log(p_probs / q_probs)))


def mutual_information(
    x: NDArray,
    y: NDArray,
    method: str = "histogram",
    bins: int = 30,
    k: int = 5,
) -> float:
    """Mutual information I(X;Y) between two continuous variables.

    method="histogram": discretize and compute MI
    method="knn": KNN-based estimation (Kraskov et al.)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "knn":
        return _mi_knn(x, y, k)

    # Histogram-based
    joint_counts, _, _ = np.histogram2d(x, y, bins=bins)
    joint_probs = joint_counts / np.sum(joint_counts) + 1e-10

    px = np.sum(joint_probs, axis=1, keepdims=True)
    py = np.sum(joint_probs, axis=0, keepdims=True)

    mi = np.sum(joint_probs * np.log(joint_probs / (px * py + 1e-10)))
    return float(max(mi, 0.0))


def _mi_knn(x: NDArray, y: NDArray, k: int = 5) -> float:
    """KNN-based mutual information estimation (Kraskov et al. 2004)."""
    from scipy.spatial import cKDTree

    n = len(x)
    points = np.column_stack([x, y])
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)  # +1 because includes self
    eps = distances[:, -1]

    # Count neighbors within eps in each marginal
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    nx = np.array([len(tree_x.query_ball_point(x[i], eps[i])) - 1 for i in range(n)])
    ny = np.array([len(tree_y.query_ball_point(y[i], eps[i])) - 1 for i in range(n)])

    # Digamma-based estimate
    from scipy.special import digamma
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(mi, 0.0))


def transfer_entropy(
    source: NDArray,
    target: NDArray,
    lag: int = 1,
    bins: int = 10,
) -> float:
    """Transfer entropy from source to target.

    TE(X→Y) = H(Y_t|Y_{t-lag}) - H(Y_t|Y_{t-lag}, X_{t-lag})
    Measures directional information flow from X to Y.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(source)

    if n <= lag + 1:
        return 0.0

    y_t = target[lag:]
    y_lag = target[:-lag]
    x_lag = source[:-lag]

    # Discretize
    def discretize(arr: NDArray, b: int) -> NDArray:
        edges = np.linspace(np.min(arr), np.max(arr), b + 1)
        return np.digitize(arr, edges[1:-1])

    yt_d = discretize(y_t, bins)
    yl_d = discretize(y_lag, bins)
    xl_d = discretize(x_lag, bins)

    # Compute conditional entropies via joint histograms
    # H(Y_t | Y_{t-lag})
    joint_yyl = np.zeros((bins, bins))
    joint_yylx = np.zeros((bins, bins, bins))

    for i in range(len(yt_d)):
        if yt_d[i] < bins and yl_d[i] < bins and xl_d[i] < bins:
            joint_yyl[yt_d[i], yl_d[i]] += 1
            joint_yylx[yt_d[i], yl_d[i], xl_d[i]] += 1

    joint_yyl = joint_yyl / max(np.sum(joint_yyl), 1)
    joint_yylx = joint_yylx / max(np.sum(joint_yylx), 1)

    # Marginals
    p_yl = np.sum(joint_yyl, axis=0)
    p_ylx = np.sum(joint_yylx, axis=0)

    # Conditional entropy H(Y_t|Y_{t-l})
    h_y_given_yl = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_yyl[i, j] > 0 and p_yl[j] > 0:
                h_y_given_yl -= joint_yyl[i, j] * np.log(joint_yyl[i, j] / p_yl[j])

    # Conditional entropy H(Y_t|Y_{t-l}, X_{t-l})
    h_y_given_ylx = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if joint_yylx[i, j, k] > 0 and p_ylx[j, k] > 0:
                    h_y_given_ylx -= joint_yylx[i, j, k] * np.log(joint_yylx[i, j, k] / p_ylx[j, k])

    return float(max(h_y_given_yl - h_y_given_ylx, 0.0))
