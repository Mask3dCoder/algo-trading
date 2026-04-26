"""Portfolio optimization: MVO, Black-Litterman, HRP, NCO, Risk Parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize, root
from scipy.spatial.distance import squareform


@dataclass
class OptimizationResult:
    weights: NDArray[np.floating]
    expected_return: float
    expected_vol: float
    sharpe_ratio: float
    method: str
    constraints_satisfied: bool = True


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# T-0326: Mean-Variance Optimization (Markowitz)
# ---------------------------------------------------------------------------


def mean_variance(
    returns: NDArray[np.floating],
    target: Literal["max_sharpe", "min_variance", "target_return"] = "max_sharpe",
    target_return: float | None = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    rf: float = 0.0,
) -> OptimizationResult:
    """Markowitz mean-variance optimization.

    Args:
        returns: Historical returns, shape (n_periods, n_assets).
        target: Optimization objective.
        target_return: Target return (for 'target_return' mode).
        long_only: If True, all weights >= 0.
        max_weight: Maximum weight per asset.
        rf: Risk-free rate (annualized).

    Returns:
        OptimizationResult with optimal weights.
    """
    r = np.asarray(returns, dtype=float)
    n_assets = r.shape[1]
    r.shape[0]

    mu = r.mean(axis=0) * TRADING_DAYS
    sigma = np.cov(r.T) * TRADING_DAYS

    if n_assets == 1:
        vol = float(np.std(r[:, 0], ddof=1) * np.sqrt(TRADING_DAYS))
        expected_ret = float(mu[0]) if mu.ndim > 0 else float(mu)
        return OptimizationResult(
            weights=np.array([1.0]),
            expected_return=expected_ret,
            expected_vol=vol,
            sharpe_ratio=float((expected_ret - rf) / vol) if vol > 0 else 0.0,
            method="single_asset",
        )

    # Constraints
    constraints = []
    bounds = [(0.0, max_weight)] * n_assets if long_only else [(-max_weight, max_weight)] * n_assets

    # Sum = 1
    constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

    if target == "target_return" and target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: float(w @ mu) - target_return,
        })

    # Objective functions
    def objective_max_sharpe(w: NDArray) -> float:
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ sigma @ w))
        if port_vol <= 0:
            return 1e10
        return -(port_ret - rf) / port_vol

    def objective_min_var(w: NDArray) -> float:
        return float(w @ sigma @ w)

    objective = objective_max_sharpe if target != "min_variance" else objective_min_var

    # Solve
    x0 = np.ones(n_assets) / n_assets
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    w = result.x
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ sigma @ w))
    sr = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights=w,
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
        method="mvo",
        constraints_satisfied=abs(w.sum() - 1.0) < 1e-6,
    )


# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------


def compute_frontier(
    returns: NDArray[np.floating],
    n_points: int = 50,
    long_only: bool = True,
) -> list[tuple[float, float, NDArray[np.floating]]]:
    """Compute the efficient frontier.

    Returns:
        List of (risk, return, weights) tuples along the frontier.
    """
    r = np.asarray(returns, dtype=float)
    n_assets = r.shape[1]

    mu = r.mean(axis=0) * TRADING_DAYS
    np.cov(r.T) * TRADING_DAYS

    # Find min var portfolio
    min_var = mean_variance(r, target="min_variance", long_only=long_only)
    # Find max return portfolio
    if long_only:
        max_ret_idx = np.argmax(mu)
        max_ret_w = np.zeros(n_assets)
        max_ret_w[max_ret_idx] = 1.0
        max_ret = float(mu[max_ret_idx])
    else:
        max_ret = float(np.max(mu))

    target_returns = np.linspace(min_var.expected_return, max_ret * 0.95, n_points)

    frontier = []
    for tr in target_returns:
        result = mean_variance(r, target="target_return", target_return=float(tr), long_only=long_only)
        frontier.append((result.expected_vol, result.expected_return, result.weights))

    return frontier


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------


def black_litterman(
    market_caps: NDArray[np.floating],
    cov: NDArray[np.floating],
    views: dict[int, float] | None = None,  # {asset_idx: expected_return}
    view_confidence: float = 0.025,  # Uncertainty in views
    tau: float = 0.05,  # Uncertainty in prior
    rf: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Black-Litterman model: combine equilibrium returns with investor views.

    Π = δ Σ w_mkt   (prior/equilibrium)
    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]   (posterior)

    Args:
        market_caps: Market capitalizations (equilibrium weights).
        cov: Covariance matrix (annualized).
        views: {asset_idx: view_return} for investor views.
        view_confidence: Uncertainty around views (smaller = more confident).
        tau: Uncertainty around prior (smaller = more confident in prior).
        rf: Risk-free rate.

    Returns:
        (posterior_returns, posterior_cov)
    """
    cov_arr = np.asarray(cov, dtype=float)
    w_mkt = np.asarray(market_caps, dtype=float).flatten()
    w_mkt = w_mkt / w_mkt.sum()

    n = len(w_mkt)

    # Market risk aversion: δ = (E[R_mkt] - rf) / σ²_mkt
    port_ret = 0.07  # Assume 7% equity risk premium
    port_var = float(w_mkt @ cov_arr @ w_mkt)
    delta = (port_ret - rf) / port_var if port_var > 0 else 2.5

    # Prior: equilibrium returns
    pi = delta * cov_arr @ w_mkt

    if views is None or len(views) == 0:
        return pi, cov_arr

    # Views
    view_indices = sorted(views.keys())
    k = len(view_indices)
    Q = np.array([views[i] for i in view_indices])

    # Pick matrix: P[k × n]
    P = np.zeros((k, n))
    for j, idx in enumerate(view_indices):
        P[j, idx] = 1.0

    # View uncertainty
    Omega = view_confidence * P @ cov_arr @ P.T * tau

    # Posterior
    tau_sigma_inv = np.linalg.inv(tau * cov_arr)
    P_omega_inv = P.T @ np.linalg.inv(Omega)

    posterior_cov_inv = tau_sigma_inv + P_omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_cov_inv)

    posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P_omega_inv @ Q)

    return posterior_mean, posterior_cov


def bl_optimize(
    posterior_returns: NDArray[np.floating],
    cov: NDArray[np.floating],
    long_only: bool = True,
) -> OptimizationResult:
    """Optimize portfolio using Black-Litterman posterior returns."""
    mu = np.asarray(posterior_returns, dtype=float).flatten()
    cov_arr = np.asarray(cov, dtype=float)
    n = len(mu)

    if n == 1:
        return OptimizationResult(
            weights=np.array([1.0]),
            expected_return=float(mu[0]),
            expected_vol=float(np.sqrt(cov_arr[0, 0])),
            sharpe_ratio=float(mu[0] / np.sqrt(cov_arr[0, 0])),
            method="black_litterman",
        )

    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: NDArray) -> float:
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov_arr @ w))
        return -ret / vol if vol > 0 else 1e10

    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})

    w = result.x
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov_arr @ w))
    sr = port_ret / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights=w,
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
        method="black_litterman",
    )


# ---------------------------------------------------------------------------
# T-0327: Hierarchical Risk Parity (HRP)
# ---------------------------------------------------------------------------


def _corr_to_dist(corr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert correlation matrix to distance matrix: D = sqrt(0.5 * (1 - ρ))."""
    return np.sqrt(0.5 * (1.0 - corr))


def hrp(
    returns: NDArray[np.floating],
) -> OptimizationResult:
    """Hierarchical Risk Parity (Lopez de Prado).

    1. Hierarchical clustering (Ward) on distance matrix
    2. Quasi-diagonalization (reorder assets by cluster)
    3. Recursive bisection: allocate risk equally at each split
    """
    r = np.asarray(returns, dtype=float)
    n_assets = r.shape[1]

    if n_assets <= 1:
        return OptimizationResult(
            weights=np.ones(n_assets) / max(n_assets, 1),
            expected_return=float(r.mean()) * TRADING_DAYS,
            expected_vol=float(r.std()) * np.sqrt(TRADING_DAYS),
            sharpe_ratio=float(r.mean() / r.std()) * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0,
            method="hrp",
        )

    cov = np.cov(r.T)
    corr = np.corrcoef(r.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # 1. Hierarchical clustering
    dist = _corr_to_dist(corr)
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method="ward")

    # 2. Quasi-diagonalization: get sorted indices
    def _get_quasi_diag(link_matrix: NDArray) -> list[int]:
        """Get quasi-diagonal order from linkage matrix."""
        link = link_matrix.astype(int)
        n = link.shape[0] + 1
        sorted_idx: list[int] = []
        for i in range(link.shape[0]):
            for j in range(2):
                val = link[i, j]
                if val < n:
                    sorted_idx.append(val)
        # Nodes not reached (single-element clusters)
        reached = set(sorted_idx)
        for i in range(n):
            if i not in reached:
                sorted_idx.append(i)
        return sorted_idx

    sorted_idx = _get_quasi_diag(link)
    if len(sorted_idx) == 0:
        sorted_idx = list(range(n_assets))

    weights = np.ones(n_assets) / n_assets

    # 3. Recursive bisection
    def _recursive_bisection(cov_sub: NDArray, indices: list[int]) -> NDArray[np.floating]:
        """Recursively bisect cluster and allocate equal risk."""
        k = len(indices)
        if k == 0:
            return np.array([])
        if k == 1:
            return np.ones(1)

        # Split into two sub-clusters
        mid = k // 2
        indices[:mid]
        indices[mid:]

        # Sub-covariances
        cov_left = cov_sub[np.ix_(range(mid), range(mid))]
        cov_right = cov_sub[np.ix_(range(mid, k), range(mid, k))]

        # Inverse-variance weights within each cluster
        w_left = _inverse_variance_weights(cov_left)
        w_right = _inverse_variance_weights(cov_right)

        # Cluster variances
        var_left = float(w_left @ cov_left @ w_left)
        var_right = float(w_right @ cov_right @ w_right)

        # Risk allocation: α_left = (1/var_left) / (1/var_left + 1/var_right)
        if var_left + var_right > 0:
            alpha_left = (1.0 / var_left) / (1.0 / var_left + 1.0 / var_right)
        else:
            alpha_left = 0.5

        w = np.zeros(k)
        if mid > 0:
            left_result = _recursive_bisection(cov_left, list(range(mid)))
            w[:mid] = alpha_left * left_result
        if k - mid > 0:
            right_result = _recursive_bisection(cov_right, list(range(k - mid)))
            w[mid:k] = (1.0 - alpha_left) * right_result

        return w

    weights = _recursive_bisection(cov, list(range(n_assets)))
    weights = weights / weights.sum()  # Normalize

    port_ret = float(r.mean(axis=0) @ weights) * TRADING_DAYS
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(TRADING_DAYS)
    sr = port_ret / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights=weights,
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
        method="hrp",
    )


def _inverse_variance_weights(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    """Inverse variance weights."""
    diag = np.diag(cov)
    ivar = 1.0 / np.maximum(diag, 1e-10)
    w = ivar / ivar.sum() if ivar.sum() > 0 else np.ones(len(diag)) / len(diag)
    return w


# ---------------------------------------------------------------------------
# Nested Clustered Optimization (NCO)
# ---------------------------------------------------------------------------


def nco(
    returns: NDArray[np.floating],
    n_clusters: int | None = None,
) -> OptimizationResult:
    """Nested Clustered Optimization (Lopez de Prado).

    1. Cluster assets
    2. Intra-cluster: optimize each cluster via MVO
    3. Inter-cluster: optimize across cluster portfolios via MVO
    """
    r = np.asarray(returns, dtype=float)
    n_assets = r.shape[1]

    if n_assets <= 2:
        return hrp(r)

    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(n_assets)))

    cov = np.cov(r.T)
    corr = np.corrcoef(r.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Cluster
    dist = _corr_to_dist(corr)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    clusters = fcluster(link, t=n_clusters, criterion="maxclust")

    # Group indices by cluster
    cluster_indices: dict[int, list[int]] = {}
    for i in range(n_assets):
        c = int(clusters[i])
        if c not in cluster_indices:
            cluster_indices[c] = []
        cluster_indices[c].append(i)

    # Intra-cluster optimization
    intra_weights: dict[int, NDArray[np.floating]] = {}
    intra_returns = np.zeros((r.shape[0], len(cluster_indices)))
    cluster_keys = sorted(cluster_indices.keys())

    for ci, ck in enumerate(cluster_keys):
        idx = cluster_indices[ck]
        if len(idx) == 1:
            intra_weights[ck] = np.ones(1)
            intra_returns[:, ci] = r[:, idx[0]]
        else:
            sub_r = r[:, idx]
            try:
                result = mean_variance(sub_r, target="max_sharpe", long_only=True)
                intra_weights[ck] = result.weights
                intra_returns[:, ci] = sub_r @ result.weights
            except Exception:
                intra_weights[ck] = np.ones(len(idx)) / len(idx)
                intra_returns[:, ci] = sub_r @ intra_weights[ck]

    # Inter-cluster optimization
    try:
        inter_result = mean_variance(intra_returns, target="max_sharpe", long_only=True)
    except Exception:
        inter_result = OptimizationResult(
            weights=np.ones(len(cluster_keys)) / len(cluster_keys),
            expected_return=0.0,
            expected_vol=0.0,
            sharpe_ratio=0.0,
            method="nco_fallback",
        )

    # Combine weights
    final_weights = np.zeros(n_assets)
    for ci, ck in enumerate(cluster_keys):
        idx = cluster_indices[ck]
        final_weights[idx] = inter_result.weights[ci] * intra_weights[ck]

    w = final_weights / final_weights.sum()
    port_ret = float(r.mean(axis=0) @ w) * TRADING_DAYS
    port_vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(TRADING_DAYS)
    sr = port_ret / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights=w,
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
        method="nco",
    )


# ---------------------------------------------------------------------------
# Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------


def risk_parity(
    cov: NDArray[np.floating],
    target_vol: float | None = None,
) -> OptimizationResult:
    """Equal Risk Contribution (Risk Parity) optimization.

    Finds weights w such that: ∂Risk/∂w_i * w_i is equal for all i.

    Uses root-finding approach: RC_i(w) = w_i * (Σw)_i
    """
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]

    if n == 1:
        return OptimizationResult(
            weights=np.array([1.0]),
            expected_return=0.0,
            expected_vol=float(np.sqrt(cov_arr[0, 0])),
            sharpe_ratio=0.0,
            method="risk_parity",
        )

    def risk_contributions(w: NDArray) -> NDArray[np.floating]:
        """Risk contribution of each asset."""
        port_vol = float(np.sqrt(w @ cov_arr @ w))
        if port_vol <= 0:
            return np.zeros_like(w)
        marginal_risk = cov_arr @ w / port_vol
        return w * marginal_risk

    def erc_objective(w: NDArray) -> NDArray[np.floating]:
        """Objective: equalize risk contributions."""
        w = np.abs(w)  # Long-only
        w = w / w.sum()
        rc = risk_contributions(w)
        rc_target = np.mean(rc)
        return rc - rc_target

    # Initial guess: 1/vol
    x0 = 1.0 / np.maximum(np.sqrt(np.diag(cov_arr)), 1e-10)
    x0 = x0 / x0.sum()

    try:
        result = root(erc_objective, x0, method="hybr", options={"maxfev": 2000})
        if result.success:
            w = np.abs(result.x)
            w = w / w.sum()
        else:
            w = x0
    except Exception:
        w = x0

    # Scale to target vol
    if target_vol is not None and target_vol > 0:
        port_vol = float(np.sqrt(w @ cov_arr @ w))
        if port_vol > 0:
            w = w * (target_vol / port_vol)
        w = w / w.sum()

    port_vol = float(np.sqrt(w @ cov_arr @ w)) * np.sqrt(TRADING_DAYS)
    sr = 0.05 / port_vol if port_vol > 0 else 0.0  # Assume 5% excess return

    return OptimizationResult(
        weights=w,
        expected_return=0.0,
        expected_vol=port_vol,
        sharpe_ratio=sr,
        method="risk_parity",
    )


# ---------------------------------------------------------------------------
# Stability test for HRP vs MVO
# ---------------------------------------------------------------------------


def stability_test(
    returns: NDArray[np.floating],
    n_perturbations: int = 100,
    perturbation_scale: float = 0.10,
    seed: int = 42,
) -> dict[str, float]:
    """Test weight stability: HRP should be more stable than MVO under covariance perturbation.

    Returns:
        {method: mean_absolute_weight_change}
    """
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)

    hrp_base = hrp(r)
    mvo_base = mean_variance(r, target="max_sharpe", long_only=True)

    hrp_changes = []
    mvo_changes = []

    for _ in range(n_perturbations):
        # Perturb returns
        noise = rng.normal(0, perturbation_scale, r.shape)
        r_pert = r + noise

        try:
            hrp_p = hrp(r_pert)
            mvo_p = mean_variance(r_pert, target="max_sharpe", long_only=True)
        except Exception:
            continue

        hrp_changes.append(float(np.sum(np.abs(hrp_base.weights - hrp_p.weights))))
        mvo_changes.append(float(np.sum(np.abs(mvo_base.weights - mvo_p.weights))))

    return {
        "hrp_mean_change": float(np.mean(hrp_changes)) if hrp_changes else 0.0,
        "mvo_mean_change": float(np.mean(mvo_changes)) if mvo_changes else 0.0,
        "stability_ratio": float(np.mean(mvo_changes) / np.mean(hrp_changes))
            if hrp_changes and np.mean(hrp_changes) > 0 else 0.0,
    }
