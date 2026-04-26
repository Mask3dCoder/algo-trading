"""Risk budgeting: allocate risk across strategies, assets, and factors.

Provides:
- Equal risk contribution (ERC / risk parity)
- Risk budget optimization (target risk allocation per asset/factor)
- Hierarchical risk parity
- Diversification ratio and risk concentration index
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sp_optimize


@dataclass
class RiskBudget:
    """Risk budget allocation result."""
    weights: NDArray[np.floating]
    risk_contributions: dict[str, float]
    risk_pcts: dict[str, float]
    total_vol: float
    diversification_ratio: float
    risk_concentration: float
    converged: bool = True
    iterations: int = 0


# ---------------------------------------------------------------------------
# Equal Risk Contribution (Risk Parity)
# ---------------------------------------------------------------------------


def equal_risk_contribution(
    cov: NDArray[np.floating],
    assets: list[str] | None = None,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> RiskBudget:
    """Compute equal risk contribution (risk parity) weights.

    Each asset contributes equally to portfolio volatility:
        σ_wi = w_i * (Σw)_i / σ_p  →  all equal

    Solved via the Maillard-Roncalli-Teiletche formulation:
        minimize  Σ_i Σ_j (w_i*(Σw)_i - w_j*(Σw)_j)^2

    Args:
        cov: Covariance matrix (n_assets, n_assets).
        assets: Optional asset names.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        RiskBudget with ERC weights and risk decomposition.
    """
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    # Initial weights
    w = np.full(n, 1.0 / n)

    def _rc(w_vec: NDArray[np.floating]) -> NDArray[np.floating]:
        """Risk contribution of each asset."""
        sigma = np.sqrt(w_vec @ cov_arr @ w_vec)
        if sigma < 1e-12:
            return np.zeros(n)
        return w_vec * (cov_arr @ w_vec) / sigma

    def _objective(w_vec: NDArray[np.floating]) -> float:
        rc = _rc(w_vec)
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    # Constraint: sum(weights) = 1
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(0, 1) for _ in range(n)]

    try:
        result = sp_optimize.minimize(
            _objective, w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": tol, "maxiter": max_iter},
        )
        w_opt = result.x
        converged = result.success
        iterations = result.nit
    except Exception:
        w_opt = w
        converged = False
        iterations = 0

    port_var = float(w_opt @ cov_arr @ w_opt)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0

    rc = _rc(w_opt) if port_vol > 0 else np.zeros(n)
    rc_dict = {assets[i]: float(rc[i]) for i in range(n)}
    rc_pct = {assets[i]: float(rc[i] / port_vol) if port_vol > 0 else 0.0 for i in range(n)}

    div_ratio = _diversification_ratio(w_opt, cov_arr)
    rci = _risk_concentration_index(rc, port_vol)

    return RiskBudget(
        weights=w_opt,
        risk_contributions=rc_dict,
        risk_pcts=rc_pct,
        total_vol=port_vol,
        diversification_ratio=div_ratio,
        risk_concentration=rci,
        converged=converged,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Risk Budget Optimization
# ---------------------------------------------------------------------------


def risk_budget_optimize(
    cov: NDArray[np.floating],
    risk_budgets: NDArray[np.floating] | None = None,
    assets: list[str] | None = None,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> RiskBudget:
    """Optimize portfolio weights to match target risk budgets.

    Target allocation: each asset i contributes risk_budgets[i] of total risk.

    Args:
        cov: Covariance matrix (n_assets, n_assets).
        risk_budgets: Target risk contributions (normalized to sum=1 if absent).
            Defaults to equal (risk parity).
        assets: Optional asset names.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        RiskBudget with optimized weights.
    """
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]

    if risk_budgets is None:
        risk_budgets = np.full(n, 1.0 / n)
    else:
        risk_budgets = np.asarray(risk_budgets, dtype=float).flatten()
        total = risk_budgets.sum()
        if total > 0:
            risk_budgets = risk_budgets / total

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    w = np.full(n, 1.0 / n)

    def _rc(w_vec: NDArray[np.floating]) -> NDArray[np.floating]:
        sigma = np.sqrt(w_vec @ cov_arr @ w_vec)
        if sigma < 1e-12:
            return np.zeros(n)
        return w_vec * (cov_arr @ w_vec) / sigma

    def _objective(w_vec: NDArray[np.floating]) -> float:
        sigma = np.sqrt(w_vec @ cov_arr @ w_vec)
        if sigma < 1e-12:
            return 1e10
        rc = w_vec * (cov_arr @ w_vec) / sigma
        rc_pct = rc / sigma
        return float(np.sum((rc_pct - risk_budgets) ** 2))

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(0, 1) for _ in range(n)]

    try:
        result = sp_optimize.minimize(
            _objective, w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": tol, "maxiter": max_iter},
        )
        w_opt = result.x
        converged = result.success
        iterations = result.nit
    except Exception:
        w_opt = w
        converged = False
        iterations = 0

    port_var = float(w_opt @ cov_arr @ w_opt)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0

    rc = _rc(w_opt) if port_vol > 0 else np.zeros(n)
    rc_dict = {assets[i]: float(rc[i]) for i in range(n)}
    rc_pct = {assets[i]: float(rc[i] / port_vol) if port_vol > 0 else 0.0 for i in range(n)}

    div_ratio = _diversification_ratio(w_opt, cov_arr)
    rci = _risk_concentration_index(rc, port_vol)

    return RiskBudget(
        weights=w_opt,
        risk_contributions=rc_dict,
        risk_pcts=rc_pct,
        total_vol=port_vol,
        diversification_ratio=div_ratio,
        risk_concentration=rci,
        converged=converged,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP)
# ---------------------------------------------------------------------------


def hierarchical_risk_parity(
    cov: NDArray[np.floating],
    assets: list[str] | None = None,
    method: str = "single",
) -> RiskBudget:
    """Hierarchical Risk Parity using recursive partitioning.

    Clusters assets by correlation distance, then allocates risk equally
    within and across clusters. Robust to estimation error.

    Args:
        cov: Covariance matrix (n_assets, n_assets).
        assets: Optional asset names.
        method: Linkage method ("single", "complete", "average", "ward").

    Returns:
        RiskBudget with HRP weights.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    # Correlation matrix
    std = np.sqrt(np.diag(cov_arr))
    std[std < 1e-12] = 1e-12
    corr = cov_arr / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Distance matrix
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    # Clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method=method)
    leaves_list(link)

    # Recursive bisection
    def _hrp_recursive(indices: list[int]) -> list[float]:
        if len(indices) == 1:
            return [1.0]
        if len(indices) == 2:
            # Allocate inversely proportional to variance
            v0 = cov_arr[indices[0], indices[0]]
            v1 = cov_arr[indices[1], indices[1]]
            w0 = v1 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            return [w0, 1.0 - w0]

        # Split cluster into two
        dist_sub = dist[np.ix_(indices, indices)]
        condensed_sub = squareform(dist_sub, checks=False)
        link_sub = linkage(condensed_sub, method=method)
        half = len(indices) // 2
        list(link_sub[: half, :2].flatten().astype(int))
        # Build cluster membership via the cut
        # Simplified: split by the linkage ordering
        idx_left = sorted(set(leaves_list(link_sub)[:half]))
        idx_right = sorted(set(leaves_list(link_sub)[half:]))

        left_w = _hrp_recursive([indices[i] for i in idx_left])
        right_w = _hrp_recursive([indices[i] for i in idx_right])

        # Compute cluster variances
        left_indices = [indices[i] for i in idx_left]
        right_indices = [indices[i] for i in idx_right]

        left_cov = cov_arr[np.ix_(left_indices, left_indices)]
        right_cov = cov_arr[np.ix_(right_indices, right_indices)]

        left_var = float(np.array(left_w) @ left_cov @ np.array(left_w))
        right_var = float(np.array(right_w) @ right_cov @ np.array(right_w))

        alpha = right_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5

        left_scaled = [w * alpha for w in left_w]
        right_scaled = [w * (1.0 - alpha) for w in right_w]

        # Reconstruct full weight list
        result: list[float] = [0.0] * len(indices)
        for j, orig_idx in enumerate(idx_left):
            result[orig_idx] = left_scaled[j]
        for j, orig_idx in enumerate(idx_right):
            result[orig_idx] = right_scaled[j]
        return result

    w_raw = _hrp_recursive(list(range(n)))
    w_opt = np.array(w_raw)
    w_opt = w_opt / w_opt.sum()

    port_var = float(w_opt @ cov_arr @ w_opt)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0

    rc = w_opt * (cov_arr @ w_opt) / port_vol if port_vol > 0 else np.zeros(n)
    rc_dict = {assets[i]: float(rc[i]) for i in range(n)}
    rc_pct = {assets[i]: float(rc[i] / port_vol) if port_vol > 0 else 0.0 for i in range(n)}
    div_ratio = _diversification_ratio(w_opt, cov_arr)
    rci = _risk_concentration_index(rc, port_vol)

    return RiskBudget(
        weights=w_opt,
        risk_contributions=rc_dict,
        risk_pcts=rc_pct,
        total_vol=port_vol,
        diversification_ratio=div_ratio,
        risk_concentration=rci,
        converged=True,
    )


# ---------------------------------------------------------------------------
# Maximum diversification
# ---------------------------------------------------------------------------


def maximum_diversification(
    cov: NDArray[np.floating],
    assets: list[str] | None = None,
    tol: float = 1e-8,
) -> RiskBudget:
    """Maximum diversification ratio portfolio.

    Maximizes DR = (Σ w_i σ_i) / σ_p  (ratio of weighted-average to portfolio vol).

    Args:
        cov: Covariance matrix (n_assets, n_assets).
        assets: Optional asset names.
        tol: Convergence tolerance.

    Returns:
        RiskBudget with max-diversification weights.
    """
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    sigmas = np.sqrt(np.diag(cov_arr))

    def _objective(w_vec: NDArray[np.floating]) -> float:
        port_vol = np.sqrt(w_vec @ cov_arr @ w_vec)
        if port_vol < 1e-12:
            return -1e10
        return -float(np.dot(w_vec, sigmas) / port_vol)

    w = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(0, 1) for _ in range(n)]

    try:
        result = sp_optimize.minimize(
            _objective, w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": tol, "maxiter": 500},
        )
        w_opt = result.x
        converged = result.success
    except Exception:
        w_opt = w
        converged = False

    port_var = float(w_opt @ cov_arr @ w_opt)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0
    rc = w_opt * (cov_arr @ w_opt) / port_vol if port_vol > 0 else np.zeros(n)
    rc_dict = {assets[i]: float(rc[i]) for i in range(n)}
    rc_pct = {assets[i]: float(rc[i] / port_vol) if port_vol > 0 else 0.0 for i in range(n)}
    div_ratio = _diversification_ratio(w_opt, cov_arr)
    rci = _risk_concentration_index(rc, port_vol)

    return RiskBudget(
        weights=w_opt,
        risk_contributions=rc_dict,
        risk_pcts=rc_pct,
        total_vol=port_vol,
        diversification_ratio=div_ratio,
        risk_concentration=rci,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Inverse volatility
# ---------------------------------------------------------------------------


def inverse_volatility(
    cov: NDArray[np.floating],
    assets: list[str] | None = None,
) -> RiskBudget:
    """Inverse volatility weighting (naive risk parity).

    w_i = (1/σ_i) / Σ (1/σ_j)

    Args:
        cov: Covariance matrix.
        assets: Optional asset names.

    Returns:
        RiskBudget with inverse-volatility weights.
    """
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    sigmas = np.sqrt(np.diag(cov_arr))
    sigmas[sigmas < 1e-12] = 1e-12
    inv_vol = 1.0 / sigmas
    w_opt = inv_vol / inv_vol.sum()

    port_var = float(w_opt @ cov_arr @ w_opt)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0
    rc = w_opt * (cov_arr @ w_opt) / port_vol if port_vol > 0 else np.zeros(n)
    rc_dict = {assets[i]: float(rc[i]) for i in range(n)}
    rc_pct = {assets[i]: float(rc[i] / port_vol) if port_vol > 0 else 0.0 for i in range(n)}
    div_ratio = _diversification_ratio(w_opt, cov_arr)
    rci = _risk_concentration_index(rc, port_vol)

    return RiskBudget(
        weights=w_opt,
        risk_contributions=rc_dict,
        risk_pcts=rc_pct,
        total_vol=port_vol,
        diversification_ratio=div_ratio,
        risk_concentration=rci,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diversification_ratio(
    weights: NDArray[np.floating],
    cov: NDArray[np.floating],
) -> float:
    """Diversification ratio: weighted-avg vol / portfolio vol."""
    sigmas = np.sqrt(np.diag(cov))
    w_vol = float(np.dot(weights, sigmas))
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol < 1e-12:
        return 1.0
    return w_vol / port_vol


def _risk_concentration_index(
    risk_contributions: NDArray[np.floating],
    total_vol: float,
) -> float:
    """Risk Concentration Index (Herfindahl of risk contributions).

    RCI = 1 - (Σ rc_i²) / (Σ rc_i)²
    Values near 0 = concentrated; near 1 = perfectly diversified.
    """
    if total_vol < 1e-12 or len(risk_contributions) < 2:
        return 1.0
    rc = np.asarray(risk_contributions, dtype=float)
    rc_pct = rc / total_vol
    hhi = float(np.sum(rc_pct ** 2))
    n = len(rc)
    # Normalize: (1 - hhi) / (1 - 1/n)
    rci = (1.0 - hhi) / (1.0 - 1.0 / n) if n > 1 else 1.0
    return max(0.0, min(1.0, rci))


def risk_decomposition(
    weights: NDArray[np.floating],
    cov: NDArray[np.floating],
    assets: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Full risk decomposition: total, component, marginal, pct.

    Returns:
        {asset: {component_vol, marginal_vol, pct_vol, component_var, ...}}
    """
    w = np.asarray(weights, dtype=float).flatten()
    cov_arr = np.asarray(cov, dtype=float)
    n = len(w)

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    port_var = float(w @ cov_arr @ w)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0

    result: dict[str, dict[str, float]] = {}
    if port_vol < 1e-12:
        for i in range(n):
            result[assets[i]] = {
                "component_vol": 0.0, "component_var": 0.0,
                "marginal_vol": 0.0, "marginal_var": 0.0,
                "pct_risk": 0.0, "weight": float(w[i]),
            }
        return result

    for i in range(n):
        marginal_var = float(2.0 * (cov_arr @ w)[i])
        marginal_vol = marginal_var / (2.0 * port_vol)
        component_var = float(w[i] * marginal_var)
        component_vol = component_var / (2.0 * port_vol)

        result[assets[i]] = {
            "component_vol": component_vol,
            "component_var": component_var,
            "marginal_vol": marginal_vol,
            "marginal_var": marginal_var,
            "pct_risk": component_vol / port_vol if port_vol > 0 else 0.0,
            "weight": float(w[i]),
        }

    return result
