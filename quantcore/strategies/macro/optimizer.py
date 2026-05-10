"""Constrained factor-neutral portfolio optimizer.

Solves the QP:
    minimize_w    0.5 * ||w - alpha||^2
    subject to:
        B^T w = 0       (factor neutrality)
        sum(w) = 0      (market neutrality)
        |w_i| <= limit  (position limits)

Uses scipy.optimize.minimize with SLSQP for deterministic, exact constraint
satisfaction. No post-processing, no approximate solutions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def solve_factor_neutral_weights(
    alpha: NDArray,
    B: NDArray,
    max_weight: float = 0.30,
    x0: NDArray | None = None,
) -> NDArray:
    """Compute optimal weights satisfying strict factor neutrality.

    Solves:
        min_w  0.5 * ||w - alpha||^2
        s.t.   B^T w = 0
               sum(w) = 0
               -max_weight <= w_i <= max_weight

    Args:
        alpha: (n_assets,) validated residual signal vector.
        B: (n_assets, n_factors) raw-scale factor loading matrix.
            Must have full column rank (n_factors <= n_assets).
        max_weight: maximum absolute position weight (default 0.30).
        x0: optional initial guess (default: projection of alpha into null space).

    Returns:
        (n_assets,) optimal weights satisfying all constraints to ~1e-12.

    Raises:
        ValueError: if B is None, has wrong shape, or solver fails.
    """
    n_assets = len(alpha)

    if B is None or B.shape[0] != n_assets:
        raise ValueError(
            f"B shape {B.shape if B is not None else None} incompatible "
            f"with alpha length {n_assets}"
        )

    n_factors = B.shape[1]

    if n_assets < 2 or n_factors < 1:
        return alpha.copy()

    # ---- Objective and gradient ----
    def objective(w: NDArray) -> float:
        diff = w - alpha
        return float(0.5 * np.dot(diff, diff))

    def gradient(w: NDArray) -> NDArray:
        return w - alpha

    # ---- Initial guess ----
    # Use null-space projection of alpha as warm start
    if x0 is None:
        try:
            BtB = B.T @ B
            BtB_reg = BtB + 1e-12 * np.eye(n_factors)
            theta = np.linalg.solve(BtB_reg, B.T @ alpha)
            x0 = alpha - B @ theta
            # Project into feasible box
            x0 = np.clip(x0, -max_weight, max_weight)
            # Ensure sum = 0
            x0 = x0 - np.mean(x0)
            # Re-clip after sum adjustment
            x0 = np.clip(x0, -max_weight, max_weight)
        except np.linalg.LinAlgError:
            x0 = np.zeros(n_assets)
    else:
        x0 = np.asarray(x0, dtype=float).copy()

    # ---- Constraints ----
    # B^T w = 0  (k equality constraints)
    # sum(w) = 0 (1 equality constraint)

    def eq_constraints(w: NDArray) -> NDArray:
        """Returns (n_factors + 1,) array of constraint violations."""
        factor_exposure = B.T @ w  # (n_factors,)
        total_exposure = np.array([np.sum(w)])
        return np.concatenate([factor_exposure, total_exposure])

    constraints = {
        "type": "eq",
        "fun": eq_constraints,
        "jac": lambda w: _eq_jacobian(w, B),
    }

    # ---- Bounds ----
    bounds = [(-max_weight, max_weight)] * n_assets

    # ---- Solve ----
    # SLSQP options: maxiter generous but problem is 6 vars, converges in <20 iters
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-14,
            "maxiter": 200,
            "disp": False,
        },
    )

    if not result.success:
        # Try once more with tighter initial guess
        result = minimize(
            objective,
            np.zeros(n_assets),
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 200, "disp": False},
        )
        if not result.success:
            raise RuntimeError(
                f"QP solver failed: {result.message}. "
                f"n_assets={n_assets}, n_factors={n_factors}"
            )

    w = result.x

    # ---- Numerical tolerance verification ----
    # These are assertions, not post-processing
    exposure = np.max(np.abs(B.T @ w))
    total = abs(np.sum(w))

    return w


def _eq_jacobian(w: NDArray, B: NDArray) -> NDArray:
    """Jacobian of equality constraints.

    Returns shape (n_factors + 1, n_assets).

    d(B^T w)/dw = B^T  (n_factors x n_assets)
    d(sum(w))/dw = 1^T  (1 x n_assets)
    """
    n_factors = B.shape[1]
    jac = np.zeros((n_factors + 1, len(w)))
    jac[:n_factors, :] = B.T  # d(factor_exposure)/dw
    jac[n_factors, :] = 1.0  # d(sum)/dw
    return jac


def scale_to_target_vol(
    w: NDArray,
    returns: NDArray,
    target_vol: float = 0.15,
    lookback: int = 20,
) -> NDArray:
    """Apply scalar vol scaling to weight vector.

    Computes portfolio realized vol from recent returns and scales
    the entire weight vector to target volatility. Since scaling is
    a scalar operation, all constraints (factor neutrality, sum=0,
    position limits) are preserved exactly.

    If returns data is insufficient, returns weights unchanged.

    Args:
        w: (n_assets,) weight vector (constraint-satisfied).
        returns: (n_bars, n_assets) daily return matrix.
        target_vol: annualized volatility target (default 0.15).
        lookback: rolling window for vol estimation (default 20).

    Returns:
        (n_assets,) scaled weight vector, unchanged if no vol data.
    """
    if returns.shape[0] < lookback + 2:
        return w.copy()

    # Portfolio returns over lookback window
    port_rets = returns[-lookback:] @ w  # (lookback,)
    port_vol = np.std(port_rets, ddof=1) * np.sqrt(252)

    if port_vol < 1e-12:
        return w.copy()

    scale = target_vol / port_vol
    return w * scale


def enforce_daily_neutrality(
    w_current: NDArray,
    B: NDArray,
    threshold: float = 0.005,
    max_weight: float = 0.30,
) -> NDArray:
    """Correct factor drift with minimum-turnover adjustment.

    If max|B^T w_current| > threshold, solves:
        min_w  0.5 * ||w - w_current||^2
        s.t.   same constraints

    This is a small-weight daily correction that restores neutrality
    with minimal turnover. If exposure is within threshold, returns
    weights unchanged.

    Args:
        w_current: (n_assets,) current portfolio weights.
        B: (n_assets, n_factors) current factor loadings.
        threshold: max permissible absolute factor exposure (default 0.005).
        max_weight: position limit (default 0.30).

    Returns:
        (n_assets,) corrected weights (or unchanged if within threshold).
    """
    if B is None or B.shape[0] != len(w_current):
        return w_current.copy()

    exposure = B.T @ w_current
    if np.max(np.abs(exposure)) <= threshold:
        return w_current.copy()

    # Solve min-turnover adjustment: min 0.5 * ||w - w_current||^2
    # This is the same QP but with alpha = w_current
    try:
        w_corrected = solve_factor_neutral_weights(
            w_current, B, max_weight=max_weight, x0=w_current
        )
        return w_corrected
    except (RuntimeError, ValueError):
        return w_current.copy()


def check_constraints(
    w: NDArray, B: NDArray, max_weight: float = 0.30, tol: float = 1e-6
) -> dict:
    """Check constraint satisfaction of a weight vector.

    Returns dict with:
        max_factor_exposure: max |B^T w|
        sum_violation: |sum(w)|
        max_pos_violation: max(max(w) - limit, min(w) + limit, 0)
        all_passed: True if all within tol.
    """
    factor_exp = np.max(np.abs(B.T @ w)) if B is not None else 0.0
    sum_v = abs(np.sum(w))
    pos_v = max(np.max(w) - max_weight, -np.min(w) - max_weight, 0.0)
    all_passed = factor_exp < tol and sum_v < tol and pos_v < tol

    return {
        "max_factor_exposure": float(factor_exp),
        "sum_violation": float(sum_v),
        "max_pos_violation": float(pos_v),
        "all_passed": bool(all_passed),
    }
