"""Black-Litterman portfolio optimization for multi-strategy allocation.

Implements the Idzorek (2002) approach: start from market-equilibrium weights
(the "prior"), then tilt based on investor views on specific strategies.

Key formulas:
    Equilibrium returns:  Pi = lambda * Sigma @ w_mkt
    Posterior returns:    mu_hat = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
                                 * [(tau*Sigma)^-1 Pi + P' Omega^-1 Q]
    Posterior weights:    w_BL = (lambda * Sigma)^-1 @ mu_hat

When no explicit views are provided, default views use each strategy's
trailing Sharpe * 0.5 as the expected return (conservative shrinkage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bl_weights(
    returns: pd.DataFrame,
    views: list[dict] | None = None,
    market_weights: np.ndarray | None = None,
    tau: float = 0.025,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """Compute Black-Litterman portfolio weights.

    Args:
        returns: DataFrame of strategy daily returns (T x N).
        views: List of views, each a dict with:
               - 'type': 'absolute' or 'relative'
               - 'strategy': column name (absolute) or 'strategies': [long, short] (relative)
               - 'value': expected annualized return for the view
               If None, uses default conservative views (Sharpe * 0.5).
        market_weights: Prior weights (default: equal weight).
        tau: Prior uncertainty scalar (default 0.025).
        risk_aversion: Risk aversion parameter (default 2.5).

    Returns:
        Array of weights summing to 1, non-negative (no shorting).
    """
    if returns.empty or returns.shape[1] == 0:
        return np.array([])

    names = list(returns.columns)
    n = len(names)
    r = returns.values

    # Drop columns with zero variance
    col_stds = np.std(r, axis=0, ddof=1)
    valid_mask = col_stds > 1e-12
    if valid_mask.sum() == 0:
        return np.ones(n) / max(n, 1)

    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)
    r_valid = r[:, valid_idx]

    # 1. Annualized covariance matrix (252-day trailing)
    cov = np.cov(r_valid, rowvar=False, ddof=1)
    if cov.ndim == 1:
        cov = cov.reshape(1, 1)
    # Annualize
    cov_annual = cov * 252.0

    # Regularize if near-singular
    reg = np.eye(n_valid) * 1e-8
    cov_annual = cov_annual + reg

    # 2. Market weights (prior)
    if market_weights is not None:
        w_mkt = np.asarray(market_weights, dtype=float).flatten()
        if len(w_mkt) == n:
            w_mkt_valid = w_mkt[valid_idx]
        else:
            w_mkt_valid = np.ones(n_valid) / n_valid
    else:
        w_mkt_valid = np.ones(n_valid) / n_valid

    total = w_mkt_valid.sum()
    if total > 0:
        w_mkt_valid = w_mkt_valid / total

    # 3. Implied equilibrium returns: Pi = lambda * Sigma @ w_mkt
    pi = risk_aversion * cov_annual @ w_mkt_valid

    # 4. Build views
    if views is None or len(views) == 0:
        views = _default_views(r_valid, names, valid_idx)

    # 5. Build pick matrix P (k x n_valid) and view vector Q (k,)
    P, Q = _build_pick_matrix(views, names, valid_idx, n_valid)

    k = P.shape[0]  # number of views

    if k == 0:
        # No views — return equilibrium weights
        w_bl = w_mkt_valid.copy()
    else:
        # 6. View uncertainty Omega (k x k diagonal)
        # Idzorek approach: Omega = diag(P @ (tau * Sigma) @ P')
        # This scales view uncertainty proportionally to the variance
        # of the view portfolio — more volatile views are less certain.
        tau_cov = tau * cov_annual
        omega_diag = np.diag(P @ tau_cov @ P.T)
        # Ensure positive
        omega_diag = np.maximum(omega_diag, 1e-12)
        Omega = np.diag(omega_diag)

        # 7. Posterior returns
        # mu_hat = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
        #        * [(tau*Sigma)^-1 Pi + P' Omega^-1 Q]
        tau_cov_inv = np.linalg.inv(tau_cov)
        Omega_inv = np.linalg.inv(Omega)

        A = tau_cov_inv + P.T @ Omega_inv @ P
        A_inv = np.linalg.inv(A)

        mu_hat = A_inv @ (tau_cov_inv @ pi + P.T @ Omega_inv @ Q)

        # 8. Unconstrained weights: w = (lambda * Sigma)^-1 @ mu_hat
        lambda_sigma_inv = np.linalg.inv(risk_aversion * cov_annual)
        w_bl = lambda_sigma_inv @ mu_hat

    # 9. Map back to full strategy array
    weights = np.zeros(n)
    for j, idx in enumerate(valid_idx):
        weights[idx] = w_bl[j]

    # 10. Clip negative weights and renormalize (no shorting)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        # Fallback: equal weight
        weights = np.ones(n) / max(n, 1)

    return weights


def _default_views(
    r_valid: np.ndarray,
    names: list[str],
    valid_idx: np.ndarray,
) -> list[dict]:
    """Generate default views: each strategy's expected return = Sharpe * 0.5.

    This is a conservative shrinkage — the view is that each strategy will
    deliver half its historical Sharpe ratio as excess return going forward.
    """
    views = []
    for j, orig_idx in enumerate(valid_idx):
        col = r_valid[:, j]
        mean_ret = float(np.mean(col)) * 252.0  # annualized
        std_ret = float(np.std(col, ddof=1)) * np.sqrt(252.0)
        if std_ret > 1e-10:
            sharpe = mean_ret / std_ret
        else:
            sharpe = 0.0

        # Conservative: expected return = Sharpe * 0.5 * vol
        # This is equivalent to "half the historical Sharpe persists"
        expected_annual = sharpe * 0.5 * std_ret

        views.append({
            "type": "absolute",
            "strategy": names[orig_idx],
            "value": expected_annual,
        })

    return views


def _build_pick_matrix(
    views: list[dict],
    names: list[str],
    valid_idx: np.ndarray,
    n_valid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the pick matrix P (k x n_valid) and view vector Q (k,)."""
    name_to_valid_pos = {names[valid_idx[j]]: j for j in range(n_valid)}

    P_rows = []
    Q_vals = []

    for view in views:
        view_type = view.get("type", "absolute")
        value = float(view.get("value", 0.0))

        row = np.zeros(n_valid)

        if view_type == "absolute":
            strat_name = view.get("strategy", "")
            if strat_name in name_to_valid_pos:
                row[name_to_valid_pos[strat_name]] = 1.0
            else:
                continue  # skip views on non-valid strategies
        elif view_type == "relative":
            long_strat = view.get("strategies", ["", ""])[0]
            short_strat = view.get("strategies", ["", ""])[1]
            if long_strat in name_to_valid_pos and short_strat in name_to_valid_pos:
                row[name_to_valid_pos[long_strat]] = 1.0
                row[name_to_valid_pos[short_strat]] = -1.0
            else:
                continue
        else:
            continue

        P_rows.append(row)
        Q_vals.append(value)

    if not P_rows:
        return np.zeros((0, n_valid)), np.array([])

    return np.vstack(P_rows), np.array(Q_vals)
