"""Implementation Shortfall (Almgren-Chriss) optimal execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ISConfig:
    """Implementation Shortfall configuration."""
    risk_aversion: float = 1e-6  # λ: risk aversion parameter
    temporary_impact: float = 0.01  # η: temporary impact (bps)
    permanent_impact: float = 0.005  # γ: permanent impact (bps)
    volatility: float = 0.02  # σ: annualized volatility
    alpha_decay: float = 0.0  # κ: signal decay rate (per day)


@dataclass
class ISSchedule:
    """Optimal execution trajectory."""
    times: NDArray[np.floating]
    shares_remaining: NDArray[np.floating]
    trading_rate: NDArray[np.floating]
    total_shares: float
    T: float
    expected_cost: float
    cost_variance: float


def compute_almgren_chriss(
    total_shares: float,
    T: float,  # Time horizon in days (e.g., 1/6.5 for 1 hour)
    sigma: float,  # Annualized volatility
    eta: float,  # Temporary impact parameter
    gamma: float,  # Permanent impact parameter
    lambda_: float,  # Risk aversion
    n_steps: int = 100,
) -> ISSchedule:
    """Compute the optimal Almgren-Chriss execution trajectory.

    Analytical solution for linear temporary + permanent impact:
    x(t) = X * sinh(κ(T-t)) / sinh(κT)

    where κ = sqrt(λσ²/η)

    Args:
        total_shares: Total shares to execute.
        T: Time horizon (in years-equivalent, or days).
        sigma: Annualized volatility.
        eta: Temporary impact (cost per unit of trading rate).
        gamma: Permanent impact (cost per share, independent of rate).
        lambda_: Risk aversion parameter.
        n_steps: Number of discrete steps.

    Returns:
        ISSchedule with optimal trajectory.
    """
    X = total_shares
    T_days = T * 252  # Convert to day fraction

    # Adjusted parameters
    sigma_daily = sigma / np.sqrt(252)

    # Almgren-Chriss κ parameter
    kappa_sq = (lambda_ * sigma_daily**2) / eta
    kappa = np.sqrt(max(kappa_sq, 1e-16))

    # Discrete time points
    t = np.linspace(0, T, n_steps)

    # Optimal remaining shares: x(t) = X * sinh(κ(T-t)) / sinh(κT)
    if kappa * T < 1e-8:
        # Near-zero κ → uniform schedule (TWAP)
        x_t = X * (1.0 - t / T)
    else:
        x_t = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

    # Trading rate: -dx/dt
    rate = np.zeros(n_steps)
    rate[0] = (x_t[0] - x_t[1]) / (t[1] - t[0]) if n_steps > 1 else 0.0
    for i in range(1, n_steps - 1):
        rate[i] = (x_t[i - 1] - x_t[i + 1]) / (2.0 * (t[1] - t[0]))
    rate[-1] = (x_t[-2] - x_t[-1]) / (t[-1] - t[-2]) if n_steps > 1 else 0.0

    # Expected cost
    tau = T / n_steps
    expected_cost = 0.0
    for i in range(n_steps):
        ri = rate[i]
        xi = x_t[i]
        # Temporary impact cost: η * r_i² * τ
        temp_cost = eta * ri**2 * tau
        # Permanent impact: γ * r_i * X (integrated)
        perm_cost = gamma * ri * xi * tau
        # Risk: λ * σ² * x_i² * τ
        risk_cost = lambda_ * sigma_daily**2 * xi**2 * tau
        expected_cost += temp_cost + perm_cost + risk_cost

    # Cost variance
    cost_var = 0.0
    for i in range(n_steps):
        cost_var += sigma_daily**2 * x_t[i]**2 * tau

    return ISSchedule(
        times=t,
        shares_remaining=x_t,
        trading_rate=rate,
        total_shares=X,
        T=T,
        expected_cost=float(expected_cost),
        cost_variance=float(cost_var),
    )


def compute_trajectory(
    total_shares: float,
    T: float,
    sigma: float,
    eta: float,
    gamma: float,
    lambda_: float,
    n_steps: int = 100,
) -> dict[str, NDArray[np.floating]]:
    """Compute Almgren-Chriss optimal trajectory.

    Returns dict with 'times', 'remaining', 'rate'.
    """
    sched = compute_almgren_chriss(total_shares, T, sigma, eta, gamma, lambda_, n_steps)
    return {
        "times": sched.times,
        "remaining": sched.shares_remaining,
        "rate": sched.trading_rate,
        "expected_cost": np.full(1, sched.expected_cost),
    }


def compute_trajectory_with_alpha(
    total_shares: float,
    T: float,
    sigma: float,
    eta: float,
    gamma: float,
    lambda_: float,
    alpha: float,  # Expected alpha per day
    alpha_decay: float = 0.0,  # Decay rate per day
    n_steps: int = 100,
) -> ISSchedule:
    """Compute optimal trajectory incorporating alpha decay.

    If signal decays fast (alpha_decay > 0), front-load execution to capture alpha
    before it decays.

    Modified objective includes alpha term:
    max ∫ [α_t * x_t - η * r_t²] dt - λ * σ² * ∫ x_t² dt
    """
    X = total_shares
    t = np.linspace(0, T, n_steps)
    tau = T / n_steps

    sigma_daily = sigma / np.sqrt(252)

    # Alpha at each time point
    alpha_t = alpha * np.exp(-alpha_decay * t)

    # Base AC without alpha
    base = compute_almgren_chriss(X, T, sigma, eta, gamma, lambda_, n_steps)

    # Adjust trajectory for alpha: shift shares toward times with higher alpha
    # Simplified: add a front-loading term proportional to alpha
    kappa_sq = (lambda_ * sigma_daily**2) / eta
    kappa = np.sqrt(max(kappa_sq, 1e-16))

    if kappa * T < 1e-8 or alpha == 0:
        return base

    # Alpha-adjusted trajectory
    alpha_weight = alpha_t - np.mean(alpha_t)
    adjustment = alpha_weight * X * tau / (2.0 * eta) if eta > 0 else np.zeros_like(t)

    x_adj = base.shares_remaining - adjustment
    x_adj = np.clip(x_adj, 0, X)

    # Renormalize
    x_adj = x_adj / x_adj[0] * X

    # Rates from adjusted trajectory
    rate_adj = np.zeros(n_steps)
    rate_adj[0] = (x_adj[0] - x_adj[1]) / tau if n_steps > 1 else 0.0
    for i in range(1, n_steps - 1):
        rate_adj[i] = (x_adj[i - 1] - x_adj[i + 1]) / (2.0 * tau)
    rate_adj[-1] = (x_adj[-2] - x_adj[-1]) / tau if n_steps > 1 else 0.0

    # Recompute expected cost
    expected_cost = 0.0
    for i in range(n_steps):
        temp_cost = eta * rate_adj[i]**2 * tau
        perm_cost = gamma * rate_adj[i] * x_adj[i] * tau
        risk_cost = lambda_ * sigma_daily**2 * x_adj[i]**2 * tau
        alpha_benefit = -alpha_t[i] * x_adj[i] * tau  # Negative = benefit
        expected_cost += temp_cost + perm_cost + risk_cost + alpha_benefit

    cost_var = 0.0
    for i in range(n_steps):
        cost_var += sigma_daily**2 * x_adj[i]**2 * tau

    return ISSchedule(
        times=t,
        shares_remaining=x_adj,
        trading_rate=rate_adj,
        total_shares=X,
        T=T,
        expected_cost=float(expected_cost),
        cost_variance=float(cost_var),
    )
