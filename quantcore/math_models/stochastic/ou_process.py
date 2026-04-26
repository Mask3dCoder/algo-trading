"""Ornstein-Uhlenbeck process: simulation, analytical properties, MLE estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def simulate_ou(
    X0: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float = 1.0 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate OU process using exact discretization.

    dX = theta*(mu - X)*dt + sigma*dW

    Exact: X_{t+dt} = mu + (X_t - mu)*exp(-theta*dt) + sigma*sqrt((1-exp(-2*theta*dt))/(2*theta))*Z
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    z = rng.standard_normal((n_paths, n_steps))

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = X0

    decay = np.exp(-theta * dt)
    vol = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))

    for t in range(n_steps):
        paths[:, t + 1] = mu + (paths[:, t] - mu) * decay + vol * z[:, t]

    return paths


def analytical_mean(X0: float, theta: float, mu: float, t: float) -> float:
    """Analytical expected value: E[X_t] = mu + (X0 - mu)*exp(-theta*t)."""
    return mu + (X0 - mu) * np.exp(-theta * t)


def analytical_variance(theta: float, sigma: float, t: float) -> float:
    """Analytical variance: Var[X_t] = sigma^2/(2*theta)*(1 - exp(-2*theta*t))."""
    return sigma**2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * t))


def half_life(theta: float) -> float:
    """Half-life of mean reversion: ln(2)/theta."""
    if theta <= 0:
        return float("inf")
    return np.log(2.0) / theta


def estimate_params(series: NDArray, dt: float = 1.0 / 252) -> tuple[float, float, float]:
    """Estimate OU parameters (theta, mu, sigma) via MLE.

    Uses the regression-based approach:
    X_{t+dt} = a + b*X_t + epsilon
    where theta = -ln(b)/dt, mu = a/(1-b), sigma = std(epsilon)*sqrt(-2*ln(b)/(dt*(1-b^2)))
    """
    X = np.asarray(series, dtype=float)
    X_prev = X[:-1]
    X_next = X[1:]

    # OLS regression: X_next = a + b * X_prev
    n = len(X_prev)
    sum_x = np.sum(X_prev)
    sum_y = np.sum(X_next)
    sum_xy = np.sum(X_prev * X_next)
    sum_x2 = np.sum(X_prev**2)

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    a = (sum_y - b * sum_x) / n

    # Residuals
    residuals = X_next - a - b * X_prev
    sigma_eps = np.std(residuals, ddof=1)

    # Convert to OU parameters
    if b <= 0 or b >= 1:
        # Non-mean-reverting — return boundary values
        return 0.001, float(np.mean(X)), sigma_eps * 10.0

    theta_est = -np.log(b) / dt
    mu_est = a / (1.0 - b)
    sigma_est = sigma_eps * np.sqrt(-2.0 * np.log(b) / (dt * (1.0 - b**2)))

    return float(theta_est), float(mu_est), float(sigma_est)
