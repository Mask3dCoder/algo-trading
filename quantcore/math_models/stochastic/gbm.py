"""Geometric Brownian Motion: simulation and parameter estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float = 1.0 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate GBM paths via Euler-Maruyama.

    Returns array of shape (n_paths, n_steps+1) with S0 in first column.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    z = rng.standard_normal((n_paths, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * z
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)
    return np.column_stack([np.full(n_paths, S0), prices])


def simulate_gbm_analytical(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate GBM using the analytical solution.

    S(T) = S0 * exp((mu - sigma^2/2)*T + sigma*W(T))
    Returns array of shape (n_paths,) — terminal prices only.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    log_ST = np.log(S0) + (mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z
    return np.exp(log_ST)


def simulate_gbm_milstein(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float = 1.0 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate GBM via Milstein scheme (higher-order correction).

    S_{t+dt} = S_t * (1 + mu*dt + sigma*sqrt(dt)*Z + 0.5*sigma^2*(Z^2 - 1)*dt)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    z = rng.standard_normal((n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        dW = sqrt_dt * z[:, t]
        paths[:, t + 1] = paths[:, t] * (
            1.0 + mu * dt + sigma * dW + 0.5 * sigma**2 * (dW**2 - dt) / dt * dt
        )
    return paths


def estimate_params(prices: NDArray) -> tuple[float, float]:
    """Estimate GBM parameters (mu, sigma) via MLE from log returns.

    For GBM, the MLE of mu and sigma from log returns r_t = ln(S_t/S_{t-1}):
      mu_hat = mean(r) / dt + 0.5 * var(r) / dt
      sigma_hat = std(r) / sqrt(dt)
    where dt is inferred from the data frequency (assumed daily = 1/252).
    """
    log_returns = np.diff(np.log(prices))
    dt = 1.0 / 252
    mu = np.mean(log_returns) / dt + 0.5 * np.var(log_returns, ddof=1) / dt
    sigma = np.std(log_returns, ddof=1) / np.sqrt(dt)
    return float(mu), float(sigma)


def estimate_params_from_returns(
    log_returns: NDArray,
    dt: float = 1.0 / 252,
) -> tuple[float, float]:
    """Estimate mu and sigma from a series of log returns."""
    mu = np.mean(log_returns) / dt + 0.5 * np.var(log_returns, ddof=1) / dt
    sigma = np.std(log_returns, ddof=1) / np.sqrt(dt)
    return float(mu), float(sigma)
