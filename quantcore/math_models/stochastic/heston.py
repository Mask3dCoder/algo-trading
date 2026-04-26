"""Heston stochastic volatility model: simulation, characteristic function, calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import minimize


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float      # initial variance
    kappa: float    # mean reversion speed
    theta: float    # long-run variance
    xi: float       # vol of vol
    rho: float      # correlation between price and variance
    mu: float = 0.0  # drift


# Need dataclass import


def simulate_heston(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    dt: float = 1.0 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Simulate Heston model using full truncation scheme.

    Returns (prices, variances) each of shape (n_paths, n_steps+1).
    """
    from dataclasses import dataclass as _dc  # noqa — already imported above

    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    prices = np.zeros((n_paths, n_steps + 1))
    variances = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    variances[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # Correlated Brownian motions
        dW_s = z1
        dW_v = rho * z1 + np.sqrt(1.0 - rho**2) * z2

        v = variances[:, t]
        # Full truncation: use max(v, 0) in drift but allow v to go negative
        v_pos = np.maximum(v, 0.0)

        # Variance process: Euler scheme
        variances[:, t + 1] = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW_v * sqrt_dt
        # Reflect at zero (full truncation)
        variances[:, t + 1] = np.maximum(variances[:, t + 1], 0.0)

        # Price process
        s = prices[:, t]
        prices[:, t + 1] = s * np.exp(
            (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_s * sqrt_dt
        )

    return prices, variances


def heston_characteristic_function(
    u: complex,
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
) -> complex:
    """Heston model characteristic function (log-price).

    Using the formulation from Albrecher et al. (the "little H" formulation
    for numerical stability).
    """
    i = 1j
    x = np.log(S0)

    d = np.sqrt(
        (rho * xi * i * u - kappa) ** 2 + xi**2 * (i * u + u**2)
    )
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

    C = (
        r * i * u * T
        + kappa * theta / xi**2 * ((kappa - rho * xi * i * u - d) * T - 2.0 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    )
    D = (kappa - rho * xi * i * u - d) / xi**2 * (1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T))

    return np.exp(C + D * v0 + i * u * x)


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_points: int = 200,
) -> float:
    """European call price via Heston characteristic function (FFT-free integration)."""
    i = 1j

    def integrand_p1(u: float) -> complex:
        cf = heston_characteristic_function(u - i, S0, v0, r, kappa, theta, xi, rho, T)
        numerator = np.exp(-i * u * np.log(K)) * cf / (i * u * S0 * np.exp(r * T))
        return numerator.real

    def integrand_p2(u: float) -> complex:
        cf = heston_characteristic_function(u, S0, v0, r, kappa, theta, xi, rho, T)
        numerator = np.exp(-i * u * np.log(K)) * cf / (i * u)
        return numerator.real

    # Numerical integration
    p1 = 0.5 + 1.0 / np.pi * quad(integrand_p1, 0, 200, limit=n_points)[0]
    p2 = 0.5 + 1.0 / np.pi * quad(integrand_p2, 0, 200, limit=n_points)[0]

    return float(S0 * p1 - K * np.exp(-r * T) * p2)


def calibrate_heston(
    S0: float,
    market_prices: NDArray,
    strikes: NDArray,
    expiries: NDArray,
    r: float,
    initial_params: tuple = (0.04, 2.0, 0.04, 0.3, -0.5),
) -> HestonParams:
    """Calibrate Heston model to market option prices.

    Uses Levenberg-Marquardt (L-BFGS-B with bounds).
    """
    def objective(x: NDArray) -> float:
        v0, kappa, theta, xi, rho = x
        if kappa <= 0 or theta <= 0 or xi <= 0 or v0 <= 0:
            return 1e10
        total = 0.0
        for price, K, T in zip(market_prices, strikes, expiries):
            model_price = heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho)
            total += (model_price - price) ** 2
        return total

    bounds = [
        (1e-4, 1.0),    # v0
        (0.1, 10.0),    # kappa
        (1e-4, 1.0),    # theta
        (0.01, 2.0),    # xi
        (-0.999, 0.0),  # rho (typically negative for equity)
    ]

    result = minimize(
        objective,
        initial_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    return HestonParams(
        v0=result.x[0],
        kappa=result.x[1],
        theta=result.x[2],
        xi=result.x[3],
        rho=result.x[4],
    )
