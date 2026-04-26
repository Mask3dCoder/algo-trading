"""SABR model: simulation, Hagan implied vol approximation, calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def simulate_sabr(
    F0: float,
    sigma0: float,
    alpha: float,
    beta: float,
    rho: float,
    T: float,
    dt: float = 1.0 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Simulate SABR model: dF = sigma*F^beta*dW1, dsigma = alpha*sigma*dW2.

    Returns (forward_paths, vol_paths).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    F = np.zeros((n_paths, n_steps + 1))
    sigma = np.zeros((n_paths, n_steps + 1))
    F[:, 0] = F0
    sigma[:, 0] = sigma0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW1 = z1
        dW2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2

        sig = np.maximum(sigma[:, t], 1e-10)
        f = np.maximum(F[:, t], 1e-10)

        # Vol process (lognormal)
        sigma[:, t + 1] = sigma[:, t] * np.exp(alpha * dW2 * sqrt_dt - 0.5 * alpha**2 * dt)

        # Forward process
        if beta == 1.0:
            F[:, t + 1] = F[:, t] * np.exp(sig * dW1 * sqrt_dt - 0.5 * sig**2 * dt)
        else:
            F[:, t + 1] = F[:, t] + sig * f**beta * dW1 * sqrt_dt

    return F, sigma


def implied_vol_sabr(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    sigma0: float,
) -> float:
    """Hagan et al. (2002) approximation for SABR implied volatility.

    Valid for ATM and near-ATM. May produce negative vols for deep OTM —
    use the adjusted formula from Oblój (2008) for better behavior.
    """
    if F <= 0 or K <= 0 or T <= 0:
        return np.nan

    z = alpha / sigma0 * (F * K) ** ((1.0 - beta) / 2.0) * np.log(F / K)

    # Avoid division by zero at ATM
    if abs(F - K) < 1e-10:
        # ATM implied vol
        iv = sigma0 / F ** (1.0 - beta) * (
            1.0 + ((1.0 - beta) ** 2 / 24.0 * sigma0**2 / F ** (2.0 - 2.0 * beta)
                   + 0.25 * rho * beta * alpha * sigma0 / F ** (1.0 - beta)
                   + (2.0 - 3.0 * rho**2) / 24.0 * alpha**2) * T
        )
        return float(iv)

    x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

    sigma_b = alpha * z / x if abs(x) > 1e-10 else sigma0

    iv = sigma_b * (
        1.0
        + ((1.0 - beta) ** 2 / 24.0 * sigma0**2 / (F * K) ** (1.0 - beta)
           + 0.25 * rho * beta * alpha * sigma0 / (F * K) ** ((1.0 - beta) / 2.0)
           + (2.0 - 3.0 * rho**2) / 24.0 * alpha**2) * T
    )

    return float(max(iv, 1e-8))


def calibrate_sabr(
    strikes: NDArray,
    ivs: NDArray,
    F: float,
    T: float,
    beta: float = 0.5,
    initial_params: tuple = (0.3, 0.0, 0.2),
) -> tuple[float, float, float]:
    """Calibrate SABR to market implied vols.

    Returns (alpha, rho, sigma0).
    beta is typically fixed exogenously (0.5 for rates, ~1 for equities).
    """
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(ivs, dtype=float)

    def objective(x: NDArray) -> float:
        a, rho_val, sig0 = x
        total = 0.0
        for K, market_iv in zip(strikes, vols):
            model_iv = implied_vol_sabr(F, K, T, a, beta, rho_val, sig0)
            total += (model_iv - market_iv) ** 2
        return total

    bounds = [(0.01, 2.0), (-0.999, 0.999), (0.001, 2.0)]
    result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000})
    return float(result.x[0]), float(result.x[1]), float(result.x[2])
