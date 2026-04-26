"""Merton jump-diffusion model: simulation and characteristic function."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def simulate_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    dt: float = 1.0 / 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate Merton jump-diffusion: GBM + compound Poisson jumps.

    dS/S = (mu - lambda*kappa)*dt + sigma*dW + J*dN
    where J ~ lognormal(mu_j, sigma_j^2), kappa = exp(mu_j + sigma_j^2/2) - 1
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)

    # Compensator
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):
        s = paths[:, t]
        # Diffusion component
        z = rng.standard_normal(n_paths)
        diffusion = (mu - lam * kappa - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

        # Jump component
        n_jumps = rng.poisson(lam * dt, n_paths)
        jump_sizes = np.zeros(n_paths)
        for i in range(n_paths):
            if n_jumps[i] > 0:
                jump_z = rng.standard_normal(n_jumps[i])
                jump_log_sizes = mu_j + sigma_j * jump_z
                jump_sizes[i] = np.sum(jump_log_sizes)

        log_s = np.log(s) + diffusion + jump_sizes
        paths[:, t + 1] = np.exp(log_s)

    return paths


def jd_characteristic_function(
    u: complex,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    r: float,
) -> complex:
    """Characteristic function of log-price under Merton jump-diffusion."""
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    i = 1j
    # GBM component
    cf_gbm = np.exp(
        i * u * (r - 0.5 * sigma**2 - lam * kappa) * T
        - 0.5 * sigma**2 * u**2 * T
    )
    # Jump component
    cf_jump = np.exp(lam * T * (np.exp(i * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1.0))
    return cf_gbm * cf_jump


def jd_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    n_terms: int = 50,
) -> float:
    """European call price via Merton series expansion.

    C = sum_{k=0}^{inf} P(N_T=k) * BSM_call(S0*exp(-k*kappa_lambda*T+k*mu_j+k*sigma_j^2/2), K, T, r_k, sigma_k)
    where sigma_k^2 = sigma^2 + k*sigma_j^2/T, r_k = r - k*kappa/T + k*mu_j/T + k*sigma_j^2/(2T)
    """
    from scipy.stats import poisson

    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    total = 0.0
    for k in range(n_terms):
        prob_k = poisson.pmf(k, lam * T)
        if prob_k < 1e-12:
            break
        r_k = r + k * (mu_j + 0.5 * sigma_j**2) / T - k * kappa / T
        sigma_k = np.sqrt(sigma**2 + k * sigma_j**2 / T)
        S_adj = S0 * np.exp(k * mu_j + 0.5 * k * sigma_j**2 - k * kappa)

        from quantcore.math_models.options.bsm import bsm_call as _bsm_call
        price_k = float(_bsm_call(S_adj, K, T, r_k, sigma_k))
        total += prob_k * price_k

    return total
