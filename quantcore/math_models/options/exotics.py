"""Exotic options pricing: Asian, barrier, American, binary, lookback."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from quantcore.math_models.options.bsm import bsm_call, bsm_put
from quantcore.math_models.stochastic.monte_carlo import MonteCarloEngine

# ---------------------------------------------------------------------------
# Asian Options
# ---------------------------------------------------------------------------


def asian_geometric_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int,
) -> float:
    """Analytical price for geometric average Asian call.

    Uses the fact that geometric average of lognormal is lognormal
    with adjusted drift and volatility.
    """
    sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_adj = (r - 0.5 * sigma**2) * 0.5 + 0.5 * sigma_adj**2
    d1 = (np.log(S / K) + (mu_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)
    return np.exp(-r * T) * (S * np.exp(mu_adj * T) * norm.cdf(d1) - K * norm.cdf(d2))


def asian_geometric_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int,
) -> float:
    """Analytical price for geometric average Asian put via put-call parity."""
    c = asian_geometric_call(S, K, T, r, sigma, n)
    # Adjusted forward for geometric average
    sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_adj = (r - 0.5 * sigma**2) * 0.5 + 0.5 * sigma_adj**2
    fwd = S * np.exp(mu_adj * T)
    return c - fwd * np.exp(-r * T) + K * np.exp(-r * T)


def asian_arithmetic_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int,
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo price for arithmetic average Asian call.

    Returns (price, standard_error).
    """
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n, seed=seed)
    paths = engine.generate_gbm_paths(S, r, sigma, T)

    # Arithmetic average of price path (excluding S0)
    avg_prices = np.mean(paths[:, 1:], axis=1)
    payoffs = np.maximum(avg_prices - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return price, std_err


def asian_arithmetic_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int,
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo price for arithmetic average Asian put."""
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n, seed=seed)
    paths = engine.generate_gbm_paths(S, r, sigma, T)
    avg_prices = np.mean(paths[:, 1:], axis=1)
    payoffs = np.maximum(K - avg_prices, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return price, std_err


# ---------------------------------------------------------------------------
# Barrier Options
# ---------------------------------------------------------------------------


def barrier_call_down_out(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Analytical price for down-and-out call (H < S, H < K)."""
    if H >= S:
        return 0.0  # Already knocked out
    if H >= K:
        # Use reflection principle
        return _barrier_do_call_h_above_k(S, K, H, T, r, sigma)
    return _barrier_do_call_h_below_k(S, K, H, T, r, sigma)


def _barrier_do_call_h_below_k(
    S: float, K: float, H: float, T: float, r: float, sigma: float,
) -> float:
    """Down-and-out call when H <= K."""
    lam = (r + 0.5 * sigma**2) / sigma**2
    x = np.log(S / H) / (sigma * np.sqrt(T))
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T))

    A = S * norm.cdf(x + sigma * np.sqrt(T)) - K * np.exp(-r * T) * norm.cdf(x)
    B = S * (H / S) ** (2 * lam) * norm.cdf(-y - sigma * np.sqrt(T))
    C = K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(-y)
    return float(A - B + C)


def _barrier_do_call_h_above_k(
    S: float, K: float, H: float, T: float, r: float, sigma: float,
) -> float:
    """Down-and-out call when H > K (uses in-out parity)."""
    # Down-and-in call
    lam = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T))
    np.log(S / H) / (sigma * np.sqrt(T))

    din = S * (H / S) ** (2 * lam) * norm.cdf(-y + sigma * np.sqrt(T)) \
        - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(-y)
    # In-out parity: down-and-out = vanilla - down-and-in
    vanilla = float(bsm_call(S, K, T, r, sigma))
    return vanilla - din


def barrier_put_up_out(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Analytical price for up-and-out put (H > S, H > K)."""
    if H <= S:
        return 0.0
    lam = (r + 0.5 * sigma**2) / sigma**2
    x = np.log(S / H) / (sigma * np.sqrt(T))
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T))

    # Up-and-in put components
    uin_A = -S * norm.cdf(-x - sigma * np.sqrt(T)) + K * np.exp(-r * T) * norm.cdf(-x)
    uin_B = S * (H / S) ** (2 * lam) * (norm.cdf(y + sigma * np.sqrt(T)) - 1) \
        - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * (norm.cdf(y) - 1)

    if H > K:
        uin = -uin_A + uin_B + float(bsm_call(S, H, T, r, sigma)) \
            + (H - K) * np.exp(-r * T) * norm.cdf(x - sigma * np.sqrt(T) + 2 * lam * sigma * np.sqrt(T))
    else:
        uin = uin_B

    vanilla = float(bsm_put(S, K, T, r, sigma))
    return vanilla - uin


# ---------------------------------------------------------------------------
# American Options — Longstaff-Schwartz Method
# ---------------------------------------------------------------------------


def american_option_lsm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 50_000,
    n_steps: int = 200,
    is_call: bool = False,
    seed: int = 42,
    poly_degree: int = 3,
) -> float:
    """American option via Longstaff-Schwartz least-squares Monte Carlo.

    Defaults to put (most common American option).
    """
    dt = T / n_steps
    np.exp(-r * dt)
    rng = np.random.default_rng(seed)

    # Generate paths
    z = rng.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * z
    log_prices = np.log(S) + np.cumsum(log_returns, axis=1)
    paths = np.exp(log_prices)

    # Insert S0
    paths = np.column_stack([np.full(n_paths, S), paths])

    # Payoff at each step
    if is_call:
        payoffs = np.maximum(paths - K, 0.0)
    else:
        payoffs = np.maximum(K - paths, 0.0)

    # LSM backward induction
    cashflows = payoffs[:, -1].copy()
    exercise_times = np.full(n_paths, n_steps, dtype=int)

    for t in range(n_steps - 1, 0, -1):
        in_the_money = payoffs[:, t] > 0
        if not np.any(in_the_money):
            continue

        # Continuation value from discounted future cashflows
        exercise_times_masked = exercise_times[in_the_money]
        cont_values = cashflows[in_the_money] * np.exp(-r * dt * (exercise_times_masked - t))

        # Regression of continuation value on polynomial of price
        X = paths[in_the_money, t]
        A = np.vander(X - np.mean(X), N=poly_degree + 1, increasing=True)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, cont_values, rcond=None)
            predicted_cont = A @ coeffs
        except np.linalg.LinAlgError:
            continue

        # Exercise where immediate payoff > predicted continuation
        exercise = payoffs[in_the_money, t] > predicted_cont
        indices = np.where(in_the_money)[0]
        for idx, ex in zip(indices, exercise):
            if ex:
                cashflows[idx] = payoffs[idx, t]
                exercise_times[idx] = t

    # Discount cashflows back to t=0
    total_discount = np.exp(-r * dt * exercise_times)
    return float(np.mean(cashflows * total_discount))


# ---------------------------------------------------------------------------
# Binary / Digital Options
# ---------------------------------------------------------------------------


def binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    payout: float = 1.0,
) -> float:
    """Cash-or-nothing binary call: pays `payout` if S_T > K."""
    from quantcore.math_models.options.bsm import _d2
    d2 = float(_d2(S, K, T, r, sigma))
    return payout * np.exp(-r * T) * norm.cdf(d2)


def binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    payout: float = 1.0,
) -> float:
    """Cash-or-nothing binary put: pays `payout` if S_T < K."""
    from quantcore.math_models.options.bsm import _d2
    d2 = float(_d2(S, K, T, r, sigma))
    return payout * np.exp(-r * T) * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# Lookback Options (Monte Carlo)
# ---------------------------------------------------------------------------


def lookback_call_floating(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> tuple[float, float]:
    """Floating-strike lookback call: payoff = S_T - min(S_t).

    Returns (price, standard_error).
    """
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed)
    paths = engine.generate_gbm_paths(S, r, sigma, T)
    min_prices = np.min(paths, axis=1)
    payoffs = paths[:, -1] - min_prices
    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return price, std_err


def lookback_put_floating(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> tuple[float, float]:
    """Floating-strike lookback put: payoff = max(S_t) - S_T."""
    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=seed)
    paths = engine.generate_gbm_paths(S, r, sigma, T)
    max_prices = np.max(paths, axis=1)
    payoffs = max_prices - paths[:, -1]
    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return price, std_err
