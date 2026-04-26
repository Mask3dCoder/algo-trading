"""Implied volatility solvers: Newton-Raphson, Brent, and auto-select."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq

from quantcore.math_models.options.bsm import bsm_call, bsm_put, vega


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    tol: float = 1e-12,
    max_iter: int = 100,
    initial_guess: float | None = None,
) -> float:
    """Newton-Raphson implied vol solver using analytical vega.

    Converges quadratically near the root. Falls back to Brent if vega ≈ 0.
    """
    if T <= 0:
        return np.nan

    price_fn = bsm_call if is_call else bsm_put
    intrinsic = max(0.0, S - K * np.exp(-r * T)) if is_call else max(0.0, K * np.exp(-r * T) - S)

    if market_price < intrinsic - 1e-10:
        warnings.warn("Market price below intrinsic value", stacklevel=2)
        return np.nan

    sigma = initial_guess if initial_guess is not None else _initial_guess(market_price, S, K, T, r, is_call)
    sigma = np.clip(sigma, 1e-6, 10.0)

    for _ in range(max_iter):
        model_price = float(price_fn(S, K, T, r, sigma))
        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma
        v = float(vega(S, K, T, r, sigma))
        if v < 1e-12:
            break
        sigma -= diff / v
        sigma = np.clip(sigma, 1e-6, 10.0)

    return implied_vol_brent(market_price, S, K, T, r, is_call, tol)


def implied_vol_brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    tol: float = 1e-12,
) -> float:
    """Brent's method IV solver — derivative-free, robust for edge cases."""
    if T <= 0:
        return np.nan

    price_fn = bsm_call if is_call else bsm_put

    def objective(sigma: float) -> float:
        return float(price_fn(S, K, T, r, sigma)) - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except ValueError:
        warnings.warn("IV solver: no root in [1e-6, 10.0]", stacklevel=2)
        return np.nan


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    tol: float = 1e-12,
) -> float:
    """Auto-select IV solver. Newton first (fast), Brent fallback (robust)."""
    return implied_vol_newton(market_price, S, K, T, r, is_call, tol)


def _initial_guess(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
) -> float:
    """Rational approximation initial guess (simplified Jaeckel-style).

    Falls back to sigma=0.2 if the approximation gives nonsense.
    """
    # Simple moneyness-based guess
    fwd = S * np.exp(r * T)
    moneyness = K / fwd
    # At-the-money-forward: use Brenner-Subrahmanyam approximation
    # ATM: sigma ≈ price * sqrt(2π) / (S * sqrt(T))
    x = moneyness - 1.0
    if abs(x) < 0.01:
        sigma_atm = market_price * np.sqrt(2.0 * np.pi) / (S * np.sqrt(T))
        return max(min(sigma_atm, 5.0), 0.01)

    # Away from ATM: rough guess based on moneyness
    guess = 0.2 + abs(x)
    return max(min(guess, 5.0), 0.01)
