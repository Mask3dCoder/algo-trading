"""Black-Scholes-Merton option pricing and analytical Greeks.

All formulas follow Hull, "Options, Futures, and Other Derivatives".
Vectorized — accepts numpy arrays for S, K, T, r, sigma.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

ScalarOrArray = float | NDArray[np.floating]


def _d1(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Compute d1 in the BSM formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Compute d2 in the BSM formula."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bsm_call(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """BSM European call price.

    C = S*N(d1) - K*exp(-rT)*N(d2)
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_put(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """BSM European put price.

    P = K*exp(-rT)*N(-d2) - S*N(-d1)
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# First-order Greeks
# ---------------------------------------------------------------------------


def call_delta(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Call delta: N(d1)."""
    return norm.cdf(_d1(S, K, T, r, sigma))


def put_delta(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Put delta: N(d1) - 1."""
    return norm.cdf(_d1(S, K, T, r, sigma)) - 1.0


def gamma(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Gamma (same for call and put): N'(d1) / (S * sigma * sqrt(T))."""
    d1_val = _d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def vega(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Vega (same for call and put): S * N'(d1) * sqrt(T).

    Convention: dV/dsigma per 1-unit change in sigma (i.e. per 100% vol move).
    For dV/dsigma per 1% vol move, divide by 100.
    """
    d1_val = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1_val) * np.sqrt(T)


def call_theta(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Call theta (per year). Divide by 252 for per-calendar-day, by 365 for per-day."""
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    return (
        -S * norm.pdf(d1_val) * sigma / (2.0 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2_val)
    )


def put_theta(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Put theta (per year)."""
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    return (
        -S * norm.pdf(d1_val) * sigma / (2.0 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2_val)
    )


def call_rho(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Call rho: K*T*exp(-rT)*N(d2)."""
    d2_val = _d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2_val)


def put_rho(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Put rho: -K*T*exp(-rT)*N(-d2)."""
    d2_val = _d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2_val)


# ---------------------------------------------------------------------------
# Second- and third-order Greeks
# ---------------------------------------------------------------------------


def vanna(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Vanna: d²V/(dS dσ) = -vega / S * d1 * d2 / (σ² * T).

    Simplified: vega * (1 - d1/(σ√T)) / S  ... or  -N'(d1)*d2/σ
    """
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    return -norm.pdf(d1_val) * d2_val / sigma


def volga(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Volga (Vomma): d²V/dσ² = vega * d1*d2 / σ."""
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    v = vega(S, K, T, r, sigma)
    return v * d1_val * d2_val / sigma


def charm(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
    is_call: bool = True,
) -> ScalarOrArray:
    """Charm (delta decay): -dDelta/dT.

    For call: -N'(d1)*(2(r-σ²/2)T - σ√T*d2) / (2σT^(3/2)) + r*exp(-rT)*N(d2)
    """
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1_val)
    term1 = -pdf_d1 * (2.0 * (r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * d2_val) / (
        2.0 * sigma * T ** 1.5
    )
    cdf_sign = 1.0 if is_call else -1.0
    term2 = r * np.exp(-r * T) * norm.cdf(cdf_sign * d2_val)
    return term1 - cdf_sign * term2


def speed(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Speed: d³V/dS³ = -Gamma/S * (d1/(σ√T) + 1)."""
    d1_val = _d1(S, K, T, r, sigma)
    g = gamma(S, K, T, r, sigma)
    return -g / S * (d1_val / (sigma * np.sqrt(T)) + 1.0)


def color(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Color: dGamma/dT = -Gamma/(2T) * (2(r-σ²/2)T - σ√T*d2)/(σ√T) + σ²/2 * Gamma."""
    _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    g = gamma(S, K, T, r, sigma)
    return -g / (2.0 * T) * (
        2.0 * (r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * d2_val
    ) / (sigma * np.sqrt(T)) + 0.5 * sigma**2 * g


def ultima(
    S: ScalarOrArray,
    K: ScalarOrArray,
    T: ScalarOrArray,
    r: ScalarOrArray,
    sigma: ScalarOrArray,
) -> ScalarOrArray:
    """Ultima: d³V/dσ³ = -volga/σ² * (d1*d2*(d1*d2 - σ²*T) - d1² - d2²)."""
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    v = volga(S, K, T, r, sigma)
    sst = sigma**2 * T
    return -v / sigma**2 * (d1_val * d2_val * (d1_val * d2_val - sst) - d1_val**2 - d2_val**2)


# ---------------------------------------------------------------------------
# Numerical Greeks (finite differences — for verification)
# ---------------------------------------------------------------------------


def numerical_greek(
    price_fn,
    param: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    epsilon: float | None = None,
    is_call: bool = True,
) -> float:
    """Compute a Greek numerically via central finite differences.

    price_fn: one of bsm_call or bsm_put
    param: one of 'S', 'K', 'T', 'r', 'sigma'
    """
    defaults = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
    eps_map = {"S": 0.01, "K": 0.01, "T": 1e-6, "r": 1e-6, "sigma": 1e-6}
    eps = epsilon or eps_map[param]

    up = {**defaults, param: defaults[param] + eps}
    down = {**defaults, param: defaults[param] - eps}
    price_up = price_fn(**up)
    price_down = price_fn(**down)
    if param == "T":
        # Theta is negative of dP/dT
        return -(float(price_up) - float(price_down)) / (2.0 * eps)
    return (float(price_up) - float(price_down)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# Put-call parity verification
# ---------------------------------------------------------------------------


def put_call_parity_check(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Return |C - P - S + K*exp(-rT)|. Should be zero (machine precision)."""
    c = float(bsm_call(S, K, T, r, sigma))
    p = float(bsm_put(S, K, T, r, sigma))
    return abs(c - p - S + K * np.exp(-r * T))
