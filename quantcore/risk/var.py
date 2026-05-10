"""Value-at-Risk and Expected Shortfall engine.

Methods:
  - Historical simulation
  - Parametric (variance-covariance) with EWMA covariance
  - Cornish-Fisher adjustment
  - Monte Carlo (copula-based)
  - Filtered historical simulation (GARCH + historical on residuals)
  - CVaR / Expected Shortfall (marginal, component, incremental)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from quantcore.math_models.distributions.copulas import GaussianCopula
from quantcore.math_models.statistics.garch import fit_garch


@dataclass
class VaRResult:
    method: str
    confidence: float
    horizon_days: int
    var: float
    cvar: float | None = None
    std_error: float | None = None


# ---------------------------------------------------------------------------
# T-0301: Historical Simulation VaR
# ---------------------------------------------------------------------------


def historical_var(
    returns: NDArray[np.floating],
    confidence: float = 0.95,
    horizon_days: int = 1,
    scaling: str = "direct",
) -> VaRResult:
    """Historical simulation VaR.

    Args:
        returns: Array of historical returns (shape: n_periods,) or (n_periods, n_assets).
        confidence: VaR confidence level.
        horizon_days: Horizon in days. VaR scaled via √T (parametric) or direct simulation.
        scaling: 'sqrt' for √T scaling, 'direct' for overlapping-return simulation.

    Returns:
        VaRResult with var (negative = loss) at the given confidence level.
    """
    r = np.asarray(returns, dtype=float).flatten()
    if len(r) == 0:
        return VaRResult(
            method="historical",
            confidence=confidence,
            horizon_days=horizon_days,
            var=float("nan"),
            cvar=float("nan"),
        )
    if scaling == "direct" and horizon_days > 1 and len(r) > horizon_days + 1:
        # Direct simulation: construct overlapping h-period returns
        h_returns = np.array([
            np.sum(r[i : i + horizon_days]) for i in range(len(r) - horizon_days + 1)
        ])
    elif scaling == "sqrt" or horizon_days == 1:
        h_returns = r * np.sqrt(horizon_days)
    else:
        h_returns = r

    percentile = (1.0 - confidence) * 100.0
    var_val = float(np.percentile(h_returns, percentile))

    # CVaR from same historical data
    tail = h_returns[h_returns <= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return VaRResult(
        method="historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var=var_val,
        cvar=cvar_val,
    )


# ---------------------------------------------------------------------------
# T-0302: Parametric VaR + Cornish-Fisher
# ---------------------------------------------------------------------------


def _ewma_covariance(
    returns: NDArray[np.floating],
    lambda_: float = 0.94,
) -> NDArray[np.floating]:
    """Exponentially weighted moving average covariance matrix (RiskMetrics).

    Returns covariance matrix for the latest observation.
    """
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    # Remove unconditional mean
    r_demean = r - r.mean(axis=0)
    # EWMA weights
    decay = np.power(lambda_, np.arange(n)[::-1])
    w = (1.0 - lambda_) * decay / (1.0 - lambda_**n)
    return np.asarray(r_demean.T @ (r_demean * w[:, np.newaxis]))


def parametric_var(
    mu: float | NDArray[np.floating],
    sigma: float | NDArray[np.floating],
    confidence: float = 0.95,
    horizon_days: int = 1,
    weights: NDArray[np.floating] | None = None,
) -> VaRResult:
    """Parametric (Gaussian) VaR: VaR_α = -μ - σ * Φ⁻¹(1-α).

    Args:
        mu: Expected return (scalar or per-asset).
        sigma: Volatility (scalar or covariance matrix).
        confidence: VaR confidence level.
        horizon_days: Scaling horizon.
        weights: Portfolio weights. When provided with a covariance matrix,
            computes portfolio vol as sqrt(w'Σw) instead of averaging.
    """
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=float))
    mu_scalar = float(np.asarray(mu).mean())

    if sigma_arr.ndim <= 1:
        port_sigma = float(sigma_arr.mean())  # For scalar case
    elif weights is not None:
        w = np.asarray(weights, dtype=float).flatten()
        port_sigma = float(np.sqrt(w @ sigma_arr @ w))
    else:
        port_sigma = float(np.sqrt(np.mean(sigma_arr)))  # Simplified fallback

    z_alpha = sp_stats.norm.ppf(1.0 - confidence)
    var_val = mu_scalar * horizon_days + port_sigma * np.sqrt(horizon_days) * z_alpha
    cvar_val = mu_scalar * horizon_days - port_sigma * np.sqrt(horizon_days) * sp_stats.norm.pdf(z_alpha) / (1.0 - confidence)

    return VaRResult(
        method="parametric",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var_val),
        cvar=float(cvar_val),
    )


def portfolio_parametric_var(
    weights: NDArray[np.floating],
    cov: NDArray[np.floating],
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult:
    """Parametric VaR for a weighted portfolio.

    Portfolio variance = w' Σ w.
    """
    w = np.asarray(weights, dtype=float).flatten()
    sigma = np.asarray(cov, dtype=float)
    port_var = float(w @ sigma @ w)
    port_sigma = np.sqrt(port_var)

    z_alpha = sp_stats.norm.ppf(1.0 - confidence)
    var_val = port_sigma * np.sqrt(horizon_days) * z_alpha
    cvar_val = -port_sigma * np.sqrt(horizon_days) * sp_stats.norm.pdf(z_alpha) / (1.0 - confidence)

    return VaRResult(
        method="parametric_portfolio",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var_val),
        cvar=float(cvar_val),
    )


def cornish_fisher_var(
    mu: float,
    sigma: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult:
    """Cornish-Fisher VaR adjustment for non-normality.

    CF expansion: z_cf = z_α + (z_α² - 1)γ/6 + (z_α³ - 3z_α)κ/24 - (2z_α³ - 5z_α)γ²/36
    """
    z_alpha = sp_stats.norm.ppf(1.0 - confidence)
    z_cf = (
        z_alpha
        + (z_alpha**2 - 1.0) * skewness / 6.0
        + (z_alpha**3 - 3.0 * z_alpha) * (kurtosis - 3.0) / 24.0
        - (2.0 * z_alpha**3 - 5.0 * z_alpha) * skewness**2 / 36.0
    )

    var_val = mu * horizon_days + sigma * np.sqrt(horizon_days) * z_cf
    return VaRResult(
        method="cornish_fisher",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var_val),
    )


# ---------------------------------------------------------------------------
# T-0303: Monte Carlo VaR + Filtered Historical Simulation
# ---------------------------------------------------------------------------


def monte_carlo_var(
    returns: NDArray[np.floating],
    n_scenarios: int = 100_000,
    confidence: float = 0.95,
    horizon_days: int = 1,
    seed: int = 42,
) -> VaRResult:
    """Monte Carlo VaR via fitted distribution simulation.

    Fits a Student-t distribution to returns, generates n_scenarios,
    and computes the percentile loss.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim > 1:
        r = r.flatten()

    # Fit Student-t distribution
    df, loc, scale = sp_stats.t.fit(r)
    rng = np.random.default_rng(seed)
    simulated = sp_stats.t.rvs(df, loc=loc, scale=scale, size=n_scenarios, random_state=rng)
    simulated_h = simulated * np.sqrt(horizon_days)

    percentile = (1.0 - confidence) * 100.0
    var_val = float(np.percentile(simulated_h, percentile))
    tail = simulated_h[simulated_h <= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    # Standard error via bootstrap
    boot_vars = np.array([
        float(np.percentile(
            simulated_h[np.random.randint(0, n_scenarios, n_scenarios)], percentile))
        for _ in range(200)
    ])
    std_err = float(np.std(boot_vars))

    return VaRResult(
        method="monte_carlo",
        confidence=confidence,
        horizon_days=horizon_days,
        var=var_val,
        cvar=cvar_val,
        std_error=std_err,
    )


def monte_carlo_var_copula(
    returns: NDArray[np.floating],
    weights: NDArray[np.floating],
    n_scenarios: int = 100_000,
    confidence: float = 0.95,
    horizon_days: int = 1,
    seed: int = 42,
) -> VaRResult:
    """Monte Carlo portfolio VaR using Gaussian copula for multi-asset dependence.

    Steps:
    1. Fit marginal distributions (Student-t) per asset
    2. Fit Gaussian copula to capture cross-sectional dependence
    3. Generate joint scenarios
    4. Compute portfolio returns → VaR
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float).flatten()
    n_assets = r.shape[1]

    if r.shape[0] < 10 or n_assets < 2:
        return monte_carlo_var(r @ w, n_scenarios, confidence, horizon_days, seed)

    # Fit marginals
    marginals = []
    for i in range(n_assets):
        params = sp_stats.t.fit(r[:, i])
        marginals.append(params)

    # Fit Gaussian copula
    copula = GaussianCopula()
    copula.fit(r)

    # Generate joint scenarios
    np.random.default_rng(seed)
    copula_samples = copula.sample(n_scenarios, seed=seed)
    joint = np.zeros((n_scenarios, n_assets))
    for i in range(n_assets):
        df_i, loc_i, scale_i = marginals[i]
        joint[:, i] = sp_stats.t.ppf(copula_samples[:, i], df_i, loc=loc_i, scale=scale_i)

    port_returns = joint @ w
    port_returns_h = port_returns * np.sqrt(horizon_days)

    percentile = (1.0 - confidence) * 100.0
    var_val = float(np.percentile(port_returns_h, percentile))
    tail = port_returns_h[port_returns_h <= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    boot_vars = np.array([
        float(np.percentile(
            port_returns_h[np.random.randint(0, n_scenarios, n_scenarios)], percentile))
        for _ in range(200)
    ])

    return VaRResult(
        method="monte_carlo_copula",
        confidence=confidence,
        horizon_days=horizon_days,
        var=var_val,
        cvar=cvar_val,
        std_error=float(np.std(boot_vars)),
    )


def filtered_historical_var(
    returns: NDArray[np.floating],
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> VaRResult:
    """Filtered Historical Simulation VaR (GARCH-filter + HS on residuals).

    1. Fit GARCH(1,1) to returns
    2. Extract standardized residuals
    3. Bootstrap from historical residuals (preserving fat tails + clustering)
    4. Reapply GARCH conditional volatility forecast
    5. Compute VaR from simulated paths
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)

    # Fit GARCH
    try:
        garch_result = fit_garch(r, p=1, q=1)
    except Exception:
        # Fallback: use standard historical if GARCH fails
        return historical_var(r, confidence, horizon_days)

    cond_vol = garch_result.conditional_volatility
    residuals = r / np.maximum(cond_vol[:n], 1e-10)

    # Simulate: draw from standardized residuals
    rng = np.random.default_rng(42)
    n_sim = 100_000
    idx = rng.integers(0, n, n_sim)
    resampled_residuals = residuals[idx]

    # Apply current conditional vol
    current_vol = cond_vol[-1]
    forecast_vol = current_vol * np.sqrt(horizon_days)
    sim_returns = resampled_residuals * forecast_vol

    percentile = (1.0 - confidence) * 100.0
    var_val = float(np.percentile(sim_returns, percentile))
    tail = sim_returns[sim_returns <= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return VaRResult(
        method="filtered_historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var=var_val,
        cvar=cvar_val,
    )


# ---------------------------------------------------------------------------
# T-0304: CVaR (Expected Shortfall) — marginal, component, incremental
# ---------------------------------------------------------------------------


def cvar(
    returns: NDArray[np.floating],
    confidence: float = 0.975,
) -> float:
    """Expected shortfall at the given confidence level.

    CVaR_α = E[r | r ≤ VaR_α]
    """
    r = np.asarray(returns, dtype=float).flatten()
    var_val = float(np.percentile(r, (1.0 - confidence) * 100.0))
    tail = r[r <= var_val]
    return float(np.mean(tail))


def marginal_cvar(
    returns: NDArray[np.floating],
    weights: NDArray[np.floating],
    confidence: float = 0.975,
) -> NDArray[np.floating]:
    """Marginal CVaR: sensitivity of portfolio CVaR to each position weight.

    M-CVaR_i = ∂CVaR/∂w_i = E[r_i | r_port ≤ VaR_α]

    Returns array of shape (n_assets,).
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float).flatten()
    n_assets = r.shape[1]

    if r.ndim == 1:
        r = r.reshape(-1, 1)

    port_returns = r @ w
    var_val = float(np.percentile(port_returns, (1.0 - confidence) * 100.0))
    tail_mask = port_returns <= var_val

    mcvar = np.zeros(n_assets)
    for i in range(n_assets):
        if np.any(tail_mask):
            mcvar[i] = float(np.mean(r[tail_mask, i]))
        else:
            mcvar[i] = float(np.mean(r[:, i]))

    return mcvar


def component_cvar(
    returns: NDArray[np.floating],
    weights: NDArray[np.floating],
    confidence: float = 0.975,
) -> NDArray[np.floating]:
    """Component CVaR: decompose portfolio CVaR into position contributions.

    C-CVaR_i = w_i × M-CVaR_i
    Sum(C-CVaR_i) = Portfolio CVaR (Euler allocation)

    Returns array of shape (n_assets,).
    """
    w = np.asarray(weights, dtype=float).flatten()
    mcvar = marginal_cvar(returns, w, confidence)
    return w * mcvar


def incremental_cvar(
    returns: NDArray[np.floating],
    weights: NDArray[np.floating],
    new_trade_asset: int,
    new_trade_weight: float,
    confidence: float = 0.975,
) -> float:
    """Incremental CVaR: impact of adding a proposed trade.

    IV-CVaR = CVaR(new portfolio) - CVaR(current portfolio)

    Args:
        new_trade_asset: Index of the asset being added/increased.
        new_trade_weight: Weight change for that asset.
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float).flatten().astype(float)

    current_port = r @ w
    current_cvar_val = cvar(current_port, confidence)

    w_new = w.copy()
    w_new[new_trade_asset] += new_trade_weight
    new_port = r @ w_new
    new_cvar_val = cvar(new_port, confidence)

    return float(new_cvar_val - current_cvar_val)
