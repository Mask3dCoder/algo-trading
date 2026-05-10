"""Performance and risk attribution: Brinson, factor-based, and risk decomposition.

Decomposes portfolio P&L into:
- Allocation effect: how much comes from over/under-weighting sectors
- Selection effect: how much comes from picking winners within sectors
- Interaction effect: cross-term between allocation and selection
- CAPM decomposition: alpha, beta, systematic / idiosyncratic
- Risk contribution: how much risk each position contributes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class AttributionResult:
    """Full performance attribution breakdown."""
    total_return: float = 0.0
    benchmark_return: float = 0.0
    active_return: float = 0.0
    # Brinson
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    # By segment
    allocation_by_segment: dict[str, float] = field(default_factory=dict)
    selection_by_segment: dict[str, float] = field(default_factory=dict)
    # Risk attribution
    risk_contributions: dict[str, float] = field(default_factory=dict)
    marginal_var: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskAttribution:
    """Decomposition of portfolio risk."""
    total_vol: float = 0.0
    total_var: float = 0.0
    component_var: dict[str, float] = field(default_factory=dict)
    component_var_pct: dict[str, float] = field(default_factory=dict)
    marginal_var: dict[str, float] = field(default_factory=dict)
    incremental_var: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PerformanceAttribution (CAPM + multi-factor)
# ---------------------------------------------------------------------------


@dataclass
class PerformanceAttribution:
    """CAPM and multi-factor performance decomposition.

    CAPM: R_p - R_f = α + β * (R_m - R_f) + ε
    Systematic return = β * mean(R_m - R_f)
    Idiosyncratic return = α (the manager's skill component)
    """
    alpha: float = 0.0
    beta: float = 0.0
    systematic_return: float = 0.0
    idiosyncratic_return: float = 0.0
    r_squared: float = 0.0
    factor_attribution: dict[str, float] = field(default_factory=dict)


def capm_attribution(
    portfolio_returns: NDArray[np.floating],
    market_returns: NDArray[np.floating],
    risk_free: float = 0.0,
    annualize: bool = True,
) -> PerformanceAttribution:
    """CAPM decomposition of portfolio returns.

    R_p = R_f + β * (R_m - R_f) + α

    Args:
        portfolio_returns: (n_periods,) portfolio returns.
        market_returns: (n_periods,) market benchmark returns.
        risk_free: Annualized risk-free rate.
        annualize: If True, alpha is annualized.

    Returns:
        PerformanceAttribution with alpha, beta, systematic/idio returns.
    """
    r = np.asarray(portfolio_returns, dtype=float).flatten()
    m = np.asarray(market_returns, dtype=float).flatten()
    n = min(len(r), len(m))
    r = r[:n]
    m = m[:n]

    from quantcore.risk.metrics import TRADING_DAYS
    rf_daily = risk_free / TRADING_DAYS

    r_excess = r - rf_daily
    m_excess = m - rf_daily

    cov = np.cov(r_excess, m_excess)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

    mean_r = float(np.mean(r_excess))
    mean_m = float(np.mean(m_excess))

    alpha_daily = mean_r - beta * mean_m
    ann_factor = TRADING_DAYS if annualize else 1.0
    alpha = alpha_daily * ann_factor
    systematic_return = beta * mean_m * ann_factor
    idiosyncratic_return = alpha  # α is the manager-specific component

    # R² = explained variance / total variance
    total_var = float(np.var(r_excess))
    if total_var > 0:
        predicted = beta * m_excess
        residual_var = float(np.var(r_excess - predicted))
        r_squared = max(0.0, 1.0 - residual_var / total_var)
    else:
        r_squared = 0.0

    return PerformanceAttribution(
        alpha=alpha,
        beta=beta,
        systematic_return=systematic_return,
        idiosyncratic_return=idiosyncratic_return,
        r_squared=r_squared,
    )


# ---------------------------------------------------------------------------
# Brinson attribution
# ---------------------------------------------------------------------------


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    segment_returns: dict[str, float],
    benchmark_segment_returns: dict[str, float] | None = None,
) -> AttributionResult:
    """Brinson performance attribution.

    Decomposes active return into allocation, selection, and interaction effects.

    Allocation = Σ (w_p,i - w_b,i) * R_b,i      ← sector bets
    Selection   = Σ w_b,i * (R_p,i - R_b,i)      ← stock picking
    Interaction = Σ (w_p,i - w_b,i) * (R_p,i - R_b,i)  ← cross-term

    Args:
        portfolio_weights: {segment: weight} for the portfolio.
        benchmark_weights: {segment: weight} for the benchmark.
        segment_returns: {segment: return} for portfolio segments.
        benchmark_segment_returns: {segment: return} for benchmark segments
            (defaults to segment_returns if not provided).

    Returns:
        AttributionResult with full decomposition.
    """
    if benchmark_segment_returns is None:
        benchmark_segment_returns = segment_returns

    segments = set(portfolio_weights) | set(benchmark_weights)

    allocation = 0.0
    selection = 0.0
    interaction = 0.0
    alloc_by_seg: dict[str, float] = {}
    sel_by_seg: dict[str, float] = {}

    for seg in segments:
        wp = portfolio_weights.get(seg, 0.0)
        wb = benchmark_weights.get(seg, 0.0)
        rp = segment_returns.get(seg, 0.0)
        rb = benchmark_segment_returns.get(seg, 0.0)

        alloc = (wp - wb) * rb
        sel = wb * (rp - rb)
        inter = (wp - wb) * (rp - rb)

        allocation += alloc
        selection += sel
        interaction += inter
        alloc_by_seg[seg] = alloc
        sel_by_seg[seg] = sel

    total_return = sum(
        portfolio_weights.get(s, 0) * segment_returns.get(s, 0)
        for s in segments
    )
    benchmark_return = sum(
        benchmark_weights.get(s, 0) * benchmark_segment_returns.get(s, 0)
        for s in segments
    )

    return AttributionResult(
        total_return=total_return,
        benchmark_return=benchmark_return,
        active_return=total_return - benchmark_return,
        allocation_effect=allocation,
        selection_effect=selection,
        interaction_effect=interaction,
        allocation_by_segment=alloc_by_seg,
        selection_by_segment=sel_by_seg,
    )


# ---------------------------------------------------------------------------
# Factor-based attribution
# ---------------------------------------------------------------------------


def factor_attribution(
    portfolio_returns: NDArray[np.floating],
    factor_returns: NDArray[np.floating],
    factor_exposures: NDArray[np.floating],
    factor_names: list[str] | None = None,
) -> dict[str, float]:
    """Factor-based return attribution.

    R_portfolio = Σ β_k * F_k + ε  (idiosyncratic)
    Attribute return to each factor: β_k * mean(F_k).

    Args:
        portfolio_returns: (n_periods,) array of portfolio returns.
        factor_returns: (n_periods, n_factors) array of factor returns.
        factor_exposures: (n_factors,) array of factor betas.
        factor_names: Optional list of factor names.

    Returns:
        {factor_name: attributed_return}
    """
    r = np.asarray(portfolio_returns, dtype=float).flatten()
    F = np.asarray(factor_returns, dtype=float)
    betas = np.asarray(factor_exposures, dtype=float).flatten()

    n_factors = F.shape[1] if F.ndim > 1 else 1
    if F.ndim == 1:
        F = F.reshape(-1, 1)

    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(n_factors)]

    result: dict[str, float] = {}

    for k in range(n_factors):
        factor_mean = float(np.mean(F[:, k]))
        attributed = betas[k] * factor_mean
        result[factor_names[k]] = attributed

    # Idiosyncratic
    predicted = F @ betas
    n = min(len(r), len(predicted))
    residual = float(np.mean(r[:n] - predicted[:n]))
    result["idiosyncratic"] = residual
    result["total"] = float(np.mean(r[:n]))

    return result


# ---------------------------------------------------------------------------
# Risk attribution (component VaR / marginal VaR)
# ---------------------------------------------------------------------------


def risk_attribution(
    weights: NDArray[np.floating],
    cov: NDArray[np.floating],
    confidence: float = 0.95,
    asset_names: list[str] | None = None,
) -> RiskAttribution:
    """Decompose portfolio risk into component and marginal contributions.

    Component VaR_i = w_i * ∂VaR/∂w_i = w_i * (Σw)_i / σ_p * z_α
    Marginal VaR_i  = ∂VaR/∂w_i = (Σw)_i / σ_p * z_α
    Incremental VaR_i = VaR(w + Δ_i) - VaR(w)

    Args:
        weights: Portfolio weights (n_assets,).
        cov: Covariance matrix (n_assets, n_assets).
        confidence: Confidence level for VaR.
        asset_names: Optional asset names.

    Returns:
        RiskAttribution with full decomposition.
    """
    w = np.asarray(weights, dtype=float).flatten()
    cov_arr = np.asarray(cov, dtype=float)
    n = len(w)

    if asset_names is None:
        asset_names = [f"asset_{i}" for i in range(n)]

    from scipy import stats as sp_stats
    z_alpha = -sp_stats.norm.ppf(1.0 - confidence)

    port_var = float(w @ cov_arr @ w)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0
    total_var = port_vol * z_alpha if port_vol > 0 else 0.0

    component_var: dict[str, float] = {}
    component_var_pct: dict[str, float] = {}
    marginal_var_dict: dict[str, float] = {}
    incremental_var: dict[str, float] = {}

    if port_vol > 0:
        for i in range(n):
            # Marginal VaR
            mvar_i = float((cov_arr @ w)[i]) / port_vol * z_alpha
            marginal_var_dict[asset_names[i]] = mvar_i

            # Component VaR
            cvar_i = w[i] * mvar_i
            component_var[asset_names[i]] = cvar_i

            # Component VaR %
            component_var_pct[asset_names[i]] = cvar_i / total_var if total_var > 0 else 0.0

            # Incremental VaR (remove asset i)
            w_without = w.copy()
            w_without[i] = 0
            w_without = w_without / w_without.sum() if w_without.sum() > 0 else w_without
            var_without = float(np.sqrt(w_without @ cov_arr @ w_without)) * z_alpha if port_vol > 0 else 0.0
            incremental_var[asset_names[i]] = total_var - var_without

    return RiskAttribution(
        total_vol=port_vol,
        total_var=total_var,
        component_var=component_var,
        component_var_pct=component_var_pct,
        marginal_var=marginal_var_dict,
        incremental_var=incremental_var,
    )


# ---------------------------------------------------------------------------
# P&L explain
# ---------------------------------------------------------------------------


def pnl_explain(
    pnl: float,
    market_move: float,
    delta: float,
    gamma: float = 0.0,
    theta: float = 0.0,
    vega: float = 0.0,
    vol_change: float = 0.0,
) -> dict[str, float]:
    """Explain daily P&L via Greeks decomposition.

    PnL ≈ Delta * ΔS + 0.5 * Gamma * (ΔS)² + Theta * Δt + Vega * Δσ

    Args:
        pnl: Actual daily P&L.
        market_move: Change in underlying price.
        delta: Position delta.
        gamma: Position gamma.
        theta: Position theta (daily).
        vega: Position vega.
        vol_change: Change in implied volatility.

    Returns:
        {component: attributed_pnl}
    """
    delta_pnl = delta * market_move
    gamma_pnl = 0.5 * gamma * (market_move**2)
    theta_pnl = theta  # Daily theta
    vega_pnl = vega * vol_change

    explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
    unexplained = pnl - explained

    return {
        "delta": delta_pnl,
        "gamma": gamma_pnl,
        "theta": theta_pnl,
        "vega": vega_pnl,
        "explained": explained,
        "unexplained": unexplained,
        "total": pnl,
        "explained_pct": explained / pnl if abs(pnl) > 1e-10 else 0.0,
    }
