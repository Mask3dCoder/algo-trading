"""Position sizing methods: Kelly, vol targeting, risk parity, ATR, CVaR."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------


def kelly_criterion(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Full Kelly fraction.

    f* = (bp - q) / b
    where b = avg_win / |avg_loss|, p = win_prob, q = 1 - p

    Args:
        win_prob: Probability of winning trade (0-1).
        avg_win: Average winning trade return (positive).
        avg_loss: Average losing trade return (positive, absolute value).

    Returns:
        Kelly fraction f*.
    """
    if avg_loss <= 0:
        return 0.0

    b = avg_win / avg_loss  # Odds ratio
    p = win_prob
    q = 1.0 - p

    f = (b * p - q) / b if b > 0 else 0.0
    return max(0.0, min(f, 1.0))  # Clamp to [0, 1]


def fractional_kelly(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly: f_fraction = fraction × f*."""
    f_star = kelly_criterion(win_prob, avg_win, avg_loss)
    return f_star * fraction


# ---------------------------------------------------------------------------
# Volatility Targeting
# ---------------------------------------------------------------------------


def volatility_target(
    asset_vol: float,
    target_vol: float = 0.15,  # 15% annualized
    n_assets: int = 1,
    correlation: float | NDArray[np.floating] | None = None,
) -> float:
    """Volatility-targeted portfolio weight.

    weight = target_vol / (portfolio_vol)
    Where portfolio_vol estimates the vol of the combined position.

    For simple case: weight ≈ target_vol / (asset_vol × √n_assets)

    Args:
        asset_vol: Annualized volatility of the asset.
        target_vol: Target annualized portfolio volatility.
        n_assets: Number of assets (assumes equal diversification).
        correlation: Optional correlation matrix for more precise sizing.

    Returns:
        Portfolio weight (fraction of capital).
    """
    if asset_vol <= 0:
        return 0.0

    # Simple: assume equal risk contribution
    weight = target_vol / (asset_vol * max(np.sqrt(n_assets), 1.0))

    # If correlation matrix provided, adjust
    if correlation is not None and n_assets > 1:
        corr = np.atleast_2d(np.asarray(correlation, dtype=float))
        avg_corr = (corr.sum() - n_assets) / (n_assets * (n_assets - 1)) if n_assets > 1 else 0.0
        avg_corr = max(0.0, min(avg_corr, 0.99))
        port_vol = asset_vol * np.sqrt((1.0 + (n_assets - 1) * avg_corr) / n_assets)
        weight = target_vol / port_vol if port_vol > 0 else 0.0

    return max(0.0, min(weight, 1.0))


# ---------------------------------------------------------------------------
# Risk Parity Sizing
# ---------------------------------------------------------------------------


def risk_parity_size(
    volatilities: NDArray[np.floating],
    correlations: NDArray[np.floating],
    target_vol: float = 0.10,
) -> NDArray[np.floating]:
    """Risk parity portfolio weights (equal risk contribution).

    Iterative solve for weights w such that:
    ∂Risk/∂w_i × w_i is equal for all i.

    Uses naive risk parity (inverse-vol) for simplicity.
    For exact ERC, use scipy optimization in portfolio/optimization.py.

    Args:
        volatilities: Annualized vols for each asset.
        correlations: Correlation matrix.
        target_vol: Target portfolio volatility.

    Returns:
        Array of portfolio weights.
    """
    vols = np.asarray(volatilities, dtype=float).flatten()
    corr = np.asarray(correlations, dtype=float)

    n = len(vols)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Naive risk parity: weight ∝ 1/volatility
    inv_vol = 1.0 / np.maximum(vols, 1e-10)
    weights = inv_vol / inv_vol.sum()

    # Scale to target vol
    cov = np.diag(vols) @ corr @ np.diag(vols)
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol > 0:
        weights *= target_vol / port_vol

    return weights


# ---------------------------------------------------------------------------
# ATR-based Sizing
# ---------------------------------------------------------------------------


def atr_size(
    account_value: float,
    risk_pct: float,  # Percentage of account to risk per trade (e.g., 0.01 = 1%)
    atr: float,  # Average True Range in price units
    multiplier: float = 2.0,  # Stop-loss multiple of ATR
) -> float:
    """ATR-based position sizing.

    shares = (account_value × risk_pct) / (ATR × multiplier)

    Args:
        account_value: Total account value.
        risk_pct: Fraction of account to risk (e.g., 0.01 for 1%).
        atr: Average True Range.
        multiplier: Stop-loss distance in ATR multiples.

    Returns:
        Number of shares.
    """
    if atr <= 0 or multiplier <= 0:
        return 0.0

    risk_amount = account_value * risk_pct
    stop_distance = atr * multiplier
    shares = risk_amount / stop_distance

    return shares


# ---------------------------------------------------------------------------
# CVaR-based Sizing
# ---------------------------------------------------------------------------


def cvar_size(
    returns: NDArray[np.floating],
    max_cvar: float,  # Maximum acceptable CVaR (dollar amount)
    account_value: float,
    confidence: float = 0.975,
) -> float:
    """CVaR-constrained position size.

    Computes the maximum position such that CVaR ≤ max_cvar.

    Args:
        returns: Historical asset returns.
        max_cvar: Maximum CVaR in dollars.
        account_value: Total account value.
        confidence: CVaR confidence level.

    Returns:
        Maximum position as fraction of account.
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < 20:
        return 0.0

    var = float(np.percentile(r, (1.0 - confidence) * 100))
    tail = r[r <= var]
    cvar_per_unit = float(np.mean(tail)) if len(tail) > 0 else var

    if abs(cvar_per_unit) < 1e-10:
        return 0.0

    # cvar_dollar = cvar_per_unit * position_value
    max_position_value = max_cvar / abs(cvar_per_unit)
    max_weight = max_position_value / account_value if account_value > 0 else 0.0

    return min(1.0, max(0.0, max_weight))


# ---------------------------------------------------------------------------
# Optimal-f (Vince's method)
# ---------------------------------------------------------------------------


def optimal_f(
    returns: NDArray[np.floating],
    n_bins: int = 100,
) -> float:
    """Ralph Vince's Optimal-f: maximize geometric growth rate.

    Searches for the optimal fraction f that maximizes:
    TWR(f) = ∏ (1 + f × R_i)

    Args:
        returns: Historical trade returns.
        n_bins: Number of f values to test.

    Returns:
        Optimal f value.
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < 5:
        return 0.25  # Conservative default

    # Normalize returns by the worst loss
    worst_loss = float(np.min(r))
    if worst_loss >= 0 or abs(worst_loss) < 1e-10:
        return 1.0  # No losing trades → full investment

    normalized = r / abs(worst_loss)

    best_f = 0.0
    best_twr = 0.0

    for f in np.linspace(0.01, 1.0, n_bins):
        # Terminal Wealth Relative
        returns_scaled = 1.0 + f * normalized
        if np.any(returns_scaled <= 0):
            continue
        twr = float(np.prod(returns_scaled)**(1.0 / len(r)))
        if twr > best_twr:
            best_twr = twr
            best_f = f

    return best_f
