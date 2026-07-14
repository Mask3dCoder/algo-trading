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


# ---------------------------------------------------------------------------
# Dynamic Vol-Targeting for Options (Audit Mandate 2026-07-12, Phase 6)
# ---------------------------------------------------------------------------


def compute_ewma_vol(returns: NDArray[np.floating], span: int = 20) -> float:
    """Annualized EWMA volatility of a daily return series.

    Uses a RiskMetrics-style exponentially-weighted variance with decay
    ``lambda = (span - 1) / (span + 1)`` (pandas ``ewm(span=...)`` convention),
    then annualizes by sqrt(252).

    Returns ``nan`` when there are too few observations to estimate vol.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    lam = (span - 1.0) / (span + 1.0)
    weights = lam ** np.arange(r.size - 1, -1, -1)
    weights /= weights.sum()
    mean = float(np.dot(weights, r))
    var = float(np.dot(weights, (r - mean) ** 2))
    if var <= 0:
        return float("nan")
    return float(np.sqrt(var) * np.sqrt(252.0))


def calculate_contract_size(
    capital: float,
    returns: NDArray[np.floating],
    current_underlying_price: float,
    *,
    target_annual_vol: float = 0.12,
    vix_close: float | None = None,
    risk_per_trade_pct: float = 0.02,
    span: int = 20,
    max_contracts: int = 50,
) -> int:
    """Volatility-aware option contract sizing.

    Sizing scales INVERSELY with recent realized volatility so that dollar risk
    stays near ``target_annual_vol``.  A VIX term-structure adjustment shrinks
    size further when the VIX prints above realized (stress / backwardation).

        scaling_factor = target_annual_vol / realized_vol
        base_notional  = capital * risk_per_trade_pct
        contracts      = int(base_notional * scaling_factor / (price * 100))

    Args:
        capital: Deployable capital.
        returns: Daily returns of the underlying (for EWMA realized vol).
        current_underlying_price: Spot price of the underlying (e.g. SPY).
        target_annual_vol: Target annualized volatility (default 12%).
        vix_close: Latest VIX close (as a fraction, e.g. 0.18 for 18). If given
            as points (>1.5) it is auto-scaled to a fraction.
        risk_per_trade_pct: Base notional as a fraction of capital (default 2%).
        span: EWMA span for realized vol (default 20 trading days).
        max_contracts: Hard cap on contracts (sanity clamp).

    Returns:
        Number of contracts (minimum 1).
    """
    if capital <= 0 or current_underlying_price <= 0:
        return 1

    realized_vol = compute_ewma_vol(returns, span=span)
    if not np.isfinite(realized_vol) or realized_vol <= 1e-6:
        realized_vol = target_annual_vol  # neutral fallback -> scaling ~1x

    scaling_factor = target_annual_vol / realized_vol

    # VIX term-structure haircut: if forward-looking VIX exceeds trailing
    # realized vol, the market is pricing stress -> reduce size proportionally.
    if vix_close is not None and vix_close > 0:
        vix_frac = vix_close / 100.0 if vix_close > 1.5 else float(vix_close)
        if vix_frac > realized_vol > 0:
            scaling_factor *= realized_vol / vix_frac

    scaling_factor = float(np.clip(scaling_factor, 0.1, 3.0))

    base_notional = capital * risk_per_trade_pct
    contracts = int(base_notional * scaling_factor / (current_underlying_price * 100.0))
    return max(1, min(contracts, max_contracts))


# ---------------------------------------------------------------------------
# Equity Market-Neutral Sizing (Phase 8)
# ---------------------------------------------------------------------------

def calculate_equity_size(
    capital: float,
    target_vol: float = 0.12,
    realized_vol: float = 0.15,
    n_stocks: int = 50,
) -> float:
    """Volatility-scaled equal-notional sizing for an equity long/short sleeve.

    Per-name notional = (capital * 0.02) / n_stocks, scaled by the ratio
    ``target_vol / realized_vol`` (inverse to realized risk) and floored/capped so
    the combined gross notional can never exceed the deployable capital.

    NOTE: The equity_factor_neutral strategy runs its first backtest with plain
    equal-weighted notional (gross = ``gross_target * capital``) and a hard 2%
    per-name cap. This helper is the optional vol-targeting overlay; wire it in
    once the base walk-forward is green.

    Args:
        capital: Deployable capital.
        target_vol: Target annualized volatility of the sleeve (default 12%).
        realized_vol: Recent realized annualized vol of the sleeve/spread.
        n_stocks: Total names on BOTH sides (longs + shorts).

    Returns:
        Notional dollars per stock (>= 0).
    """
    if capital <= 0 or n_stocks <= 0:
        return 0.0

    base = (capital * 0.02) / n_stocks
    if realized_vol and realized_vol > 0:
        vol_scale = target_vol / float(realized_vol)
    else:
        vol_scale = 1.0
    vol_scale = float(np.clip(vol_scale, 0.1, 3.0))

    notional = base * vol_scale
    # Hard cap: one side's notional must not exceed 95% of capital / n_stocks.
    return min(notional, capital * 0.95 / n_stocks)
