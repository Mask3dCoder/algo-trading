"""Market impact models: square-root, linear, Kyle's lambda, Amihud, Roll spread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class CostEstimate:
    total_bps: float
    spread_cost_bps: float
    impact_cost_bps: float
    timing_cost_bps: float = 0.0
    total_dollars: float = 0.0


# ---------------------------------------------------------------------------
# Square-root impact model
# ---------------------------------------------------------------------------


def sqrt_impact(
    shares: float,
    adv: float,  # Average daily volume (shares)
    sigma: float,  # Daily volatility
) -> float:
    """Square-root impact model (Almgren et al.).

    ΔP/P = σ × √(Q / ADV)

    Returns impact in decimal (e.g., 0.001 = 10 bps).
    """
    if adv <= 0:
        return 0.0

    q_adv = shares / adv
    if q_adv <= 0:
        return 0.0

    return float(sigma * np.sqrt(q_adv))


def sqrt_impact_bps(
    shares: float,
    adv: float,
    sigma: float,
) -> float:
    """Square-root impact in basis points."""
    return sqrt_impact(shares, adv, sigma) * 10_000


# ---------------------------------------------------------------------------
# Linear impact (Kyle's model)
# ---------------------------------------------------------------------------


def linear_impact(
    shares: float,
    adv: float,
    kyle_lambda: float,
) -> float:
    """Linear impact model: ΔP = λ × Q.

    Kyle's lambda: permanent price impact per share traded.

    Returns impact in decimal.
    """
    return float(kyle_lambda * shares)


def estimate_kyle_lambda(
    returns: NDArray[np.floating],
    volume: NDArray[np.floating],
    signed_volume: NDArray[np.floating] | None = None,
) -> float:
    """Estimate Kyle's lambda from trade data.

    Kyle's model: ΔP_t = λ × SignedVolume_t + ε_t

    Uses OLS regression of price changes on signed volume.

    Args:
        returns: Price changes (or use returns).
        volume: Total volume.
        signed_volume: Trade-signed volume (buys - sells). If None, uses volume * sign(returns).

    Returns:
        Kyle's lambda estimate.
    """
    r = np.asarray(returns, dtype=float).flatten()
    v = np.asarray(volume, dtype=float).flatten()

    if signed_volume is None:
        signed_volume = v * np.sign(r)

    sv = np.asarray(signed_volume, dtype=float).flatten()

    # OLS: return = λ × signed_volume
    cov = np.cov(r, sv)
    if cov[1, 1] <= 0:
        return 0.0

    lamb = cov[0, 1] / cov[1, 1]
    return float(lamb)


# ---------------------------------------------------------------------------
# Amihud illiquidity measure
# ---------------------------------------------------------------------------


def amihud_illiquidity(
    returns: NDArray[np.floating],
    dollar_volume: NDArray[np.floating],
) -> float:
    """Amihud (2002) illiquidity measure.

    ILLIQ = mean(|r_t| / dollar_volume_t)

    Higher values = more illiquid. Good for cross-sectional comparison.
    """
    r = np.asarray(returns, dtype=float).flatten()
    dv = np.asarray(dollar_volume, dtype=float).flatten()

    n = min(len(r), len(dv))
    r = r[:n]
    dv = dv[:n]

    mask = dv > 0
    if not np.any(mask):
        return 0.0

    illiq = np.mean(np.abs(r[mask]) / dv[mask])
    return float(illiq)


# ---------------------------------------------------------------------------
# Roll bid-ask spread estimator
# ---------------------------------------------------------------------------


def roll_spread(
    prices: NDArray[np.floating],
) -> float:
    """Roll (1984) bid-ask spread estimator.

    Spread = 2 × √(-cov(Δp_t, Δp_{t-1}))

    If the covariance is positive, spread = 0 (invalid estimate).

    Returns spread in price units.
    """
    p = np.asarray(prices, dtype=float).flatten()
    if len(p) < 3:
        return 0.0

    dp = np.diff(p)
    if len(dp) < 2:
        return 0.0

    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    if cov >= 0:
        return 0.0

    return float(2.0 * np.sqrt(-cov))


def roll_spread_bps(prices: NDArray[np.floating]) -> float:
    """Roll spread in basis points of the average price."""
    spread = roll_spread(prices)
    avg_price = float(np.mean(prices))
    if avg_price <= 0 or spread <= 0:
        return 0.0
    return float(spread / avg_price * 10_000)


# ---------------------------------------------------------------------------
# Predicted total cost
# ---------------------------------------------------------------------------


def predict_cost(
    shares: float,
    price: float,
    adv: float,
    sigma: float,
    model: Literal["sqrt", "linear"] = "sqrt",
    kyle_lambda: float | None = None,
    half_spread_bps: float = 5.0,
) -> CostEstimate:
    """Predict total execution cost for an order.

    Total cost = spread_cost + impact_cost

    Args:
        shares: Order size in shares.
        price: Current price.
        adv: Average daily volume (shares).
        sigma: Daily volatility.
        model: Impact model.
        kyle_lambda: Kyle's lambda (for linear model).
        half_spread_bps: Half-spread in basis points (default 5).

    Returns:
        CostEstimate with breakdown.
    """
    order_value = shares * price

    # Spread cost (half-spread × side)
    spread_cost_bps = half_spread_bps  # One side only

    # Impact cost
    if model == "linear" and kyle_lambda is not None:
        impact_bps = linear_impact(shares, adv, kyle_lambda) * 10_000
    else:
        impact_bps = sqrt_impact_bps(shares, adv, sigma)

    total_bps = spread_cost_bps + impact_bps
    total_dollars = order_value * total_bps / 10_000

    return CostEstimate(
        total_bps=total_bps,
        spread_cost_bps=spread_cost_bps,
        impact_cost_bps=impact_bps,
        total_dollars=total_dollars,
    )


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------


def calibrate_impact(
    trades: list[dict],  # Each: {shares, arrival_price, avg_fill_price, adv, sigma}
    model: str = "sqrt",
) -> dict[str, float]:
    """Calibrate impact model parameters from trade data.

    For sqrt model: ΔP/P = σ × c × √(Q/ADV) → estimate c.
    For linear model: ΔP = λ × Q → estimate λ.

    Returns calibrated parameters.
    """
    if model == "sqrt":
        x = []
        y = []
        for trade in trades:
            q_adv = trade["shares"] / trade["adv"] if trade["adv"] > 0 else 0
            if q_adv <= 0:
                continue
            x.append(trade["sigma"] * np.sqrt(q_adv))
            impact = abs(trade["avg_fill_price"] - trade["arrival_price"]) / trade["arrival_price"]
            y.append(impact)

        if len(x) < 2:
            return {"c": 1.0}

        x = np.array(x)
        y = np.array(y)
        c = float(np.sum(y) / np.sum(x)) if np.sum(x) > 0 else 1.0
        return {"c": c}

    else:
        # Linear model
        x = []
        y = []
        for trade in trades:
            q = trade["shares"]
            impact = trade["avg_fill_price"] - trade["arrival_price"]
            x.append(q)
            y.append(impact)

        if len(x) < 2:
            return {"lambda": 0.0}

        x = np.array(x)
        y = np.array(y)
        lamb = float(np.sum(x * y) / np.sum(x**2)) if np.sum(x**2) > 0 else 0.0
        return {"lambda": lamb}
