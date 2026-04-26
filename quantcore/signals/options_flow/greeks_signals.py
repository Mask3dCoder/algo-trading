"""Greeks-based signals: GEX, gamma flip, charm/vanna flows, pin risk."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Gamma Exposure (GEX)
# ---------------------------------------------------------------------------


def gamma_exposure(
    gamma: NDArray,
    open_interest: NDArray,
    spot_price: float,
    contract_multiplier: float = 100.0,
) -> float:
    """Compute total gamma exposure (GEX) at a given spot price.

    GEX = sum(gamma_i * OI_i * spot^2 * 100) for all options.
    Positive GEX = market makers are long gamma → stabilizing.
    Negative GEX = market makers short gamma → destabilizing.

    Args:
        gamma: per-contract gamma values for all strikes/expiries.
        open_interest: corresponding open interest.
        spot_price: underlying spot price.
        contract_multiplier: options multiplier (default 100).
    """
    valid = ~(np.isnan(gamma) | np.isnan(open_interest))
    g = gamma[valid]
    oi = open_interest[valid]

    # GEX = gamma * OI * S^2 * 100 (dollar gamma exposure)
    total_gex = np.sum(g * oi * spot_price ** 2 * contract_multiplier)
    return float(total_gex)


def gex_profile(
    strikes: NDArray,
    gamma: NDArray,
    open_interest: NDArray,
    spot_price: float,
    price_range: float = 0.10,
    n_points: int = 50,
    contract_multiplier: float = 100.0,
) -> tuple[NDArray, NDArray]:
    """Compute GEX across a range of spot prices.

    Returns (price_levels, gex_values).
    """
    low = spot_price * (1 - price_range)
    high = spot_price * (1 + price_range)
    prices = np.linspace(low, high, n_points)
    gex_vals = np.zeros(n_points, dtype=float)

    for i, p in enumerate(prices):
        gex_vals[i] = gamma_exposure(gamma, open_interest, p, contract_multiplier)

    return prices, gex_vals


# ---------------------------------------------------------------------------
# Gamma Flip Level
# ---------------------------------------------------------------------------


def gamma_flip_level(
    strikes: NDArray,
    gamma: NDArray,
    open_interest: NDArray,
    spot_price: float,
    contract_multiplier: float = 100.0,
) -> float | None:
    """Find the price level where GEX flips from positive to negative.

    This is a key level: above it, dealers buy dips (stabilizing).
    Below it, dealers sell rallies (accelerating selloff).
    """
    prices, gex_vals = gex_profile(
        strikes, gamma, open_interest, spot_price,
        price_range=0.15, n_points=100,
        contract_multiplier=contract_multiplier,
    )

    # Find zero crossing from positive to negative (going down in price)
    for i in range(len(gex_vals) - 1):
        if gex_vals[i] > 0 and gex_vals[i + 1] <= 0:
            # Linear interpolation
            if gex_vals[i] - gex_vals[i + 1] != 0:
                frac = gex_vals[i] / (gex_vals[i] - gex_vals[i + 1])
                return float(prices[i] + frac * (prices[i + 1] - prices[i]))

    return None


# ---------------------------------------------------------------------------
# Charm (dDelta/dTime) Signal
# ---------------------------------------------------------------------------


def charm_exposure(
    charm: NDArray,
    open_interest: NDArray,
    contract_multiplier: float = 100.0,
) -> float:
    """Total charm exposure: how delta shifts with time decay.

    Charm = dDelta/dTime. Significant charm near expiry means
    dealers need to rebalance hedges as time passes.

    Args:
        charm: per-contract charm values.
        open_interest: corresponding OI.
        contract_multiplier: options multiplier.
    """
    valid = ~(np.isnan(charm) | np.isnan(open_interest))
    c = charm[valid]
    oi = open_interest[valid]
    return float(np.sum(c * oi * contract_multiplier))


# ---------------------------------------------------------------------------
# Vanna (dDelta/dVol) Signal
# ---------------------------------------------------------------------------


def vanna_exposure(
    vanna: NDArray,
    open_interest: NDArray,
    contract_multiplier: float = 100.0,
) -> float:
    """Total vanna exposure: how delta shifts with volatility.

    Vanna = dDelta/dVol. Dealers with large vanna exposure need
    to buy/sell underlying when vol changes.

    Args:
        vanna: per-contract vanna values.
        open_interest: corresponding OI.
        contract_multiplier: options multiplier.
    """
    valid = ~(np.isnan(vanna) | np.isnan(open_interest))
    v = vanna[valid]
    oi = open_interest[valid]
    return float(np.sum(v * oi * contract_multiplier))


# ---------------------------------------------------------------------------
# Pin Risk Signal
# ---------------------------------------------------------------------------


def pin_risk_score(
    strikes: NDArray,
    open_interest: NDArray,
    spot_price: float,
    time_to_expiry: float,
    implied_vol: float,
) -> NDArray:
    """Estimate pin risk at each strike.

    Pin risk = probability that the underlying closes near a strike at expiry.
    Higher OI at a strike + closer to current price + near expiry = higher pin risk.

    Uses Black-Scholes probability density.

    Returns array of pin risk scores (same shape as strikes).
    """

    n = len(strikes)
    pin_scores = np.zeros(n, dtype=float)

    if time_to_expiry <= 0 or implied_vol <= 0:
        return pin_scores

    sqrt_t = np.sqrt(time_to_expiry)
    vol_sqrt_t = implied_vol * sqrt_t

    for i in range(n):
        if np.isnan(strikes[i]) or np.isnan(open_interest[i]):
            continue

        # Approximate probability of closing at this strike
        # Use BSM probability density
        d1_val = (np.log(spot_price / strikes[i]) + 0.5 * implied_vol ** 2 * time_to_expiry) / vol_sqrt_t

        # Standard normal PDF at d1 gives probability density
        pdf_val = np.exp(-0.5 * d1_val ** 2) / np.sqrt(2.0 * np.pi)

        # Weight by open interest and normalize by vol
        pin_scores[i] = open_interest[i] * pdf_val / vol_sqrt_t

    # Normalize to [0, 1]
    max_score = np.max(pin_scores)
    if max_score > 0:
        pin_scores /= max_score

    return pin_scores


# ---------------------------------------------------------------------------
# Dealer Positioning Signal
# ---------------------------------------------------------------------------


def dealer_positioning_signal(
    total_gex: float,
    gex_flip_level: float | None,
    spot_price: float,
    lookback_gex: NDArray | None = None,
) -> float:
    """Composite dealer positioning signal based on GEX analysis.

    Positive signal = dealers long gamma → market support → bullish.
    Negative signal = dealers short gamma → accelerant → bearish.

    Args:
        total_gex: current total gamma exposure.
        gex_flip_level: price where GEX flips sign (None if no flip found).
        spot_price: current underlying price.
        lookback_gex: historical GEX values for normalization.

    Returns:
        Signal in [-1, +1].
    """
    signal = 0.0

    # GEX direction
    if lookback_gex is not None and len(lookback_gex) > 20:
        valid = lookback_gex[~np.isnan(lookback_gex)]
        if len(valid) > 10:
            m = np.mean(valid)
            s = np.std(valid, ddof=1)
            if s > 0:
                z = (total_gex - m) / s
                signal += np.clip(z / 3.0, -1.0, 1.0) * 0.5
    else:
        # Raw sign
        if total_gex > 0:
            signal += 0.5
        else:
            signal -= 0.5

    # Distance from gamma flip level
    if gex_flip_level is not None and gex_flip_level > 0:
        distance = (spot_price - gex_flip_level) / spot_price
        # Above flip = positive GEX zone = bullish
        signal += np.clip(distance * 10.0, -0.5, 0.5)

    return np.clip(signal, -1.0, 1.0)
