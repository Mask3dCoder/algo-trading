"""Basket arbitrage signals: ETF arb, index arb, ADR arb, fair value NAV."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Fair Value / NAV Estimate
# ---------------------------------------------------------------------------


def fair_value(
    constituent_prices: NDArray,
    weights: NDArray,
) -> NDArray:
    """Compute NAV from constituent prices and weights.

    Args:
        constituent_prices: (n_periods, n_constituents) price matrix.
        weights: (n_constituents,) weight vector (sums to 1.0).

    Returns:
        (n_periods,) NAV series.
    """
    if constituent_prices.ndim == 1:
        return constituent_prices * weights[0]
    return constituent_prices @ weights


# ---------------------------------------------------------------------------
# ETF Arbitrage Signal
# ---------------------------------------------------------------------------


def etf_arb_signal(
    etf_price: NDArray,
    nav_estimate: NDArray,
    lookback: int = 252,
) -> NDArray:
    """ETF premium/discount z-score signal.

    premium = (etf_price - nav) / nav
    Positive = ETF trading at premium (sell ETF, buy basket).
    Negative = ETF trading at discount (buy ETF, sell basket).

    Z-scored against rolling history.
    """
    n = len(etf_price)
    premium = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isnan(etf_price[i]) and not np.isnan(nav_estimate[i]) and nav_estimate[i] > 0:
            premium[i] = (etf_price[i] - nav_estimate[i]) / nav_estimate[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = premium[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(premium[i]):
            z = (premium[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Index Arbitrage Signal
# ---------------------------------------------------------------------------


def index_arb_signal(
    futures_price: NDArray,
    spot_index: NDArray,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.015,
    days_to_expiry: NDArray | None = None,
    constant_dte: float = 90.0,
    lookback: int = 252,
) -> NDArray:
    """Index arbitrage: futures basis relative to fair value.

    Fair value = spot * exp((r - q) * T)
    Basis = futures_price - fair_value

    Positive basis = futures expensive → sell futures, buy spot.
    Negative basis = futures cheap → buy futures, sell spot.

    Returns z-scored basis signal.
    """
    n = len(futures_price)
    basis = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if np.isnan(futures_price[i]) or np.isnan(spot_index[i]) or spot_index[i] <= 0:
            continue

        dte = days_to_expiry[i] if days_to_expiry is not None else constant_dte
        T = dte / 365.0
        fair_val = spot_index[i] * np.exp((risk_free_rate - dividend_yield) * T)
        basis[i] = (futures_price[i] - fair_val) / spot_index[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = basis[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(basis[i]):
            z = (basis[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# ADR Arbitrage Signal
# ---------------------------------------------------------------------------


def adr_arb_signal(
    adr_price: NDArray,
    foreign_price: NDArray,
    fx_rate: NDArray,
    adr_ratio: float = 1.0,
    lookback: int = 252,
) -> NDArray:
    """ADR arbitrage: premium/discount of ADR vs foreign share.

    Implied foreign in USD = foreign_price * fx_rate * adr_ratio
    Spread = (adr_price - implied) / adr_price

    Positive = ADR expensive relative to foreign listing.
    Negative = ADR cheap.
    """
    n = len(adr_price)
    spread = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if (np.isnan(adr_price[i]) or np.isnan(foreign_price[i]) or
                np.isnan(fx_rate[i]) or adr_price[i] <= 0):
            continue

        implied_usd = foreign_price[i] * fx_rate[i] * adr_ratio
        spread[i] = (adr_price[i] - implied_usd) / adr_price[i]

    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(spread[i]):
            z = (spread[i] - m) / s
            result[i] = np.clip(z / 3.0, -1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Pairs Spread Signal (Basket)
# ---------------------------------------------------------------------------


def basket_spread_signal(
    prices_a: NDArray,
    prices_b: NDArray,
    hedge_ratio: float | None = None,
    lookback: int = 60,
) -> NDArray:
    """Z-score of spread between two price series (for basket arbitrage).

    If hedge_ratio is None, estimate via OLS on available data.
    """
    n = len(prices_a)
    if hedge_ratio is None:
        # OLS estimate
        valid = ~(np.isnan(prices_a) | np.isnan(prices_b))
        if np.sum(valid) < 30:
            return np.full(n, np.nan, dtype=float)
        y = prices_a[valid]
        x = prices_b[valid]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        hedge_ratio = ss_xy / ss_xx if ss_xx > 0 else 1.0

    spread = prices_a - hedge_ratio * prices_b
    return spread_zscore(spread, lookback)


def spread_zscore(spread: NDArray, lookback: int = 60) -> NDArray:
    """Rolling z-score of a spread series."""
    n = len(spread)
    result = np.full(n, np.nan, dtype=float)
    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        m = np.mean(valid)
        s = np.std(valid, ddof=1)
        if s > 0 and not np.isnan(spread[i]):
            result[i] = np.clip((spread[i] - m) / s / 3.0, -1.0, 1.0)
    return result
