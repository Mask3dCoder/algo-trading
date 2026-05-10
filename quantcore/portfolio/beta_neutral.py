"""Beta-neutral portfolio construction.

Adjusts portfolio weights so that net market beta ≈ 0. Uses rolling OLS
regression of each asset's returns against a market proxy (typically SPY).
If no market returns are provided, falls back to the equal-weight portfolio
as a proxy.

The adjustment subtracts proportional beta exposure from weights while
preserving dollar-neutrality (weights sum to zero).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def neutralize_beta(
    weights: NDArray[np.floating],
    asset_returns: NDArray[np.floating],
    market_returns: NDArray[np.floating] | None = None,
    window: int = 60,
) -> NDArray[np.floating]:
    """Adjust weights so portfolio beta ≈ 0.

    Rolling regression: beta_i = Cov(r_i, r_m) / Var(r_m)
    Portfolio beta = sum(w_i * beta_i)
    Adjustment: subtract proportional beta exposure from weights.

    If market_returns is None, falls back to equal-weight portfolio as proxy.
    Returns adjusted weights preserving dollar-neutrality (sum ≈ 0).
    """
    w = np.array(weights, dtype=float).copy()
    rets = np.asarray(asset_returns, dtype=float)
    if rets.ndim == 1:
        rets = rets.reshape(-1, 1)

    n_periods, n_assets = rets.shape
    if n_assets != len(w):
        return w

    if n_periods < max(window, 10):
        return w

    mkt = _resolve_market_returns(rets, market_returns)
    use_len = min(window, n_periods)
    seg_rets = rets[-use_len:]
    seg_mkt = mkt[-use_len:]

    betas = _compute_betas(seg_rets, seg_mkt)
    if betas is None:
        return w

    port_beta = float(np.dot(w, betas))
    if abs(port_beta) < 1e-8:
        return w

    beta_sum_sq = float(np.dot(betas, betas))
    if beta_sum_sq < 1e-10:
        return w

    adjustment = betas * (port_beta / beta_sum_sq)
    w = w - adjustment
    w = w - np.mean(w)

    return w


def compute_rolling_betas(
    asset_returns: NDArray[np.floating],
    market_returns: NDArray[np.floating],
    window: int = 60,
) -> NDArray[np.floating]:
    """Rolling OLS beta for each asset vs market.

    Returns array of shape (n_periods, n_assets) with NaN for预热 periods.
    """
    rets = np.asarray(asset_returns, dtype=float)
    mkt = np.asarray(market_returns, dtype=float).flatten()

    if rets.ndim == 1:
        rets = rets.reshape(-1, 1)

    n_periods, n_assets = rets.shape
    betas = np.full((n_periods, n_assets), np.nan)

    for t in range(window - 1, n_periods):
        seg_rets = rets[t - window + 1: t + 1]
        seg_mkt = mkt[t - window + 1: t + 1]
        b = _compute_betas(seg_rets, seg_mkt)
        if b is not None:
            betas[t, :] = b

    return betas


def _resolve_market_returns(
    asset_returns: NDArray[np.floating],
    market_returns: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    if market_returns is not None:
        return np.asarray(market_returns, dtype=float).flatten()
    # Fallback: equal-weight portfolio as market proxy
    return np.nanmean(asset_returns, axis=1)


def _compute_betas(
    asset_returns: NDArray[np.floating],
    market_returns: NDArray[np.floating],
) -> NDArray[np.floating] | None:
    rets = np.asarray(asset_returns, dtype=float)
    mkt = np.asarray(market_returns, dtype=float).flatten()

    if rets.ndim == 1:
        rets = rets.reshape(-1, 1)

    n_periods, n_assets = rets.shape
    if len(mkt) != n_periods or n_periods < 10:
        return None

    mkt_var = np.var(mkt, ddof=1)
    if mkt_var < 1e-10:
        return None

    betas = np.zeros(n_assets)
    for i in range(n_assets):
        valid = ~np.isnan(rets[:, i]) & ~np.isnan(mkt)
        if valid.sum() < 10:
            continue
        cov = np.cov(rets[valid, i], mkt[valid], ddof=1)
        if cov.ndim == 2 and cov.shape == (2, 2):
            betas[i] = cov[0, 1] / mkt_var
        else:
            betas[i] = 0.0

    return betas
