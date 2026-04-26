"""Portfolio management API routes: optimization, sizing, rebalance, tax."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from quantcore.portfolio.sizing import (
    kelly_criterion, fractional_kelly, volatility_target, atr_size, cvar_size,
)
from quantcore.portfolio.optimization import (
    mean_variance, compute_frontier, black_litterman, bl_optimize,
    hrp, nco, risk_parity, stability_test,
)
from quantcore.portfolio.rebalance import RebalanceEngine, RebalanceConfig, RebalanceTrigger

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@router.post("/sizing/kelly")
def api_kelly(
    win_prob: float = Query(..., gt=0, le=1),
    avg_win: float = Query(..., gt=0),
    avg_loss: float = Query(..., gt=0),
    fraction: float | None = None,
) -> dict[str, float]:
    """Compute Kelly criterion position size."""
    if fraction is not None:
        f = fractional_kelly(win_prob, avg_win, avg_loss, fraction)
    else:
        f = kelly_criterion(win_prob, avg_win, avg_loss)
    return {"kelly_fraction": f, "fractional": fraction is not None}


@router.post("/sizing/vol-target")
def api_vol_target(
    asset_vol: float = Query(..., gt=0),
    target_vol: float = Query(0.15, gt=0),
    n_assets: int = Query(1, ge=1),
) -> dict[str, float]:
    """Compute volatility-targeted position weight."""
    w = volatility_target(asset_vol, target_vol, n_assets)
    return {"weight": w, "target_vol": target_vol}


@router.post("/sizing/atr")
def api_atr_size(
    account_value: float = Query(..., gt=0),
    risk_pct: float = Query(0.01, gt=0, le=1),
    atr: float = Query(..., gt=0),
    multiplier: float = Query(2.0, gt=0),
) -> dict[str, float]:
    """Compute ATR-based position sizing."""
    shares = atr_size(account_value, risk_pct, atr, multiplier)
    return {"shares": shares, "risk_amount": account_value * risk_pct}


@router.post("/sizing/cvar")
def api_cvar_size(
    returns: list[float],
    max_cvar: float = Query(..., gt=0),
    account_value: float = Query(..., gt=0),
    confidence: float = Query(0.975),
) -> dict[str, float]:
    """Compute CVaR-constrained position size."""
    r = np.array(returns)
    w = cvar_size(r, max_cvar, account_value, confidence)
    return {"weight": w, "max_cvar": max_cvar}


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


@router.post("/optimize/mvo")
def api_mean_variance(
    returns: list[list[float]],
    target: str = Query("max_sharpe", pattern="^(max_sharpe|min_variance)$"),
    long_only: bool = True,
    rf: float = 0.0,
) -> dict[str, Any]:
    """Run mean-variance optimization."""
    r = np.array(returns)
    result = mean_variance(r, target=target, long_only=long_only, rf=rf)
    return {
        "weights": result.weights.tolist(),
        "expected_return": result.expected_return,
        "expected_vol": result.expected_vol,
        "sharpe_ratio": result.sharpe_ratio,
        "method": result.method,
    }


@router.post("/optimize/frontier")
def api_frontier(
    returns: list[list[float]],
    n_points: int = Query(50, le=200),
    long_only: bool = True,
) -> dict[str, list]:
    """Compute the efficient frontier."""
    r = np.array(returns)
    frontier = compute_frontier(r, n_points, long_only)
    return {
        "points": [
            {"risk": float(risk), "return": float(ret), "weights": w.tolist()}
            for risk, ret, w in frontier
        ]
    }


@router.post("/optimize/black-litterman")
def api_black_litterman(
    returns: list[list[float]],
    market_caps: list[float],
    views: dict[int, float] | None = None,
    long_only: bool = True,
) -> dict[str, Any]:
    """Run Black-Litterman optimization."""
    r = np.array(returns)
    mc = np.array(market_caps)
    cov = np.cov(r.T) * 252
    posterior_ret, posterior_cov = black_litterman(mc, cov, views=views)
    result = bl_optimize(posterior_ret, posterior_cov, long_only)
    return {
        "weights": result.weights.tolist(),
        "expected_return": result.expected_return,
        "expected_vol": result.expected_vol,
        "sharpe_ratio": result.sharpe_ratio,
        "posterior_returns": posterior_ret.tolist(),
    }


@router.post("/optimize/hrp")
def api_hrp(returns: list[list[float]]) -> dict[str, Any]:
    """Run Hierarchical Risk Parity optimization."""
    r = np.array(returns)
    result = hrp(r)
    return {
        "weights": result.weights.tolist(),
        "expected_return": result.expected_return,
        "expected_vol": result.expected_vol,
        "sharpe_ratio": result.sharpe_ratio,
        "method": "hrp",
    }


@router.post("/optimize/nco")
def api_nco(
    returns: list[list[float]],
    n_clusters: int | None = None,
) -> dict[str, Any]:
    """Run Nested Clustered Optimization."""
    r = np.array(returns)
    result = nco(r, n_clusters)
    return {
        "weights": result.weights.tolist(),
        "expected_return": result.expected_return,
        "expected_vol": result.expected_vol,
        "sharpe_ratio": result.sharpe_ratio,
        "method": "nco",
    }


@router.post("/optimize/risk-parity")
def api_risk_parity(
    returns: list[list[float]],
    target_vol: float | None = None,
) -> dict[str, Any]:
    """Run Risk Parity optimization."""
    r = np.array(returns)
    cov = np.cov(r.T) * 252
    result = risk_parity(cov, target_vol)
    return {
        "weights": result.weights.tolist(),
        "expected_vol": result.expected_vol,
        "sharpe_ratio": result.sharpe_ratio,
    }


@router.post("/optimize/stability")
def api_stability_test(
    returns: list[list[float]],
    n_perturbations: int = Query(100, le=500),
) -> dict[str, float]:
    """Test MVO vs HRP weight stability."""
    r = np.array(returns)
    result = stability_test(r, n_perturbations=n_perturbations)
    return result


# ---------------------------------------------------------------------------
# Rebalance
# ---------------------------------------------------------------------------


@router.post("/rebalance/check")
def api_rebalance_check(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    drift_threshold: float = Query(0.05, gt=0),
    tolerance: float = Query(0.005),
) -> dict[str, Any]:
    """Check if rebalancing is needed."""
    engine = RebalanceEngine(RebalanceConfig(
        trigger=RebalanceTrigger.THRESHOLD,
        drift_threshold=drift_threshold,
        tolerance=tolerance,
    ))
    should, reason = engine.should_rebalance(current_weights, target_weights)
    return {"should_rebalance": should, "reason": reason}


@router.post("/rebalance/compute")
def api_rebalance_compute(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value: float = Query(..., gt=0),
    tolerance: float = Query(0.005),
) -> dict[str, Any]:
    """Compute rebalancing trades."""
    engine = RebalanceEngine(RebalanceConfig(tolerance=tolerance))
    trades = engine.compute_trades(
        current_weights, target_weights, portfolio_value,
    )
    return {"trades": trades}
