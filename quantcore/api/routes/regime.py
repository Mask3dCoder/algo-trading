"""Regime detection API routes: HMM, features, changepoint, volatility."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from quantcore.regime.hmm import RegimeHMM, select_hmm_states
from quantcore.regime.features import (
    realized_volatility, rolling_correlation, adx,
    hurst_exponent, turnover_ratio, bid_ask_spread,
    credit_spread, yield_curve_slope, compute_all_features,
)
from quantcore.regime.changepoint import cusum
from quantcore.regime.volatility_regime import vix_regime, garch_regime, vol_percentile_regime

router = APIRouter(prefix="/regime", tags=["regime"])


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------


@router.post("/hmm/fit")
def api_hmm_fit(
    returns: list[float],
    n_states: int = Query(2, ge=2, le=5),
) -> dict[str, Any]:
    """Fit a Hidden Markov Model for regime detection."""
    try:
        hmm = RegimeHMM(n_states=n_states)
        result = hmm.fit(np.array(returns))
        return {
            "n_states": result.n_states,
            "current_state": result.current_state,
            "current_probability": result.current_probability,
            "state_labels": result.state_labels,
            "transition_matrix": result.transition_matrix.tolist(),
            "means": result.means.tolist(),
            "volatilities": result.volatilities.tolist(),
            "aic": result.aic,
            "bic": result.bic,
            "states_summary": {
                label: {"mean_return": float(result.means[i]), "volatility": float(result.volatilities[i])}
                for i, label in enumerate(result.state_labels)
            },
        }
    except ImportError:
        raise HTTPException(501, "hmmlearn not installed")
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/hmm/select")
def api_hmm_select(
    returns: list[float],
    max_states: int = Query(5, ge=2, le=8),
) -> dict[str, Any]:
    """Select optimal number of HMM states via AIC/BIC."""
    try:
        results = select_hmm_states(np.array(returns), max_states)
        return {
            "results": results,
            "best_aic": results[0] if results else None,
        }
    except ImportError:
        raise HTTPException(501, "hmmlearn not installed")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@router.post("/features/compute")
def api_compute_features(
    returns: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
    volume: list[float] | None = None,
) -> dict[str, float]:
    """Compute all regime detection features."""
    r = np.array(returns)
    h_arr = np.array(high) if high else np.random.normal(0, 1, len(r))
    l_arr = np.array(low) if low else -np.abs(np.random.normal(0, 1, len(r)))
    c_arr = np.array(close) if close else np.cumsum(r) + 100
    v_arr = np.array(volume) if volume else np.ones(len(r)) * 1000

    features = compute_all_features(r, high=h_arr, low=l_arr, close=c_arr, volume=v_arr)
    return {k: float(v) if v is not None else None for k, v in features.items()}


@router.post("/features/volatility")
def api_realized_vol(
    returns: list[float],
    window: int = Query(21, ge=5, le=252),
) -> dict[str, float]:
    """Compute realized volatility."""
    r = np.array(returns)
    vol = realized_volatility(r, window=window)
    return {"realized_volatility": vol, "window": window}


@router.post("/features/hurst")
def api_hurst(
    returns: list[float],
    max_lag: int = Query(20, ge=5, le=100),
) -> dict[str, float]:
    """Compute Hurst exponent (mean-reversion vs trending)."""
    r = np.array(returns)
    h = hurst_exponent(r[:500], max_lag=max_lag)
    return {"hurst_exponent": h, "interpretation": "mean-reverting" if h < 0.5 else "trending" if h > 0.5 else "random-walk"}


@router.post("/features/adx")
def api_adx(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = Query(14, ge=5, le=50),
) -> dict[str, float]:
    """Compute Average Directional Index (trend strength)."""
    adx_val = adx(np.array(high), np.array(low), np.array(close), period)
    return {"adx": adx_val, "trend_strength": "strong" if adx_val > 25 else "weak"}


@router.post("/features/correlation")
def api_rolling_corr(
    returns1: list[float],
    returns2: list[float],
    window: int = Query(63, ge=10),
) -> dict[str, float]:
    """Compute rolling correlation between two return series."""
    corr = rolling_correlation(np.array(returns1), np.array(returns2), window)
    return {"correlation": corr, "window": window}


# ---------------------------------------------------------------------------
# Changepoint
# ---------------------------------------------------------------------------


@router.post("/changepoint/cusum")
def api_cusum(
    data: list[float],
    threshold: float = Query(5.0, gt=0),
    drift: float = Query(1.0, gt=0),
) -> dict[str, Any]:
    """Detect changepoints using CUSUM."""
    result = cusum(np.array(data), threshold=threshold, drift=drift)
    return {
        "n_changepoints": result.n_changepoints,
        "changepoints": result.changepoints if hasattr(result, "changepoints") else [],
    }


# ---------------------------------------------------------------------------
# Volatility Regime
# ---------------------------------------------------------------------------


@router.get("/volatility/vix")
def api_vix_regime(vix: float = Query(..., gt=0)) -> dict[str, str]:
    """Classify volatility regime from VIX level."""
    result = vix_regime(vix)
    return {"regime": result.regime.value, "vix_level": vix}


@router.post("/volatility/garch")
def api_garch_regime(returns: list[float]) -> dict[str, Any]:
    """Classify volatility regime using GARCH model."""
    try:
        result = garch_regime(np.array(returns))
        return {
            "regime": result.regime.value,
            "current_vol": result.current_vol if hasattr(result, "current_vol") else None,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/volatility/percentile")
def api_vol_percentile(
    returns: list[float],
    window: int = Query(252, ge=20),
) -> dict[str, Any]:
    """Classify volatility regime by historical percentile."""
    result = vol_percentile_regime(np.array(returns), window=window)
    return {
        "regime": result.regime.value,
        "percentile": result.percentile,
        "current_vol": result.current_vol if hasattr(result, "current_vol") else None,
    }
