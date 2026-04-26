"""Extreme Value Theory: GPD, GEV, and EVT-based risk measures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genextreme, genpareto


@dataclass
class GPDResult:
    """Generalized Pareto Distribution fit result."""
    shape: float  # xi (tail index)
    scale: float  # sigma
    threshold: float
    n_exceedances: int
    loglik: float


@dataclass
class GEVResult:
    """Generalized Extreme Value Distribution fit result."""
    shape: float  # xi
    loc: float    # mu
    scale: float  # sigma
    loglik: float


def fit_gpd(
    losses: NDArray,
    threshold: float | None = None,
    quantile_threshold: float = 0.95,
) -> GPDResult:
    """Fit GPD to tail losses exceeding a threshold (Peaks Over Threshold).

    If threshold is None, use the quantile_threshold quantile of losses.
    """
    x = np.asarray(losses, dtype=float)

    if threshold is None:
        threshold = float(np.quantile(x, quantile_threshold))

    exceedances = x[x > threshold] - threshold
    n_exceed = len(exceedances)

    if n_exceed < 10:
        return GPDResult(shape=0.0, scale=0.0, threshold=threshold, n_exceedances=n_exceed, loglik=-np.inf)

    # MLE fit
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    loglik = np.sum(genpareto.logpdf(exceedances, shape, 0, scale))

    return GPDResult(
        shape=float(shape),
        scale=float(scale),
        threshold=threshold,
        n_exceedances=n_exceed,
        loglik=float(loglik),
    )


def fit_gev(block_maxima: NDArray) -> GEVResult:
    """Fit GEV to block maxima (e.g., monthly max losses)."""
    x = np.asarray(block_maxima, dtype=float)
    shape, loc, scale = genextreme.fit(-x)  # Negative because genextreme fits maxima convention
    loglik = np.sum(genextreme.logpdf(-x, shape, loc, scale))

    return GEVResult(
        shape=float(shape),
        loc=float(loc),
        scale=float(scale),
        loglik=float(loglik),
    )


def evt_var(gpd_result: GPDResult, confidence: float = 0.999, n_total: int = 0) -> float:
    """EVT-based Value at Risk using fitted GPD.

    VaR_alpha = threshold + (sigma/xi) * ((n/N * (1-alpha))^(-xi) - 1)
    """
    if gpd_result.n_exceedances == 0 or gpd_result.scale == 0:
        return np.nan

    n = n_total or gpd_result.n_exceedances * 10
    xi = gpd_result.shape
    sigma = gpd_result.scale
    u = gpd_result.threshold
    n_u = gpd_result.n_exceedances

    if abs(xi) < 1e-8:
        # Exponential tail
        return float(u + sigma * np.log(n / n_u * (1.0 - confidence)))

    return float(u + sigma / xi * ((n / n_u * (1.0 - confidence)) ** (-xi) - 1.0))


def evt_cvar(gpd_result: GPDResult, confidence: float = 0.999, n_total: int = 0) -> float:
    """EVT-based Conditional VaR (Expected Shortfall).

    CVaR = VaR + (sigma + xi*(VaR - u)) / (1 - xi)
    """
    var = evt_var(gpd_result, confidence, n_total)
    if np.isnan(var):
        return np.nan

    xi = gpd_result.shape
    sigma = gpd_result.scale
    u = gpd_result.threshold

    return float(var + (sigma + xi * (var - u)) / (1.0 - xi)) if abs(1.0 - xi) > 1e-8 else float(var + sigma + xi * (var - u))


def mean_excess_plot(losses: NDArray, n_thresholds: int = 50) -> tuple[NDArray, NDArray]:
    """Mean excess function for threshold selection.

    Returns (thresholds, mean_excess) arrays.
    """
    x = np.sort(np.asarray(losses, dtype=float))
    len(x)

    thresholds = np.linspace(x[0], x[-2], n_thresholds)
    mean_excess = np.zeros(n_thresholds)

    for i, u in enumerate(thresholds):
        exceedances = x[x > u] - u
        mean_excess[i] = np.mean(exceedances) if len(exceedances) > 0 else 0.0

    return thresholds, mean_excess
