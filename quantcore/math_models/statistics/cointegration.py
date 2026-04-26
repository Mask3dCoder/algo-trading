"""Cointegration tests and pair scanning: Engle-Granger, Johansen, VECM."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import linregress
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class CointResult:
    """Engle-Granger cointegration test result."""
    statistic: float
    pvalue: float
    hedge_ratio: float
    is_cointegrated: bool
    half_life: float


@dataclass
class JohansenResult:
    """Johansen cointegration test result."""
    trace_stats: NDArray
    trace_crit_vals: NDArray
    max_eigen_stats: NDArray
    max_eigen_crit_vals: NDArray
    n_cointegrating: int
    eigenvectors: NDArray


@dataclass
class CointPair:
    """Cointegrated pair with metadata."""
    symbol_a: str
    symbol_b: str
    pvalue: float
    hedge_ratio: float
    half_life: float
    score: float  # Composite score (lower is better)


def engle_granger(
    y: NDArray,
    x: NDArray,
    significance: float = 0.05,
) -> CointResult:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS regression y = alpha + beta*x + epsilon
    Step 2: ADF test on residuals
    """
    # Step 1: OLS hedge ratio
    slope, intercept, _, _, _ = linregress(x, y)
    residuals = y - (intercept + slope * x)

    # Step 2: ADF on residuals
    adf_result = adfuller(residuals, maxlag=1, regression="c")
    pval = adf_result[1]

    # Half-life of mean reversion in spread
    spread_lag = np.roll(residuals, 1)
    spread_lag[0] = spread_lag[1]
    spread_ret = residuals - spread_lag
    reg = linregress(spread_lag, spread_ret)
    theta = -reg.slope
    hl = np.log(2) / theta if theta > 0 else float("inf")

    return CointResult(
        statistic=float(adf_result[0]),
        pvalue=float(pval),
        hedge_ratio=float(slope),
        is_cointegrated=pval < significance,
        half_life=float(hl),
    )


def johansen_test(
    data: NDArray,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Johansen cointegration test (trace and max eigenvalue statistics).

    data: (T, n) matrix of time series (each column is a series)
    """
    result = coint_johansen(data, det_order, k_ar_diff)

    # Count cointegrating relationships at 5% level
    trace_stats = result.lr1
    trace_crit = result.cvt  # columns: 90%, 95%, 99%
    n_coint = int(np.sum(trace_stats > trace_crit[:, 1]))  # 95% critical values

    return JohansenResult(
        trace_stats=trace_stats,
        trace_crit_vals=trace_crit,
        max_eigen_stats=result.lr2,
        max_eigen_crit_vals=result.cvm,
        n_cointegrating=n_coint,
        eigenvectors=result.evec,
    )


def vecm(data: NDArray, coint_rank: int = 1, k_ar_diff: int = 1) -> dict:
    """Fit Vector Error Correction Model.

    Returns dict with coefficients, cointegrating vectors, and adjustment speeds.
    """
    from statsmodels.tsa.vector_ar.vecm import VECM

    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="ci")
    result = model.fit()

    return {
        "beta": result.beta,         # cointegrating vectors
        "alpha": result.alpha,       # adjustment speeds
        "gamma": result.gamma,       # short-run coefficients
        "sigma_u": result.sigma_u,   # residual covariance
    }


def scan_pairs(
    returns_matrix: NDArray,
    symbols: list[str],
    method: str = "engle_granger",
    significance: float = 0.01,
    min_half_life: float = 1.0,
    max_half_life: float = 252.0,
) -> list[CointPair]:
    """Scan all pairs in a universe for cointegration.

    returns_matrix: (T, N) array of price series
    symbols: list of N symbol names
    """
    pairs: list[CointPair] = []
    n = len(symbols)

    for i, j in combinations(range(n), 2):
        y = returns_matrix[:, i]
        x = returns_matrix[:, j]

        if method == "engle_granger":
            result = engle_granger(y, x, significance=significance)
            if result.is_cointegrated and min_half_life <= result.half_life <= max_half_life:
                score = result.pvalue * abs(result.half_life - 20)
                pairs.append(CointPair(
                    symbol_a=symbols[i],
                    symbol_b=symbols[j],
                    pvalue=result.pvalue,
                    hedge_ratio=result.hedge_ratio,
                    half_life=result.half_life,
                    score=score,
                ))

    pairs.sort(key=lambda p: p.score)
    return pairs


def rolling_cointegration(
    y: NDArray,
    x: NDArray,
    window: int = 252,
) -> NDArray:
    """Rolling cointegration p-values."""
    n = len(y)
    pvals = np.full(n, np.nan)

    for t in range(window, n):
        y_w = y[t - window:t]
        x_w = x[t - window:t]
        try:
            _, pv, _ = coint(y_w, x_w)
            pvals[t] = pv
        except Exception:
            pvals[t] = np.nan

    return pvals
