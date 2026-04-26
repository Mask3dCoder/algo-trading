"""Stationarity tests: ADF, KPSS, Phillips-Perron, variance ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller, kpss, phillips_ouliolis


@dataclass
class TestResult:
    """Result of a statistical test."""
    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    is_stationary: bool
    interpretation: str


@dataclass
class StationarityReport:
    """Results from all stationarity tests."""
    adf: TestResult
    kpss: TestResult
    phillips_perron: TestResult | None
    variance_ratio: TestResult | None
    consensus: str  # "stationary", "non-stationary", "inconclusive"


def adf_test(
    series: NDArray,
    maxlag: int | None = None,
    significance: float = 0.05,
) -> TestResult:
    """Augmented Dickey-Fuller test for unit root.

    H0: series has a unit root (non-stationary)
    Reject H0 → stationary
    """
    result = adfuller(series, maxlag=maxlag, autolag="AIC")
    stat = result[0]
    pval = result[1]
    crit = result[4]
    is_stat = pval < significance

    return TestResult(
        statistic=float(stat),
        pvalue=float(pval),
        critical_values={k: float(v) for k, v in crit.items()},
        is_stationary=is_stat,
        interpretation=f"ADF stat={stat:.4f}, p={pval:.4f} — {'stationary' if is_stat else 'non-stationary'}",
    )


def kpss_test(
    series: NDArray,
    regression: str = "c",
    significance: float = 0.05,
) -> TestResult:
    """KPSS test for stationarity.

    H0: series is level or trend stationary
    Fail to reject H0 → stationary (opposite of ADF)
    """
    result = kpss(series, regression=regression, nlags="auto")
    stat = result[0]
    pval = result[1]
    crit = result[3]
    is_stat = pval >= significance

    return TestResult(
        statistic=float(stat),
        pvalue=float(pval),
        critical_values={k: float(v) for k, v in crit.items()},
        is_stationary=is_stat,
        interpretation=f"KPSS stat={stat:.4f}, p={pval:.4f} — {'stationary' if is_stat else 'non-stationary'}",
    )


def phillips_perron_test(
    series: NDArray,
    significance: float = 0.05,
) -> TestResult:
    """Phillips-Perron test for unit root (non-parametric correction).

    H0: unit root (non-stationary), like ADF.
    """
    result = phillips_ouliolis(series)
    # PP wraps same as ADF
    stat = float(result[0])
    pval = float(result[1])
    crit = result[2] if len(result) > 2 else {}
    is_stat = pval < significance

    return TestResult(
        statistic=stat,
        pvalue=pval,
        critical_values={k: float(v) for k, v in crit.items()} if isinstance(crit, dict) else {},
        is_stationary=is_stat,
        interpretation=f"PP stat={stat:.4f}, p={pval:.4f} — {'stationary' if is_stat else 'non-stationary'}",
    )


def variance_ratio_test(
    series: NDArray,
    lags: int = 12,
    significance: float = 0.05,
) -> TestResult:
    """Lo-MacKinlay variance ratio test for random walk.

    H0: series is a random walk (VR = 1)
    Reject H0 → not a random walk (either mean-reverting or trending)
    """
    from statsmodels.tsa.stattools import zivot_andrews  # noqa — we use a custom implementation

    # Custom implementation
    n = len(series)
    returns = np.diff(np.log(series)) if np.all(series > 0) else np.diff(series)
    n_r = len(returns)

    mu = np.mean(returns)
    var_1 = np.sum((returns - mu) ** 2) / (n_r - 1)

    # Variance at `lags` period
    lagged_returns = returns[lags:] - returns[:-lags] if lags < n_r else returns
    var_q = np.var(returns[::lags] if lags > 1 else returns, ddof=1)
    # Alternative computation
    q_returns = np.convolve(returns, np.ones(lags), mode='valid')
    var_q = np.var(q_returns, ddof=1) / lags if len(q_returns) > 1 else var_1

    vr = var_q / var_1 if var_1 > 0 else 1.0

    # Z-statistic
    phi = 2.0 * (2 * lags - 1) * (lags - 1) / (3 * lags * n_r)
    z_stat = (vr - 1.0) / np.sqrt(phi) if phi > 0 else 0.0

    from scipy.stats import norm
    pval = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    is_stat = pval < significance

    crit_vals = {"1%": 2.576, "5%": 1.960, "10%": 1.645}

    return TestResult(
        statistic=float(z_stat),
        pvalue=float(pval),
        critical_values=crit_vals,
        is_stationary=is_stat,
        interpretation=f"VR={vr:.4f}, z={z_stat:.4f}, p={pval:.4f} — {'not random walk' if is_stat else 'random walk'}",
    )


def test_stationarity(
    series: NDArray,
    significance: float = 0.05,
) -> StationarityReport:
    """Run all stationarity tests and report consensus."""
    adf_result = adf_test(series, significance=significance)
    kpss_result = kpss_test(series, significance=significance)

    try:
        pp_result = phillips_perron_test(series, significance=significance)
    except Exception:
        pp_result = None

    try:
        vr_result = variance_ratio_test(series, significance=significance)
    except Exception:
        vr_result = None

    # Consensus: both ADF and KPSS agree
    votes = [adf_result.is_stationary, kpss_result.is_stationary]
    if pp_result:
        votes.append(pp_result.is_stationary)

    if sum(votes) >= len(votes) * 0.5:
        consensus = "stationary" if sum(votes) > len(votes) * 0.5 else "inconclusive"
    else:
        consensus = "non-stationary"

    if sum(votes) == len(votes) or sum(votes) == 0:
        consensus = "stationary" if all(votes) else "non-stationary"

    return StationarityReport(
        adf=adf_result,
        kpss=kpss_result,
        phillips_perron=pp_result,
        variance_ratio=vr_result,
        consensus=consensus,
    )
