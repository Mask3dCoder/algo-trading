"""Serial correlation analysis: ACF, PACF, Ljung-Box, ARCH test."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf


def compute_acf(series: NDArray, nlags: int = 40) -> NDArray:
    """Compute autocorrelation function."""
    return sm_acf(series, nlags=nlags)


def compute_pacf(series: NDArray, nlags: int = 40) -> NDArray:
    """Compute partial autocorrelation function."""
    return sm_pacf(series, nlags=nlags)


@dataclass
class LjungBoxResult:
    statistic: float
    pvalue: float
    is_significant: bool


def ljung_box_test(series: NDArray, lags: int = 10) -> LjungBoxResult:
    """Ljung-Box test for serial correlation.

    H0: no serial correlation (independent)
    Reject H0 → serial correlation exists
    """
    result = acorr_ljungbox(series, lags=[lags], return_df=True)
    stat = float(result.iloc[0]["lb_stat"])
    pval = float(result.iloc[0]["lb_pvalue"])
    return LjungBoxResult(
        statistic=stat,
        pvalue=pval,
        is_significant=pval < 0.05,
    )


@dataclass
class ARCHTestResult:
    statistic: float
    pvalue: float
    has_arch_effects: bool


def arch_test(series: NDArray, lags: int = 5) -> ARCHTestResult:
    """Engle's LM test for ARCH effects (heteroskedasticity).

    H0: no ARCH effects (constant variance)
    Reject H0 → ARCH/GARCH effects present
    """
    result = het_arch(series, nlags=lags)
    stat = float(result[0])
    pval = float(result[1])
    return ARCHTestResult(
        statistic=stat,
        pvalue=pval,
        has_arch_effects=pval < 0.05,
    )
