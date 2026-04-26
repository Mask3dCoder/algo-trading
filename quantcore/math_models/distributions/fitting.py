"""Distribution fitting and comparison: Normal, t, NIG, Skewed-t."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kstest, norm
from scipy.stats import t as student_t


@dataclass
class FitResult:
    """Result of fitting a distribution to data."""
    distribution: str
    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float


def fit_normal(data: NDArray) -> FitResult:
    """Fit Normal distribution via MLE."""
    x = np.asarray(data, dtype=float)
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    n = len(x)
    loglik = np.sum(norm.logpdf(x, mu, sigma))
    k = 2
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    ks_stat, ks_pval = kstest(x, "norm", args=(mu, sigma))

    return FitResult(
        distribution="Normal",
        params={"mu": float(mu), "sigma": float(sigma)},
        loglik=float(loglik),
        aic=float(aic),
        bic=float(bic),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
    )


def fit_student_t(data: NDArray) -> FitResult:
    """Fit Student-t distribution via MLE (using scipy.fit)."""
    x = np.asarray(data, dtype=float)
    n = len(x)
    df, loc, scale = student_t.fit(x)
    loglik = np.sum(student_t.logpdf(x, df, loc, scale))
    k = 3
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    ks_stat, ks_pval = kstest(x, "t", args=(df, loc, scale))

    return FitResult(
        distribution="Student-t",
        params={"df": float(df), "loc": float(loc), "scale": float(scale)},
        loglik=float(loglik),
        aic=float(aic),
        bic=float(bic),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
    )


def fit_skewed_t(data: NDArray) -> FitResult:
    """Fit skewed-t distribution (using scipy skewt if available, else fallback)."""
    try:
        from scipy.stats import skewt
        x = np.asarray(data, dtype=float)
        n = len(x)
        a, df, loc, scale = skewt.fit(x)
        loglik = np.sum(skewt.logpdf(x, a, df, loc, scale))
        k = 4
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik
        ks_stat, ks_pval = kstest(x, "skewt", args=(a, df, loc, scale))
        return FitResult(
            distribution="Skewed-t",
            params={"a": float(a), "df": float(df), "loc": float(loc), "scale": float(scale)},
            loglik=float(loglik), aic=float(aic), bic=float(bic),
            ks_statistic=float(ks_stat), ks_pvalue=float(ks_pval),
        )
    except ImportError:
        return fit_student_t(data)  # Fallback


def compare_distributions(data: NDArray) -> list[FitResult]:
    """Fit multiple distributions and rank by AIC (lower is better)."""
    results = [
        fit_normal(data),
        fit_student_t(data),
    ]
    try:
        results.append(fit_skewed_t(data))
    except Exception:
        pass

    results.sort(key=lambda r: r.aic)
    return results


def qq_plot_data(data: NDArray, distribution: str = "normal") -> tuple[NDArray, NDArray]:
    """Return (theoretical_quantiles, sample_quantiles) for QQ plot."""
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    theoretical = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)

    if distribution == "normal":
        quantiles = norm.ppf(theoretical) * np.std(x, ddof=1) + np.mean(x)
    elif distribution == "student_t":
        df, loc, scale = student_t.fit(x)
        quantiles = student_t.ppf(theoretical, df, loc, scale)
    else:
        quantiles = norm.ppf(theoretical) * np.std(x, ddof=1) + np.mean(x)

    return quantiles, x
