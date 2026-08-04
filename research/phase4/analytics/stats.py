"""Phase 4 — Statistical Validation (Stage 4).

Bootstrap confidence intervals, permutation null tests, and
Benjamini-Hochberg False Discovery Rate correction for multiple testing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


@dataclass
class StatTestResult:
    ic_mean: float
    ic_std: float
    ic_tstat: float
    ic_pvalue: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    permutation_pvalue: float
    n_observations: int
    reject_null: bool          # True if both p-values < 0.10


def bootstrap_ic_ci(
    ic_series: pd.Series,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean IC.

    Resamples the daily IC time series (with replacement) and computes
    the empirical distribution of mean IC.

    Returns:
        (lower_bound, upper_bound) at the given CI level.
    """
    rng = np.random.default_rng(seed)
    ic_arr = ic_series.dropna().values
    if len(ic_arr) < 20:
        return (float("nan"), float("nan"))

    boot_means = np.array([
        np.mean(rng.choice(ic_arr, size=len(ic_arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def permutation_pvalue(
    ic_series: pd.Series,
    n_permutations: int = 500,
    seed: int = 42,
) -> float:
    """Permutation test p-value for the mean IC.

    Under the null hypothesis (signal has no predictive power), randomly
    shuffling the daily IC values should produce mean IC ≈ 0. The p-value
    is the fraction of permuted means that exceed the observed mean IC.

    This tests: H0: E[IC] = 0  vs  H1: E[IC] > 0.
    """
    rng = np.random.default_rng(seed)
    ic_arr = ic_series.dropna().values
    if len(ic_arr) < 20:
        return float("nan")

    observed_mean = float(np.mean(ic_arr))
    null_means = np.array([
        np.mean(rng.permutation(ic_arr))
        for _ in range(n_permutations)
    ])
    # One-sided: fraction of null means >= observed mean
    pvalue = float(np.mean(null_means >= observed_mean))
    return pvalue


def t_test_ic(ic_series: pd.Series) -> tuple[float, float]:
    """One-sample t-test: H0: mean(IC) = 0.

    Returns:
        (t_stat, p_value) — two-sided.
    """
    ic_arr = ic_series.dropna().values
    if len(ic_arr) < 5:
        return (float("nan"), float("nan"))
    tstat, pval = scipy_stats.ttest_1samp(ic_arr, popmean=0.0)
    return float(tstat), float(pval)


def run_full_stat_test(
    factor_name: str,
    ic_series: pd.Series,
    n_bootstrap: int = 1000,
    n_permutations: int = 500,
) -> StatTestResult:
    """Run all statistical tests for a single factor's daily IC series."""
    ic_arr = ic_series.dropna().values
    ic_mean = float(np.mean(ic_arr)) if len(ic_arr) > 0 else float("nan")
    ic_std  = float(np.std(ic_arr)) if len(ic_arr) > 0 else float("nan")

    tstat, pval_t = t_test_ic(ic_series)
    ci_lo, ci_hi = bootstrap_ic_ci(ic_series, n_bootstrap=n_bootstrap)
    pval_perm = permutation_pvalue(ic_series, n_permutations=n_permutations)

    # Reject null if: t-test p < 0.10 AND permutation p < 0.10 AND CI excludes 0
    reject = (
        (not np.isnan(pval_t) and pval_t < 0.10)
        and (not np.isnan(pval_perm) and pval_perm < 0.10)
        and (not np.isnan(ci_lo) and ci_lo > 0)
    )

    log.info(
        "  %s: IC=%.4f t=%.2f p_t=%.4f p_perm=%.4f CI=[%.4f,%.4f] → %s",
        factor_name, ic_mean, tstat, pval_t, pval_perm, ci_lo, ci_hi,
        "REJECT NULL" if reject else "FAIL TO REJECT",
    )

    return StatTestResult(
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_tstat=tstat,
        ic_pvalue=pval_t,
        bootstrap_ci_lower=ci_lo,
        bootstrap_ci_upper=ci_hi,
        permutation_pvalue=pval_perm,
        n_observations=len(ic_arr),
        reject_null=reject,
    )


def benjamini_hochberg_correction(
    pvalues: dict[str, float],
    alpha: float = 0.10,
) -> dict[str, bool]:
    """Apply Benjamini-Hochberg FDR correction across multiple factors.

    Controls the expected proportion of false discoveries among rejected null
    hypotheses. Essential when testing 13+ factors simultaneously.

    Returns:
        dict mapping factor_name → True if null is rejected after FDR correction.
    """
    names = [k for k, v in pvalues.items() if not np.isnan(v)]
    pvals = np.array([pvalues[n] for n in names])

    m = len(pvals)
    if m == 0:
        return {}

    # BH procedure: sort, compute threshold, find largest k where p(k) <= k/m * alpha
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    significant = sorted_pvals <= thresholds

    # All tests up to the largest significant one are also significant
    if np.any(significant):
        last_sig = np.where(significant)[0][-1]
        significant[:last_sig + 1] = True

    # Map back to original order
    result: dict[str, bool] = {n: False for n in pvalues}
    for rank, orig_idx in enumerate(sorted_idx):
        result[names[orig_idx]] = bool(significant[rank])

    rejected = [n for n, v in result.items() if v]
    log.info(
        "BH FDR correction (α=%.2f): %d/%d factors survive multiple-testing adjustment: %s",
        alpha, len(rejected), len(names), rejected,
    )
    return result
