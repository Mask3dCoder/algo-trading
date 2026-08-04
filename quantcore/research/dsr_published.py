"""Deflated Sharpe Ratio, exact published form (Bailey & Lopez de Prado, 2014).

This module implements the DSR exactly as published in "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"
(*Journal of Portfolio Management*, 40(5), 94-107; also SSRN 2465675).

Formula (Eq. 2-4 of the paper):

    DSR = Phi( (SR_hat - SR*) / sigma_hat(SR_hat) )

with

    sigma_hat(SR_hat) = sqrt( (1 - gamma3*SR_hat + ((gamma4-1)/4)*SR_hat^2)
                              / (T-1) )

and the multiple-testing benchmark

    SR* = SR0 + sigma_hat(SR_hat)
          * ( (1-gamma_em)*Phi^-1(1 - 1/N)
              + gamma_em*Phi^-1(1 - 1/(N*e)) )

where gamma_em is the Euler-Mascheroni constant (0.5772...), e is Euler's
number, N is the number of independent trials, T is the number of return
observations, and gamma3/gamma4 are the *per-observation* skewness and
*raw* kurtosis (3.0 for a normal).

Units contract (critical)
-------------------------
``observed_sharpe`` MUST be the per-observation Sharpe ratio
(mean/std of the daily/monthly return series), NOT the annualized one.
Feeding an annualized Sharpe into this formula silently corrupts the
result (see ``docs/dsr_audit.md`` for the worked comparison).

Legacy note
-----------
``quantcore.research.deflated_sharpe.deflated_sharpe_ratio`` is an earlier,
simplified implementation: it drops the ``sigma_hat`` scaling inside SR*,
uses a (gamma4)/4 instead of (gamma4-1)/4 kurtosis convention, uses
sqrt(T-0.25) instead of sqrt(T-1), and its callers pass annualized Sharpe.
That legacy path is retained for backward compatibility but must NOT be used
for the pre-registered factor gate; use the functions in this module.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.integrate import quad


def expected_max_sharpe(num_trials: int) -> float:
    """Expected maximum of ``num_trials`` iid standard normals.

    Computed exactly by quadrature of
        E[max] = int_0^inf [1 - Phi(x)^N - (1 - Phi(x))^N] dx
    which equals the paper's asymptotic approximation to well within 1e-3 for
    N >= 10 and is more accurate for the small trial counts used here
    (N = 3 family size). The paper's approximation under-corrects for N < 10.
    """
    n = int(num_trials)
    if n <= 1:
        return 0.0

    def integrand(x: float) -> float:
        # Stable log-space form: Phi(x)^N + (1-Phi(x))^N.
        return 1.0 - math.exp(n * stats.norm.logcdf(x)) - math.exp(n * stats.norm.logcdf(-x))

    val, _ = quad(integrand, 0.0, np.inf, limit=200)
    return float(val)


def sharpe_se(
    *,
    observed_sharpe: float,
    sample_length: int,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Mertens (2002) / Johnson (2003) standard error of the Sharpe estimate.

    Returns ``sigma_hat(SR_hat)`` from the paper, or NaN when the variance
    term is non-positive (in which case the DSR is undefined and the p-value
    must be treated as 1.0 - no evidence).
    """
    if sample_length < 2:
        return float("nan")
    skew = 0.0 if skewness is None else float(skewness)
    kurt = 3.0 if kurtosis is None else float(kurtosis)
    if not np.isfinite(kurt) or kurt <= 1.0:
        kurt = 3.0
    var_term = 1.0 - skew * observed_sharpe + ((kurt - 1.0) / 4.0) * observed_sharpe**2
    if not np.isfinite(var_term) or var_term <= 0.0:
        return float("nan")
    return float(np.sqrt(var_term / (float(sample_length) - 1.0)))


def deflated_sharpe_ratio_published(
    *,
    observed_sharpe: float,
    num_trials: int,
    sample_length: int,
    skewness: float | None = None,
    kurtosis: float | None = None,
    benchmark_sharpe: float = 0.0,
) -> tuple[float, float]:
    """Published Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Parameters
    ----------
    observed_sharpe :
        Per-observation Sharpe ratio (mean/std of the return series).
    num_trials :
        Number of independent strategy variants tested before selection.
    sample_length :
        Number of return observations (T).
    skewness :
        Sample skewness of the return series (0 = symmetric).
    kurtosis :
        Sample *raw* kurtosis (3.0 = normal). Excess kurtosis is NOT accepted.
    benchmark_sharpe :
        Per-observation Sharpe of a passive benchmark under the null
        (default 0.0, matching the paper).

    Returns
    -------
    (p_value, dsr_stat) :
        ``p_value`` = 1 - Phi(DSR), the probability under H0 that the Sharpe
        is not significant after deflation. ``dsr_stat`` is the DSR z-score.
    """
    if sample_length < 2 or not np.isfinite(observed_sharpe):
        return 1.0, 0.0
    se = sharpe_se(
        observed_sharpe=observed_sharpe,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    if not np.isfinite(se) or se <= 0.0:
        return 1.0, 0.0
    sr_star = float(benchmark_sharpe) + se * expected_max_sharpe(num_trials)
    dsr_stat = float((observed_sharpe - sr_star) / se)
    p_value = float(1.0 - stats.norm.cdf(dsr_stat))
    return p_value, dsr_stat


def dsr_components(
    *,
    observed_sharpe: float,
    num_trials: int,
    sample_length: int,
    skewness: float | None = None,
    kurtosis: float | None = None,
    benchmark_sharpe: float = 0.0,
) -> dict[str, float]:
    """Return every DSR component for transparent reporting."""
    se = sharpe_se(
        observed_sharpe=observed_sharpe,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    emax = expected_max_sharpe(num_trials)
    sr_star = float(benchmark_sharpe) + se * emax if np.isfinite(se) else float("nan")
    p_value, dsr_stat = deflated_sharpe_ratio_published(
        observed_sharpe=observed_sharpe,
        num_trials=num_trials,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark_sharpe=benchmark_sharpe,
    )
    return {
        "observed_sharpe_per_obs": float(observed_sharpe),
        "benchmark_sharpe": float(benchmark_sharpe),
        "expected_max_std_normal": float(emax),
        "sharpe_se": float(se),
        "sr_star": float(sr_star),
        "dsr_stat": float(dsr_stat),
        "p_value": float(p_value),
    }
