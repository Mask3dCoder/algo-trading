"""Deflated Sharpe Ratio (Bailey & López de Prado, 2012).

⚠ DEPRECATED / STATISTICALLY INCORRECT — DO NOT USE FOR VERDICTS ⚠

This legacy implementation inflates the DSR statistic by sqrt(T-0.25), omits
the sigma_hat scaling inside SR*, uses a mixed kurtosis convention, and its
callers historically passed ANNUALIZED Sharpe where the formula requires the
per-observation Sharpe. Every DSR p-value computed through it is
**uninformative** (see FREEZE.md DSR correction note, 2026-08-05, and
docs/dsr_audit.md).

The corrected, published Bailey & López de Prado (2014) implementation is
``quantcore.research.dsr_published`` and MUST be used for any gate, triage, or
promotion decision. This module is retained only for backward compatibility
with archived research scripts.

Provides the ``deflated_sharpe_ratio`` function, which computes the
probability that an observed Sharpe ratio is statistically significant
given the number of independent trials inherent in the backtest
optimisation process.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import stats

_log = logging.getLogger(__name__)


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    num_trials: int,
    sample_length: int,
    skewness: float | None = None,
    kurtosis_: float | None = None,
) -> tuple[float, float]:
    """Compute the Bailey & López de Prado Deflated Sharpe Ratio (DSR).

    .. deprecated::
        Statistically incorrect legacy instrument — results are uninformative.
        Use ``quantcore.research.dsr_published.deflated_sharpe_ratio_published``
        (per-observation Sharpe, RAW kurtosis) for any verdict.

    Parameters
    ----------
    observed_sharpe :
        The Sharpe ratio observed in the final backtest.
    num_trials :
        Number of independent strategy variations (trials) tested
        before selecting the best.  A conservative lower bound is fine.
    sample_length :
        Number of observations in the full return series (e.g. days).
    skewness :
        Sample skewness of the returns.  If *None*, it is computed from
        the ``returns`` array.
    kurtosis_ :
        Sample excess kurtosis of the returns.  If *None*, it is
        computed from the ``returns`` array.

    Returns
    -------
    p_value :
        Probability under H₀ that the Sharpe is not significant.
    dsr_stat :
        The deflated Sharpe statistic (same scale as ``observed_sharpe``).
    """
    warnings.warn(
        "quantcore.research.deflated_sharpe.deflated_sharpe_ratio is a legacy, "
        "statistically incorrect instrument (sqrt(T-0.25) inflation; callers "
        "historically passed annualized Sharpe). Its p-values are uninformative. "
        "Use quantcore.research.dsr_published.deflated_sharpe_ratio_published "
        "for any verdict (FREEZE.md DSR correction note, 2026-08-05).",
        UserWarning,
        stacklevel=2,
    )
    n = float(sample_length)

    # Expected maximum Sharpe under the null (max of N standard normals)
    expected_max = _expected_max_sharpe(num_trials)

    # Variance adjustment (Eq. 4 in B&LdP, 2012)
    if skewness is None or kurtosis_ is None:
        # Fall back to standard normal assumption
        variance_term = 1.0
    else:
        variance_term = np.sqrt(
            1.0
            - skewness * observed_sharpe
            + ((kurtosis_ + 3.0) / 4.0) * observed_sharpe ** 2
        )

    dsr = (observed_sharpe - expected_max) * np.sqrt(n - 0.25) / variance_term

    p_value = 1.0 - stats.norm.cdf(dsr)

    return float(p_value), float(dsr)


def _expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of N independent standard normals (FBST approximation)."""
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649015329
    log_n = np.log(n_trials)
    return (
        np.sqrt(2.0 * log_n)
        - (np.log(log_n) + np.log(4.0 * np.pi)) / (2.0 * np.sqrt(2.0 * log_n))
        + euler_mascheroni / np.sqrt(2.0 * log_n)
    )


def min_backtest_length(
    observed_sharpe: float,
    num_trials: int,
    skewness: float = 0.0,
    kurtosis_: float = 3.0,
    target_prob: float = 0.95,
) -> int:
    """Minimum required backtest length (in years) for significance.

    Solves for the smallest T such that P(DSR > d) >= ``target_prob``.
    """
    z = stats.norm.ppf(target_prob)
    var_term = np.sqrt(
        1.0 - skewness * observed_sharpe
        + ((kurtosis_) / 4.0) * observed_sharpe ** 2
    )
    years = ((z * var_term) / (observed_sharpe - _expected_max_sharpe(num_trials))) ** 2
    return max(1, int(np.ceil(years)))
