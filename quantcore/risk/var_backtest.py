"""VaR backtesting: Kupiec POF, Christoffersen conditional coverage, Basel traffic light."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


@dataclass
class VaRBacktestReport:
    method: str
    n_observations: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    traffic_light: str  # green, yellow, red
    violation_dates: list[str] = field(default_factory=list)
    violation_returns: list[float] = field(default_factory=list)
    avg_violation_loss: float = 0.0


# ---------------------------------------------------------------------------
# Kupiec POF (Proportion of Failures) test
# ---------------------------------------------------------------------------


def kupiec_pof_test(
    violations: int | NDArray[np.integer],
    n_obs: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Kupiec Proportion of Failures (unconditional coverage) test.

    H0: violation rate = 1 - confidence.
    Test statistic: LR = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)]

    Returns:
        (lr_stat, p_value): if p_value < 0.05, reject H0 (model is miscalibrated).
    """
    x = int(np.sum(violations)) if hasattr(violations, '__iter__') else int(violations)
    n = n_obs
    p = 1.0 - confidence

    if x == 0 or x == n:
        # LR undefined when x=0 or x=n; use approximation
        observed_rate = max(x / n, 1.0 / n)
    else:
        observed_rate = x / n
    # Clamp away from boundaries to prevent log(0)
    observed_rate = max(min(observed_rate, 1.0 - 1e-12), 1e-12)

    # Log-likelihood under H0 vs H1
    ll_h0 = (n - x) * np.log(1.0 - p) + x * np.log(p) if x > 0 else (n - x) * np.log(1.0 - p)
    ll_h1 = (n - x) * np.log(1.0 - observed_rate) + x * np.log(observed_rate) if x > 0 else (n - x) * np.log(1.0 - observed_rate)

    lr_stat = -2.0 * (ll_h0 - ll_h1)
    # Kuiper's correction for boundary cases
    lr_stat = max(0.0, lr_stat)
    p_value = 1.0 - sp_stats.chi2.cdf(lr_stat, df=1)

    return float(lr_stat), float(p_value)


# ---------------------------------------------------------------------------
# Christoffersen conditional coverage (independence) test
# ---------------------------------------------------------------------------


def christoffersen_test(
    violations: NDArray[np.integer],
) -> tuple[float, float]:
    """Christoffersen conditional coverage test.

    Tests independence of violations (violations should not cluster).

    Returns:
        (lr_stat, p_value): if p_value < 0.05, reject H0 (violations are clustered).
    """
    v = np.asarray(violations, dtype=int)
    n = len(v)

    if n < 3 or np.sum(v) == 0:
        return 0.0, 1.0

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        if v[t - 1] == 0 and v[t] == 0:
            n00 += 1
        elif v[t - 1] == 0 and v[t] == 1:
            n01 += 1
        elif v[t - 1] == 1 and v[t] == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        return 0.0, 1.0

    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    pi_total = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    # Log-likelihood under H0 (violations independent)
    def safe_log(x: float) -> float:
        return float(np.log(max(x, 1e-10)))

    ll_h0 = n00 * safe_log(1.0 - pi_total) + n01 * safe_log(pi_total) + \
            n10 * safe_log(1.0 - pi_total) + n11 * safe_log(pi_total)

    # Log-likelihood under H1 (violations conditionally dependent)
    ll_h1 = n00 * safe_log(1.0 - pi0) + n01 * safe_log(pi0) + \
            n10 * safe_log(1.0 - pi1) + n11 * safe_log(pi1)

    lr_stat = -2.0 * (ll_h0 - ll_h1)
    lr_stat = max(0.0, lr_stat)
    p_value = 1.0 - sp_stats.chi2.cdf(lr_stat, df=1)

    return float(lr_stat), float(p_value)


# ---------------------------------------------------------------------------
# Basel traffic light test
# ---------------------------------------------------------------------------


def traffic_light(
    violations: int | NDArray[np.integer],
    n_obs: int,
    confidence: float = 0.99,
) -> str:
    """Basel Committee traffic light test for VaR model.

    Green:  0-4 violations  (model OK)
    Yellow: 5-9 violations  (model needs review)
    Red:    10+ violations  (model rejected)

    Based on 99% VaR over 250 days.
    """
    x = int(np.sum(violations)) if hasattr(violations, '__iter__') else int(violations)

    # Adjust thresholds for different n_obs
    scaling = n_obs / 250.0
    if x <= max(1, int(4 * scaling)):
        return "green"
    elif x <= max(2, int(9 * scaling)):
        return "yellow"
    else:
        return "red"


# ---------------------------------------------------------------------------
# Full backtest report
# ---------------------------------------------------------------------------


def var_backtest_report(
    returns: NDArray[np.floating],
    var_series: NDArray[np.floating],
    confidence: float = 0.95,
) -> VaRBacktestReport:
    """Generate a complete VaR backtest report.

    Args:
        returns: Actual portfolio returns (typically daily).
        var_series: VaR forecasts at each point in time (same length as returns).
        confidence: Confidence level used for VaR.

    Returns:
        VaRBacktestReport with all test results.
    """
    r = np.asarray(returns, dtype=float).flatten()
    v = np.asarray(var_series, dtype=float).flatten()
    n = min(len(r), len(v))

    r = r[:n]
    v = v[:n]

    # Violations: return < -VaR (VaR is typically reported as positive loss)
    violations = (r < -v).astype(int)
    # Also handle case where VaR is reported as a negative number
    violations_neg = (r < v).astype(int)
    if np.sum(violations_neg) > 0:
        violations = violations_neg

    n_violations = int(np.sum(violations))

    kupiec_stat, kupiec_pv = kupiec_pof_test(n_violations, n, confidence)
    if n_violations > 0 and n_violations < n - 1:
        chris_stat, chris_pv = christoffersen_test(violations)
    else:
        chris_stat, chris_pv = 0.0, 1.0

    tl = traffic_light(n_violations, n, confidence)

    violation_idx = np.where(violations == 1)[0]
    violation_returns = [float(r[i]) for i in violation_idx]

    return VaRBacktestReport(
        method="Mixed",
        n_observations=n,
        n_violations=n_violations,
        violation_rate=float(n_violations / n) if n > 0 else 0.0,
        expected_rate=1.0 - confidence,
        kupiec_stat=kupiec_stat,
        kupiec_pvalue=kupiec_pv,
        christoffersen_stat=chris_stat,
        christoffersen_pvalue=float(chris_pv),
        traffic_light=tl,
        violation_returns=violation_returns,
        avg_violation_loss=float(np.mean(violation_returns)) if violation_returns else 0.0,
    )
