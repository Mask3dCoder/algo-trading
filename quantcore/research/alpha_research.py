"""Alpha research tools: deflated Sharpe, multiple testing corrections, overfitting detection.

Implements the Bailey & Lopez de Prado family of statistical tests for strategy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class DeflatedSharpeResult:
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    significant: bool
    haircut_pct: float
    min_backtest_length: int
    recommended_trials: int


def deflated_sharpe(
    returns: NDArray[np.floating],
    num_trials: int = 100,
    skewness: float | None = None,
    kurtosis: float | None = None,
    variance: float | None = None,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Accounts for multiple testing by estimating E[max(Sharpe)] under the null.
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)
    if n < 10:
        return DeflatedSharpeResult(
            observed_sharpe=0.0, deflated_sharpe=0.0, p_value=1.0,
            significant=False, haircut_pct=0.0, min_backtest_length=0, recommended_trials=0,
        )

    sr_obs = float(np.mean(r) / np.std(r, ddof=1)) if np.std(r, ddof=1) > 0 else 0.0

    if sr_obs == 0.0:
        return DeflatedSharpeResult(
            observed_sharpe=0.0, deflated_sharpe=0.0, p_value=1.0,
            significant=False, haircut_pct=0.0,
            min_backtest_length=_min_backtest_length(0.0, num_trials, variance, 0.0, 0.0),
            recommended_trials=num_trials,
        )

    if skewness is None:
        skewness = float(stats.skew(r))
    if kurtosis is None:
        kurtosis = float(stats.kurtosis(r, fisher=True))
    if variance is None:
        variance = float(np.var(r, ddof=1))

    # Expected maximum Sharpe under multiple testing
    euler_mascheroni = 0.5772156649
    if num_trials <= 1:
        expected_max = 0.0
    else:
        log_trials = np.log(num_trials)
        expected_max = np.sqrt(2 * log_trials)
        expected_max = expected_max - (np.log(log_trials) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_trials))
        expected_max = expected_max + euler_mascheroni / np.sqrt(2 * log_trials)

    # Deflated Sharpe
    numerator = (sr_obs - expected_max) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skewness * sr_obs + (kurtosis + 2) / 4 * sr_obs**2)
    if denominator <= 0:
        return DeflatedSharpeResult(
            observed_sharpe=sr_obs, deflated_sharpe=0.0, p_value=1.0,
            significant=False, haircut_pct=0.0,
            min_backtest_length=_min_backtest_length(sr_obs, num_trials, variance, skewness, kurtosis),
            recommended_trials=num_trials,
        )
    dsr = numerator / denominator

    p_value = 1.0 - stats.norm.cdf(dsr)
    significant = bool(p_value < 0.05)
    haircut_pct = max(0.0, 1.0 - dsr / sr_obs) if sr_obs > 0 else 0.0

    return DeflatedSharpeResult(
        observed_sharpe=float(sr_obs),
        deflated_sharpe=float(dsr),
        p_value=float(p_value),
        significant=significant,
        haircut_pct=float(haircut_pct),
        min_backtest_length=_min_backtest_length(sr_obs, num_trials, variance, skewness, kurtosis),
        recommended_trials=num_trials,
    )


def minimum_backtest_length(
    observed_sharpe: float,
    num_trials: int,
    variance: float | None = None,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> int:
    """Minimum backtest length (in years) for statistical significance."""
    return _min_backtest_length(observed_sharpe, num_trials, variance, skewness, kurtosis)


def _min_backtest_length(
    sr: float, trials: int, var: float | None, skew: float | None, kurt: float | None,
) -> int:
    if sr <= 0:
        return 999
    z_crit = stats.norm.ppf(1.0 - 0.05 / max(trials, 1))
    adj = 1.0
    if skew is not None and kurt is not None:
        adj = np.sqrt(1 - (skew or 0) * sr + ((kurt or 0) + 2) / 4 * sr**2)
    years = (z_crit * adj / sr) ** 2
    return max(1, int(np.ceil(years)))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    num_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark SR)."""
    if num_observations < 2:
        return 0.0
    denominator = np.sqrt(
        (1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
        / (num_observations - 1)
    )
    if denominator <= 0:
        return 0.0
    psr = (observed_sharpe - benchmark_sharpe) / denominator
    return float(stats.norm.cdf(psr))


def fdr_correction(
    p_values: NDArray[np.floating], method: str = "bh",
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """FDR correction: Benjamini-Hochberg or Benjamini-Yekutieli."""
    p = np.asarray(p_values, dtype=float).flatten()
    n = len(p)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    order = np.argsort(p)
    p_sorted = p[order]
    if method == "by":
        c = sum(1.0 / k for k in range(1, n + 1))
        thresholds = np.arange(1, n + 1) * 0.05 / (n * c)
    else:
        thresholds = np.arange(1, n + 1) * 0.05 / n
    rejected = np.zeros(n, dtype=bool)
    last_reject = -1
    for i in range(n - 1, -1, -1):
        if p_sorted[i] <= thresholds[i]:
            last_reject = i
            break
    if last_reject >= 0:
        rejected[order[:last_reject + 1]] = True
    adjusted = np.minimum(1.0, p_sorted * n / np.arange(1, n + 1))
    adj_in_order = np.zeros(n, dtype=float)
    adj_in_order[order] = adjusted
    return adj_in_order, rejected


def holm_correction(
    p_values: NDArray[np.floating], alpha: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Holm-Bonferroni step-down correction."""
    p = np.asarray(p_values, dtype=float).flatten()
    n = len(p)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    order = np.argsort(p)
    rejected = np.zeros(n, dtype=bool)
    thresholds = alpha / np.arange(n, 0, -1)
    for k, i in enumerate(order):
        if p[i] <= thresholds[k]:
            rejected[i] = True
        else:
            break
    adjusted = np.minimum(1.0, p * np.arange(n, 0, -1))
    return adjusted, rejected


def multiple_testing_summary(
    p_values: NDArray[np.floating],
    strategy_names: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize multiple testing results across correction methods."""
    p = np.asarray(p_values, dtype=float).flatten()
    n = len(p)
    bonf_rejected = p <= alpha / max(n, 1)
    _, holm_rejected = holm_correction(p, alpha)
    _, bh_rejected = fdr_correction(p, "bh")

    def _names(mask: NDArray[np.bool_]) -> list[str | int]:
        idx = np.where(mask)[0]
        if strategy_names and len(strategy_names) >= n:
            return [strategy_names[i] for i in idx]
        return idx.tolist()

    return {
        "n_tests": n,
        "bonferroni_significant": int(np.sum(bonf_rejected)),
        "holm_significant": int(np.sum(holm_rejected)),
        "bh_significant": int(np.sum(bh_rejected)),
        "bonferroni_names": _names(bonf_rejected),
        "holm_names": _names(holm_rejected),
        "bh_names": _names(bh_rejected),
    }


def haircut_sharpe(
    annualized_sharpe: float, num_trials: int, num_observations: int,
) -> float:
    """Simple empirical Sharpe haircut based on number of trials.

    The more strategies you try, the more you haircut the observed Sharpe.
    """
    penalty = np.sqrt(np.log(max(num_trials, 1)) / max(num_observations, 1))
    return max(0.0, annualized_sharpe - penalty)


def overfitting_probability(returns_cv: list[NDArray[np.floating]]) -> float:
    """CSCV (Combinatorially Symmetric Cross-Validation) overfitting probability.

    Args:
        returns_cv: List of return arrays, one per strategy (or one per CV fold).

    Returns:
        Probability of overfitting, where 0 = no overfitting, 1 = severe.
    """
    if len(returns_cv) < 2:
        return 0.0
    n = len(returns_cv)
    sr_values = np.zeros(n, dtype=float)
    for i, r in enumerate(returns_cv):
        r_f = np.asarray(r, dtype=float).flatten()
        if len(r_f) < 3:
            sr_values[i] = 0.0
            continue
        mu = np.mean(r_f)
        sig = np.std(r_f, ddof=1)
        sr_values[i] = mu / sig if sig > 0 else 0.0

    best_is = np.argmax(sr_values)

    # Split data in half, compare rank stability
    n_obs = min(len(r) for r in returns_cv)
    half = n_obs // 2
    if half < 5:
        return 0.0

    sr_is = np.zeros(n, dtype=float)
    sr_oos = np.zeros(n, dtype=float)
    for i, r in enumerate(returns_cv):
        r_f = np.asarray(r, dtype=float).flatten()
        mu_is = np.mean(r_f[:half])
        sig_is = np.std(r_f[:half], ddof=1)
        sr_is[i] = mu_is / sig_is if sig_is > 0 else 0.0
        mu_oos = np.mean(r_f[half:])
        sig_oos = np.std(r_f[half:], ddof=1)
        sr_oos[i] = mu_oos / sig_oos if sig_oos > 0 else 0.0

    best_oos = np.argmax(sr_oos)
    if best_is == best_oos:
        return 0.0

    # Compute rank correlation
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(sr_is, sr_oos)
    if np.isnan(rank_corr):
        return 0.5
    return float(1.0 - max(0.0, rank_corr))
