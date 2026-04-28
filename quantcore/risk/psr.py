"""Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

Based on:
- Bailey & Lopez de Prado (2012): "The Sharpe Ratio Efficient Frontier"
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"

PSR answers: "Given the skewness and kurtosis of returns, what is the probability
that the true Sharpe ratio exceeds a benchmark?"

DSR extends PSR to correct for multiple testing (data snooping). When N strategies
are tested, the best one's Sharpe is biased upward — DSR deflates the benchmark
to account for selection bias.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from quantcore.risk.metrics import sharpe_ratio

EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


def probabilistic_sharpe_ratio(
    returns: NDArray[np.floating],
    benchmark_sr: float = 0.0,
    annualize: bool = True,
) -> float:
    """Compute the Probabilistic Sharpe Ratio.

    PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - γ3*SR_hat + ((γ4-1)/4)*SR_hat²) )

    Where SR_hat is the annualized Sharpe and SR* is the benchmark Sharpe.

    Args:
        returns: Array of periodic returns.
        benchmark_sr: Benchmark Sharpe ratio to test against (default 0).
        annualize: If True, SR_hat is annualized and benchmark_sr is the annual target.

    Returns:
        Probability (0 to 1) that the true Sharpe > benchmark_sr.
    """
    r = np.asarray(returns, dtype=float).flatten()
    T = len(r)

    if T < 3:
        return 0.5  # Not enough data — no information

    sr_hat = sharpe_ratio(r, annualize=annualize)

    # If returns are constant (or near-constant), Sharpe is undefined
    if np.std(r, ddof=1) < 1e-15:
        mean_r = float(np.mean(r))
        return 1.0 if mean_r > 0 else (0.0 if mean_r < 0 else 0.5)

    # Skewness and excess kurtosis
    skew = float(sp_stats.skew(r))
    ex_kurt = float(sp_stats.kurtosis(r))  # excess kurtosis (normal = 0)

    # PSR numerator
    numerator = (sr_hat - benchmark_sr) * math.sqrt(T - 1)

    # PSR denominator: sqrt(1 - γ3*SR + (γ4-1)/4 * SR²)
    denom = math.sqrt(max(0.0, 1.0 - skew * sr_hat + ((ex_kurt - 1.0) / 4.0) * sr_hat**2))

    if denom <= 0:
        return 1.0 if numerator > 0 else 0.0

    psr_stat = numerator / denom
    return float(sp_stats.norm.cdf(psr_stat))


def min_track_record_length(
    sr_hat: float,
    benchmark_sr: float = 0.0,
    target_probability: float = 0.95,
    skew: float = 0.0,
    ex_kurt: float = 0.0,
) -> int:
    """Minimum number of observations needed to achieve target PSR confidence.

    T* = 1 + (Z_alpha²) * (1 - γ3*SR + (γ4-1)/4 * SR²) / (SR - SR*)²

    Args:
        sr_hat: Estimated Sharpe ratio (annualized).
        benchmark_sr: Benchmark Sharpe to beat.
        target_probability: Desired confidence level (default 0.95).
        skew: Skewness of returns (default 0 = normal).
        ex_kurt: Excess kurtosis of returns (default 0 = normal).

    Returns:
        Minimum number of observations required.
    """
    if sr_hat <= benchmark_sr:
        return 2**31 - 1  # Effectively infinite

    z_alpha = float(sp_stats.norm.ppf(target_probability))
    numerator = z_alpha**2 * (1.0 - skew * sr_hat + (ex_kurt - 1.0) / 4.0 * sr_hat**2)
    denominator = (sr_hat - benchmark_sr) ** 2

    if denominator <= 0:
        return 2**31 - 1

    return max(2, int(1 + math.ceil(numerator / denominator)))


def deflated_sharpe_ratio(
    all_trial_sharpes: list[float],
    best_sharpe: float,
    best_returns: NDArray[np.floating],
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    The DSR adjusts the Probabilistic Sharpe Ratio to account for multiple testing.
    When N trials are performed, the maximum Sharpe is inflated by selection bias.

    DSR = PSR(SR*_deflated)

    Where SR*_deflated accounts for the number of trials.

    Args:
        all_trial_sharpes: List of Sharpe ratios from all trials/strategies tested.
        best_sharpe: The maximum Sharpe ratio (the one selected).
        best_returns: Returns of the best-performing strategy (for PSR computation).

    Returns:
        DSR value (probability 0 to 1) — the probability that the selected strategy's
        true Sharpe exceeds what's expected from random search.
    """
    sharpes = np.asarray(all_trial_sharpes, dtype=float).flatten()
    N = len(sharpes)

    if N < 2:
        return probabilistic_sharpe_ratio(best_returns, benchmark_sr=0.0)

    # Variance of Sharpe estimates across trials
    var_sr = float(np.var(sharpes, ddof=1))

    if var_sr <= 0:
        return probabilistic_sharpe_ratio(best_returns, benchmark_sr=0.0)

    # Expected maximum Sharpe from N independent trials:
    # SR*_deflated = sqrt(Var(SR)) * ((1-γ)*Z^(-1)(1 - 1/N) + γ*Z^(-1)(1 - 1/(N*e)))
    std_sr = math.sqrt(var_sr)
    z1 = float(sp_stats.norm.ppf(1.0 - 1.0 / N))
    z2 = float(sp_stats.norm.ppf(1.0 - 1.0 / (N * math.e)))

    sr_deflated = std_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    return probabilistic_sharpe_ratio(best_returns, benchmark_sr=sr_deflated)


def expected_max_sharpe(
    n_trials: int,
    sharpe_mean: float = 0.0,
    sharpe_std: float = 1.0,
) -> float:
    """Expected maximum Sharpe ratio from N independent trials under the null.

    E[max(SR)] ≈ mean(SR) + std(SR) * ((1-γ)*Z^(-1)(1 - 1/N) + γ*Z^(-1)(1 - 1/(N*e)))

    Args:
        n_trials: Number of independent trials.
        sharpe_mean: Mean of Sharpe distribution under null (default 0).
        sharpe_std: Standard deviation of Sharpe distribution (default 1).

    Returns:
        Expected maximum Sharpe from N trials.
    """
    if n_trials < 2:
        return sharpe_mean

    # Clamp to avoid numerical issues with very large N
    n = min(n_trials, 10_000_000)
    z1 = float(sp_stats.norm.ppf(1.0 - 1.0 / n))
    z2 = float(sp_stats.norm.ppf(1.0 - 1.0 / (n * math.e)))

    correction = (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
    return sharpe_mean + sharpe_std * correction


def selection_bias_estimate(
    n_trials: int,
    sharpe_std: float,
) -> float:
    """Estimate the expected selection bias from testing N strategies.

    Bias ≈ E[max(SR)] - E[SR] when null is true.

    Args:
        n_trials: Number of strategy configurations tested.
        sharpe_std: Standard deviation of Sharpe estimates across trials.

    Returns:
        Expected upward bias in the selected Sharpe.
    """
    return expected_max_sharpe(n_trials, 0.0, sharpe_std)


# ---------------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------------


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Bonferroni correction for multiple hypothesis testing.

    Controls family-wise error rate (FWER) by dividing alpha by number of tests.

    Args:
        p_values: List of p-values from each test.
        alpha: Desired family-wise error rate (default 0.05).

    Returns:
        List of booleans: True if the null is rejected at the corrected threshold.
    """
    n = max(len(p_values), 1)
    threshold = alpha / n
    return [p <= threshold for p in p_values]


def benjamini_hochberg_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Benjamini-Hochberg procedure for false discovery rate control.

    Less conservative than Bonferroni. Controls the expected proportion of
    false rejections among all rejections.

    Args:
        p_values: List of p-values from each test.
        alpha: Desired false discovery rate (default 0.05).

    Returns:
        List of booleans: True if the null is rejected at the corrected threshold.
    """
    n = len(p_values)
    if n == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=float)[sorted_indices]

    thresholds = np.arange(1, n + 1) * alpha / n
    reject = sorted_p <= thresholds

    # Find the largest k where p_k <= k/n * alpha
    # All tests with p <= p_k are rejected
    cumulative = np.maximum.accumulate(reject)
    result = np.zeros(n, dtype=bool)
    result[sorted_indices] = cumulative
    return result.tolist()


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Holm-Bonferroni step-down procedure.

    More powerful than standard Bonferroni while still controlling FWER.
    Sorts p-values ascending and rejects if p_k <= alpha / (n - k + 1).

    Args:
        p_values: List of p-values from each test.
        alpha: Desired family-wise error rate (default 0.05).

    Returns:
        List of booleans: True if the null is rejected at the corrected threshold.
    """
    n = len(p_values)
    if n == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=float)[sorted_indices]

    result = np.zeros(n, dtype=bool)
    for i, idx in enumerate(sorted_indices):
        threshold = alpha / (n - i)
        if sorted_p[i] <= threshold:
            result[idx] = True
        else:
            break  # Remaining all fail to reject

    return result.tolist()
