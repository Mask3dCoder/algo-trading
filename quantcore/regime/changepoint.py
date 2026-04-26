"""Changepoint detection: PELT, BOCPD, CUSUM, SPRT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


@dataclass
class ChangepointResult:
    algorithm: str
    changepoints: list[int]
    n_changepoints: int
    probabilities: NDArray[np.floating] | None = None
    run_length: NDArray[np.floating] | None = None


# ---------------------------------------------------------------------------
# PELT (Pruned Exact Linear Time)
# ---------------------------------------------------------------------------


def pelt_changepoints(
    series: NDArray[np.floating],
    penalty: float = 10.0,
    model: str = "l2",
) -> ChangepointResult:
    """PELT algorithm via the ruptures library.

    Args:
        series: Time series data.
        penalty: Complexity penalty (higher = fewer changepoints).
        model: Cost function: "l2" (mean shift), "l1", "rbf", "normal".

    Returns:
        ChangepointResult with detected changepoint indices.
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("ruptures not installed. Run: pip install ruptures")

    signal = np.asarray(series, dtype=float)
    if signal.ndim > 1:
        signal = signal.flatten()

    if model == "l2":
        algo = rpt.Pelt(model="l2", min_size=2, jump=1).fit(signal)
    elif model == "l1":
        algo = rpt.Pelt(model="l1", min_size=2, jump=1).fit(signal)
    elif model == "rbf":
        algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal)
    else:
        algo = rpt.Pelt(model="normal", min_size=2, jump=1).fit(signal)

    result = algo.predict(pen=penalty)
    # ruptures returns end indices; convert to start of new regime
    cps = [int(r) for r in result[:-1]]  # Last point is always n

    return ChangepointResult(
        algorithm="pelt",
        changepoints=cps,
        n_changepoints=len(cps),
    )


# ---------------------------------------------------------------------------
# Bayesian Online Changepoint Detection (BOCPD)
# ---------------------------------------------------------------------------


def bocpd(
    series: NDArray[np.floating],
    hazard_rate: float = 0.01,
    max_run_length: int | None = None,
) -> ChangepointResult:
    """Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Maintains a belief distribution over run lengths. Uses a Gaussian
    model with conjugate Normal-Inverse-Gamma prior.

    Args:
        series: Time series data.
        hazard_rate: Constant hazard rate (probability of changepoint at each step).
        max_run_length: Maximum run length to track.

    Returns:
        ChangepointResult with run-length distribution.
    """
    y = np.asarray(series, dtype=float).flatten()
    n = len(y)

    if max_run_length is None:
        max_run_length = min(n, 500)

    maxrl = min(max_run_length, n)

    # Run-length posterior
    rl_posterior = np.zeros((n, maxrl))

    # Initialize: run_length=0 with probability 1 at t=0
    rl_posterior[0, 0] = 1.0

    # Prior hyperparameters
    mu_prior = np.mean(y)
    kappa_prior = 1.0
    alpha_prior = 3.0
    beta_prior = np.var(y) if np.var(y) > 0 else 1.0

    changepoints = []

    for t in range(1, n):
        x_t = y[t]

        # Prediction for each possible run length
        pred_probs = np.zeros(maxrl)

        for r in range(min(t, maxrl)):
            # Conjugate update parameters for run length r
            # Using all data since last changepoint (at t - r)
            data_since_cp = y[t - r:t]

            n_r = len(data_since_cp)
            xbar_r = np.mean(data_since_cp)

            # Posterior parameters
            kappa_n = kappa_prior + n_r
            mu_n = (kappa_prior * mu_prior + n_r * xbar_r) / kappa_n
            alpha_n = alpha_prior + n_r / 2.0
            ss = np.sum((data_since_cp - xbar_r)**2)
            beta_n = beta_prior + 0.5 * ss + (kappa_prior * n_r * (xbar_r - mu_prior)**2) / (2.0 * kappa_n)

            # Predictive probability (Student-t)
            scale = beta_n * (kappa_n + 1) / (kappa_n * alpha_n)
            try:
                pred_probs[r] = sp_stats.t.pdf(x_t, df=2 * alpha_n, loc=mu_n, scale=np.sqrt(scale))
            except Exception:
                pred_probs[r] = 1e-10

        # Growth probabilities
        growth_probs = np.zeros(maxrl)
        growth_probs[0] = hazard_rate  # Changepoint probability
        for r in range(1, maxrl):
            growth_probs[r] = rl_posterior[t - 1, r - 1] * pred_probs[r - 1] * (1.0 - hazard_rate)

        growth_probs[0] *= np.sum(rl_posterior[t - 1, :])

        # Normalize
        total = np.sum(growth_probs)
        if total > 0:
            rl_posterior[t, :] = growth_probs / total

        # Detect changepoint
        if growth_probs[0] / total > 0.5:  # More than 50% chance of reset
            changepoints.append(t)

    return ChangepointResult(
        algorithm="bocpd",
        changepoints=changepoints,
        n_changepoints=len(changepoints),
        run_length=rl_posterior[:, 0],  # Probability of immediate changepoint
    )


# ---------------------------------------------------------------------------
# CUSUM control chart
# ---------------------------------------------------------------------------


def cusum(
    series: NDArray[np.floating],
    threshold: float = 5.0,
    drift: float = 1.0,
    reset_after: bool = False,
) -> ChangepointResult:
    """CUSUM (Cumulative Sum) control chart for changepoint detection.

    Detects upward (or downward) shifts in the mean.

    Args:
        series: Time series data.
        threshold: Decision threshold.
        drift: Allowable drift before resetting (smaller = more sensitive).
        reset_after: Reset CUSUM after each detection.

    Returns:
        ChangepointResult with changepoint indices.
    """
    y = np.asarray(series, dtype=float).flatten()
    n = len(y)

    if n < 10:
        return ChangepointResult("cusum", [], 0)

    # Estimate in-control mean and std
    warmup = min(30, n // 3)
    mu0 = float(np.mean(y[:warmup]))
    sigma = float(np.std(y[:warmup], ddof=1)) or 1.0

    # Standardize
    z = (y - mu0) / sigma

    # CUSUM statistics (upper/lower)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    changepoints = []

    for t in range(1, n):
        s_pos[t] = max(0, s_pos[t - 1] + z[t] - drift / 2)
        s_neg[t] = max(0, s_neg[t - 1] - z[t] - drift / 2)

        if s_pos[t] > threshold:
            changepoints.append(t)
            if reset_after:
                s_pos[t] = 0
        if s_neg[t] > threshold:
            changepoints.append(t)
            if reset_after:
                s_neg[t] = 0

    return ChangepointResult(
        algorithm="cusum",
        changepoints=changepoints,
        n_changepoints=len(changepoints),
    )


# ---------------------------------------------------------------------------
# Sequential Probability Ratio Test (SPRT)
# ---------------------------------------------------------------------------


def sprt_test(
    series: NDArray[np.floating],
    h0_mean: float = 0.0,
    h1_mean: float = 1.0,
    alpha: float = 0.05,
    beta: float = 0.10,
) -> ChangepointResult:
    """Sequential Probability Ratio Test for mean shift detection.

    Tests H0: mean = h0_mean vs H1: mean = h1_mean.

    Args:
        series: Time series data.
        h0_mean: Null hypothesis mean.
        h1_mean: Alternative hypothesis mean.
        alpha: Type-I error rate.
        beta: Type-II error rate.

    Returns:
        ChangepointResult with detection indices.
    """
    y = np.asarray(series, dtype=float).flatten()
    n = len(y)

    if n < 2:
        return ChangepointResult("sprt", [], 0)

    sigma = float(np.std(y, ddof=1)) or 1.0

    A = (1.0 - beta) / alpha
    B = beta / (1.0 - alpha)

    lr = np.zeros(n)
    changepoints = []

    for t in range(n):
        # Log-likelihood ratio
        lh1 = sp_stats.norm.logpdf(y[t], loc=h1_mean, scale=sigma)
        lh0 = sp_stats.norm.logpdf(y[t], loc=h0_mean, scale=sigma)
        lr[t] = lr[t - 1] + (lh1 - lh0) if t > 0 else lh1 - lh0

        if lr[t] > np.log(A):
            changepoints.append(t)
            lr[t:] = 0  # Reset
        elif lr[t] < np.log(B):
            lr[t:] = 0  # Reset (accept H0)

    return ChangepointResult(
        algorithm="sprt",
        changepoints=changepoints,
        n_changepoints=len(changepoints),
    )
