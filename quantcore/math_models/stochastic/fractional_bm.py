"""Fractional Brownian Motion: simulation and Hurst exponent estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky


def simulate_fbm_cholesky(
    H: float,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate fBM via Cholesky decomposition of the covariance matrix.

    Cov(W^H_s, W^H_t) = 0.5 * (s^(2H) + t^(2H) - |s-t|^(2H))
    Returns shape (n_paths, n_steps+1) on [0, T].
    """
    T / n_steps
    times = np.linspace(0, T, n_steps + 1)[1:]  # Exclude 0

    n = len(times)
    # Build covariance matrix
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s, t = times[i], times[j]
            cov[i, j] = 0.5 * (s ** (2 * H) + t ** (2 * H) - abs(s - t) ** (2 * H))

    # Add small diagonal for numerical stability
    cov += 1e-12 * np.eye(n)
    L = cholesky(cov, lower=True)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n))
    increments = (z @ L.T)  # shape (n_paths, n)

    # Build paths with B^H(0) = 0
    fbm = np.zeros((n_paths, n_steps + 1))
    fbm[:, 1:] = increments

    return fbm


def simulate_fbm_hosking(
    H: float,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Simulate fBM via Hosking method (sequential conditional sampling).

    More memory-efficient than Cholesky for large n_steps.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    gamma = _autocovariance_fgn(H, n_steps)

    fbm = np.zeros((n_paths, n_steps + 1))

    for p in range(n_paths):
        fgn = _hosking_sample(gamma, n_steps, rng)
        # FGN increments integrated give fBM
        fbm[p, 1:] = np.cumsum(fgn) * dt**H

    return fbm


def _autocovariance_fgn(H: float, n: int) -> NDArray:
    """Autocovariance function of fractional Gaussian noise."""
    gamma = np.zeros(n)
    for k in range(n):
        if k == 0:
            gamma[k] = 1.0
        else:
            gamma[k] = 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))
    return gamma


def _hosking_sample(gamma: NDArray, n: int, rng: np.random.Generator) -> NDArray:
    """Generate FGN via Hosking's method."""
    fgn = np.zeros(n)
    fgn[0] = rng.standard_normal()

    # Initialize
    sigma2 = gamma[0]
    phi = np.array([gamma[1] / gamma[0]])
    mu = phi[0] * fgn[0]

    fgn[1] = mu + np.sqrt(sigma2 - phi[0]**2 * gamma[0]) * rng.standard_normal()

    for t in range(2, n):
        # Update phi using Durbin-Levinson
        r = gamma[1:t]  # autocorrelations
        new_phi_num = gamma[t] - np.dot(phi, r[::-1])
        new_phi_den = 1.0 - np.dot(phi, r)
        new_phi = new_phi_num / new_phi_den if abs(new_phi_den) > 1e-12 else 0.0
        phi = np.append(phi - new_phi * phi[::-1], new_phi)

        mu = np.dot(phi, fgn[t-1:0:-1])
        var = gamma[0] - np.dot(phi, gamma[1:t+1][::-1][:len(phi)])
        var = max(var, 1e-12)
        fgn[t] = mu + np.sqrt(var) * rng.standard_normal()

    return fgn


def estimate_hurst_rs(series: NDArray) -> float:
    """Estimate Hurst exponent via R/S (rescaled range) analysis."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    max_k = int(np.log2(n))

    rs_values = []
    sizes = []

    for k in range(2, max_k + 1):
        size = n // (2 ** k)
        if size < 8:
            continue
        n_sub = n // size
        rs_sub = []
        for i in range(n_sub):
            sub = series[i * size:(i + 1) * size]
            mean_sub = np.mean(sub)
            deviations = np.cumsum(sub - mean_sub)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_sub.append(R / S)
        if rs_sub:
            rs_values.append(np.log(np.mean(rs_sub)))
            sizes.append(np.log(size))

    if len(sizes) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(sizes, rs_values, 1)
    return float(np.clip(coeffs[0], 0.01, 0.99))


def estimate_hurst_dfa(series: NDArray) -> float:
    """Estimate Hurst exponent via Detrended Fluctuation Analysis (DFA)."""
    series = np.asarray(series, dtype=float)
    n = len(series)

    # Cumulative sum
    cumsum = np.cumsum(series - np.mean(series))

    # Segment sizes (log-spaced)
    min_size = max(8, n // 50)
    max_size = n // 4
    sizes = np.unique(np.logspace(np.log10(min_size), np.log10(max_size), 20).astype(int))

    fluctuations = []
    for size in sizes:
        n_segments = n // size
        if n_segments < 2:
            continue
        f_vals = []
        for i in range(n_segments):
            segment = cumsum[i * size:(i + 1) * size]
            x = np.arange(size)
            # Linear detrending
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            f_vals.append(np.sqrt(np.mean(detrended**2)))
        fluctuations.append(np.mean(f_vals))

    if len(sizes) < 3 or len(fluctuations) < 3:
        return 0.5

    # Regression: log(F) = H * log(size) + c
    valid = np.array(fluctuations) > 0
    if np.sum(valid) < 2:
        return 0.5
    log_sizes = np.log(sizes[:len(fluctuations)][valid])
    log_fluct = np.log(np.array(fluctuations)[valid])
    coeffs = np.polyfit(log_sizes, log_fluct, 1)
    return float(np.clip(coeffs[0], 0.01, 0.99))
