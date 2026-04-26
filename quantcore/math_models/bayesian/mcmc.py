"""MCMC samplers: Metropolis-Hastings, convergence diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler."""

    def __init__(
        self,
        log_posterior,
        proposal_std: float | NDArray = 1.0,
        seed: int = 42,
    ) -> None:
        self.log_posterior = log_posterior
        self.proposal_std = np.atleast_1d(proposal_std)
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        init: NDArray,
        n_samples: int = 10_000,
        burn_in: int = 1000,
    ) -> NDArray:
        """Run MH sampler.

        Returns array of shape (n_samples - burn_in, dim).
        """
        dim = len(init)
        chain = np.zeros((n_samples, dim))
        current = np.array(init, dtype=float)
        std = np.resize(self.proposal_std, dim)
        current_lp = self.log_posterior(current)

        for i in range(n_samples):
            # Propose
            proposal = current + self.rng.normal(0, std, dim)
            proposal_lp = self.log_posterior(proposal)

            # Accept/reject
            log_alpha = proposal_lp - current_lp
            if np.log(self.rng.uniform()) < log_alpha:
                current = proposal
                current_lp = proposal_lp

            chain[i] = current

        return chain[burn_in:]


def gelman_rubin(chains: list[NDArray]) -> float:
    """Gelman-Rubin R-hat statistic for convergence diagnosis.

    R-hat < 1.1 indicates convergence.
    chains: list of M arrays each of shape (N, dim) — M independent chains.
    Returns scalar R-hat (averaged across dimensions if dim > 1).
    """
    M = len(chains)
    N = chains[0].shape[0]

    # Per-chain means and overall mean
    chain_means = np.array([np.mean(c, axis=0) for c in chains])
    overall_mean = np.mean(chain_means, axis=0)

    # Between-chain variance
    B = N / (M - 1) * np.sum((chain_means - overall_mean)**2, axis=0)

    # Within-chain variance
    W = np.mean([np.var(c, axis=0, ddof=1) for c in chains], axis=0)

    # Pooled variance estimate
    var_plus = (N - 1) / N * W + B / N

    # R-hat
    rhat = np.sqrt(var_plus / W)

    return float(np.mean(rhat))


def effective_sample_size(chain: NDArray) -> float:
    """Effective sample size for a single MCMC chain.

    Uses autocorrelation-based estimation.
    """
    x = np.asarray(chain, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]  # Use first dimension

    n = len(x)
    mean = np.mean(x)
    var = np.var(x, ddof=1)

    if var < 1e-12:
        return float(n)

    # Compute autocorrelations using FFT
    x_centered = x - mean
    fft = np.fft.fft(x_centered, n=2 * n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n].real / (var * n)

    # Sum autocorrelations until first negative pair
    tau = 1.0
    for lag in range(1, n // 2):
        rho = acf[lag]
        if rho < 0.05:
            break
        tau += 2 * rho

    return float(n / tau)
