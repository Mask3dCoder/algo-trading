"""Sequential Monte Carlo (Particle Filter) for nonlinear state-space models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ParticleFilter:
    """Particle filter with systematic resampling."""

    def __init__(
        self,
        n_particles: int = 1000,
        state_dim: int = 1,
        seed: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.particles: NDArray = np.zeros((n_particles, state_dim))
        self.weights: NDArray = np.ones(n_particles) / n_particles

    def init_gaussian(
        self,
        mean: float | NDArray = 0.0,
        std: float | NDArray = 1.0,
    ) -> None:
        """Initialize particles from a Gaussian distribution."""
        self.particles = self.rng.normal(mean, std, (self.n_particles, self.state_dim))
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, transition_fn) -> None:
        """Propagate particles through the state transition model.

        transition_fn: callable(particles, rng) -> new_particles
        """
        self.particles = transition_fn(self.particles, self.rng)

    def update(self, observation: float | NDArray, likelihood_fn) -> None:
        """Update particle weights using the observation likelihood.

        likelihood_fn: callable(particles, observation) -> likelihoods (n_particles,)
        """
        likelihoods = likelihood_fn(self.particles, observation)
        self.weights *= likelihoods
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Resample if ESS below threshold
        if self.effective_sample_size < self.n_particles / 2:
            self.resample()

    @property
    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2)."""
        return float(1.0 / np.sum(self.weights**2))

    @property
    def state_estimate(self) -> NDArray:
        """Weighted mean of particles."""
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def state_covariance(self) -> NDArray:
        """Weighted covariance of particles."""
        mean = self.state_estimate
        diff = self.particles - mean
        return np.average([np.outer(d, d) for d in diff], weights=self.weights, axis=0)

    def resample(self) -> None:
        """Systematic resampling (low variance)."""
        n = self.n_particles
        positions = (self.rng.uniform(0, 1) + np.arange(n)) / n
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # Ensure exact sum

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(n) / n

    def filter(
        self,
        observations: NDArray,
        transition_fn,
        likelihood_fn,
    ) -> NDArray:
        """Run the full particle filter on a sequence of observations.

        Returns array of state estimates (len(observations), state_dim).
        """
        estimates = []
        for obs in observations:
            self.predict(transition_fn)
            self.update(obs, likelihood_fn)
            estimates.append(self.state_estimate.copy())
        return np.array(estimates)


def online_volatility(
    returns: NDArray,
    n_particles: int = 1000,
    seed: int = 42,
) -> NDArray:
    """Online volatility estimation using particle filter.

    Stochastic volatility model:
      h_t = mu + phi*(h_{t-1} - mu) + sigma_eta * eta_t
      r_t = exp(h_t/2) * epsilon_t
    """
    r = np.asarray(returns, dtype=float)

    # Hyperparameters
    mu = 0.0
    phi = 0.95
    sigma_eta = 0.2

    pf = ParticleFilter(n_particles=n_particles, state_dim=1, seed=seed)
    pf.init_gaussian(mean=np.log(np.var(r)), std=1.0)

    def transition(particles: NDArray, rng: np.random.Generator) -> NDArray:
        noise = rng.normal(0, sigma_eta, (len(particles), 1))
        return mu + phi * (particles - mu) + noise

    def likelihood(particles: NDArray, obs: float) -> NDArray:
        h = particles[:, 0]
        vol = np.exp(h / 2.0)
        return np.exp(-0.5 * (obs / vol)**2) / (vol * np.sqrt(2 * np.pi) + 1e-10)

    return pf.filter(r, transition, likelihood)
