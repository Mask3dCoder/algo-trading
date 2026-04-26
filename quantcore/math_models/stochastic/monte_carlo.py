"""Monte Carlo simulation engine with variance reduction techniques."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc


@dataclass
class MCResult:
    """Monte Carlo pricing result."""
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    variance_reduction_ratio: float = 1.0


class MonteCarloEngine:
    """Configurable Monte Carlo engine with variance reduction.

    Supports:
    - Standard pseudo-random
    - Antithetic variates
    - Control variates
    - Quasi-Monte Carlo (Sobol sequences)
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int = 42,
    ) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_gbm_paths(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        method: str = "standard",
    ) -> NDArray[np.floating]:
        """Generate GBM price paths.

        Returns array of shape (n_paths, n_steps+1) where col 0 is S0.
        """
        dt = T / self.n_steps
        if method == "antithetic":
            return self._gbm_antithetic(S0, mu, sigma, dt)
        elif method == "sobol":
            return self._gbm_sobol(S0, mu, sigma, dt)
        else:
            return self._gbm_standard(S0, mu, sigma, dt)

    def _gbm_standard(
        self, S0: float, mu: float, sigma: float, dt: float,
    ) -> NDArray[np.floating]:
        z = self._rng.standard_normal((self.n_paths, self.n_steps))
        return self._paths_from_z(S0, mu, sigma, dt, z)

    def _gbm_antithetic(
        self, S0: float, mu: float, sigma: float, dt: float,
    ) -> NDArray[np.floating]:
        half = self.n_paths // 2
        z = self._rng.standard_normal((half, self.n_steps))
        z_full = np.vstack([z, -z])
        return self._paths_from_z(S0, mu, sigma, dt, z_full)

    def _gbm_sobol(
        self, S0: float, mu: float, sigma: float, dt: float,
    ) -> NDArray[np.floating]:
        sampler = qmc.Sobol(d=self.n_steps, scramble=True)
        u = sampler.random(self.n_paths)
        z = self._uniform_to_normal(u)
        return self._paths_from_z(S0, mu, sigma, dt, z)

    @staticmethod
    def _paths_from_z(
        S0: float,
        mu: float,
        sigma: float,
        dt: float,
        z: NDArray,
    ) -> NDArray[np.floating]:
        """Convert standard normal samples to GBM price paths."""
        n_paths, n_steps = z.shape
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        log_returns = drift + diffusion * z
        log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
        prices = np.exp(log_prices)
        # Prepend S0
        return np.column_stack([np.full(n_paths, S0), prices])

    @staticmethod
    def _uniform_to_normal(u: NDArray) -> NDArray:
        """Convert Sobol uniform samples to standard normal via inverse CDF."""
        # Clip to avoid infinities at exactly 0 or 1
        u_clipped = np.clip(u, 1e-10, 1.0 - 1e-10)
        from scipy.stats import norm
        return norm.ppf(u_clipped)

    def compute_price(
        self,
        payoff_fn,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        r: float | None = None,
        method: str = "standard",
        control_variate: tuple | None = None,
    ) -> MCResult:
        """Price a derivative using MC with optional variance reduction.

        payoff_fn: callable(paths) -> array of payoffs
        control_variate: tuple (cv_fn, cv_expected) for control variate method
        """
        paths = self.generate_gbm_paths(S0, mu, sigma, T, method=method)
        payoffs = payoff_fn(paths)

        discount = np.exp(-(r or mu) * T)
        discounted = discount * payoffs

        if control_variate is not None:
            cv_fn, cv_expected = control_variate
            cv_values = cv_fn(paths)
            cv_discounted = discount * cv_values
            # Optimal beta
            cov_matrix = np.cov(discounted, cv_discounted)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
            adjusted = discounted - beta * (cv_discounted - cv_expected * discount)

            var_raw = float(np.var(discounted, ddof=1))
            var_adj = float(np.var(adjusted, ddof=1))
            vr = var_raw / var_adj if var_adj > 0 else 1.0
            mean_val = float(np.mean(adjusted))
            std_err = float(np.std(adjusted, ddof=1) / np.sqrt(len(adjusted)))
        else:
            mean_val = float(np.mean(discounted))
            std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
            vr = 1.0

        ci_hw = 1.96 * std_err
        return MCResult(
            price=mean_val,
            std_error=std_err,
            ci_lower=mean_val - ci_hw,
            ci_upper=mean_val + ci_hw,
            n_paths=self.n_paths,
            variance_reduction_ratio=vr,
        )
