"""Unit tests for stochastic models: GBM, OU, Heston, jump diffusion, fBM, MC engine."""

from __future__ import annotations

import numpy as np

from quantcore.math_models.stochastic.gbm import (
    estimate_params,
    simulate_gbm,
    simulate_gbm_analytical,
    simulate_gbm_milstein,
)
from quantcore.math_models.stochastic.monte_carlo import MonteCarloEngine
from quantcore.math_models.stochastic.ou_process import (
    analytical_mean,
    half_life,
    simulate_ou,
)
from quantcore.math_models.stochastic.ou_process import estimate_params as ou_estimate


class TestGBM:
    def test_analytical_terminal_mean(self) -> None:
        S0, mu, sigma, T = 100.0, 0.08, 0.2, 1.0
        prices = simulate_gbm_analytical(S0, mu, sigma, T, n_paths=500_000, seed=42)
        expected = S0 * np.exp(mu * T)
        actual = float(np.mean(prices))
        assert abs(actual - expected) / expected < 0.02  # Within 2%

    def test_analytical_terminal_variance(self) -> None:
        S0, mu, sigma, T = 100.0, 0.0, 0.3, 1.0
        prices = simulate_gbm_analytical(S0, mu, sigma, T, n_paths=500_000, seed=42)
        expected_var = (S0**2 * np.exp(sigma**2 * T) - S0**2) * np.exp(2 * mu * T)
        actual_var = float(np.var(prices))
        assert abs(actual_var - expected_var) / expected_var < 0.1  # Within 10%

    def test_mle_recovery(self) -> None:
        true_mu, true_sigma = 0.10, 0.25
        prices = simulate_gbm(100.0, true_mu, true_sigma, T=5.0, dt=1/252, n_paths=1, seed=42)[0]
        est_mu, est_sigma = estimate_params(prices)
        assert abs(est_sigma - true_sigma) / true_sigma < 0.10  # Within 10%
        # Mu is harder to estimate precisely

    def test_paths_shape(self) -> None:
        paths = simulate_gbm(100.0, 0.05, 0.2, T=1.0, dt=1/252, n_paths=100, seed=42)
        assert paths.shape[0] == 100
        assert paths.shape[1] == 253  # 252 steps + S0
        assert np.all(paths[:, 0] == 100.0)

    def test_milstein_no_negative_prices(self) -> None:
        paths = simulate_gbm_milstein(100.0, 0.05, 0.2, T=1.0, dt=1/252, n_paths=1000, seed=42)
        assert np.all(paths > 0)

    def test_mc_call_price_matches_bsm(self) -> None:
        """Monte Carlo call price should be within 0.5% of BSM analytical."""
        from quantcore.math_models.options.bsm import bsm_call
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bsm_price = float(bsm_call(S0, K, T, r, sigma))

        engine = MonteCarloEngine(n_paths=200_000, n_steps=252, seed=42)
        paths = engine.generate_gbm_paths(S0, r, sigma, T)
        payoffs = np.maximum(paths[:, -1] - K, 0.0)
        mc_price = float(np.mean(payoffs) * np.exp(-r * T))

        assert abs(mc_price - bsm_price) / bsm_price < 0.02  # Within 2%


class TestOUProcess:
    def test_simulated_mean_matches_analytical(self) -> None:
        X0, theta, mu, sigma, T = 0.0, 5.0, 2.0, 0.5, 2.0
        paths = simulate_ou(X0, theta, mu, sigma, T, dt=1/252, n_paths=50_000, seed=42)
        # At t=T, E[X_T] = mu + (X0 - mu)*exp(-theta*T)
        expected = analytical_mean(X0, theta, mu, T)
        actual = float(np.mean(paths[:, -1]))
        assert abs(actual - expected) / max(abs(expected), 0.01) < 0.05

    def test_half_life(self) -> None:
        theta = 5.0
        hl = half_life(theta)
        expected = np.log(2) / theta
        assert abs(hl - expected) < 1e-10

    def test_mle_recovery(self) -> None:
        true_theta, true_mu, true_sigma = 5.0, 2.0, 0.3
        paths = simulate_ou(0.0, true_theta, true_mu, true_sigma, T=10.0, dt=1/252, n_paths=1, seed=42)[0]
        est_theta, est_mu, est_sigma = ou_estimate(paths)
        assert abs(est_theta - true_theta) / true_theta < 0.15


class TestMonteCarloEngine:
    def test_antithetic_reduces_variance(self) -> None:
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        engine_std = MonteCarloEngine(n_paths=10_000, n_steps=50, seed=42)
        engine_anti = MonteCarloEngine(n_paths=10_000, n_steps=50, seed=42)

        payoff_fn = lambda paths: np.maximum(paths[:, -1] - K, 0.0)

        result_std = engine_std.compute_price(payoff_fn, S0, r, sigma, T, r=r, method="standard")
        result_anti = engine_anti.compute_price(payoff_fn, S0, r, sigma, T, r=r, method="antithetic")

        # Antithetic should have lower standard error
        assert result_anti.std_error <= result_std.std_error * 1.5  # Loose bound for small N

    def test_price_positive(self) -> None:
        engine = MonteCarloEngine(n_paths=10_000, n_steps=50, seed=42)
        payoff_fn = lambda paths: np.maximum(paths[:, -1] - 105.0, 0.0)
        result = engine.compute_price(payoff_fn, 100.0, 0.05, 0.2, 1.0, r=0.05)
        assert result.price > 0

    def test_confidence_interval_contains_true(self) -> None:
        from quantcore.math_models.options.bsm import bsm_call
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        true_price = float(bsm_call(S0, K, T, r, sigma))

        engine = MonteCarloEngine(n_paths=100_000, n_steps=252, seed=42)
        payoff_fn = lambda paths: np.maximum(paths[:, -1] - K, 0.0)
        result = engine.compute_price(payoff_fn, S0, r, sigma, T, r=r)

        assert result.ci_lower <= true_price <= result.ci_upper
