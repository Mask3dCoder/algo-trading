"""Unit tests for Kalman hedge ratio filter."""

from __future__ import annotations

import numpy as np


class TestKalmanHedgeRatio:
    """Tests for KalmanHedgeRatio class."""

    def test_basic_initialization(self):
        """Filter initializes with default parameters."""
        from quantcore.signals.statistical.kalman_hedge import KalmanHedgeRatio

        kf = KalmanHedgeRatio()
        assert kf.delta == 1e-4
        assert kf.Ve == 1e-1
        assert kf.current_beta == 0.0
        assert kf.current_alpha == 0.0

    def test_converges_to_true_ratio(self):
        """Filter converges to the true hedge ratio on clean data."""
        from quantcore.signals.statistical.kalman_hedge import KalmanHedgeRatio

        np.random.seed(42)
        n = 300
        true_beta = 1.5
        true_alpha = 0.5

        # Use stationary x (not random walk) for cleaner convergence test
        x = 50 + 10 * np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 2
        y = true_alpha + true_beta * x + np.random.randn(n) * 0.1

        kf = KalmanHedgeRatio(delta=1e-4, Ve=1e-2, warmup=50)
        for t in range(n):
            kf.update(y[t], x[t])

        # After warmup + convergence, beta should be close to true value
        assert abs(kf.current_beta - true_beta) < 0.15
        assert abs(kf.current_alpha - true_alpha) < 1.0

    def test_handles_nan_inputs(self):
        """Filter handles NaN inputs gracefully."""
        from quantcore.signals.statistical.kalman_hedge import KalmanHedgeRatio

        kf = KalmanHedgeRatio(warmup=5)

        # Feed some valid data first
        for _ in range(10):
            state = kf.update(1.0, 1.0)
            assert not np.isnan(state.beta)

        # Now feed NaN
        state = kf.update(np.nan, 1.0)
        assert np.isnan(state.spread)

        state = kf.update(1.0, np.nan)
        assert np.isnan(state.spread)

    def test_state_history(self):
        """Filter records full state history."""
        from quantcore.signals.statistical.kalman_hedge import KalmanHedgeRatio

        kf = KalmanHedgeRatio(warmup=5)
        for i in range(20):
            kf.update(float(i), float(i) * 0.5)

        assert len(kf.states) == 20
        assert all(hasattr(s, 'beta') for s in kf.states)
        assert all(hasattr(s, 'alpha') for s in kf.states)

    def test_uncertainty_decreases(self):
        """Uncertainty (P_beta) decreases as more data is observed."""
        from quantcore.signals.statistical.kalman_hedge import KalmanHedgeRatio

        np.random.seed(42)
        kf = KalmanHedgeRatio(warmup=10)

        # Feed consistent data
        for i in range(100):
            kf.update(float(i) * 2.0 + np.random.randn() * 0.01,
                      float(i) + np.random.randn() * 0.01)

        # Uncertainty should be lower than initial
        assert kf.beta_uncertainty < 1.0

    def test_adapts_to_structural_break(self):
        """Filter adapts when the true hedge ratio changes."""
        from quantcore.signals.statistical.kalman_hedge import KalmanHedgeRatio

        np.random.seed(42)
        n = 300
        kf = KalmanHedgeRatio(delta=1e-3, warmup=30)  # Higher delta for faster adaptation

        x = np.cumsum(np.random.randn(n)) + 100

        # Phase 1: beta = 1.0
        y = np.zeros(n)
        for t in range(150):
            y[t] = 1.0 * x[t] + np.random.randn(t + 1)[-1] * 0.1
            kf.update(y[t], x[t])

        beta_phase1 = kf.current_beta

        # Phase 2: beta = 2.0
        for t in range(150, n):
            y[t] = 2.0 * x[t] + np.random.randn(1)[0] * 0.1
            kf.update(y[t], x[t])

        beta_phase2 = kf.current_beta

        # Beta should have moved toward 2.0
        assert beta_phase2 > beta_phase1
        assert abs(beta_phase2 - 2.0) < abs(beta_phase1 - 2.0)


class TestKalmanHedgeRatioSeries:
    """Tests for kalman_hedge_ratio_series convenience function."""

    def test_returns_correct_shapes(self):
        """Returns arrays of correct shape."""
        from quantcore.signals.statistical.kalman_hedge import kalman_hedge_ratio_series

        n = 100
        y = np.random.randn(n).cumsum()
        x = np.random.randn(n).cumsum()

        betas, spreads, uncs = kalman_hedge_ratio_series(y, x, warmup=20)

        assert len(betas) == n
        assert len(spreads) == n
        assert len(uncs) == n

    def test_no_nans_after_warmup(self):
        """No NaN values after warmup period."""
        from quantcore.signals.statistical.kalman_hedge import kalman_hedge_ratio_series

        n = 100
        y = np.arange(n, dtype=float) * 2.0
        x = np.arange(n, dtype=float)

        betas, spreads, uncs = kalman_hedge_ratio_series(y, x, warmup=10)

        # After warmup, no NaN
        assert not np.any(np.isnan(betas[10:]))
        assert not np.any(np.isnan(spreads[10:]))


class TestAdaptiveKalman:
    """Tests for adaptive Kalman filter."""

    def test_adapts_delta(self):
        """Adaptive filter adjusts process noise based on innovation."""
        from quantcore.signals.statistical.kalman_hedge import adaptive_kalman_hedge_ratio

        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n)) + 100
        y = 1.0 * x + np.random.randn(n) * 0.1

        betas, spreads, uncs = adaptive_kalman_hedge_ratio(y, x, warmup=20)

        assert len(betas) == n
        assert not np.all(np.isnan(betas[20:]))
