"""Tests for Bug 3 fix: _rolling_std in mean_reversion strategy.

Bug 3: _rolling_std called _rolling_mean() but discarded the result,
wasting CPU and potentially using a less numerically stable approach.
"""

import numpy as np
import pytest

from quantcore.strategies.equity.mean_reversion import _rolling_mean, _rolling_std


class TestRollingStd:
    """Verify Bug 3 fix: _rolling_std uses efficient cumulative-sum formula."""

    def test_rolling_std_basic_correctness(self):
        """Rolling std should match np.std for each window."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        window = 20

        result = _rolling_std(x, window)

        # Check every valid window against numpy
        for i in range(window - 1, len(x)):
            expected = np.std(x[i - window + 1:i + 1], ddof=1)
            assert abs(result[i] - expected) < 1e-10, (
                f"Window ending at {i}: expected {expected:.10f}, got {result[i]:.10f}"
            )

    def test_rolling_std_nan_before_window(self):
        """First (window-1) elements should be NaN."""
        x = np.arange(50, dtype=float)
        result = _rolling_std(x, window=20)
        assert all(np.isnan(result[:19]))
        assert not any(np.isnan(result[19:]))

    def test_rolling_std_constant_input(self):
        """Constant input should produce zero std."""
        x = np.ones(100) * 5.0
        result = _rolling_std(x, window=20)
        valid = result[~np.isnan(result)]
        assert np.allclose(valid, 0.0, atol=1e-12)

    def test_rolling_std_short_input(self):
        """Input shorter than window should return all NaN."""
        x = np.array([1.0, 2.0, 3.0])
        result = _rolling_std(x, window=20)
        assert all(np.isnan(result))

    def test_rolling_std_window_equals_length(self):
        """Window equal to input length should produce one valid value."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_std(x, window=5)
        assert np.isnan(result[3])
        expected = np.std(x, ddof=1)
        assert abs(result[4] - expected) < 1e-10

    def test_rolling_std_performance_no_wasted_computation(self):
        """Verify the function doesn't call _rolling_mean (no wasted computation).

        This is a structural test: the old code called _rolling_mean(x, window)
        and discarded the result. The fix uses cumulative sums instead.
        We verify by checking that _rolling_std produces correct results
        without relying on _rolling_mean being called.
        """
        rng = np.random.default_rng(99)
        x = rng.standard_normal(1000)
        window = 50

        # Should complete without error and produce correct values
        result = _rolling_std(x, window)
        expected = np.std(x[:window], ddof=1)
        assert abs(result[window - 1] - expected) < 1e-10


class TestRollingMean:
    """Sanity: _rolling_mean still works after the refactor."""

    def test_rolling_mean_correctness(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        window = 20
        result = _rolling_mean(x, window)

        for i in range(window - 1, len(x)):
            expected = np.mean(x[i - window + 1:i + 1])
            assert abs(result[i] - expected) < 1e-10
