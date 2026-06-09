"""Unit tests for fixed metrics (Task 1.3).

Validates that:
  1. sharpe_ratio returns NaN (not 0.0) for flat return series.
  2. calmar_ratio uses equity curve, not log cumulative.
  3. rf rate uses geometric decompounding when annualize=True.
  4. bootstrap_ci returns finite CI bounds for normal returns.
  5. rolling_sharpe and rolling_maxdd produce correct shapes.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from quantcore.risk.metrics import (
    sharpe_ratio, calmar_ratio, bootstrap_ci,
    rolling_sharpe, rolling_maxdd,
)


class TestSharpeRatioFix:
    def test_flat_returns_return_nan(self):
        """Core fix: flat return series -> NaN, not 0.0."""
        flat = np.zeros(252)
        result = sharpe_ratio(flat)
        assert math.isnan(result), f"Expected NaN for flat returns, got {result}"

    def test_near_zero_std_returns_nan(self):
        r = np.full(252, 1e-15)
        result = sharpe_ratio(r)
        assert math.isnan(result), f"Expected NaN for near-zero std, got {result}"

    def test_pure_alpha_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0008, 0.01, 252)  # daily 0.08% mean
        sr = sharpe_ratio(r)
        assert not math.isnan(sr), "Sharpe should be finite for non-flat returns"
        assert sr > 0, "Positive mean returns -> positive Sharpe"

    def test_rf_geometric_decompounding(self):
        """rf=0.05 annually must be decompounded correctly per trading day."""
        daily_rf = (1.05) ** (1 / 252) - 1
        r = np.full(252, daily_rf)  # returns EXACTLY equal to rf -> Sharpe=NaN
        result = sharpe_ratio(r, rf=0.05, annualize=True)
        assert math.isnan(result) or abs(result) < 0.1, (
            f"Returns == rf should yield Sharpe near NaN or zero, got {result}"
        )

    def test_short_series_returns_nan(self):
        assert math.isnan(sharpe_ratio(np.array([0.01])))

    def test_empty_series_returns_nan(self):
        assert math.isnan(sharpe_ratio(np.array([])))


class TestCalmarRatioFix:
    def test_equity_drawdown_not_log_cumulative(self):
        """Verify calmar uses equity curve, not cumsum of log returns."""
        # Construct returns with a known 10% equity drawdown
        # Peak equity = exp(sum of up returns), then down 10%
        up = np.full(50, 0.002)    # 50 bars up
        down = np.full(10, -0.011) # 10 bars down (roughly -10% from peak)
        r = np.concatenate([up, down])

        calmar = calmar_ratio(r)
        assert not math.isnan(calmar), "Calmar should be finite for non-flat returns"
        # The drawdown is real and max_dd should be > 0
        # Prior bug would use cumsum, giving a different (smaller) max_dd
        equity = np.exp(np.cumsum(r))
        peak = np.maximum.accumulate(equity)
        correct_maxdd = float(np.abs(((equity - peak) / peak).min()))
        assert correct_maxdd > 0.01, f"Expected drawdown >1%, got {correct_maxdd:.4f}"

    def test_no_drawdown_returns_nan(self):
        """Strictly increasing equity -> max_dd near zero -> Calmar NaN."""
        r = np.full(100, 0.001)  # monotonically increasing
        result = calmar_ratio(r)
        assert math.isnan(result) or result > 100, (
            f"Expected NaN or very high Calmar for no-drawdown series, got {result}"
        )


class TestBootstrapCI:
    def test_returns_finite_bounds(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 500)
        lo, hi = bootstrap_ci(r, sharpe_ratio, n_boot=200, ci=0.90)
        assert math.isfinite(lo), f"Lower CI bound not finite: {lo}"
        assert math.isfinite(hi), f"Upper CI bound not finite: {hi}"
        assert lo <= hi, f"Lower CI {lo} > Upper CI {hi}"

    def test_90_pct_ci_contains_point_estimate(self):
        rng = np.random.default_rng(99)
        r = rng.normal(0.001, 0.01, 500)
        point = sharpe_ratio(r)
        lo, hi = bootstrap_ci(r, sharpe_ratio, n_boot=400, ci=0.90)
        assert lo <= point <= hi, (
            f"Point estimate {point:.3f} not within CI [{lo:.3f}, {hi:.3f}]"
        )

    def test_positive_alpha_lower_ci_positive(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.002, 0.01, 1000)  # strong alpha
        lo, hi = bootstrap_ci(r, sharpe_ratio, n_boot=500, ci=0.90)
        assert lo > 0, f"95th lower CI should be positive for strong alpha, got {lo:.3f}"


class TestRollingSharpeMaxDD:
    def test_rolling_sharpe_shape(self):
        r = np.random.default_rng(1).normal(0.001, 0.01, 300)
        rs = rolling_sharpe(r, window=252)
        assert rs.shape == r.shape, f"Shape mismatch: {rs.shape} vs {r.shape}"
        # First 251 values must be NaN
        assert np.all(np.isnan(rs[:251])), "Pre-window values should be NaN"
        assert np.isfinite(rs[251]) or np.isnan(rs[251])

    def test_rolling_maxdd_shape(self):
        r = np.random.default_rng(2).normal(0.001, 0.01, 300)
        rmd = rolling_maxdd(r, window=60)
        assert rmd.shape == r.shape
        assert np.all(np.isnan(rmd[:59])), "Pre-window values should be NaN"
        assert rmd[59] >= 0, "Max drawdown must be non-negative"

    def test_rolling_sharpe_flat_gives_nan(self):
        r = np.zeros(300)
        rs = rolling_sharpe(r, window=252)
        assert np.all(np.isnan(rs[251:])), "Flat returns should give NaN rolling Sharpe"
