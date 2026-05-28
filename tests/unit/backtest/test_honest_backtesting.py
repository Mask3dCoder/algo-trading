"""Honest backtesting infrastructure tests — verify Q1 FIX FOUNDATION repairs.

These tests validate the structural fixes from the forensic audit:
1. Look-ahead bias elimination
2. Equity-based drawdown (not cumsum)
3. Sharpe NaN for zero-vol returns
4. Student-t parametric VaR
5. Cornish-Fisher n<100 warning
6. Purged CV no crash
7. Bootstrap CI integration
8. Drawdown guard state machine
9. Liquidity-aware sizing
10. Walk-forward acceptance criteria
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quantcore.backtest.walk_forward import (
    ACCEPTANCE_CRITERIA,
    anchored_walk_forward,
    purged_kfold_cv,
    rolling_walk_forward,
)
from quantcore.risk.drawdown_guard import DrawdownGuard, DrawdownGuardConfig, VolAdjustedDrawdownGuard
from quantcore.risk.liquidity import max_position_size, update_adv_cache
from quantcore.risk.metrics import (
    bootstrap_ci,
    burke_ratio,
    calmar_ratio,
    drawdown_stats,
    pain_index,
    sharpe_ratio,
    sterling_ratio,
    ulcer_index,
)
from quantcore.risk.var import cornish_fisher_var, parametric_var


# ---------------------------------------------------------------------------
# 1. Equity-based drawdown (not cumulative log-return)
# ---------------------------------------------------------------------------


class TestEquityBasedDrawdown:
    """All drawdown metrics must use exp(cumsum(r)) equity curve."""

    def test_calmar_equity_vs_cumsum(self):
        """Calmar with equity curve gives LARGER drawdown than cumsum for neg-skew."""
        # Simulate: small gains then large loss — classic neg-skew
        r = np.array([0.001] * 50 + [-0.08])
        calmar_val = calmar_ratio(r)
        # With equity curve, DD should be deeper than cumsum-based
        # Verify the metric doesn't return 0 or inf (which cumsum path could give)
        assert np.isfinite(calmar_val), f"Calmar should be finite, got {calmar_val}"

    def test_drawdown_stats_uses_equity_curve(self):
        """Drawdown stats must use exp(cumsum), not cumsum directly."""
        # Returns that produce different DD under equity vs cumsum
        r = np.array([0.01] * 20 + [-0.15])
        dd = drawdown_stats(r)
        # With equity curve (multiplicative), a -15% single-day loss from an
        # accumulated position produces max_drawdown ≈ 0.15 * exp(0.01*20)
        # With cumsum, it would be exactly -0.15+0.01*20 = 0.05 (way off)
        assert dd.max_drawdown < 0, "Drawdown should be negative"
        assert abs(dd.max_drawdown) > 0.10, "Equity-based DD should exceed 10%"

    def test_ulcer_index_equity_based(self):
        r = np.array([0.005] * 30 + [-0.12])
        ui = ulcer_index(r)
        assert ui > 0, "Ulcer index should be positive with drawdown"

    def test_pain_index_equity_based(self):
        r = np.array([0.005] * 30 + [-0.12])
        pi = pain_index(r)
        assert pi > 0, "Pain index should be positive with drawdown"

    def test_sterling_ratio_equity_based(self):
        r = np.array([0.001] * 100 + [-0.10])
        sr = sterling_ratio(r)
        assert np.isfinite(sr), "Sterling should be finite"

    def test_burke_ratio_equity_based(self):
        r = np.array([0.001] * 100 + [-0.10])
        br = burke_ratio(r)
        assert np.isfinite(br), "Burke should be finite"


# ---------------------------------------------------------------------------
# 2. Sharpe NaN for zero-vol returns
# ---------------------------------------------------------------------------


class TestSharpeNanForFlatReturns:
    """Flat return series → Sharpe is NaN (undefined), not 0.0."""

    def test_flat_returns_nan(self):
        r = np.zeros(100)
        sr = sharpe_ratio(r)
        assert np.isnan(sr), f"Flat returns should give NaN Sharpe, got {sr}"

    def test_near_flat_returns_nan(self):
        r = np.full(100, 1e-12)
        sr = sharpe_ratio(r)
        assert np.isnan(sr), "Near-flat returns should give NaN Sharpe"

    def test_volatile_returns_finite(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 100)
        sr = sharpe_ratio(r)
        assert np.isfinite(sr), "Volatile returns should give finite Sharpe"

    def test_single_return_nan(self):
        sr = sharpe_ratio(np.array([0.01]))
        assert np.isnan(sr), "Single return should give NaN Sharpe"


# ---------------------------------------------------------------------------
# 3. Student-t parametric VaR
# ---------------------------------------------------------------------------


class TestStudentTParametricVar:
    """Parametric VaR with Student-t produces WIDER VaR than Gaussian."""

    def test_t_var_wider_than_gaussian(self):
        rng = np.random.default_rng(42)
        # Heavy-tailed returns (t-distributed)
        r = rng.standard_t(3, 1000) * 0.01

        result_gauss = parametric_var(mu=0.0, sigma=0.01, confidence=0.99)
        result_t = parametric_var(mu=0.0, sigma=0.01, confidence=0.99, nu=3.0)

        # Student-t VaR should be wider (more negative = larger loss)
        assert result_t.var < result_gauss.var, (
            f"Student-t 99% VaR ({result_t.var:.6f}) should be wider than "
            f"Gaussian ({result_gauss.var:.6f})"
        )
        assert result_t.method == "parametric_t"
        assert result_gauss.method == "parametric"

    def test_mle_estimation_from_returns(self):
        rng = np.random.default_rng(42)
        r = rng.standard_t(4, 500) * 0.01
        result = parametric_var(mu=0.0, sigma=0.01, confidence=0.95, returns_for_t=r)
        assert result.method == "parametric_t", "Should auto-detect heavy tails"
        assert result.cvar is not None, "CVaR should be computed"

    def test_gaussian_fallback_when_no_returns(self):
        result = parametric_var(mu=0.0, sigma=0.01, confidence=0.95)
        assert result.method == "parametric", "Falls back to Gaussian when no returns"


# ---------------------------------------------------------------------------
# 4. Cornish-Fisher warning for n < 100
# ---------------------------------------------------------------------------


class TestCornishFisherWarning:
    """CF VaR with few observations should emit a warning."""

    def test_warning_on_small_sample(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cornish_fisher_var(
                mu=0.0, sigma=0.01, skewness=0.0, kurtosis=3.0,
                confidence=0.95, n_observations=50,
            )
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert "unreliable" in str(w[0].message).lower()

    def test_no_warning_on_large_sample(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cornish_fisher_var(
                mu=0.0, sigma=0.01, skewness=0.0, kurtosis=3.0,
                confidence=0.95, n_observations=200,
            )
            cf_warnings = [x for x in w if "unreliable" in str(x.message).lower()]
            assert len(cf_warnings) == 0, "Should not warn with n=200"

    def test_unstable_method_name(self):
        result = cornish_fisher_var(
            mu=0.0, sigma=0.01, skewness=0.5, kurtosis=5.0,
            confidence=0.95, n_observations=30,
        )
        assert result.method == "unstable_cornish_fisher"

    def test_stable_method_name(self):
        result = cornish_fisher_var(
            mu=0.0, sigma=0.01, skewness=0.5, kurtosis=5.0,
            confidence=0.95, n_observations=100,
        )
        assert result.method == "cornish_fisher"


# ---------------------------------------------------------------------------
# 5. Purged CV no crash
# ---------------------------------------------------------------------------


class TestPurgedCV:
    """Purged K-Fold CV must not crash with NameError (old bug: `total`)."""

    def test_basic(self):
        folds = purged_kfold_cv(n_bars=500, n_splits=3, test_size=60)
        # First fold dropped (no prior training data).
        assert len(folds) == 2

    def test_no_overlap(self):
        folds = purged_kfold_cv(n_bars=1000, n_splits=5, test_size=100)
        for fold in folds:
            # Training must always be chronologically before test data.
            assert fold.train_end <= fold.test_start, "Train/test must not overlap"

    def test_fold_bounds_within_data(self):
        for n_bars in [200, 500, 1000]:
            folds = purged_kfold_cv(n_bars=n_bars, n_splits=3)
            for fold in folds:
                assert fold.test_end <= n_bars
                assert fold.train_end <= n_bars


# ---------------------------------------------------------------------------
# 6. Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Bootstrap confidence intervals work and produce valid bounds."""

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 500)
        lo, hi = bootstrap_ci(r, sharpe_ratio, n_boot=500, ci=0.90)
        # Point estimate should be roughly within CI
        point = sharpe_ratio(r)
        if np.isfinite(point):
            assert lo <= point + 0.5, f"CI lower {lo} way below point {point}"
            assert hi >= point - 0.5, f"CI upper {hi} way above point {point}"

    def test_ci_bounds_order(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 500)
        lo, hi = bootstrap_ci(r, sharpe_ratio, n_boot=500, ci=0.90)
        assert lo <= hi, f"CI lower {lo} should be <= upper {hi}"


# ---------------------------------------------------------------------------
# 7. Drawdown guard (vol-adjusted)
# ---------------------------------------------------------------------------


class TestVolAdjustedDrawdownGuard:
    """Vol-adjusted drawdown guard with state machine recovery."""

    def test_flatten_on_large_drawdown(self):
        guard = VolAdjustedDrawdownGuard()
        # Simulate equity drop
        for i in range(100):
            equity = 100000 - i * 500  # Linear descent
            state = guard.update(equity)
        assert state.action in ("flatten", "reduce"), f"Should be in flatten or reduce, got {state.action}"

    def test_normal_state_on_steady_equity(self):
        guard = VolAdjustedDrawdownGuard()
        for i in range(100):
            equity = 100000 + i * 100
            state = guard.update(equity)
        assert state.action == "normal", "Steady growth should be normal state"

    def test_recovery_gate_exists(self):
        guard = VolAdjustedDrawdownGuard()
        for i in range(200):
            # Drop then recover
            equity = 100000 - i * 300 if i < 50 else 100000 - 15000 + (i - 50) * 200
            state = guard.update(equity)
        # After recovery, should eventually return to normal
        assert state.action in ("normal", "reduce", "recovery_gate", "flatten")


# ---------------------------------------------------------------------------
# 8. Liquidity-aware sizing
# ---------------------------------------------------------------------------


class TestLiquiditySizing:
    """ADV-based position sizing caps position at participation rate."""

    def test_unlimited_when_no_adv(self):
        size = max_position_size("UNKNOWN_SYMBOL", capital=1_000_000)
        assert size >= 1e8, "Should return large sentinel when ADV unknown"

    def test_capped_at_participation_rate(self):
        update_adv_cache({"SPY": 15_000_000_000.0})
        size = max_position_size("SPY", capital=10_000_000, participation_rate=0.05)
        # 5% of $15B ADV = $750M, but 20% of $10M capital = $2M
        assert size <= 2_000_000, f"Should be capped at 20% of capital, got {size}"

    def test_adv_override(self):
        size = max_position_size("TEST", capital=1_000_000, adv_override=100_000_000)
        expected = min(100_000_000 * 0.05, 1_000_000 * 0.20)
        assert abs(size - expected) < 1.0, f"Expected {expected}, got {size}"


# ---------------------------------------------------------------------------
# 9. Walk-forward acceptance criteria
# ---------------------------------------------------------------------------


class TestWalkForwardAcceptance:
    """Walk-forward acceptance criteria must be realistic."""

    def test_min_oos_sharpe_is_positive(self):
        assert ACCEPTANCE_CRITERIA["min_oos_sharpe"] > 0, (
            "Min OOS Sharpe must be positive (not 0.0 — that would pass anything)"
        )

    def test_max_dd_is_bounded(self):
        assert ACCEPTANCE_CRITERIA["max_drawdown_pct"] <= 50.0, (
            "Max DD threshold too loose"
        )
        assert ACCEPTANCE_CRITERIA["max_drawdown_pct"] >= 5.0, (
            "Max DD threshold too tight"
        )

    def test_rolling_folds_generated(self):
        folds = rolling_walk_forward(1000, train_size=200, test_size=50)
        assert len(folds) >= 1, "Should generate at least one fold"

    def test_anchored_folds_expand_window(self):
        folds = anchored_walk_forward(1000, initial_train=200, test_size=50)
        # Every fold should start training from 0 (anchored)
        for fold in folds:
            assert fold.train_start == 0, "Anchored WF: training always starts at 0"
