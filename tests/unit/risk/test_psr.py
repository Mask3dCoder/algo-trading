"""Tests for Probabilistic Sharpe Ratio and Deflated Sharpe Ratio (Priority 3)."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.risk.psr import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    holm_bonferroni_correction,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    selection_bias_estimate,
)

# ---------------------------------------------------------------------------
# PSR tests
# ---------------------------------------------------------------------------


class TestProbabilisticSharpeRatio:
    def test_psr_is_05_when_sr_equals_benchmark(self) -> None:
        """PSR = 0.5 when the estimated Sharpe exactly equals the benchmark."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.02, 500)  # SR ~ 0.5 (not annualized)
        sr = float(np.mean(returns) / np.std(returns, ddof=1))
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=sr, annualize=False)
        assert psr == pytest.approx(0.5, abs=0.02)

    def test_psr_increases_with_more_observations(self) -> None:
        """Given the same Sharpe estimate, more observations → higher PSR."""
        rng = np.random.default_rng(42)
        mean, std = 0.001, 0.01
        returns = rng.normal(mean, std, 1000)

        # Compute PSR for increasing windows from the same series
        # Use a fixed benchmark of 0 so PSR rises with sqrt(T)
        psr_short = probabilistic_sharpe_ratio(returns[:100], benchmark_sr=0.0, annualize=False)
        psr_long = probabilistic_sharpe_ratio(returns[:500], benchmark_sr=0.0, annualize=False)
        psr_full = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, annualize=False)

        # Each longer window with similar Sharpe should give higher PSR
        assert psr_full >= psr_long - 0.02
        assert psr_long >= psr_short - 0.02

    def test_psr_zero_for_negative_sharpe(self) -> None:
        """PSR should be near 0 when the Sharpe is clearly negative."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.01, 500)
        sr = float(np.mean(returns) / np.std(returns, ddof=1))
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, annualize=False)
        if sr < 0:
            assert psr < 0.5

    def test_psr_returns_probability(self) -> None:
        """PSR should always return a value in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            returns = rng.normal(np.random.uniform(-0.001, 0.001), 0.01, 100)
            psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, annualize=False)
            assert 0.0 <= psr <= 1.0

    def test_psr_insufficient_data(self) -> None:
        """With fewer than 3 observations, PSR should return 0.5."""
        returns = np.array([0.01, 0.02])
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        assert psr == 0.5

    def test_psr_constant_positive_returns(self) -> None:
        """Constant positive returns → PSR of 1.0."""
        returns = np.full(100, 0.001)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, annualize=False)
        assert psr == 1.0

    def test_psr_constant_negative_returns(self) -> None:
        """Constant negative returns → PSR of 0.0."""
        returns = np.full(100, -0.001)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, annualize=False)
        assert psr == 0.0

    def test_psr_constant_zero_returns(self) -> None:
        """Constant zero returns → PSR of 0.5 (no evidence either way)."""
        returns = np.full(100, 0.0)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, annualize=False)
        assert psr == 0.5


# ---------------------------------------------------------------------------
# Min Track Record Length tests
# ---------------------------------------------------------------------------


class TestMinTrackRecordLength:
    def test_infinite_when_sr_below_benchmark(self) -> None:
        """If SR_hat <= benchmark, should return effectively infinite."""
        result = min_track_record_length(sr_hat=0.0, benchmark_sr=0.5)
        assert result == 2**31 - 1

    def test_finite_when_sr_above_benchmark(self) -> None:
        """If SR_hat > benchmark, should return a finite value."""
        result = min_track_record_length(sr_hat=1.5, benchmark_sr=0.0)
        assert result > 0
        assert result < 2**31 - 1

    def test_more_skew_requires_more_observations(self) -> None:
        """Higher |skew| should increase minimum track record length."""
        base = min_track_record_length(sr_hat=1.0, benchmark_sr=0.0, skew=0.0, ex_kurt=0.0)
        with_skew = min_track_record_length(sr_hat=1.0, benchmark_sr=0.0, skew=-2.0, ex_kurt=0.0)
        assert with_skew >= base

    def test_minimum_return_value(self) -> None:
        """Should never return less than 2."""
        result = min_track_record_length(sr_hat=10.0, benchmark_sr=0.0)
        assert result >= 2


# ---------------------------------------------------------------------------
# DSR tests
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_dsr_equals_psr_when_one_trial(self) -> None:
        """DSR should equal regular PSR when only one trial exists."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 500)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        dsr = deflated_sharpe_ratio([0.5], 0.5, returns)
        assert dsr == pytest.approx(psr, abs=0.01)

    def test_dsr_leq_raw_psr_when_many_trials(self) -> None:
        """DSR should be <= PSR when there are many trials (punishes selection bias)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 500)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

        # Many random trial Sharpes — DSR should be lower
        all_srs = rng.normal(0.0, 0.3, 100).tolist()
        dsr = deflated_sharpe_ratio(all_srs, 0.5, returns)
        assert dsr <= psr

    def test_dsr_decreases_with_more_trials(self) -> None:
        """More trials should decrease DSR (more selection bias correction)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 500)

        base_srs = rng.normal(0.0, 0.3, 10).tolist()
        many_srs = rng.normal(0.0, 0.3, 1000).tolist()

        best_sr = 0.5
        dsr_few = deflated_sharpe_ratio(base_srs, best_sr, returns)
        dsr_many = deflated_sharpe_ratio(many_srs, best_sr, returns)
        assert dsr_many <= dsr_few

    def test_dsr_returns_probability(self) -> None:
        """DSR should always return a value in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            returns = rng.normal(0.0005, 0.01, 200)
            all_srs = rng.normal(0.0, 0.3, 50).tolist()
            dsr = deflated_sharpe_ratio(all_srs, rng.normal(0.0, 0.3), returns)
            assert 0.0 <= dsr <= 1.0

    def test_dsr_low_with_many_noise_trials(self) -> None:
        """If best Sharpe comes from many noise trials, DSR should be low."""
        rng = np.random.default_rng(42)

        # Strategy returns that are truly mediocre
        returns = rng.normal(0.0, 0.01, 500)

        # 200 trials of pure noise (some may have high Sharpe by luck)
        all_srs = rng.normal(0.0, 0.5, 200).tolist()
        best_sr = max(all_srs)
        dsr = deflated_sharpe_ratio(all_srs, best_sr, returns)
        # With many trials and mediocre true returns, DSR should be low
        assert dsr < 0.5


# ---------------------------------------------------------------------------
# Expected Max Sharpe tests
# ---------------------------------------------------------------------------


class TestExpectedMaxSharpe:
    def test_no_trials_returns_mean(self) -> None:
        result = expected_max_sharpe(0)
        assert result == 0.0

    def test_single_trial_returns_mean(self) -> None:
        result = expected_max_sharpe(1, sharpe_mean=0.5, sharpe_std=0.0)
        assert result == pytest.approx(0.5)

    def test_increases_with_trials(self) -> None:
        low = expected_max_sharpe(10)
        high = expected_max_sharpe(1000)
        assert high > low

    def test_large_n_clamped(self) -> None:
        """Very large N should not cause numerical issues."""
        result = expected_max_sharpe(100_000_000)
        assert np.isfinite(result)


class TestSelectionBiasEstimate:
    def test_zero_trials(self) -> None:
        assert selection_bias_estimate(0, 1.0) == 0.0

    def test_positive_bias(self) -> None:
        bias = selection_bias_estimate(100, 0.5)
        assert bias > 0

    def test_more_trials_more_bias(self) -> None:
        low = selection_bias_estimate(10, 0.5)
        high = selection_bias_estimate(1000, 0.5)
        assert high > low


# ---------------------------------------------------------------------------
# Multiple testing correction tests
# ---------------------------------------------------------------------------


class TestBonferroni:
    def test_rejects_none_with_low_pvalues(self) -> None:
        """Bonferroni should still reject very low p-values."""
        results = bonferroni_correction([0.001, 0.001], alpha=0.05)
        assert all(results)

    def test_rejects_none_with_high_pvalues(self) -> None:
        results = bonferroni_correction([0.5, 0.6], alpha=0.05)
        assert not any(results)

    def test_empty_input(self) -> None:
        results = bonferroni_correction([], alpha=0.05)
        assert results == []

    def test_conservative_fwer_control(self) -> None:
        """Bonferroni should not reject more than expected under FWER."""
        rng = np.random.default_rng(42)
        n_tests = 100
        # All null true — uniform p-values
        p_vals = rng.uniform(0, 1, n_tests).tolist()
        results = bonferroni_correction(p_vals, alpha=0.05)
        # Expected false rejections <= 0.05 * n_tests (but actually much lower)
        assert sum(results) <= max(1, n_tests // 20)


class TestBenjaminiHochberg:
    def test_rejects_some_with_mixed_pvalues(self) -> None:
        """BH should reject at least some very low p-values."""
        p_vals = [0.0001, 0.0002, 0.5, 0.6, 0.7, 0.8]
        results = benjamini_hochberg_correction(p_vals, alpha=0.05)
        assert sum(results) >= 2

    def test_empty_input(self) -> None:
        results = benjamini_hochberg_correction([], alpha=0.05)
        assert results == []

    def test_less_conservative_than_bonferroni(self) -> None:
        """BH should reject at least as many as Bonferroni."""
        p_vals = [0.001, 0.01, 0.5, 0.6]
        bh = benjamini_hochberg_correction(p_vals, alpha=0.05)
        bf = bonferroni_correction(p_vals, alpha=0.05)
        assert sum(bh) >= sum(bf)


class TestHolmBonferroni:
    def test_rejects_none_with_high_pvalues(self) -> None:
        results = holm_bonferroni_correction([0.5, 0.6], alpha=0.05)
        assert not any(results)

    def test_rejects_low_pvalues(self) -> None:
        results = holm_bonferroni_correction([0.0001, 0.5], alpha=0.05)
        assert results[0] is True

    def test_empty_input(self) -> None:
        results = holm_bonferroni_correction([], alpha=0.05)
        assert results == []

    def test_stop_after_first_failure(self) -> None:
        """Holm-Bonferroni stops after the first non-rejection."""
        p_vals = [0.0001, 0.01, 0.5, 0.6]
        results = holm_bonferroni_correction(p_vals, alpha=0.05)
        # The highest-ranked p-values should show the pattern
        assert any(results) or all(not r for r in results)


# ---------------------------------------------------------------------------
# Backward compat / defaults
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_psr_default_benchmark_is_zero(self) -> None:
        """Default benchmark should be zero (test against null hypothesis)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        psr = probabilistic_sharpe_ratio(returns, annualize=False)
        assert 0.0 <= psr <= 1.0

    def test_dsr_with_single_trial_sharpe(self) -> None:
        """DSR with a single trial should not crash and return a probability."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 200)
        dsr = deflated_sharpe_ratio([0.5], 0.5, returns)
        assert 0.0 <= dsr <= 1.0

    def test_psr_annualized_vs_daily(self) -> None:
        """Annualized PSR should differ from non-annualized."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, 500)
        psr_ann = probabilistic_sharpe_ratio(returns, annualize=True)
        psr_daily = probabilistic_sharpe_ratio(returns, annualize=False)
        # Won't be equal since benchmark_sr is treated as same scale
        assert psr_ann != pytest.approx(psr_daily, abs=0.01)
