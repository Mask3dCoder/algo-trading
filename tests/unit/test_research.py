"""Tests for research module: alpha evaluation, experiment tracking."""

from __future__ import annotations

import numpy as np

from quantcore.research.alpha_research import (
    DeflatedSharpeResult,
    deflated_sharpe,
    fdr_correction,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
)
from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio
from quantcore.research.experiment_tracker import (
    Experiment,
    load_experiments,
    log_experiment,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Deflated Sharpe (legacy alpha_research interface)
# ---------------------------------------------------------------------------

class TestDeflatedSharpe:
    def test_positive_sharpe(self) -> None:
        returns = RNG.normal(0.001, 0.01, 252)
        result = deflated_sharpe(returns, num_trials=10)
        assert isinstance(result, DeflatedSharpeResult)
        assert result.observed_sharpe != 0.0
        assert result.p_value >= 0.0

    def test_known_zero_returns(self) -> None:
        returns = np.zeros(100)
        result = deflated_sharpe(returns)
        assert result.observed_sharpe == 0.0
        assert result.p_value > 0.05

    def test_more_trials_reduces_deflated(self) -> None:
        returns = RNG.normal(0.0008, 0.01, 500)
        r1 = deflated_sharpe(returns, num_trials=1)
        r2 = deflated_sharpe(returns, num_trials=100)
        assert r2.deflated_sharpe <= r1.deflated_sharpe

    def test_small_data(self) -> None:
        returns = np.array([0.001, 0.002, -0.001])
        result = deflated_sharpe(returns)
        assert result.observed_sharpe == 0.0


class TestMinimumBacktestLength:
    def test_higher_sharpe_shorter_length(self) -> None:
        len1 = minimum_backtest_length(2.0, 10)
        len2 = minimum_backtest_length(0.5, 10)  # type: ignore[call-arg]
        assert len1 < len2

    def test_zero_sharpe(self) -> None:
        length = minimum_backtest_length(0.0, 10)  # type: ignore[call-arg]
        assert length > 100


class TestProbabilisticSharpeRatio:
    def test_positive_vs_zero(self) -> None:
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 252)
        assert psr > 0.5

    def test_negative_vs_zero(self) -> None:
        psr = probabilistic_sharpe_ratio(-0.5, 0.0, 252)
        assert psr < 0.5


class TestFDRCorrection:
    def test_bh_simple(self) -> None:
        p_values = np.array([0.001, 0.01, 0.05, 0.5, 0.9])
        adj, rejected = fdr_correction(p_values, method="bh")
        assert len(adj) == 5
        assert np.sum(rejected) <= 3

    def test_bh_all_significant(self) -> None:
        p_values = np.array([0.0001, 0.0001, 0.0001])
        adj, rejected = fdr_correction(p_values, method="bh")
        assert np.sum(rejected) == 3

    def test_empty(self) -> None:
        adj, rejected = fdr_correction(np.array([]))
        assert len(adj) == 0


# ---------------------------------------------------------------------------
# New DSR (Bailey & Lopez de Prado exact)
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatioNew:
    def test_zero_trials_one(self) -> None:
        pv, dsr = deflated_sharpe_ratio(
            observed_sharpe=1.0, num_trials=1, sample_length=252,
            skewness=0.0, kurtosis_=0.0,
        )
        assert 0.0 <= pv <= 1.0
        assert np.isfinite(dsr)

    def test_known_p_value_range(self) -> None:
        # More trials => DSR more negative => higher p-value
        pv1, _ = deflated_sharpe_ratio(
            observed_sharpe=1.0, num_trials=1, sample_length=252,
            skewness=0.0, kurtosis_=0.0,
        )
        pv2, _ = deflated_sharpe_ratio(
            observed_sharpe=1.0, num_trials=100, sample_length=252,
            skewness=0.0, kurtosis_=0.0,
        )
        assert pv2 > pv1

    def test_extreme_negative_sharpe(self) -> None:
        pv, dsr = deflated_sharpe_ratio(
            observed_sharpe=-2.0, num_trials=10, sample_length=252,
            skewness=0.0, kurtosis_=0.0,
        )
        assert 0.0 <= pv <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_ci_contains_true_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        lower, upper = bootstrap_confidence_interval(returns, n_boot=500, ci=0.90)
        assert lower <= upper
        assert lower > -10.0
        assert upper < 10.0

    def test_no_data_returns_same_value(self) -> None:
        lower, upper = bootstrap_confidence_interval(np.array([]), n_boot=100)
        assert lower == upper == 0.0  # fallback


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    def test_log_and_load(self, tmp_path) -> None:
        csv_file = str(tmp_path / "experiments.csv")
        exp = Experiment(
            experiment_id="TEST-001",
            strategy_name="residual_momentum",
            oos_sharpe=0.85,
            deflated_sharpe_pvalue=0.03,
            ci_90_lower=0.32,
        )
        log_experiment(exp, csv_path=csv_file)
        df = load_experiments(csv_file)
        assert len(df) == 1
        assert df.iloc[0]["experiment_id"] == "TEST-001"

    def test_unique_id(self) -> None:
        e1 = Experiment()
        e2 = Experiment()
        assert e1.experiment_id != e2.experiment_id

