"""Tests for research module: alpha evaluation, experiment tracking."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from quantcore.research.alpha_research import (
    DeflatedSharpeResult,
    deflated_sharpe,
    fdr_correction,
    haircut_sharpe,
    holm_correction,
    minimum_backtest_length,
    multiple_testing_summary,
    overfitting_probability,
    probabilistic_sharpe_ratio,
)
from quantcore.research.experiment_tracker import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTracker,
    find_best_parameters,
    sensitivity_analysis,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Deflated Sharpe tests
# ---------------------------------------------------------------------------

class TestDeflatedSharpe:
    def test_positive_sharpe(self):
        returns = RNG.normal(0.001, 0.01, 252)
        result = deflated_sharpe(returns, num_trials=10)
        assert isinstance(result, DeflatedSharpeResult)
        assert result.observed_sharpe != 0.0
        assert result.p_value >= 0.0

    def test_known_zero_returns(self):
        returns = np.zeros(100)
        result = deflated_sharpe(returns)
        assert result.observed_sharpe == 0.0
        assert result.p_value > 0.05

    def test_more_trials_reduces_deflated(self):
        returns = RNG.normal(0.0008, 0.01, 500)
        r1 = deflated_sharpe(returns, num_trials=1)
        r2 = deflated_sharpe(returns, num_trials=100)
        assert r2.deflated_sharpe <= r1.deflated_sharpe

    def test_small_data(self):
        returns = np.array([0.001, 0.002, -0.001])
        result = deflated_sharpe(returns)
        assert result.observed_sharpe == 0.0


class TestMinimumBacktestLength:
    def test_higher_sharpe_shorter_length(self):
        len1 = minimum_backtest_length(2.0, 10)
        len2 = minimum_backtest_length(0.5, 10)
        assert len1 < len2

    def test_zero_sharpe(self):
        length = minimum_backtest_length(0.0, 10)
        assert length > 100


class TestProbabilisticSharpeRatio:
    def test_positive_vs_zero(self):
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 252)
        assert psr > 0.5

    def test_negative_vs_zero(self):
        psr = probabilistic_sharpe_ratio(-0.5, 0.0, 252)
        assert psr < 0.5


class TestFDRCorrection:
    def test_bh_simple(self):
        p_values = np.array([0.001, 0.01, 0.05, 0.5, 0.9])
        adj, rejected = fdr_correction(p_values, method="bh")
        assert len(adj) == 5
        assert np.sum(rejected) <= 3

    def test_bh_all_significant(self):
        p_values = np.array([0.0001, 0.0001, 0.0001])
        adj, rejected = fdr_correction(p_values, method="bh")
        assert np.sum(rejected) == 3

    def test_empty(self):
        adj, rejected = fdr_correction(np.array([]))
        assert len(adj) == 0


class TestHolmCorrection:
    def test_simple(self):
        p_values = np.array([0.001, 0.01, 0.5])
        adj, rejected = holm_correction(p_values)
        assert len(rejected) == 3
        assert np.sum(rejected) >= 1


class TestMultipleTestingSummary:
    def test_output_format(self):
        p_values = np.array([0.001, 0.05, 0.5])
        summary = multiple_testing_summary(p_values)
        assert "n_tests" in summary
        assert "bonferroni_significant" in summary
        assert "bh_significant" in summary

    def test_with_names(self):
        p_values = np.array([0.001, 0.5])
        summary = multiple_testing_summary(p_values, strategy_names=["Alpha", "Beta"])
        assert len(summary["bh_names"]) >= 0


class TestHaircutSharpe:
    def test_more_trials_bigger_haircut(self):
        h1 = haircut_sharpe(1.0, 10, 252)
        h2 = haircut_sharpe(1.0, 1000, 252)
        assert h2 <= h1


class TestOverfittingProbability:
    def test_random_returns(self):
        r1 = RNG.normal(0.0005, 0.01, 500)
        r2 = RNG.normal(0.0003, 0.01, 500)
        prob = overfitting_probability([r1, r2])
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Experiment Tracker tests
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    def test_create_experiment(self):
        tracker = ExperimentTracker()
        config = ExperimentConfig(name="Test Strategy", strategy_name="momentum")
        exp = tracker.create_experiment(config)
        assert exp.experiment_id != ""
        assert exp.status == ExperimentStatus.DRAFT

    def test_start_and_complete(self):
        tracker = ExperimentTracker()
        config = ExperimentConfig(name="Test", strategy_name="mean_rev")
        exp = tracker.create_experiment(config)
        tracker.start_experiment(exp.experiment_id)
        result = tracker.complete_experiment(
            exp.experiment_id,
            metrics={"sharpe": 1.5, "max_dd": -0.1},
            trades_count=100,
        )
        assert result is not None
        assert result.status == ExperimentStatus.COMPLETED
        assert result.metrics["sharpe"] == 1.5

    def test_fail_experiment(self):
        tracker = ExperimentTracker()
        config = ExperimentConfig(name="Failing", strategy_name="test")
        exp = tracker.create_experiment(config)
        tracker.fail_experiment(exp.experiment_id, "Division by zero")
        fetched = tracker.get_experiment(exp.experiment_id)
        assert fetched is not None
        assert fetched.status == ExperimentStatus.FAILED
        assert fetched.error_message == "Division by zero"

    def test_list_by_status(self):
        tracker = ExperimentTracker()
        e1 = tracker.create_experiment(ExperimentConfig(name="E1", strategy_name="s1"))
        tracker.start_experiment(e1.experiment_id)
        tracker.create_experiment(ExperimentConfig(name="E2", strategy_name="s2"))
        running = tracker.list_experiments(status=ExperimentStatus.RUNNING)
        assert len(running) == 1
        drafts = tracker.list_experiments(status=ExperimentStatus.DRAFT)
        assert len(drafts) == 1

    def test_list_by_strategy(self):
        tracker = ExperimentTracker()
        tracker.create_experiment(ExperimentConfig(name="A", strategy_name="momentum"))
        tracker.create_experiment(ExperimentConfig(name="B", strategy_name="mean_rev"))
        results = tracker.list_experiments(strategy_name="momentum")
        assert len(results) == 1

    def test_list_by_tags(self):
        tracker = ExperimentTracker()
        tracker.create_experiment(ExperimentConfig(name="A", strategy_name="s", tags=["live", "equity"]))
        tracker.create_experiment(ExperimentConfig(name="B", strategy_name="s", tags=["paper"]))
        results = tracker.list_experiments(tags=["live"])
        assert len(results) == 1

    def test_best_experiment(self):
        tracker = ExperimentTracker()
        config = ExperimentConfig(name="Best", strategy_name="momentum")
        exp = tracker.create_experiment(config)
        tracker.complete_experiment(exp.experiment_id, {"sharpe": 0.5, "max_dd": -0.1})
        best = tracker.best_experiment(metric="sharpe")
        assert best is not None

    def test_compare_experiments(self):
        tracker = ExperimentTracker()
        e1 = tracker.create_experiment(ExperimentConfig(name="E1", strategy_name="s"))
        e2 = tracker.create_experiment(ExperimentConfig(name="E2", strategy_name="s"))
        tracker.complete_experiment(e1.experiment_id, {"sharpe": 1.0, "sortino": 1.2})
        tracker.complete_experiment(e2.experiment_id, {"sharpe": 0.8, "sortino": 0.9})
        comp = tracker.compare_experiments([e1.experiment_id, e2.experiment_id])
        assert len(comp) == 2
        assert comp[e1.experiment_id]["sharpe"] == 1.0

    def test_summary(self):
        tracker = ExperimentTracker()
        config = ExperimentConfig(name="Summary", strategy_name="s")
        exp = tracker.create_experiment(config)
        tracker.complete_experiment(exp.experiment_id, {"sharpe": 1.2})
        s = tracker.summary()
        assert s["total"] == 1
        assert s["best_sharpe"] == 1.2

    def test_delete_experiment(self):
        tracker = ExperimentTracker()
        exp = tracker.create_experiment(ExperimentConfig(name="Delete Me", strategy_name="s"))
        tracker.delete_experiment(exp.experiment_id)
        assert tracker.get_experiment(exp.experiment_id) is None

    def test_export_csv(self):
        tracker = ExperimentTracker()
        exp = tracker.create_experiment(ExperimentConfig(name="CSV Test", strategy_name="s"))
        tracker.complete_experiment(exp.experiment_id, {"sharpe": 1.0, "max_dd": -0.1})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        try:
            tracker.export_csv(tmp)
            assert os.path.getsize(tmp) > 0
        finally:
            os.unlink(tmp)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            tracker = ExperimentTracker(storage_path=tmp)
            config = ExperimentConfig(name="Persist", strategy_name="s")
            exp = tracker.create_experiment(config)
            tracker.complete_experiment(exp.experiment_id, {"sharpe": 0.9})

            tracker2 = ExperimentTracker(storage_path=tmp)
            assert tracker2.get_experiment(exp.experiment_id) is not None
        finally:
            os.unlink(tmp)


class TestSensitivityAnalysis:
    def test_extract_params(self):
        results = [
            ExperimentResult(
                config=ExperimentConfig(name="A", parameters={"lr": 0.001}),
                metrics={"sharpe": 1.0}, status=ExperimentStatus.COMPLETED,
            ),
            ExperimentResult(
                config=ExperimentConfig(name="B", parameters={"lr": 0.01}),
                metrics={"sharpe": 1.2}, status=ExperimentStatus.COMPLETED,
            ),
        ]
        params, metrics = sensitivity_analysis(results, "lr")
        assert len(params) == 2
        assert params[0] == 0.001
        assert params[1] == 0.01


class TestFindBestParameters:
    def test_returns_best(self):
        results = [
            ExperimentResult(
                config=ExperimentConfig(name="A", parameters={"lr": 0.001}),
                metrics={"sharpe": 1.0}, status=ExperimentStatus.COMPLETED,
            ),
            ExperimentResult(
                config=ExperimentConfig(name="B", parameters={"lr": 0.01}),
                metrics={"sharpe": 2.0}, status=ExperimentStatus.COMPLETED,
            ),
        ]
        best = find_best_parameters(results)
        assert best == {"lr": 0.01}

    def test_empty(self):
        assert find_best_parameters([]) == {}
