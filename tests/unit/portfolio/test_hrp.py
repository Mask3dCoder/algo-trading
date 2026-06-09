"""Tests for compute_hrp_weights wrapper (quantcore.portfolio.hrp)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.portfolio.hrp import compute_hrp_weights


@pytest.fixture
def strategy_returns():
    """5 strategies x 252 days of correlated returns."""
    rng = np.random.default_rng(42)
    n, m = 252, 5
    cov = np.eye(m) * 0.0001
    cov += np.ones((m, m)) * 0.00002
    data = rng.multivariate_normal(np.full(m, 0.001), cov, n)
    return pd.DataFrame(data, columns=[f"strat_{i}" for i in range(m)])


@pytest.fixture
def two_strategy_returns():
    """2 strategies x 100 days."""
    rng = np.random.default_rng(99)
    data = rng.multivariate_normal(
        [0.001, 0.0008],
        [[0.0001, 0.00003], [0.00003, 0.00009]],
        100,
    )
    return pd.DataFrame(data, columns=["alpha", "beta"])


class TestComputeHrpWeights:
    def test_weights_sum_to_one(self, strategy_returns):
        w = compute_hrp_weights(strategy_returns)
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_returns_dict_with_correct_keys(self, strategy_returns):
        w = compute_hrp_weights(strategy_returns)
        assert set(w.keys()) == set(strategy_returns.columns)

    def test_all_weights_positive(self, strategy_returns):
        w = compute_hrp_weights(strategy_returns)
        for v in w.values():
            assert v >= -1e-10

    def test_two_strategies(self, two_strategy_returns):
        w = compute_hrp_weights(two_strategy_returns)
        assert len(w) == 2
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert all(v >= -1e-10 for v in w.values())

    def test_single_strategy(self):
        df = pd.DataFrame({"only": np.random.default_rng(0).normal(0.001, 0.01, 100)})
        w = compute_hrp_weights(df)
        assert w == {"only": 1.0}

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        w = compute_hrp_weights(df)
        assert w == {}

    def test_constant_returns_fallback(self):
        """All-zero returns should still produce valid weights."""
        df = pd.DataFrame({"a": np.zeros(100), "b": np.zeros(100)})
        w = compute_hrp_weights(df)
        # Constant columns get filtered; should fallback to equal weight
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_more_strategies(self):
        """10 strategies — stress test clustering."""
        rng = np.random.default_rng(123)
        n, m = 200, 10
        cov = np.eye(m) * 0.0001
        # Block correlation: first 5 correlated, last 5 correlated
        cov[:5, :5] += 0.00005
        cov[5:, 5:] += 0.00005
        data = rng.multivariate_normal(np.full(m, 0.001), cov, n)
        df = pd.DataFrame(data, columns=[f"s{i}" for i in range(m)])
        w = compute_hrp_weights(df)
        assert len(w) == m
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_deterministic(self, strategy_returns):
        """Same input produces same output."""
        w1 = compute_hrp_weights(strategy_returns)
        w2 = compute_hrp_weights(strategy_returns)
        for k in w1:
            assert abs(w1[k] - w2[k]) < 1e-12

    def test_hrp_prefers_diversification(self):
        """HRP should give non-trivial weight to uncorrelated assets."""
        rng = np.random.default_rng(77)
        n = 300
        # Asset 0: high vol, asset 1: low vol, uncorrelated
        a = rng.normal(0.001, 0.02, n)
        b = rng.normal(0.001, 0.005, n)
        df = pd.DataFrame({"high_vol": a, "low_vol": b})
        w = compute_hrp_weights(df)
        # Both should get meaningful weight (not 100/0)
        assert w["high_vol"] > 0.01
        assert w["low_vol"] > 0.01
