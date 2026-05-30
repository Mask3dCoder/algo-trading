"""Unit tests for HMM regime filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestBuildRegimeFeatures:
    """Tests for build_regime_features function."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with correct columns."""
        from quantcore.regime.hmm_regime_filter import build_regime_features

        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        vix = pd.Series(20 + np.random.randn(500) * 5, index=dates).clip(10, 80)
        spy = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)

        features = build_regime_features(vix, spy)

        assert isinstance(features, pd.DataFrame)
        assert "vix_level" in features.columns
        assert "vix_change" in features.columns
        assert "spy_rvol" in features.columns

    def test_no_nans_after_warmup(self):
        """No NaN values after warmup period."""
        from quantcore.regime.hmm_regime_filter import build_regime_features

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        vix = pd.Series(20 + np.random.randn(500) * 5, index=dates).clip(10, 80)
        spy = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)

        features = build_regime_features(vix, spy)

        # After warmup (60 rows), no NaN
        assert not features.iloc[60:].isna().any().any()

    def test_with_optional_data(self):
        """Works with HYG and TLT data."""
        from quantcore.regime.hmm_regime_filter import build_regime_features

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        vix = pd.Series(20 + np.random.randn(500) * 5, index=dates).clip(10, 80)
        spy = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)
        hyg = pd.Series(np.cumsum(np.random.randn(500) * 0.005) + 80, index=dates)
        tlt = pd.Series(np.cumsum(np.random.randn(500) * 0.003) + 100, index=dates)

        features = build_regime_features(vix, spy, hyg, tlt)

        assert "credit_proxy" in features.columns
        assert "yc_proxy" in features.columns
        assert len(features) > 100


class TestHMMRegimeFilter:
    """Tests for HMMRegimeFilter class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing (small for speed)."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2018-01-01", periods=n, freq="B")

        # VIX with regime changes
        vix = np.zeros(n)
        vix[:150] = 15 + np.random.randn(150) * 3  # Low vol
        vix[150:300] = 25 + np.random.randn(150) * 5  # Moderate
        vix[300:] = 35 + np.random.randn(n - 300) * 10  # High vol
        vix = pd.Series(np.clip(vix, 10, 80), index=dates)

        # SPY
        spy = pd.Series(np.cumsum(np.random.randn(n) * 0.01) + 100, index=dates)

        return vix, spy, dates

    def test_train_succeeds(self, sample_data):
        """Training completes without error."""
        from quantcore.regime.hmm_regime_filter import HMMRegimeFilter, RegimeFilterConfig

        vix, spy, dates = sample_data
        config = RegimeFilterConfig(n_restarts=2, min_train_samples=200)
        filt = HMMRegimeFilter(config)
        filt.train(vix, spy)

        assert filt.is_trained
        assert filt.n_states == 3

    def test_state_labels(self, sample_data):
        """State labels are assigned correctly."""
        from quantcore.regime.hmm_regime_filter import HMMRegimeFilter, RegimeFilterConfig

        vix, spy, dates = sample_data
        config = RegimeFilterConfig(n_restarts=2, min_train_samples=200)
        filt = HMMRegimeFilter(config)
        filt.train(vix, spy)

        labels = filt.get_state_labels()
        assert len(labels) == 3

        label_names = [sl.label for sl in labels]
        assert "low_vol_bull" in label_names
        assert "crisis" in label_names

    def test_allocation_scalar_range(self, sample_data):
        """Allocation scalars are in [0, 1]."""
        from quantcore.regime.hmm_regime_filter import HMMRegimeFilter, RegimeFilterConfig

        vix, spy, dates = sample_data
        config = RegimeFilterConfig(n_restarts=2, min_train_samples=200)
        filt = HMMRegimeFilter(config)
        filt.train(vix, spy)

        for date in dates[200:210]:
            scalar = filt.allocation_scalar(date)
            assert 0.0 <= scalar <= 1.0

    def test_state_probabilities_sum_to_one(self, sample_data):
        """State probabilities sum to approximately 1.0."""
        from quantcore.regime.hmm_regime_filter import HMMRegimeFilter, RegimeFilterConfig

        vix, spy, dates = sample_data
        config = RegimeFilterConfig(n_restarts=2, min_train_samples=200)
        filt = HMMRegimeFilter(config)
        filt.train(vix, spy)

        probs = filt.state_probabilities(dates[300])
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


class TestRegimeFilterConfig:
    """Tests for RegimeFilterConfig."""

    def test_default_config(self):
        """Default config has correct values."""
        from quantcore.regime.hmm_regime_filter import RegimeFilterConfig

        cfg = RegimeFilterConfig()
        assert cfg.n_states == 3
        assert cfg.allocation_scalars == [1.0, 0.5, 0.0]
        assert cfg.min_train_samples == 252
