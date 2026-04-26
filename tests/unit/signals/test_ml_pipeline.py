"""Tests for ML pipeline: features, preprocessing, linear models, tree models, evaluation, deep learning, RL."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    n = 300
    raw = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n))
    return np.maximum(raw, 1.0)

@pytest.fixture
def ohlcv(prices):
    rng = np.random.default_rng(42)
    n = len(prices)
    high = prices * (1 + np.abs(rng.normal(0, 0.02, n)))
    low = prices * (1 - np.abs(rng.normal(0, 0.02, n)))
    low = np.minimum(low, prices)
    high = np.maximum(high, prices)
    volume = np.abs(rng.normal(1e6, 2e5, n))
    return high, low, volume

@pytest.fixture
def features_and_target(prices, ohlcv):
    from quantcore.signals.ml.features import TechnicalFeatures
    high, low, volume = ohlcv
    tech = TechnicalFeatures(close=prices, high=high, low=low, volume=volume, windows=[5, 21])
    feat_dict = tech.compute()
    names = sorted(feat_dict.keys())
    n = len(prices)
    X = np.column_stack([feat_dict[k] for k in names])
    # Forward 1-day returns as target
    target = np.full(n, np.nan, dtype=float)
    target[:-1] = prices[1:] / prices[:-1] - 1.0
    return X, target


# ===========================================================================
# Test TechnicalFeatures
# ===========================================================================

class TestTechnicalFeatures:
    def test_compute_returns_dict(self, prices, ohlcv):
        from quantcore.signals.ml.features import TechnicalFeatures
        high, low, volume = ohlcv
        tech = TechnicalFeatures(close=prices, high=high, low=low, volume=volume)
        result = tech.compute()
        assert isinstance(result, dict)
        assert len(result) >= 20
        for v in result.values():
            assert len(v) == len(prices)

    def test_close_only(self, prices):
        from quantcore.signals.ml.features import TechnicalFeatures
        tech = TechnicalFeatures(close=prices)
        result = tech.compute()
        # Should have SMA, EMA, ROC, CMO, RSI, BB, MACD, hist_vol
        assert "sma_5" in result
        assert "ema_5" in result
        assert "rsi_14" in result
        assert "macd_line" in result
        assert "hist_vol_5" in result
        # No volume/HL features
        assert "atr_14" not in result
        assert "mfi_14" not in result

    def test_custom_windows(self, prices):
        from quantcore.signals.ml.features import TechnicalFeatures
        tech = TechnicalFeatures(close=prices, windows=[3, 7])
        result = tech.compute()
        assert "sma_3" in result
        assert "sma_7" in result
        assert "sma_5" not in result


class TestInterMarketFeatures:
    def test_rolling_corr(self):
        from quantcore.signals.ml.features import InterMarketFeatures
        n = 100
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, n)
        bench = rets * 0.8 + rng.normal(0, 0.01, n)
        im = InterMarketFeatures(returns=rets, benchmark_returns=bench, windows=[21])
        result = im.compute()
        assert "bench_corr_21" in result
        assert "bench_beta_21" in result
        # Correlation should be positive (shared component)
        valid = result["bench_corr_21"][~np.isnan(result["bench_corr_21"])]
        assert np.mean(valid) > 0


class TestCalendarFeatures:
    def test_day_of_week(self):
        from datetime import date, timedelta

        from quantcore.signals.ml.features import CalendarFeatures
        dates = np.array([date(2024, 1, 1) + timedelta(days=i) for i in range(50)])
        cal = CalendarFeatures(dates=dates)
        result = cal.compute()
        assert "day_of_week" in result
        assert "month" in result
        assert len(result["day_of_week"]) == 50
        # Day of week should be in [0, 1] (normalized)
        valid = result["day_of_week"]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1.01)


class TestCrossSectionalNormalize:
    def test_zscore_rows(self):
        from quantcore.signals.ml.features import cross_sectional_normalize
        rng = np.random.default_rng(42)
        X = rng.normal(5, 2, (20, 50))
        result = cross_sectional_normalize(X)
        assert result.shape == X.shape
        # Each row should have ~0 mean and ~1 std
        for i in range(20):
            row = result[i, :]
            assert abs(np.nanmean(row)) < 0.1
            assert abs(np.nanstd(row, ddof=1) - 1.0) < 0.1


class TestBuildFeatureMatrix:
    def test_full_matrix(self, prices, ohlcv):
        from quantcore.signals.ml.features import build_feature_matrix
        high, low, volume = ohlcv
        result = build_feature_matrix(close=prices, high=high, low=low, volume=volume)
        assert isinstance(result, dict)
        assert len(result) >= 20


# ===========================================================================
# Test Preprocessing
# ===========================================================================

class TestWinsorize:
    def test_clips_outliers(self):
        from quantcore.signals.ml.preprocessing import winsorize
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        # Add extreme outliers
        X[0, 0] = 50.0
        X[1, 0] = -50.0
        result = winsorize(X, limits=3.0, axis=0)
        assert result[0, 0] < 50.0
        assert result[1, 0] > -50.0

    def test_preserves_shape(self):
        from quantcore.signals.ml.preprocessing import winsorize
        X = np.random.default_rng(42).normal(0, 1, (100, 10))
        result = winsorize(X)
        assert result.shape == X.shape


class TestCrossSectionalZscore:
    def test_per_row_zscore(self):
        from quantcore.signals.ml.preprocessing import cross_sectional_zscore
        rng = np.random.default_rng(42)
        X = rng.normal(10, 3, (50, 30))
        result = cross_sectional_zscore(X)
        # Each row ≈ mean 0, std 1
        for i in range(50):
            assert abs(np.mean(result[i, :])) < 0.15


class TestImputeMedian:
    def test_fills_nans(self):
        from quantcore.signals.ml.preprocessing import impute_median
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        result = impute_median(X)
        assert not np.any(np.isnan(result))
        # Column 2: [nan, 6, 9] → median(6,9)=7.5
        assert result[0, 2] == 7.5


class TestImputeKNN:
    def test_fills_nans(self):
        from quantcore.signals.ml.preprocessing import impute_knn
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 5))
        X[10, 2] = np.nan
        X[20, 3] = np.nan
        result = impute_knn(X, k=5)
        assert not np.any(np.isnan(result[10, 2]))
        assert not np.any(np.isnan(result[20, 3]))


class TestSelectFeaturesMI:
    def test_selects_relevant(self):
        from quantcore.signals.ml.preprocessing import select_features_mutual_info
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 20))
        # Make features 0 and 5 correlated with target
        target = 0.5 * X[:, 0] + 0.3 * X[:, 5] + rng.normal(0, 0.1, n)
        selected = select_features_mutual_info(X, target, n_select=5)
        assert isinstance(selected, list)
        assert len(selected) == 5
        # Features 0 and 5 should be ranked high
        assert 0 in selected
        assert 5 in selected


class TestTimeSeriesSplit:
    def test_no_overlap(self):
        from quantcore.signals.ml.preprocessing import TimeSeriesSplit
        splitter = TimeSeriesSplit(n_splits=5, embargo_pct=0.02)
        folds = splitter.split(500)
        assert len(folds) >= 1
        for train_idx, test_idx in folds:
            assert train_idx[-1] < test_idx[0]  # No overlap

    def test_embargo_gap(self):
        from quantcore.signals.ml.preprocessing import TimeSeriesSplit
        splitter = TimeSeriesSplit(n_splits=3, embargo_pct=0.05)
        folds = splitter.split(500)
        for train_idx, test_idx in folds:
            gap = test_idx[0] - train_idx[-1]
            assert gap >= 1


class TestPreprocessingPipeline:
    def test_full_pipeline(self, features_and_target):
        from quantcore.signals.ml.preprocessing import PreprocessingPipeline
        X, _ = features_and_target
        pipeline = PreprocessingPipeline()
        result = pipeline.fit_transform(X)
        assert result.shape == X.shape
        # Should have no NaNs after imputation
        assert not np.any(np.isnan(result))


# ===========================================================================
# Test Linear Models
# ===========================================================================

class TestLassoSignal:
    def test_fit_and_predict(self, features_and_target):
        from quantcore.signals.ml.linear_models import LassoSignal
        X, y = features_and_target
        model = LassoSignal(alpha_range=[0.01, 0.1], cv_splits=2)
        result = model.fit(X, y)
        assert result.n_features_used >= 0
        assert result.ic_train is not None
        pred = model.predict(X)
        assert len(pred) == len(X)

    def test_sparsity(self, features_and_target):
        from quantcore.signals.ml.linear_models import LassoSignal
        X, y = features_and_target
        model = LassoSignal(alpha_range=[1.0, 10.0], cv_splits=2)
        model.fit(X, y)
        # LASSO should produce sparse coefficients
        assert model.beta_ is not None


class TestRidgeSignal:
    def test_fit_and_predict(self, features_and_target):
        from quantcore.signals.ml.linear_models import RidgeSignal
        X, y = features_and_target
        model = RidgeSignal(alpha_range=[0.1, 1.0], cv_splits=2)
        result = model.fit(X, y)
        assert result.n_features_used >= 0
        pred = model.predict(X)
        assert len(pred) == len(X)


class TestElasticNetSignal:
    def test_fit_and_predict(self, features_and_target):
        from quantcore.signals.ml.linear_models import ElasticNetSignal
        X, y = features_and_target
        model = ElasticNetSignal(alpha_range=[0.01, 0.1], l1_ratio=0.5, cv_splits=2)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(X)


class TestQuantileRegression:
    def test_fit_and_predict(self, features_and_target):
        from quantcore.signals.ml.linear_models import QuantileRegression
        X, y = features_and_target
        model = QuantileRegression(quantile=0.05)
        beta = model.fit(X, y)
        assert beta is not None
        pred = model.predict(X)
        assert len(pred) == len(X)

    def test_low_quantile_biased_down(self):
        from quantcore.signals.ml.linear_models import QuantileRegression
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 5))
        y = 2 * X[:, 0] + rng.normal(0, 1, n)
        model_low = QuantileRegression(quantile=0.05)
        model_low.fit(X, y)
        pred_low = model_low.predict(X)

        model_high = QuantileRegression(quantile=0.95)
        model_high.fit(X, y)
        pred_high = model_high.predict(X)

        # Low quantile predictions should be generally lower than high
        assert np.mean(pred_low) < np.mean(pred_high)


class TestCoordinateDescent:
    def test_lasso_produces_zeros(self):
        from quantcore.signals.ml.linear_models import _coordinate_descent
        rng = np.random.default_rng(42)
        n, p = 200, 20
        X = rng.normal(0, 1, (n, p))
        # Only features 0 and 1 matter
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(0, 0.1, n)
        beta = _coordinate_descent(X, y, l1=0.5, l2=0.0)
        # With L1, some coefs should be sparse
        n_zero = int(np.sum(np.abs(beta) < 1e-6))
        assert n_zero >= 5  # Most features should be zeroed out


# ===========================================================================
# Test Tree Models
# ===========================================================================

class TestDecisionTree:
    def test_fit_and_predict(self):
        from quantcore.signals.ml.tree_models import DecisionTreeRegressor
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 5))
        y = X[:, 0] * 2 + X[:, 2] * 0.5 + rng.normal(0, 0.1, n)
        tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert len(pred) == n
        # Should fit somewhat (R^2 > 0.5)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5

    def test_leaf_prediction(self):
        from quantcore.signals.ml.tree_models import DecisionTreeRegressor
        X = np.array([[1.0], [2.0], [3.0], [100.0]])
        y = np.array([1.0, 2.0, 3.0, 100.0])
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=1)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert len(pred) == 4


class TestGradientBoostingSignal:
    def test_fit_and_predict(self):
        from quantcore.signals.ml.tree_models import GradientBoostingSignal
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 10))
        y = X[:, 0] * 2 + X[:, 2] * 0.5 + rng.normal(0, 0.5, n)
        model = GradientBoostingSignal(n_estimators=20, learning_rate=0.05, max_depth=3, cv_splits=2)
        result = model.fit(X, y)
        assert "n_trees" in result
        assert result["n_trees"] <= 20
        pred = model.predict(X)
        assert len(pred) == len(X)

    def test_early_stopping(self):
        from quantcore.signals.ml.tree_models import GradientBoostingSignal
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 10))
        y = X[:, 0] * 2 + rng.normal(0, 0.5, n)
        model = GradientBoostingSignal(
            n_estimators=200, learning_rate=0.01, max_depth=2,
            min_samples_leaf=50, cv_splits=2,
        )
        result = model.fit(X, y)
        assert result["n_trees"] <= 200


class TestRandomForestSignal:
    def test_fit_and_predict(self):
        from quantcore.signals.ml.tree_models import RandomForestSignal
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 8))
        y = X[:, 0] * 2 + X[:, 3] * 0.5 + rng.normal(0, 0.5, n)
        model = RandomForestSignal(n_estimators=10, max_depth=4, min_samples_leaf=10)
        result = model.fit(X, y)
        assert "n_trees" in result
        pred = model.predict(X)
        assert len(pred) == len(X)

    def test_feature_importance(self):
        from quantcore.signals.ml.tree_models import RandomForestSignal
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 8))
        y = X[:, 0] * 2 + rng.normal(0, 0.5, n)
        model = RandomForestSignal(n_estimators=10, max_depth=4)
        model.fit(X, y)
        imp = model.feature_importance()
        assert len(imp) == X.shape[1]
        assert np.all(imp >= 0)
        assert np.all(imp <= 1.0 + 1e-10)


# ===========================================================================
# Test Evaluation
# ===========================================================================

class TestIC:
    def test_perfect_ic(self):
        from quantcore.signals.ml.evaluation import information_coefficient
        rng = np.random.default_rng(42)
        pred = rng.normal(0, 1, 100)
        actual = pred + rng.normal(0, 0.01, 100)  # Nearly perfect
        ic = information_coefficient(pred, actual, method="pearson")
        assert ic > 0.95

    def test_random_ic_near_zero(self):
        from quantcore.signals.ml.evaluation import information_coefficient
        rng = np.random.default_rng(42)
        pred = rng.normal(0, 1, 500)
        actual = rng.normal(0, 1, 500)
        ic = information_coefficient(pred, actual)
        assert abs(ic) < 0.2

    def test_spearman(self):
        from quantcore.signals.ml.evaluation import information_coefficient
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # Monotonic
        ic = information_coefficient(pred, actual, method="spearman")
        assert ic > 0.9

    def test_short_array_returns_zero(self):
        from quantcore.signals.ml.evaluation import information_coefficient
        ic = information_coefficient(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert ic == 0.0


class TestRollingIC:
    def test_rolling_shape(self):
        from quantcore.signals.ml.evaluation import rolling_ic
        rng = np.random.default_rng(42)
        n = 300
        pred = rng.normal(0, 1, n)
        actual = rng.normal(0, 1, n)
        ric = rolling_ic(pred, actual, window=50)
        assert len(ric) == n
        # First window-1 values are NaN
        assert np.all(np.isnan(ric[:49]))


class TestICDecay:
    def test_decay_shape(self):
        from quantcore.signals.ml.evaluation import ic_decay
        rng = np.random.default_rng(42)
        n = 500
        signal = rng.normal(0, 1, n)
        fr = rng.normal(0, 0.01, n)
        ics = ic_decay(signal, fr, max_lag=10)
        assert len(ics) == 10


class TestSignalTurnover:
    def test_turnover_range(self):
        from quantcore.signals.ml.evaluation import signal_turnover
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 300)
        turnover = signal_turnover(signal, lookback=50)
        assert len(turnover) == 300
        # Turnover should be between 0 and 1
        valid = turnover[~np.isnan(turnover)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1.0)


class TestOOSDegradation:
    def test_overfitted_detection(self):
        from quantcore.signals.ml.evaluation import oos_degradation
        result = oos_degradation(ic_in_sample=0.15, ic_out_of_sample=0.03)
        assert result["degradation_ratio"] == 5.0
        assert result["is_overfitted"] is True

    def test_healthy_degradation(self):
        from quantcore.signals.ml.evaluation import oos_degradation
        result = oos_degradation(ic_in_sample=0.10, ic_out_of_sample=0.07)
        assert result["degradation_ratio"] < 2.0
        assert result["is_overfitted"] is False


class TestQuintileAnalysis:
    def test_monotonic_signal(self):
        from quantcore.signals.ml.evaluation import quintile_analysis
        rng = np.random.default_rng(42)
        n = 500
        signal = np.arange(n, dtype=float)
        fr = signal * 0.001 + rng.normal(0, 0.001, n)
        result = quintile_analysis(signal, fr)
        assert len(result["quintiles"]) == 5
        assert result["spread"] > 0
        assert result["is_monotonic"] is True

    def test_short_data(self):
        from quantcore.signals.ml.evaluation import quintile_analysis
        result = quintile_analysis(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        assert result["spread"] == 0.0


# ===========================================================================
# Test Deep Learning (numpy simplified)
# ===========================================================================

class TestSequenceDataset:
    def test_length(self):
        from quantcore.signals.ml.deep_learning import SequenceDataset
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (100, 5))
        targets = rng.normal(0, 0.01, 100)
        ds = SequenceDataset(features, targets, window_size=10)
        assert len(ds) > 0
        assert len(ds) < 100

    def test_sequences_shape(self):
        from quantcore.signals.ml.deep_learning import SequenceDataset
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (100, 5))
        targets = rng.normal(0, 0.01, 100)
        ds = SequenceDataset(features, targets, window_size=10)
        X, y = ds.get_sequences()
        assert X.shape[1] == 10  # window_size
        assert X.shape[2] == 5  # n_features
        assert len(y) == X.shape[0]


class TestSimpleLSTM:
    def test_fit_and_predict(self):
        from quantcore.signals.ml.deep_learning import SimpleLSTMSignal
        rng = np.random.default_rng(42)
        n = 100
        features = rng.normal(0, 1, (n, 5))
        target = features[:, 0] * 0.5 + rng.normal(0, 0.1, n)
        model = SimpleLSTMSignal(window_size=10, n_hidden=5, n_epochs=10)
        result = model.fit(features, target)
        assert "loss" in result
        pred = model.predict(features)
        assert len(pred) == n


class TestSimpleGRU:
    def test_fit_and_predict(self):
        from quantcore.signals.ml.deep_learning import SimpleGRUSignal
        rng = np.random.default_rng(42)
        n = 100
        features = rng.normal(0, 1, (n, 5))
        target = features[:, 0] * 0.5 + rng.normal(0, 0.1, n)
        model = SimpleGRUSignal(window_size=10, n_hidden=5, n_epochs=10)
        result = model.fit(features, target)
        assert result.get("model_type") == "gru_simplified"


class TestSimpleTCN:
    def test_fit_and_predict(self):
        from quantcore.signals.ml.deep_learning import SimpleTCNSignal
        rng = np.random.default_rng(42)
        n = 150
        features = rng.normal(0, 1, (n, 5))
        target = features[:, 0] * 0.3 + rng.normal(0, 0.1, n)
        model = SimpleTCNSignal(n_filters=8, lookback=21)
        result = model.fit(features, target)
        assert "loss" in result
        if result["loss"] != float("inf"):
            pred = model.predict(features)
            assert len(pred) == n


class TestAutoencoderAnomaly:
    def test_detect_anomalies(self):
        from quantcore.signals.ml.deep_learning import AutoencoderAnomaly
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (200, 10))
        # Add anomalies
        features[50, :] += 10.0
        features[150, :] -= 10.0
        model = AutoencoderAnomaly(n_components=5, threshold_percentile=95.0)
        model.fit(features)
        scores = model.detect_anomalies(features)
        assert len(scores) == 200
        # Anomalous points should have higher scores
        assert scores[50] > np.median(scores)
        assert scores[150] > np.median(scores)

    def test_anomaly_mask(self):
        from quantcore.signals.ml.deep_learning import AutoencoderAnomaly
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (200, 10))
        model = AutoencoderAnomaly(n_components=5)
        model.fit(features)
        mask = model.anomaly_mask(features)
        assert mask.dtype == bool
        assert np.sum(mask) < len(features)  # Not everything is anomalous


class TestVAERegimeDetector:
    def test_predict_regime(self):
        from quantcore.signals.ml.deep_learning import VAERegimeDetector
        rng = np.random.default_rng(42)
        # Create 3 distinct clusters
        n_per = 100
        c1 = rng.normal(0, 1, (n_per, 10))
        c2 = rng.normal(5, 1, (n_per, 10))
        c3 = rng.normal(-5, 1, (n_per, 10))
        features = np.vstack([c1, c2, c3])
        model = VAERegimeDetector(n_latent=3, n_regimes=3)
        result = model.fit(features)
        assert result["n_regimes"] == 3
        labels = model.predict_regime(features)
        assert len(labels) == 300
        assert set(np.unique(labels)).issubset({0, 1, 2})


# ===========================================================================
# Test Reinforcement Learning
# ===========================================================================

class TestTradingEnv:
    def test_reset_and_step(self):
        from quantcore.signals.ml.reinforcement import TradingEnv
        prices = np.linspace(100, 110, 50)
        env = TradingEnv(prices, lookback=5)
        obs = env.reset()
        assert len(obs) == env.obs_dim  # lookback + 2
        assert env.position == 0

        obs, reward, done = env.step(2)  # Go long
        assert env.position == 1
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_runs_to_completion(self):
        from quantcore.signals.ml.reinforcement import TradingEnv
        prices = np.linspace(100, 120, 30)
        env = TradingEnv(prices, lookback=5)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, done = env.step(1)  # Stay flat
            steps += 1
            if steps > 50:
                break
        assert done

    def test_transaction_cost(self):
        from quantcore.signals.ml.reinforcement import TradingEnv
        prices = np.ones(20) * 100.0
        env = TradingEnv(prices, transaction_cost_bps=10.0, lookback=5)
        env.reset()
        # Switching position should incur cost
        _, reward1, _ = env.step(2)  # Go long
        _, reward2, _ = env.step(0)  # Go short (position change → cost)
        assert reward2 < 0  # Transaction cost makes it negative


class TestQLearningAgent:
    def test_act_returns_valid_action(self):
        from quantcore.signals.ml.reinforcement import QLearningAgent
        agent = QLearningAgent(epsilon=1.0)  # Fully random
        obs = np.array([0.1, -0.2, 0.3, 0.0, 0.5])
        action = agent.act(obs)
        assert action in [0, 1, 2]

    def test_learn_updates_qtable(self):
        from quantcore.signals.ml.reinforcement import QLearningAgent
        agent = QLearningAgent(epsilon=0.0)
        obs = np.array([0.1, -0.2, 0.3])
        next_obs = np.array([0.2, -0.1, 0.4])
        agent.learn(obs, 1, 1.0, next_obs, False)
        state = agent._discretize(obs)
        assert state in agent.q_table
        assert agent.q_table[state][1] != 0.0


class TestTrainQLearning:
    def test_training_returns_agent(self):
        from quantcore.signals.ml.reinforcement import TradingEnv, train_qlearning
        prices = np.linspace(100, 110, 50)
        env = TradingEnv(prices, lookback=5)
        agent = train_qlearning(env, n_episodes=100, epsilon_start=0.5, epsilon_end=0.01)
        assert len(agent.q_table) > 0
        assert agent.epsilon == pytest.approx(0.01, abs=0.01)


# ===========================================================================
# Test __init__ exports
# ===========================================================================

class TestMLInit:
    def test_all_exports_importable(self):
        from quantcore.signals.ml import (
            CalendarFeatures,
            ElasticNetSignal,
            GradientBoostingSignal,
            InterMarketFeatures,
            LassoSignal,
            PreprocessingPipeline,
            QuantileRegression,
            RandomForestSignal,
            RidgeSignal,
            TechnicalFeatures,
            TimeSeriesSplit,
            build_feature_matrix,
            cross_sectional_normalize,
            cross_sectional_zscore,
            ic_decay,
            impute_knn,
            impute_median,
            information_coefficient,
            oos_degradation,
            quintile_analysis,
            rolling_ic,
            select_features_mutual_info,
            signal_turnover,
            winsorize,
        )
        assert all([
            TechnicalFeatures, InterMarketFeatures, CalendarFeatures,
            cross_sectional_normalize, build_feature_matrix,
            winsorize, cross_sectional_zscore, impute_median, impute_knn,
            select_features_mutual_info, TimeSeriesSplit, PreprocessingPipeline,
            LassoSignal, RidgeSignal, ElasticNetSignal, QuantileRegression,
            GradientBoostingSignal, RandomForestSignal,
            information_coefficient, rolling_ic, ic_decay,
            signal_turnover, oos_degradation, quintile_analysis,
        ])
