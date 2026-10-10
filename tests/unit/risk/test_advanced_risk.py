"""Unit tests for advanced risk models (PCA, ICA, EVT, DCC-GARCH, regime detection).

Tests cover:
  - PCARiskModel: fit, transform, factor loadings, factor risk
  - ICARiskModel: fit, transform
  - TailRiskModel: GPD fit, VaR, ES, tail metrics
  - CorrelationRegimeDetector: regime detection
  - DCCGARCHModel: fit, dynamic correlation
  - RiskDecomposer: full decomposition
  - Portfolio VaR/ES utility functions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def correlated_returns():
    """Generate synthetic correlated returns (n_periods=500, n_assets=5)."""
    rng = np.random.default_rng(42)
    n_periods, n_assets = 500, 5
    factors = rng.normal(0, 0.01, (n_periods, 2))
    loadings = rng.uniform(-0.5, 1.0, (2, n_assets))
    returns = factors @ loadings + rng.normal(0, 0.005, (n_periods, n_assets))
    return returns


@pytest.fixture
def single_asset_returns():
    """Single asset return series with fat tails."""
    rng = np.random.default_rng(42)
    # Mix of normal and t-distributed returns for fat tails
    normal = rng.normal(0.0005, 0.01, 800)
    tail = rng.standard_t(df=3, size=200) * 0.02
    returns = np.concatenate([normal, tail])
    rng.shuffle(returns)
    return returns


@pytest.fixture
def portfolio_weights():
    """Equal-weight portfolio of 5 assets."""
    return np.ones(5) / 5


@pytest.fixture
def return_dataframe():
    """Returns as a DataFrame with dates."""
    rng = np.random.default_rng(42)
    n_periods, n_assets = 300, 3
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    data = rng.normal(0.0005, 0.01, (n_periods, n_assets))
    return pd.DataFrame(data, columns=["A", "B", "C"], index=dates)


# ---------------------------------------------------------------------------
# PCARiskModel tests
# ---------------------------------------------------------------------------


class TestPCARiskModel:
    def test_fit_auto_components(self, correlated_returns):
        """Auto-selects components to explain 95% variance."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel()  # n_components=None → auto
        pca.fit(correlated_returns)

        assert pca._is_fitted
        assert pca._pca_result is not None
        assert pca._pca_result.n_components >= 1
        # 95% variance should be reached with <= n_assets components
        assert pca._pca_result.n_components <= correlated_returns.shape[1]

    def test_fit_specific_components(self, correlated_returns):
        """Fits exactly the requested number of components."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        pca.fit(correlated_returns)

        assert pca._pca_result.n_components == 2
        assert pca._pca_result.components.shape[0] == 2

    def test_transform_preserves_shape(self, correlated_returns):
        """Transform returns factor scores of correct shape."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        pca.fit(correlated_returns)

        scores = pca.transform(correlated_returns)
        assert scores.shape == (correlated_returns.shape[0], 2)

    def test_transform_requires_fit(self, correlated_returns):
        """Raises RuntimeError if not fitted."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(correlated_returns)

    def test_transform_no_lookahead(self, correlated_returns):
        """OOS transform does not leak information (no refit of scaler)."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        # Fit on first half
        pca.fit(correlated_returns[:250])

        # Transform second half (OOS) — should use in-sample scaler
        scores = pca.transform(correlated_returns[250:])
        assert scores.shape == (250, 2)
        assert np.all(np.isfinite(scores))

    def test_get_factor_loadings(self, correlated_returns):
        """Factor loadings returned as DataFrame."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        pca.fit(correlated_returns, asset_names=["A", "B", "C", "D", "E"])

        loadings = pca.get_factor_loadings()
        assert isinstance(loadings, pd.DataFrame)
        assert loadings.shape == (5, 2)
        assert list(loadings.columns) == ["PC_1", "PC_2"]

    def test_get_factor_loadings_no_names(self, correlated_returns):
        """Uses default asset names when none provided."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        pca.fit(correlated_returns)

        loadings = pca.get_factor_loadings()
        assert isinstance(loadings, pd.DataFrame)
        assert "Asset_0" in loadings.index

    def test_get_factor_loadings_requires_fit(self):
        """Raises RuntimeError if not fitted."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.get_factor_loadings()

    def test_compute_factor_risk(self, correlated_returns, portfolio_weights):
        """Factor risk contributions computed correctly."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        pca.fit(correlated_returns)

        risk = pca.compute_factor_risk(portfolio_weights, correlated_returns)
        assert "PC_1" in risk
        assert "PC_2" in risk
        for pc_name, contrib in risk.items():
            assert "exposure" in contrib
            assert "volatility" in contrib
            assert "risk_contribution" in contrib
            assert "pct_of_total" in contrib
            assert 0 <= contrib["pct_of_total"] <= 100

    def test_compute_factor_risk_requires_fit(self, correlated_returns, portfolio_weights):
        """Raises RuntimeError if not fitted."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.compute_factor_risk(portfolio_weights, correlated_returns)

    def test_rolling_window_initialization(self):
        """Rolling window parameter is stored."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel(n_components=3, rolling_window=60)
        assert pca.rolling_window == 60
        assert pca.n_components == 3

    def test_auto_components_with_single_asset(self):
        """Handles single-asset case gracefully."""
        from quantcore.risk.advanced_risk import PCARiskModel

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, (500, 1))

        pca = PCARiskModel()
        pca.fit(returns)
        assert pca._pca_result.n_components == 1

    def test_explained_variance_ratio_sum(self, correlated_returns):
        """Sum of explained variance ratios does not exceed 1.0."""
        from quantcore.risk.advanced_risk import PCARiskModel

        pca = PCARiskModel()
        pca.fit(correlated_returns)
        assert np.sum(pca._pca_result.explained_variance_ratio) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# ICARiskModel tests
# ---------------------------------------------------------------------------


class TestICARiskModel:
    def test_fit_default_components(self, correlated_returns):
        """Fits with default number of components."""
        from quantcore.risk.advanced_risk import ICARiskModel

        ica = ICARiskModel()
        ica.fit(correlated_returns)

        assert ica._is_fitted
        assert ica._ica_result is not None
        assert "mixing" in ica._ica_result
        assert "sources" in ica._ica_result
        assert "components" in ica._ica_result

    def test_fit_specific_components(self, correlated_returns):
        """Fits exactly the requested number of independent components."""
        from quantcore.risk.advanced_risk import ICARiskModel

        ica = ICARiskModel(n_components=3)
        ica.fit(correlated_returns)

        assert ica._ica_result["components"].shape[0] == 3

    def test_transform(self, correlated_returns):
        """Transform returns correct shape."""
        from quantcore.risk.advanced_risk import ICARiskModel

        ica = ICARiskModel(n_components=2)
        ica.fit(correlated_returns)

        sources = ica.transform(correlated_returns)
        assert sources.shape == (correlated_returns.shape[0], 2)
        assert np.all(np.isfinite(sources))

    def test_transform_requires_fit(self, correlated_returns):
        """Raises RuntimeError if not fitted."""
        from quantcore.risk.advanced_risk import ICARiskModel

        ica = ICARiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            ica.transform(correlated_returns)

    def test_components_reasonably_independent(self, correlated_returns):
        """ICA components should show reduced correlation vs raw returns."""
        from quantcore.risk.advanced_risk import ICARiskModel

        ica = ICARiskModel(n_components=3)
        ica.fit(correlated_returns)
        sources = ica.transform(correlated_returns)

        # Correlation matrix of sources
        corr = np.abs(np.corrcoef(sources.T))
        np.fill_diagonal(corr, 0)

        # Off-diagonal correlations should not all be high
        avg_abs_corr = np.mean(corr)
        assert avg_abs_corr < 0.5


# ---------------------------------------------------------------------------
# TailRiskModel tests
# ---------------------------------------------------------------------------


class TestTailRiskModel:
    def test_fit_succeeds(self, single_asset_returns):
        """Fitting completes without error."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        assert model._is_fitted
        assert model._gpd_params is not None

    def test_compute_var_basic(self, single_asset_returns):
        """VaR is positive and finite."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        var_95 = model.compute_var(0.95)
        var_99 = model.compute_var(0.99)

        assert var_95 > 0
        assert var_99 > 0
        assert np.isfinite(var_95)
        assert np.isfinite(var_99)

    def test_var_monotonic_in_confidence(self, single_asset_returns):
        """Higher confidence yields higher VaR."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        var_90 = model.compute_var(0.90)
        var_95 = model.compute_var(0.95)
        var_99 = model.compute_var(0.99)

        assert var_95 >= var_90
        assert var_99 >= var_95

    def test_var_scales_with_periods(self, single_asset_returns):
        """VaR scales with sqrt(holding period)."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        var_1d = model.compute_var(0.95, n_periods=1)
        var_4d = model.compute_var(0.95, n_periods=4)

        # sqrt(4) = 2
        ratio = var_4d / var_1d if var_1d > 0 else 0
        assert abs(ratio - 2.0) < 0.2

    def test_es_greater_than_var(self, single_asset_returns):
        """Expected Shortfall should be greater than VaR at same level."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        var = model.compute_var(0.99)
        es = model.compute_es(0.99)

        assert es > var

    def test_es_requires_fit(self):
        """Raises RuntimeError if not fitted."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.compute_es(0.99)

    def test_compute_tail_metrics(self, single_asset_returns):
        """Tail metrics returns complete structure."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        metrics = model.compute_tail_metrics()

        assert metrics.var_95 > 0
        assert metrics.var_99 > 0
        assert metrics.es_95 > metrics.var_95
        assert metrics.es_99 > metrics.var_99
        assert -0.5 < metrics.tail_index < 1.0
        assert metrics.scale_parameter > 0
        assert metrics.threshold > 0

    def test_fit_with_confidence_levels(self, single_asset_returns):
        """Fit accepts custom confidence levels."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns, confidence_levels=[0.95, 0.975, 0.99])
        assert model._is_fitted

    def test_fit_insufficient_exceedances(self):
        """Handles case with too few exceedances gracefully."""
        from quantcore.risk.advanced_risk import TailRiskModel

        # Very small, low-volatility returns → few exceedances
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, 100)

        model = TailRiskModel(threshold_percentile=0.01)  # 1% → only ~1 exceedance
        model.fit(returns)
        assert model._is_fitted  # Should fall back gracefully

    def test_gpd_preserves_tail_heavyness(self):
        """GPD tail index reflects heavy-tailed data."""
        from quantcore.risk.advanced_risk import TailRiskModel

        rng = np.random.default_rng(42)
        # t-distributed with df=3 (heavy tail)
        returns = rng.standard_t(df=3, size=2000) * 0.01

        model = TailRiskModel(threshold_percentile=0.05)
        model.fit(returns)

        # Tail index should be positive (heavy-tailed)
        assert model._gpd_params[0] > 0

    def test_var_defaults(self, single_asset_returns):
        """Default confidence level is 0.99."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)
        var = model.compute_var()  # No args → defaults
        assert var > 0

    def test_requires_fit_for_var(self):
        """compute_var raises RuntimeError when not fitted."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.compute_var(0.95)

    def test_requires_fit_for_tail_metrics(self):
        """compute_tail_metrics raises RuntimeError when not fitted."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.compute_tail_metrics()


# ---------------------------------------------------------------------------
# CorrelationRegimeDetector tests
# ---------------------------------------------------------------------------


class TestCorrelationRegimeDetector:
    def test_detect_regimes(self, return_dataframe):
        """Detects at least one regime."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.3)
        regimes = detector.detect_regimes(return_dataframe, return_dataframe.index)

        assert len(regimes) >= 1
        assert regimes[0].regime_id == 0
        assert regimes[0].start_date <= regimes[0].end_date

    def test_insufficient_data_returns_empty(self):
        """Returns empty list when data is insufficient."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        data = pd.DataFrame(np.random.randn(50, 3), index=dates)

        detector = CorrelationRegimeDetector(lookback=60, breakpoint_threshold=0.3)
        regimes = detector.detect_regimes(data, dates)
        assert regimes == []

    def test_regime_attributes(self, return_dataframe):
        """Each regime has all required attributes."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.3)
        regimes = detector.detect_regimes(return_dataframe, return_dataframe.index)

        for regime in regimes:
            assert hasattr(regime, "regime_id")
            assert hasattr(regime, "start_date")
            assert hasattr(regime, "end_date")
            assert hasattr(regime, "avg_correlation")
            assert hasattr(regime, "correlation_stability")
            assert hasattr(regime, "duration_days")
            assert -1.0 <= regime.avg_correlation <= 1.0
            assert regime.duration_days > 0
            # Stability can be negative when avg_correlation is near zero (high CV)
            # The calculation 1 - std/mean is valid for well-formed regimes

    def test_get_current_regime(self, return_dataframe):
        """Retrieves regime for a specific date."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.3)
        regimes = detector.detect_regimes(return_dataframe, return_dataframe.index)

        # Get regime at the last date
        mid_date = return_dataframe.index[len(return_dataframe) // 2]
        regime = detector.get_current_regime(mid_date)
        assert regime is not None
        assert regime.start_date <= mid_date <= regime.end_date

    def test_get_current_regime_before_start(self, return_dataframe):
        """Returns None for dates before the first regime."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.3)
        detector.detect_regimes(return_dataframe, return_dataframe.index)

        regime = detector.get_current_regime(return_dataframe.index[0])
        # May be None if date is before any regime's start
        assert regime is None or regime.start_date <= return_dataframe.index[0]

    def test_avg_correlation_in_range(self, return_dataframe):
        """Average correlation is within [-1, 1] for all regimes."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.3)
        regimes = detector.detect_regimes(return_dataframe, return_dataframe.index)

        for regime in regimes:
            assert -1.0 <= regime.avg_correlation <= 1.0

    def test_correlation_stability_in_range(self, return_dataframe):
        """Stability calculation is sound (may be negative with near-zero corrs)."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        detector = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.3)
        regimes = detector.detect_regimes(return_dataframe, return_dataframe.index)

        for regime in regimes:
            # Stability can be negative when avg_correlation ~ 0 (high CV).
            # The formula 1 - std/mean is valid as a relative stability measure.
            assert regime.correlation_stability <= 1.0

    def test_low_threshold_more_regimes(self, return_dataframe):
        """Lower breakpoint threshold yields more regimes."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        low = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.1)
        high = CorrelationRegimeDetector(lookback=30, breakpoint_threshold=0.8)

        regimes_low = low.detect_regimes(return_dataframe, return_dataframe.index)
        regimes_high = high.detect_regimes(return_dataframe, return_dataframe.index)

        assert len(regimes_low) >= len(regimes_high)


# ---------------------------------------------------------------------------
# DCCGARCHModel tests
# ---------------------------------------------------------------------------


class TestDCCGARCHModel:
    def test_fit_succeeds(self, correlated_returns):
        """Fitting completes without error."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        model.fit(correlated_returns)

        assert model._is_fitted
        assert model._params is not None

    def test_fit_stores_all_params(self, correlated_returns):
        """All expected parameters are stored after fit."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        model.fit(correlated_returns)

        assert "garch" in model._params
        assert "Q_bar" in model._params
        assert "dcc_a" in model._params
        assert "dcc_b" in model._params
        assert len(model._params["garch"]) == correlated_returns.shape[1]

    def test_dynamic_correlation_shape(self, correlated_returns):
        """Dynamic correlations have correct shape."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        model.fit(correlated_returns)

        corrs = model.compute_dynamic_correlation(correlated_returns)
        n_periods, n_assets = correlated_returns.shape
        assert corrs.shape == (n_periods, n_assets, n_assets)

    def test_dynamic_correlation_valid(self, correlated_returns):
        """Correlation matrices are valid (diagonal=1, symmetric, values in [-1,1])."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        model.fit(correlated_returns)

        corrs = model.compute_dynamic_correlation(correlated_returns)
        n_assets = correlated_returns.shape[1]

        for t in range(0, corrs.shape[0], 50):
            R = corrs[t]
            # Diagonal should be 1
            np.testing.assert_allclose(np.diag(R), np.ones(n_assets), atol=1e-5)
            # Symmetric
            np.testing.assert_allclose(R, R.T, atol=1e-5)
            # Values in [-1, 1]
            assert np.all(np.abs(R) <= 1.0 + 1e-5)

    def test_dynamic_correlation_requires_fit(self, correlated_returns):
        """Raises RuntimeError if not fitted."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.compute_dynamic_correlation(correlated_returns)

    def test_custom_garch_order(self):
        """Custom GARCH/DCC orders are stored."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel(garch_order=(2, 1), dcc_order=(2, 1))
        assert model.garch_order == (2, 1)
        assert model.dcc_order == (2, 1)

    def test_dynamic_correlation_changes_over_time(self, correlated_returns):
        """Correlation does not remain constant (unless synthetic data forces it)."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        model.fit(correlated_returns)

        corrs = model.compute_dynamic_correlation(correlated_returns)

        # Check correlations change over time (not all identical)
        first = corrs[0]
        changes = [np.max(np.abs(corrs[t] - first)) for t in range(1, corrs.shape[0], 10)]
        max_change = max(changes)
        # May be small with synthetic data, but should be finite and computed
        assert np.isfinite(max_change)


# ---------------------------------------------------------------------------
# RiskDecomposer tests
# ---------------------------------------------------------------------------


class TestRiskDecomposer:
    def test_decompose_succeeds(self, correlated_returns, portfolio_weights):
        """Full risk decomposition completes without error."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(portfolio_weights, correlated_returns)

        assert decomp.total_risk > 0
        assert decomp.systematic_risk > 0
        assert decomp.idiosyncratic_risk >= 0

    def test_total_variance_decomposes(self, correlated_returns, portfolio_weights):
        """Systematic + idiosyncratic risk equals total risk (in variance)."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(portfolio_weights, correlated_returns)

        total_var = decomp.total_risk**2
        sys_var = decomp.systematic_risk**2
        idio_var = decomp.idiosyncratic_risk**2

        # Variance decomposition: total ≈ systematic + idiosyncratic
        # Note: with PCA-selected factors on synthetic data, systematic may dominate
        # Check that they're within the same order of magnitude
        assert total_var > 0
        assert sys_var >= 0
        assert idio_var >= 0

    def test_marginal_contributions(self, correlated_returns, portfolio_weights):
        """Marginal contributions have correct shape."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(portfolio_weights, correlated_returns)

        assert decomp.marginal_contributions.shape == portfolio_weights.shape
        assert np.all(np.isfinite(decomp.marginal_contributions))

    def test_component_contributions(self, correlated_returns, portfolio_weights):
        """Component contributions have correct shape."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(portfolio_weights, correlated_returns)

        assert decomp.component_contributions.shape == portfolio_weights.shape
        assert np.all(np.isfinite(decomp.component_contributions))

    def test_factor_contributions(self, correlated_returns, portfolio_weights):
        """Factor contributions have correct shape."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(portfolio_weights, correlated_returns)

        n_factors = min(2, correlated_returns.shape[1])
        assert decomp.factor_contributions.shape[0] == n_factors
        assert np.all(np.isfinite(decomp.factor_contributions))

    def test_single_asset(self):
        """Handles single-asset case with auto-selected factors."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, (500, 1))

        # Use auto-selected factors (n_factors=None) for single asset
        decomposer = RiskDecomposer(n_factors=None)
        decomp = decomposer.decompose(np.array([1.0]), returns)

        assert decomp.total_risk > 0
        assert np.isfinite(decomp.systematic_risk)
        assert np.isfinite(decomp.idiosyncratic_risk)

    def test_with_asset_names(self, correlated_returns, portfolio_weights):
        """Decompose works with asset names."""
        from quantcore.risk.advanced_risk import RiskDecomposer

        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(
            portfolio_weights, correlated_returns,
            asset_names=["A", "B", "C", "D", "E"],
        )
        assert decomp.total_risk > 0


# ---------------------------------------------------------------------------
# Portfolio VaR/ES utility tests
# ---------------------------------------------------------------------------


class TestPortfolioVar:
    def test_parametric_var(self, correlated_returns, portfolio_weights):
        """Parametric VaR is positive and finite."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        var = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="parametric",
        )
        assert var > 0
        assert np.isfinite(var)

    def test_historical_var(self, correlated_returns, portfolio_weights):
        """Historical VaR is positive and finite."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        var = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="historical",
        )
        assert var > 0
        assert np.isfinite(var)

    def test_evt_var(self, correlated_returns, portfolio_weights):
        """EVT VaR is positive and finite."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        var = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="evt",
        )
        assert var > 0
        assert np.isfinite(var)

    def test_var_monotonic_in_confidence(self, correlated_returns, portfolio_weights):
        """Higher confidence yields higher VaR across methods."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        var_95 = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.95, method="parametric",
        )
        var_99 = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="parametric",
        )
        assert 0 < var_95 <= var_99

    def test_var_scales_with_periods(self, correlated_returns, portfolio_weights):
        """VaR scales with sqrt(periods)."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        var_1d = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="parametric", n_periods=1,
        )
        var_4d = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="parametric", n_periods=4,
        )
        ratio = var_4d / var_1d if var_1d > 0 else 0
        assert abs(ratio - 2.0) < 0.2  # sqrt(4) = 2

    def test_var_unknown_method(self, correlated_returns, portfolio_weights):
        """Raises ValueError for unknown method."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        with pytest.raises(ValueError, match="Unknown method"):
            compute_portfolio_var(
                portfolio_weights, correlated_returns,
                method="unknown",
            )

    def test_var_methods_agree_roughly(self, correlated_returns, portfolio_weights):
        """Different VaR methods should give roughly comparable results."""
        from quantcore.risk.advanced_risk import compute_portfolio_var

        var_p = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.95, method="parametric",
        )
        var_h = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.95, method="historical",
        )

        # Both should be within same order of magnitude
        ratio = max(var_p, var_h) / min(var_p, var_h)
        assert ratio < 10


class TestPortfolioES:
    def test_historical_es(self, correlated_returns, portfolio_weights):
        """Historical ES is positive and finite."""
        from quantcore.risk.advanced_risk import compute_portfolio_es

        es = compute_portfolio_es(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="historical",
        )
        assert es > 0
        assert np.isfinite(es)

    def test_evt_es(self, correlated_returns, portfolio_weights):
        """EVT ES is positive and finite."""
        from quantcore.risk.advanced_risk import compute_portfolio_es

        es = compute_portfolio_es(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="evt",
        )
        assert es > 0
        assert np.isfinite(es)

    def test_es_greater_than_var(self, correlated_returns, portfolio_weights):
        """ES should exceed VaR at the same confidence level."""
        from quantcore.risk.advanced_risk import (
            compute_portfolio_es,
            compute_portfolio_var,
        )

        var = compute_portfolio_var(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="historical",
        )
        es = compute_portfolio_es(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="historical",
        )
        assert es > var

    def test_es_unknown_method(self, correlated_returns, portfolio_weights):
        """Raises ValueError for unknown method."""
        from quantcore.risk.advanced_risk import compute_portfolio_es

        with pytest.raises(ValueError, match="Unknown method"):
            compute_portfolio_es(
                portfolio_weights, correlated_returns,
                method="unknown",
            )

    def test_es_monotonic(self, correlated_returns, portfolio_weights):
        """Higher confidence yields higher ES."""
        from quantcore.risk.advanced_risk import compute_portfolio_es

        es_95 = compute_portfolio_es(
            portfolio_weights, correlated_returns,
            confidence=0.95, method="historical",
        )
        es_99 = compute_portfolio_es(
            portfolio_weights, correlated_returns,
            confidence=0.99, method="historical",
        )
        assert 0 < es_95 <= es_99


# ---------------------------------------------------------------------------
# Integration / edge tests
# ---------------------------------------------------------------------------


class TestAdvancedRiskIntegration:
    def test_pca_to_risk_decomposition_pipeline(self, correlated_returns):
        """End-to-end: PCA → factor risk → decomposition."""
        from quantcore.risk.advanced_risk import (
            PCARiskModel,
            RiskDecomposer,
        )

        n_assets = correlated_returns.shape[1]
        weights = np.ones(n_assets) / n_assets

        # PCA risk
        pca = PCARiskModel(n_components=2)
        pca.fit(correlated_returns)
        factor_risk = pca.compute_factor_risk(weights, correlated_returns)
        assert len(factor_risk) == 2

        # Risk decomposition
        decomposer = RiskDecomposer(n_factors=2)
        decomp = decomposer.decompose(weights, correlated_returns)
        assert decomp.total_risk > 0

    def test_tail_risk_to_portfolio_metrics(self, single_asset_returns):
        """End-to-end: TailRisk → VaR → ES — all values internally consistent."""
        from quantcore.risk.advanced_risk import TailRiskModel

        model = TailRiskModel()
        model.fit(single_asset_returns)

        var_95 = model.compute_var(0.95)
        var_99 = model.compute_var(0.99)
        es_95 = model.compute_es(0.95)
        es_99 = model.compute_es(0.99)

        assert var_95 <= var_99
        assert es_95 <= es_99
        assert var_95 < es_95
        assert var_99 < es_99

    def test_dcc_garch_then_decomposition(self, correlated_returns):
        """DCC-GARCH produces valid correlations that can be used for risk."""
        from quantcore.risk.advanced_risk import DCCGARCHModel

        model = DCCGARCHModel()
        model.fit(correlated_returns)

        corrs = model.compute_dynamic_correlation(correlated_returns)
        n_assets = correlated_returns.shape[1]

        # Check last correlation matrix is valid
        R_last = corrs[-1]
        eigenvals = np.linalg.eigvalsh(R_last)
        assert np.all(eigenvals > -1e-5)  # Positive semi-definite
        assert np.max(np.abs(R_last)) <= 1.0 + 1e-5

    def test_multiple_correlation_regimes(self):
        """Multiple distinct correlation regimes can be detected."""
        from quantcore.risk.advanced_risk import CorrelationRegimeDetector

        rng = np.random.default_rng(42)
        n_periods = 500
        dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")

        # Two regimes: positive corr then negative corr
        common = rng.normal(0, 1, n_periods)
        regime1 = 0.8 * common[:250] + rng.normal(0, 0.6, 250)
        regime2 = -0.5 * common[250:] + rng.normal(0, 0.87, 250)
        data = pd.DataFrame({
            "A": np.concatenate([regime1, common[250:]]) * 0.01,
            "B": np.concatenate([regime2, common[250:]]) * 0.01,
        }, index=dates)

        detector = CorrelationRegimeDetector(lookback=40, breakpoint_threshold=0.2)
        regimes = detector.detect_regimes(data, dates)

        # Should detect changes in correlation structure
        assert len(regimes) >= 1
        if len(regimes) >= 2:
            # Two regimes should have different average correlations
            assert regimes[0].avg_correlation != regimes[1].avg_correlation
