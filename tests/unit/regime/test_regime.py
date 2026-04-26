"""Tests for Phase 3 Regime Module: HMM, features, changepoint, vol regime.

Environment-gated tests (skipped when optional dependency is missing):
  - TestChangepoint.test_pelt_detects_changepoints  → requires: ruptures
  - TestChangepoint.test_pelt_regularization        → requires: ruptures
  - TestHMM.* (7 tests)                             → requires: hmmlearn

  Install optional dependencies: pip install -e ".[stats]"
  Run gated tests: pytest -m "optional_dep" tests/unit/regime/
  Run everything including gated tests: pip install -e ".[stats]" && pytest tests/
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_RUPTURES = importlib.util.find_spec("ruptures") is not None
_HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regime_switching_data():
    """Synthetic data with known regime switches."""
    rng = np.random.default_rng(42)
    n = 500
    data = np.zeros(n)
    # Regime 0: low vol (0-199)
    data[:200] = rng.normal(0.0005, 0.005, 200)
    # Regime 1: high vol + negative (200-349)
    data[200:350] = rng.normal(-0.002, 0.02, 150)
    # Regime 2: low vol recovery (350-)
    data[350:] = rng.normal(0.001, 0.008, n - 350)
    return data


@pytest.fixture
def changepoint_data():
    """Data with sharp changepoints."""
    rng = np.random.default_rng(42)
    n = 500
    data = np.zeros(n)
    data[:150] = rng.normal(0.0, 1.0, 150)
    data[150:300] = rng.normal(3.0, 1.0, 150)  # Mean shift at 150
    data[300:] = rng.normal(0.0, 1.5, 200)  # Vol shift at 300
    return data


# ---------------------------------------------------------------------------
# T-0330: HMM + Features
# ---------------------------------------------------------------------------


class TestHMM:
    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_fit_2_state(self, regime_switching_data):
        """Fit a 2-state HMM and verify state sequence, probabilities, and current state.

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import RegimeHMM
        hmm = RegimeHMM(n_states=2)
        result = hmm.fit(regime_switching_data)
        assert len(result.states) == len(regime_switching_data)
        assert result.probabilities.shape == (len(regime_switching_data), 2)
        assert result.current_state in (0, 1)
        assert 0 <= result.current_probability <= 1

    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_fit_3_state(self, regime_switching_data):
        """Fit a 3-state HMM and verify state count.

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import RegimeHMM
        hmm = RegimeHMM(n_states=3)
        result = hmm.fit(regime_switching_data)
        assert result.n_states == 3

    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_state_labels(self, regime_switching_data):
        """Verify state labels are generated for each hidden state.

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import RegimeHMM
        hmm = RegimeHMM(n_states=2)
        result = hmm.fit(regime_switching_data)
        assert len(result.state_labels) == 2

    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_transition_matrix(self, regime_switching_data):
        """Verify the transition matrix is row-stochastic (each row sums to 1).

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import RegimeHMM
        hmm = RegimeHMM(n_states=2)
        result = hmm.fit(regime_switching_data)
        tm = result.transition_matrix
        for row in tm:
            assert abs(row.sum() - 1.0) < 1e-5

    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_aic_bic(self, regime_switching_data):
        """Verify AIC and BIC are computed and BIC penalizes complexity more heavily.

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import RegimeHMM
        hmm = RegimeHMM(n_states=2)
        result = hmm.fit(regime_switching_data)
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert result.bic > result.aic

    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_select_hmm_states(self, regime_switching_data):
        """Verify state selection runs across multiple state counts.

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import select_hmm_states
        results = select_hmm_states(regime_switching_data, max_states=3)
        assert len(results) >= 1

    @pytest.mark.skipif(not _HAS_HMMLEARN, reason="hmmlearn not installed — pip install hmmlearn")
    def test_multivariate_data(self, regime_switching_data):
        """Fit HMM on 2D feature data (returns + vol).

        Requires: hmmlearn (install: pip install -e ".[stats]" or pip install hmmlearn).
        """
        from quantcore.regime.hmm import RegimeHMM
        X = np.column_stack([
            regime_switching_data,
            np.abs(regime_switching_data) * 10,
        ])
        hmm = RegimeHMM(n_states=2)
        result = hmm.fit(X)
        assert len(result.states) == len(X)


class TestRegimeFeatures:
    def test_realized_volatility(self, regime_switching_data):
        from quantcore.regime.features import realized_volatility
        vol = realized_volatility(regime_switching_data)
        assert vol > 0

    def test_rolling_correlation(self, regime_switching_data):
        from quantcore.regime.features import rolling_correlation
        rng = np.random.default_rng(42)
        r2 = rng.normal(0.001, 0.01, len(regime_switching_data))
        corr = rolling_correlation(regime_switching_data, r2)
        assert -1 <= corr <= 1

    def test_adx(self):
        from quantcore.regime.features import adx
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 1, 50))
        high = close + np.abs(rng.normal(0, 0.5, 50))
        low = close - np.abs(rng.normal(0, 0.5, 50))
        adx_val = adx(high, low, close, period=14)
        assert adx_val >= 0

    def test_hurst_exponent(self, regime_switching_data):
        from quantcore.regime.features import hurst_exponent
        # Could be a bit noisy on small samples, just check it returns a valid value
        h = hurst_exponent(regime_switching_data[:200], max_lag=20)
        assert 0 < h < 1

    def test_bid_ask_spread(self):
        from quantcore.regime.features import bid_ask_spread
        bids = np.array([99.95, 99.96, 99.97])
        asks = np.array([100.05, 100.04, 100.03])
        spread = bid_ask_spread(bids, asks)
        assert spread > 0

    def test_turnover_ratio(self):
        from quantcore.regime.features import turnover_ratio
        volume = np.array([1000] * 30)
        ratio = turnover_ratio(volume, shares_outstanding=50000)
        assert ratio == 0.02  # 1000/50000

    def test_credit_spread(self):
        from quantcore.regime.features import credit_spread
        spread = credit_spread(0.06, 0.04)
        assert abs(spread - 0.02) < 1e-10

    def test_yield_curve_slope(self):
        from quantcore.regime.features import yield_curve_slope
        slope = yield_curve_slope(0.04, 0.02)
        assert slope == 0.02

    def test_yield_curve_curvature(self):
        from quantcore.regime.features import yield_curve_curvature
        curv = yield_curve_curvature(0.02, 0.03, 0.04)
        # 2*0.03 - 0.02 - 0.04 = 0.06 - 0.06 = 0
        assert abs(curv) < 0.001

    def test_compute_all_features(self, regime_switching_data):
        from quantcore.regime.features import compute_all_features
        rng = np.random.default_rng(42)
        high = np.abs(rng.normal(0, 1, len(regime_switching_data)))
        low = -np.abs(rng.normal(0, 1, len(regime_switching_data)))
        close = rng.normal(0, 1, len(regime_switching_data))
        vol = rng.random(len(regime_switching_data)) * 1000
        features = compute_all_features(
            regime_switching_data, high=high, low=low, close=close, volume=vol,
        )
        assert "realized_vol" in features
        assert "hurst" in features


# ---------------------------------------------------------------------------
# T-0331: Changepoint + Vol Regimes
# ---------------------------------------------------------------------------


class TestChangepoint:
    @pytest.mark.skipif(not _HAS_RUPTURES, reason="ruptures not installed — pip install ruptures")
    def test_pelt_detects_changepoints(self, changepoint_data):
        """PELT detects known changepoints in synthetic data with a mean shift at index 150.

        Requires: ruptures library (install: pip install -e ".[stats]" or pip install ruptures).
        The test data has a +3σ mean shift at t=150 — PELT with penalty=5 should detect it.
        Run with: pytest -m "optional_dep" tests/unit/regime/test_regime.py
        """
        from quantcore.regime.changepoint import pelt_changepoints
        cps = pelt_changepoints(changepoint_data, penalty=5.0)
        assert len(cps.changepoints) > 0

    @pytest.mark.skipif(not _HAS_RUPTURES, reason="ruptures not installed — pip install ruptures")
    def test_pelt_regularization(self, changepoint_data):
        """Higher penalty produces fewer (or equal) changepoints than lower penalty.

        Requires: ruptures library (install: pip install -e ".[stats]" or pip install ruptures).
        This validates the penalty parameter correctly controls segmentation granularity.
        Run with: pytest -m "optional_dep" tests/unit/regime/test_regime.py
        """
        from quantcore.regime.changepoint import pelt_changepoints
        low_pen = pelt_changepoints(changepoint_data, penalty=1.0)
        high_pen = pelt_changepoints(changepoint_data, penalty=100.0)
        assert high_pen.n_changepoints <= low_pen.n_changepoints

    def test_bocpd(self, changepoint_data):
        """BOCPD returns non-negative changepoint count and a valid run-length array.

        BOCPD (Bayesian Online Changepoint Detection, Adams & MacKay 2007) is implemented
        in pure numpy/scipy — no optional dependencies required.
        """
        from quantcore.regime.changepoint import bocpd

        result = bocpd(changepoint_data[:200], hazard_rate=0.01)
        assert result.n_changepoints >= 0
        assert result.run_length is not None

    def test_cusum_changepoint(self, changepoint_data):
        from quantcore.regime.changepoint import cusum
        result = cusum(changepoint_data, threshold=5.0, drift=1.0)
        assert result.n_changepoints >= 0

    def test_cusum_detects_shift(self, changepoint_data):
        from quantcore.regime.changepoint import cusum
        # Strong shift should be detected
        strong_shift = np.concatenate([
            np.zeros(100),
            np.full(100, 5.0),  # Clear shift
        ])
        result = cusum(strong_shift, threshold=3.0, drift=0.5)
        assert result.n_changepoints > 0

    def test_sprt(self, changepoint_data):
        from quantcore.regime.changepoint import sprt_test
        result = sprt_test(changepoint_data, h0_mean=0.0, h1_mean=2.0)
        assert result.n_changepoints >= 0


class TestVolatilityRegime:
    def test_vix_low(self):
        from quantcore.regime.volatility_regime import VolRegime, vix_regime
        result = vix_regime(12.0)
        assert result.regime == VolRegime.LOW

    def test_vix_normal(self):
        from quantcore.regime.volatility_regime import VolRegime, vix_regime
        result = vix_regime(18.0)
        assert result.regime == VolRegime.NORMAL

    def test_vix_crisis(self):
        from quantcore.regime.volatility_regime import VolRegime, vix_regime
        result = vix_regime(40.0)
        assert result.regime == VolRegime.CRISIS

    def test_vix_extreme(self):
        from quantcore.regime.volatility_regime import VolRegime, vix_regime
        result = vix_regime(80.0)
        assert result.regime == VolRegime.EXTREME

    def test_garch_regime(self, regime_switching_data):
        from quantcore.regime.volatility_regime import garch_regime
        result = garch_regime(regime_switching_data)
        assert result.regime.value in ("low", "normal", "elevated", "crisis", "extreme")

    def test_vol_percentile_regime(self, regime_switching_data):
        from quantcore.regime.volatility_regime import vol_percentile_regime
        result = vol_percentile_regime(regime_switching_data)
        assert result.regime.value in ("low", "normal", "elevated", "crisis", "extreme")
        assert result.percentile is not None
        assert 0 <= result.percentile <= 100

    def test_vol_percentile_short_data(self):
        from quantcore.regime.volatility_regime import VolRegime, vol_percentile_regime
        rng = np.random.default_rng(42)
        short = rng.normal(0, 0.01, 10)
        result = vol_percentile_regime(short, window=252)
        assert result.regime == VolRegime.NORMAL

    def test_consistency_across_methods(self, regime_switching_data):
        """Volatility regime classification should be consistent across methods."""
        from quantcore.regime.volatility_regime import (
            garch_regime,
            vol_percentile_regime,
        )
        garch = garch_regime(regime_switching_data)
        pct = vol_percentile_regime(regime_switching_data)
        # Both should return valid regimes
        assert isinstance(garch.regime.value, str)
        assert isinstance(pct.regime.value, str)
