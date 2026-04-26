"""Unit tests for statistical arbitrage, options flow, and signal combiner."""

from __future__ import annotations

import numpy as np
import pytest

# Statistical signals
from quantcore.signals.statistical.pairs import (
    scan_cointegrated_pairs, kalman_hedge_ratio, spread_zscore,
    pairs_signal, halflife_size, cost_adjusted_threshold,
)
from quantcore.signals.statistical.basket import (
    fair_value, etf_arb_signal, index_arb_signal, adr_arb_signal,
    basket_spread_signal,
)
from quantcore.signals.statistical.factors import (
    pe_signal, pb_signal, roe_signal, composite_factor,
    factor_ic, quintile_spread,
)
# Options flow signals
from quantcore.signals.options_flow.volatility_signals import (
    iv_rank, iv_rank_series, iv_percentile, iv_percentile_series,
    vol_risk_premium, iv_skew_signal, iv_term_structure_signal,
)
from quantcore.signals.options_flow.flow_signals import (
    unusual_options_activity, volume_weighted_delta,
    put_call_ratio, put_call_ratio_signal, detect_sweeps,
    options_flow_composite,
)
from quantcore.signals.options_flow.greeks_signals import (
    gamma_exposure, gamma_flip_level, charm_exposure,
    vanna_exposure, pin_risk_score, dealer_positioning_signal,
)
# Signal combiner
from quantcore.signals.combiner import (
    equal_weight_combine, ic_weight_combine,
    regime_weight_combine, sign_agreement_combine,
    adaptive_combine,
)


# ---------------------------------------------------------------------------
# Pairs Trading Tests
# ---------------------------------------------------------------------------


class TestKalmanHedgeRatio:
    def test_tracks_constant_ratio(self) -> None:
        """KF should track a constant hedge ratio on synthetic cointegrated pair."""
        rng = np.random.default_rng(42)
        n = 500
        x = 100.0 + np.cumsum(rng.normal(0, 1, n))
        true_ratio = 0.5
        y = true_ratio * x + rng.normal(0, 0.5, n)  # Cointegrated

        ratio, spread = kalman_hedge_ratio(y, x)
        # Last 100 values should be close to true ratio
        valid = ratio[~np.isnan(ratio)]
        if len(valid) > 100:
            assert abs(np.mean(valid[-100:]) - true_ratio) < 0.2

    def test_spread_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = 100.0 + np.cumsum(rng.normal(0, 1, n))
        y = 0.7 * x + rng.normal(0, 1, n)
        _, spread = kalman_hedge_ratio(y, x, obs_noise=0.01)
        valid = spread[~np.isnan(spread)]
        # Spread should be mean-reverting (std smaller than raw series)
        assert np.std(valid) < np.std(y - 0.7 * x) * 1.5


class TestSpreadZscore:
    def test_zscore_range(self) -> None:
        rng = np.random.default_rng(42)
        spread = rng.normal(0, 1, 300)
        result = spread_zscore(spread, lookback=60)
        valid = result[~np.isnan(result)]
        # Most values within ±3
        assert np.mean(np.abs(valid) < 3.0) > 0.95

    def test_mean_reverting_series(self) -> None:
        """Spread z-score of mean-reverting series should oscillate."""
        rng = np.random.default_rng(42)
        n = 200
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = -0.1 * spread[i - 1] + rng.normal(0, 1)
        result = spread_zscore(spread, lookback=60)
        valid = result[~np.isnan(result)]
        # Should have both positive and negative values
        assert np.any(valid > 0)
        assert np.any(valid < 0)


class TestPairsSignal:
    def test_entry_at_threshold(self) -> None:
        """Signal should enter short when z > entry, long when z < -entry."""
        zscore = np.array([0.0, 0.0, 2.5, 2.5, 0.3, 0.3, -2.5, -2.5])
        signal = pairs_signal(zscore, entry=2.0, exit=0.5, stop=4.0)
        assert signal[2] == -1.0  # Short spread
        assert signal[6] == 1.0   # Long spread

    def test_exit_near_zero(self) -> None:
        zscore = np.array([0.0, 2.5, 2.5, 0.3, 0.3])
        signal = pairs_signal(zscore, entry=2.0, exit=0.5, stop=4.0)
        # After entry at 2.5, exit when |z| < 0.5
        assert signal[3] == 0.0  # Exited

    def test_stop_loss(self) -> None:
        zscore = np.array([0.0, 2.5, 5.0, 0.0])
        signal = pairs_signal(zscore, entry=2.0, exit=0.5, stop=4.0)
        # At 5.0 > stop(4.0), should exit
        assert signal[2] == 0.0


class TestHalflifeSize:
    def test_shorter_hl_larger_size(self) -> None:
        short_hl_size = halflife_size(half_life=5.0, base_size=1.0, target_hl=10.0)
        long_hl_size = halflife_size(half_life=30.0, base_size=1.0, target_hl=10.0)
        assert short_hl_size > long_hl_size

    def test_inf_hl_tiny_position(self) -> None:
        size = halflife_size(half_life=float("inf"), base_size=1.0)
        assert size < 0.2


class TestCostAdjustedThreshold:
    def test_widens_entry(self) -> None:
        adj = cost_adjusted_threshold(entry_z=2.0, spread_cost_bps=10.0, spread_std=1.0, price=100.0)
        assert adj > 2.0  # Must be wider than raw entry


# ---------------------------------------------------------------------------
# Basket Arbitrage Tests
# ---------------------------------------------------------------------------


class TestFairValue:
    def test_known_nav(self) -> None:
        prices = np.array([[100.0, 50.0], [101.0, 51.0], [102.0, 52.0]])
        weights = np.array([0.6, 0.4])
        nav = fair_value(prices, weights)
        assert abs(nav[0] - 80.0) < 0.01  # 100*0.6 + 50*0.4
        assert abs(nav[1] - 81.0) < 0.01  # 101*0.6 + 51*0.4 = 60.6+20.4


class TestETFArbSignal:
    def test_near_zero_premium(self) -> None:
        """ETF and NAV tracking closely should produce signal near zero."""
        rng = np.random.default_rng(42)
        n = 300
        nav = 100 + np.cumsum(rng.normal(0, 0.5, n))
        etf = nav + rng.normal(0, 0.1, n)  # Tiny premium
        result = etf_arb_signal(etf, nav, lookback=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # Most values near zero when premium is small
            assert np.mean(np.abs(valid) < 1.0) > 0.5


# ---------------------------------------------------------------------------
# Factor Signal Tests
# ---------------------------------------------------------------------------


class TestFactorSignals:
    def test_pe_signal_inverted(self) -> None:
        """Low P/E should get positive signal (value)."""
        pe = np.array([10.0, 15.0, 20.0, 25.0, 30.0,
                        35.0, 40.0, 12.0, 18.0, 22.0])
        result = pe_signal(pe)
        # Lowest P/E (10) should have positive signal
        assert result[0] > 0

    def test_roe_signal_higher_better(self) -> None:
        roe = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                         0.30, 0.35, 0.08, 0.12, 0.18])
        result = roe_signal(roe)
        # Highest ROE should have positive signal
        assert result[6] > 0

    def test_composite_factor(self) -> None:
        sig1 = np.array([0.5, -0.3, 0.8, -0.1, 0.0])
        sig2 = np.array([-0.2, 0.6, 0.1, -0.4, 0.3])
        result = composite_factor([sig1, sig2], weights=[0.6, 0.4])
        assert abs(result[0] - 0.22) < 0.01  # 0.5*0.6 + (-0.2)*0.4
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_factor_ic(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        scores = rng.normal(0, 1, n)
        # Make returns partially correlated with scores
        returns = scores * 0.05 + rng.normal(0, 0.02, n)
        ic, t_stat = factor_ic(scores, returns, method="pearson")
        assert ic > 0  # Should be positive
        assert t_stat > 0

    def test_quintile_spread(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        scores = rng.normal(0, 1, n)
        returns = scores * 0.03 + rng.normal(0, 0.01, n)
        spread, quintiles = quintile_spread(scores, returns, n_quantiles=5)
        # Top quintile should outperform bottom for correlated data
        assert spread > 0


# ---------------------------------------------------------------------------
# Options Volatility Signal Tests
# ---------------------------------------------------------------------------


class TestIVRank:
    def test_at_lowest(self) -> None:
        hist = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        assert iv_rank(10.0, hist) == 20.0  # 1 of 5 <= 10 = 20%

    def test_at_highest(self) -> None:
        hist = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        assert iv_rank(30.0, hist) == 100.0  # 5 of 5 <= 30 = 100%

    def test_in_middle(self) -> None:
        hist = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        rank = iv_rank(20.0, hist)
        assert 30 <= rank <= 70  # Roughly middle


class TestIVPercentile:
    def test_percentile_range(self) -> None:
        rng = np.random.default_rng(42)
        iv_series = rng.uniform(10, 50, 500)
        result = iv_percentile_series(iv_series, lookback=252)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)


class TestVolRiskPremium:
    def test_positive_when_iv_higher(self) -> None:
        n = 300
        implied = np.full(n, 0.25)
        realized = np.full(n, 0.15)
        result = vol_risk_premium(implied, realized, lookback=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # IV > RV should give positive VRP
            assert valid[-1] >= 0


# ---------------------------------------------------------------------------
# Options Flow Signal Tests
# ---------------------------------------------------------------------------


class TestUnusualOptionsActivity:
    def test_flags_high_voi(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        volume = rng.uniform(90, 110, n)
        oi = np.ones(n) * 1000.0
        volume[50] = 5000.0  # Huge volume spike
        result = unusual_options_activity(volume, oi, avg_volume_lookback=20)
        assert result[50] >= 1.0  # Should flag as unusual


class TestPutCallRatio:
    def test_higher_put_volume(self) -> None:
        put_vol = np.array([2000.0, 3000.0, 2500.0])
        call_vol = np.array([1000.0, 1500.0, 1000.0])
        result = put_call_ratio(put_vol, call_vol)
        assert np.all(result > 1.0)  # Puts dominate

    def test_signal_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        put_vol = rng.uniform(100, 500, n)
        call_vol = rng.uniform(100, 500, n)
        result = put_call_ratio_signal(put_vol, call_vol, lookback=252)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)


class TestDetectSweeps:
    def test_detects_large_trades(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        sizes = rng.uniform(1, 10, n)
        sizes[50] = 100.0  # Sweep
        result = detect_sweeps(sizes, size_threshold=5.0, lookback=20)
        assert result[50] == True
        assert result[0] == False


# ---------------------------------------------------------------------------
# Greeks Signals Tests
# ---------------------------------------------------------------------------


class TestGammaExposure:
    def test_positive_gamma(self) -> None:
        gamma = np.array([0.01, 0.02, 0.015])
        oi = np.array([1000, 500, 800])
        gex = gamma_exposure(gamma, oi, spot_price=100.0)
        assert gex > 0

    def test_negative_gamma(self) -> None:
        gamma = np.array([-0.01, -0.02, -0.015])
        oi = np.array([1000, 500, 800])
        gex = gamma_exposure(gamma, oi, spot_price=100.0)
        assert gex < 0


class TestCharmVanna:
    def test_charm_positive(self) -> None:
        charm = np.array([0.001, 0.002])
        oi = np.array([1000, 500])
        result = charm_exposure(charm, oi)
        assert result > 0

    def test_vanna_value(self) -> None:
        vanna = np.array([0.01, -0.005])
        oi = np.array([1000, 500])
        result = vanna_exposure(vanna, oi)
        assert isinstance(result, float)


class TestPinRisk:
    def test_highest_near_atm(self) -> None:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        oi = np.array([500.0, 800.0, 2000.0, 600.0, 300.0])
        result = pin_risk_score(strikes, oi, spot_price=100.0,
                                time_to_expiry=0.08, implied_vol=0.25)
        # ATM strike should have highest pin risk
        assert result[2] > result[0]
        assert result[2] > result[4]


# ---------------------------------------------------------------------------
# Signal Combiner Tests
# ---------------------------------------------------------------------------


class TestEqualWeightCombine:
    def test_averages_signals(self) -> None:
        sig1 = np.array([0.5, -0.3, 0.0])
        sig2 = np.array([0.3, -0.1, 0.2])
        result = equal_weight_combine([sig1, sig2])
        assert abs(result[0] - 0.4) < 1e-10
        assert abs(result[1] - (-0.2)) < 1e-10

    def test_handles_nan(self) -> None:
        sig1 = np.array([0.5, np.nan, 0.0])
        sig2 = np.array([0.3, -0.1, 0.2])
        result = equal_weight_combine([sig1, sig2])
        assert abs(result[0] - 0.4) < 1e-10
        assert abs(result[1] - (-0.1)) < 1e-10  # Only sig2 available

    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        signals = [rng.uniform(-1, 1, 100) for _ in range(5)]
        result = equal_weight_combine(signals)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestRegimeWeightCombine:
    def test_weighted_by_regime(self) -> None:
        sig1 = np.array([1.0, 1.0, 1.0])
        sig2 = np.array([0.0, 0.0, 0.0])
        regime = np.array([0.0, 1.0, 0.0])
        weights = {0: [1.0, 0.0], 1: [0.0, 1.0]}
        result = regime_weight_combine([sig1, sig2], regime, weights)
        assert abs(result[0] - 1.0) < 1e-10  # Regime 0: all sig1
        assert abs(result[1] - 0.0) < 1e-10  # Regime 1: all sig2


class TestSignAgreementCombine:
    def test_unanimous(self) -> None:
        sigs = [
            np.array([0.5, -0.3, 0.2]),
            np.array([0.3, -0.1, 0.4]),
            np.array([0.1, -0.5, 0.3]),
        ]
        result = sign_agreement_combine(sigs)
        assert result[0] == 1.0   # All positive
        assert result[1] == -1.0  # All negative
        assert result[2] == 1.0   # All positive

    def test_disagreement(self) -> None:
        sigs = [
            np.array([0.5]),
            np.array([-0.3]),
        ]
        result = sign_agreement_combine(sigs)
        assert result[0] == 0.0  # Equal pos/neg


class TestAdaptiveCombine:
    def test_adapts_weights(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        forward_returns = rng.normal(0.001, 0.02, n)
        # Sig1 correlates with returns, sig2 doesn't
        sig1 = forward_returns * 5 + rng.normal(0, 0.1, n)
        sig2 = rng.normal(0, 0.5, n)
        result = adaptive_combine([sig1, sig2], forward_returns, lookback=126)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.any(valid != 0)  # Should produce some signals
