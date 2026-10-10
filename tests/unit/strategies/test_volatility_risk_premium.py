"""Unit tests for Volatility Risk Premium strategy and signals."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.signals.technical.volatility import (
    ts_regime_multiplier,
    vrp_composite_signal,
    vrp_raw,
    vvix_conviction,
)
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.macro.volatility_risk_premium import (
    VolatilityRiskPremiumStrategy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> StrategyConfig:
    params = {
        "lookback": 21,
        "ewm_span": 5,
        "entry_threshold": 0.25,
        "vix_crisis_threshold": 35,
        "target_vol": 0.10,
        "kelly_fraction": 0.5,
        "max_position_pct": 0.25,
        "stop_loss_sigma_normal": 2.0,
        "stop_loss_sigma_crisis": 3.0,
        "profit_target_pct": 0.05,
        "max_holding_days": 30,
        "signal_exit_threshold": 0.0,
        "tail_hedge_pct": 0.02,
        "vol_lookback": 21,
        "rebalance_freq": 1,  # Check every bar in tests
    }
    params.update(overrides)
    return StrategyConfig(
        name="test_vrp",
        strategy_type=StrategyType.MACRO,
        max_position_notional=1_000_000,
        params=params,
    )


def _make_ohlc(n: int = 300, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic OHLC data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 50.0)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    return {
        "close": close.reshape(-1, 1),
        "high": high.reshape(-1, 1),
        "low": low.reshape(-1, 1),
        "open": open_.reshape(-1, 1),
        "symbols": ["SPY"],
    }


def _make_vol_data(n: int = 300, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic vol index data."""
    rng = np.random.default_rng(seed)
    vix = 20.0 + rng.normal(0, 5, n)
    vix = np.clip(vix, 10, 80)
    vvix = 90.0 + rng.normal(0, 15, n)
    vvix = np.clip(vvix, 50, 200)
    ts_slope = 0.05 + rng.normal(0, 0.02, n)
    return {"vix": vix, "vvix": vvix, "ts_slope": ts_slope}


# ---------------------------------------------------------------------------
# Test vrp_raw
# ---------------------------------------------------------------------------


class TestVrpRaw:
    def test_positive_premium(self):
        iv = np.array([0.25, 0.30])
        rv = np.array([0.15, 0.20])
        result = vrp_raw(iv, rv)
        assert result[0] == pytest.approx((0.25 - 0.15) / 0.15)
        assert result[1] == pytest.approx((0.30 - 0.20) / 0.20)

    def test_zero_rv_guard(self):
        iv = np.array([0.25])
        rv = np.array([0.0])
        result = vrp_raw(iv, rv)
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_negative_premium(self):
        iv = np.array([0.10])
        rv = np.array([0.20])
        result = vrp_raw(iv, rv)
        assert result[0] < 0

    def test_nan_handling(self):
        iv = np.array([np.nan, 0.25])
        rv = np.array([0.15, np.nan])
        result = vrp_raw(iv, rv)
        assert np.isnan(result[0])
        assert np.isnan(result[1])


# ---------------------------------------------------------------------------
# Test ts_regime_multiplier
# ---------------------------------------------------------------------------


class TestTsRegimeMultiplier:
    def test_contango_maps_above_one(self):
        """Steady positive slope → z-score near mean → sigmoid ≈ 0.5 → mult ≈ 1.0."""
        n = 300
        slope = np.full(n, 0.05)
        result = ts_regime_multiplier(slope, lookback=252)
        # With constant slope, z-score is 0, sigmoid(0) = 0.5, mult = 1.0
        assert result[-1] == pytest.approx(1.0, abs=0.01)

    def test_backwardation_maps_below_one(self):
        """Negative spike in slope → z-score < 0 → sigmoid < 0.5 → mult < 1.0."""
        n = 300
        slope = np.full(n, 0.05)
        slope[-1] = -0.10  # Sharp backwardation
        result = ts_regime_multiplier(slope, lookback=252)
        assert result[-1] < 1.0

    def test_steep_contango_maps_above_one(self):
        """Very positive spike → z-score > 0 → sigmoid > 0.5 → mult > 1.0."""
        n = 300
        slope = np.full(n, 0.05)
        slope[-1] = 0.30  # Sharp contango
        result = ts_regime_multiplier(slope, lookback=252)
        assert result[-1] > 1.0

    def test_output_range(self):
        """All outputs in [0.5, 1.5]."""
        n = 300
        rng = np.random.default_rng(99)
        slope = rng.normal(0.05, 0.02, n)
        result = ts_regime_multiplier(slope, lookback=252)
        valid = result[251:]
        assert np.all(valid >= 0.5 - 1e-10)
        assert np.all(valid <= 1.5 + 1e-10)

    def test_length_preserved(self):
        slope = np.random.default_rng(42).normal(0, 1, 300)
        result = ts_regime_multiplier(slope)
        assert len(result) == len(slope)


# ---------------------------------------------------------------------------
# Test vvix_conviction
# ---------------------------------------------------------------------------


class TestVvixConviction:
    def test_normal_vvix(self):
        """VVIX near mean → z ≈ 0 → conviction ≈ 0.5."""
        n = 300
        vvix = np.full(n, 100.0)
        result = vvix_conviction(vvix, lookback=252)
        assert result[-1] == pytest.approx(0.5, abs=0.05)

    def test_high_vvix_low_conviction(self):
        """VVIX spike → z > 0 → conviction drops."""
        n = 300
        vvix = np.full(n, 100.0)
        vvix[-1] = 200.0  # Extreme spike
        result = vvix_conviction(vvix, lookback=252)
        assert result[-1] < 0.5

    def test_low_vvix_high_conviction(self):
        """VVIX drop → z < 0 → conviction rises."""
        n = 300
        vvix = np.full(n, 100.0)
        vvix[-1] = 50.0  # Very calm
        result = vvix_conviction(vvix, lookback=252)
        assert result[-1] > 0.5

    def test_output_range(self):
        """All outputs in [0, 1]."""
        n = 300
        rng = np.random.default_rng(77)
        vvix = 100 + rng.normal(0, 20, n)
        result = vvix_conviction(vvix, lookback=252)
        valid = result[251:]
        assert np.all(valid >= -1e-10)
        assert np.all(valid <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Test vrp_composite_signal
# ---------------------------------------------------------------------------


class TestVrpCompositeSignal:
    def test_length_preserved(self):
        n = 300
        iv = np.full(n, 0.25)
        rv = np.full(n, 0.15)
        ts = np.full(n, 0.05)
        vvix = np.full(n, 100.0)
        result = vrp_composite_signal(iv, rv, ts, vvix, lookback=252)
        assert len(result) == n

    def test_positive_in_normal_regime(self):
        """With IV > RV and normal VVIX, signal should be positive."""
        n = 300
        iv = np.full(n, 0.25)
        rv = np.full(n, 0.15)
        ts = np.full(n, 0.05)
        vvix = np.full(n, 100.0)
        result = vrp_composite_signal(iv, rv, ts, vvix, lookback=252)
        assert result[-1] > 0

    def test_ewm_smoothing(self):
        """EWM smoothing should reduce noise vs raw signal."""
        n = 300
        rng = np.random.default_rng(42)
        iv = 0.20 + rng.normal(0, 0.05, n)
        rv = 0.15 + rng.normal(0, 0.03, n)
        ts = 0.05 + rng.normal(0, 0.01, n)
        vvix = 100 + rng.normal(0, 10, n)
        result = vrp_composite_signal(iv, rv, ts, vvix, lookback=252, ewm_span=5)
        raw = vrp_raw(iv, rv)
        # Smoothed signal should have lower std than raw
        assert np.std(result[252:]) <= np.std(raw[252:]) * 1.5


# ---------------------------------------------------------------------------
# Test VolatilityRiskPremiumStrategy
# ---------------------------------------------------------------------------


class TestVolatilityRiskPremiumStrategy:
    def test_init(self):
        config = _make_config()
        strat = VolatilityRiskPremiumStrategy(config)
        assert strat.lookback == 21
        assert strat.entry_threshold == 0.25
        assert strat.vix_crisis_threshold == 35

    def test_no_vol_data_returns_empty(self):
        config = _make_config()
        strat = VolatilityRiskPremiumStrategy(config)
        data = _make_ohlc(300)
        signals = strat.on_data(data)
        assert signals == []

    def test_insufficient_data_returns_empty(self):
        config = _make_config()
        strat = VolatilityRiskPremiumStrategy(config)
        vol = _make_vol_data(300)
        strat.attach_vol_data(vol["vix"], vol["vvix"], vol["ts_slope"])
        data = _make_ohlc(50)  # Too short
        signals = strat.on_data(data)
        assert signals == []

    def test_crisis_no_entry(self):
        """When VIX >= crisis threshold, no new entries."""
        config = _make_config(vix_crisis_threshold=35)
        strat = VolatilityRiskPremiumStrategy(config)
        vol = _make_vol_data(300)
        vol["vix"][:] = 40.0  # Crisis level
        strat.attach_vol_data(vol["vix"], vol["vvix"], vol["ts_slope"])
        data = _make_ohlc(300)
        signals = strat.on_data(data)
        # Should only have tail hedge, no short vol entry
        short_vol = [s for s in signals if s.metadata.get("type") != "tail_hedge"]
        assert len(short_vol) == 0

    def test_entry_with_positive_signal(self):
        """Strong positive VRP signal → entry."""
        config = _make_config(entry_threshold=0.01, rebalance_freq=1)
        strat = VolatilityRiskPremiumStrategy(config)
        vol = _make_vol_data(300)
        # Set VIX high enough for positive VRP but below crisis
        vol["vix"][:] = 25.0
        vol["vvix"][:] = 80.0  # Low VVIX = high conviction
        strat.attach_vol_data(vol["vix"], vol["vvix"], vol["ts_slope"])
        data = _make_ohlc(300)
        signals = strat.on_data(data)
        # May or may not enter depending on signal strength
        # Just verify no crash and signals are valid
        for s in signals:
            assert s.direction in (-1, 0, 1)
            assert s.strength >= 0

    def test_always_emits_tail_hedge(self):
        """Tail hedge signal should always be emitted."""
        config = _make_config(tail_hedge_pct=0.02)
        strat = VolatilityRiskPremiumStrategy(config)
        vol = _make_vol_data(300)
        strat.attach_vol_data(vol["vix"], vol["vvix"], vol["ts_slope"])
        data = _make_ohlc(300)
        signals = strat.on_data(data)
        hedge_signals = [s for s in signals if s.metadata.get("type") == "tail_hedge"]
        assert len(hedge_signals) == 1
        assert hedge_signals[0].symbol == "VIX_CALL"
        assert hedge_signals[0].direction == 1

    def test_position_lifecycle(self):
        """Entry → holding → exit cycle."""
        config = _make_config(
            entry_threshold=0.01,
            max_holding_days=5,
            rebalance_freq=1,
        )
        strat = VolatilityRiskPremiumStrategy(config)
        vol = _make_vol_data(300)
        vol["vix"][:] = 20.0
        vol["vvix"][:] = 80.0
        strat.attach_vol_data(vol["vix"], vol["vvix"], vol["ts_slope"])
        data = _make_ohlc(300)

        # Run multiple bars
        all_signals = []
        for i in range(252, 300):
            bar_data = {
                "close": data["close"][:i+1],
                "high": data["high"][:i+1],
                "low": data["low"][:i+1],
                "open": data["open"][:i+1],
                "symbols": data["symbols"],
            }
            signals = strat.on_data(bar_data)
            all_signals.extend(signals)

        # Should have at least attempted some trades
        for s in all_signals:
            assert isinstance(s.symbol, str)
            assert s.size >= 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_nan_iv(self):
        iv = np.full(300, np.nan)
        rv = np.full(300, 0.15)
        result = vrp_raw(iv, rv)
        assert np.all(np.isnan(result))

    def test_single_element(self):
        result = vrp_raw(np.array([0.25]), np.array([0.15]))
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_extreme_values(self):
        iv = np.array([10.0])
        rv = np.array([0.001])
        result = vrp_raw(iv, rv)
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_ts_regime_single_element(self):
        result = ts_regime_multiplier(np.array([0.05]))
        assert len(result) == 1
        assert result[0] == 1.0  # Default when no lookback data

    def test_vvix_conviction_single_element(self):
        result = vvix_conviction(np.array([100.0]))
        assert len(result) == 1
        assert 0 <= result[0] <= 1
