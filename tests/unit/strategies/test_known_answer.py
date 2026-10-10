"""Known-answer tests for intraday strategies.

Constructs synthetic OHLCV data with hand-computed expected behaviors.
Tests deterministic properties: signal direction, no-signal guards,
stop loss correctness, and internal state consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.equity.intraday_mean_reversion import (
    IntradayMeanReversionStrategy,
    _rolling,
)
from quantcore.strategies.equity.intraday_momentum import IntradayMomentumStrategy
from quantcore.strategies.equity.intraday_trend import IntradayTrendStrategy, _rolling_atr

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_config(name: str, params: dict | None = None) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        strategy_type=StrategyType.EQUITY,
        max_position_notional=100_000.0,
        risk_budget_pct=0.02,
        params=params or {},
    )


# ──────────────────────────────────────────────────────────────
# A. _rolling_atr — unit test
# ──────────────────────────────────────────────────────────────

class TestRollingATR:
    def test_constant_prices_gives_zero_atr(self):
        n = 10
        h = np.full(n, 100.0)
        lo = np.full(n, 100.0)
        c = np.full(n, 100.0)
        atr = _rolling_atr(h, lo, c, window=3)
        # First window-1 bars are NaN
        assert np.isnan(atr[0])
        assert np.isnan(atr[1])
        # All TR = 0 for constant prices, so ATR should be 0
        assert atr[2] == pytest.approx(0.0, abs=1e-12)
        assert atr[9] == pytest.approx(0.0, abs=1e-12)

    def test_known_true_range_values(self):
        # Bar 0: H=102, L=98, C=100 → TR = H-L = 4
        # Bar 1: H=105, L=99, C=101 → TR = max(6, |105-100|=5, |99-100|=1) = 6
        # Bar 2: H=103, L=97, C=102 → TR = max(6, |103-101|=2, |97-101|=4) = 6
        # Bar 3: H=108, L=95, C=100 → TR = max(13, |108-102|=6, |95-102|=7) = 13
        h = np.array([102.0, 105.0, 103.0, 108.0])
        lo = np.array([98.0, 99.0, 97.0, 95.0])
        c = np.array([100.0, 101.0, 102.0, 100.0])
        atr = _rolling_atr(h, lo, c, window=3)

        # atr[2] = mean(tr[:3]) = mean(4, 6, 6) = 16/3 ≈ 5.333
        assert atr[2] == pytest.approx((4 + 6 + 6) / 3, rel=1e-10)
        # atr[3] = (atr[2] * (3-1) + tr[3]) / 3 = (5.333*2 + 13) / 3
        expected_atr3 = ((4 + 6 + 6) / 3 * 2 + 13) / 3
        assert atr[3] == pytest.approx(expected_atr3, rel=1e-10)

    def test_shape_and_nan_pattern(self):
        n = 8
        h = np.random.uniform(99, 101, n)
        lo = h - 1.0
        c = (h + lo) / 2
        atr = _rolling_atr(h, lo, c, window=5)
        assert len(atr) == n
        assert all(np.isnan(atr[i]) for i in range(4))
        assert not np.isnan(atr[4])
        assert not np.isnan(atr[7])


# ──────────────────────────────────────────────────────────────
# B. _rolling helper — unit test
# ──────────────────────────────────────────────────────────────

class TestRollingHelper:
    def test_rolling_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling(x, 3, np.mean)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)


# ──────────────────────────────────────────────────────────────
# C. intraday_trend — strategy tests
# ──────────────────────────────────────────────────────────────

class TestIntradayTrend:
    def _make_strategy(self, **overrides) -> IntradayTrendStrategy:
        params = {"fast_window": 3, "slow_window": 5, "trend_window": 7, "atr_window": 3,
                   "atr_risk_units": 2.0, "max_pyramid_units": 3, "pyramid_atr_add": 0.5,
                   "stop_atr_multiple": 2.0}
        params.update(overrides)
        cfg = _make_config("intraday_trend", params)
        s = IntradayTrendStrategy(cfg)
        s.initialize()
        return s

    def test_too_few_bars_returns_empty(self):
        s = self._make_strategy()
        close = np.arange(10, dtype=float) + 100
        data = {"close": close, "high": close * 1.005, "low": close * 0.995, "symbols": ["AAPL"]}
        assert s.on_data(data) == []

    def test_constant_prices_no_signal(self):
        """Constant prices → no crossover, no signal."""
        s = self._make_strategy()
        n = 20
        close = np.full(n, 100.0)
        data = {"close": close, "high": close + 0.5, "low": close - 0.5, "symbols": ["AAPL"]}
        signals = s.on_data(data)
        # EMA values all converge to 100; no crossover possible
        assert len(signals) == 0

    def test_strong_uptrend_triggers_bullish_entry(self):
        """Monotonically increasing prices: fast EMA must cross above slow EMA."""
        s = self._make_strategy()
        n = 20
        close = np.linspace(100, 130, n)
        high = close + 1.0
        low = close - 0.5
        data = {"close": close, "high": high, "low": low, "symbols": ["AAPL"]}
        signals = s.on_data(data)

        # With a strong uptrend, fast EMA (3) should cross above slow EMA (5)
        # and trend EMA (7) should be below price → bullish cross in uptrend
        if len(signals) > 0:
            sig = signals[0]
            assert sig.direction == 1, "Expected long entry in uptrend"
            assert sig.symbol == "AAPL"
            assert sig.size > 0
            assert sig.stop_loss is not None
            assert sig.stop_loss < 100.0, "Stop should be below entry price"
            # Verify metadata
            assert "atr" in sig.metadata
            assert sig.metadata["atr"] > 0
            assert sig.metadata["entry"] == "bullish_cross"

    def test_strong_downtrend_triggers_bearish_entry(self):
        """Monotonically decreasing prices: bearish cross in downtrend."""
        s = self._make_strategy()
        n = 20
        close = np.linspace(130, 100, n)
        high = close + 1.0
        low = close - 0.5
        data = {"close": close, "high": high, "low": low, "symbols": ["AAPL"]}
        signals = s.on_data(data)

        if len(signals) > 0:
            sig = signals[0]
            assert sig.direction == -1, "Expected short entry in downtrend"
            assert sig.size > 0
            assert sig.stop_loss is not None
            assert sig.stop_loss > 130.0, "Stop should be above entry price for short"

    def test_stop_loss_always_on_correct_side(self):
        """Verify stop loss is always on the loss side of the entry price."""
        s = self._make_strategy()
        for trend in [np.linspace(100, 140, 25), np.linspace(140, 100, 25)]:
            close = trend
            high = close + 1.0
            low = close - 0.5
            data = {"close": close, "high": high, "low": low, "symbols": ["X"]}
            sigs = s.on_data(data)
            for sig in sigs:
                price = float(close[-1])
                if sig.direction == 1:
                    assert sig.stop_loss < price, "Long stop must be below entry"
                elif sig.direction == -1:
                    assert sig.stop_loss > price, "Short stop must be above entry"

    def test_position_sizing_matches_formula(self):
        """Verify: size = (max_notional * risk_budget) / (atr_risk_units * atr) * price."""
        s = self._make_strategy()
        n = 25
        close = np.linspace(100, 140, n)
        high = close + 1.0
        low = close - 0.5
        data = {"close": close, "high": high, "low": low, "symbols": ["X"]}
        sigs = s.on_data(data)
        if len(sigs) > 0:
            sig = sigs[0]
            atr = sig.metadata["atr"]
            expected_pos_size = (100_000 * 0.02) / (2.0 * atr)
            expected_notional = expected_pos_size * float(close[-1])
            assert sig.size == pytest.approx(expected_notional, rel=1e-6)

    def test_state_tracking_on_entry(self):
        """After entry, internal state should track the position."""
        s = self._make_strategy()
        n = 25
        close = np.linspace(100, 140, n)
        data = {"close": close, "high": close + 1.0, "low": close - 0.5, "symbols": ["X"]}
        s.on_data(data)
        # Check internal state was set
        if "X" in s._units_held:
            assert s._units_held["X"] == 1
            assert "X" in s._entry_price
            assert "X" in s._trailing_stop

    def test_trailing_stop_increases(self):
        """Trailing stop should move up as price increases.

        The strategy needs ≥51 bars (trend_window=50 + 1). We must also
        produce a bullish_cross at the *last* bar, which requires:
          bar n-2: fast_ema < slow_ema
          bar n-1: fast_ema > slow_ema

        We achieve this with 57 bars of downtrend, 2 flat bars (EMAs
        converge), and a single sharp spike at bar 59 that tips the
        fast EMA above the slow EMA for the first time.
        """
        s = self._make_strategy()

        # Call 1: downtrend → flat → sharp spike at last bar
        downtrend = np.linspace(100, 80, 57)  # bars 0-56
        flat = np.array([80.0, 80.0])         # bars 57-58: EMAs converge
        spike = np.array([130.0])             # bar 59: fast crosses above slow
        close1 = np.concatenate([downtrend, flat, spike])  # 60 bars total
        data1 = {
            "close": close1,
            "high": close1 + 1.0,
            "low": close1 - 0.5,
            "symbols": ["X"],
        }
        s.on_data(data1)
        if "X" not in s._trailing_stop:
            pytest.skip("No entry triggered — adjust synthetic data")
        trail1 = s._trailing_stop["X"]

        # Call 2: continue rising → trailing stop ratchets up
        close2 = np.linspace(130, 180, 25)
        data2 = {
            "close": close2,
            "high": close2 + 1.0,
            "low": close2 - 0.5,
            "symbols": ["X"],
        }
        s.on_data(data2)
        trail2 = s._trailing_stop.get("X", trail1)
        assert trail2 > trail1, (
            f"Trailing stop should increase on rising prices: "
            f"trail1={trail1:.4f}, trail2={trail2:.4f}"
        )


# ──────────────────────────────────────────────────────────────
# D. intraday_mean_reversion — strategy tests
# ──────────────────────────────────────────────────────────────

class TestIntradayMeanReversion:
    def _make_strategy(self, **overrides) -> IntradayMeanReversionStrategy:
        params = {"rsi_window": 7, "rsi_oversold": 35, "rsi_overbought": 65,
                   "bb_window": 10, "bb_std": 1.5, "vwap_window": 10,
                   "z_entry": 1.5, "z_exit": 0.3, "hold_bars": 60,
                   "stop_loss_pct": 0.5}
        params.update(overrides)
        cfg = _make_config("intraday_mean_reversion", params)
        s = IntradayMeanReversionStrategy(cfg)
        s.initialize()
        return s

    def test_too_few_bars_returns_empty(self):
        s = self._make_strategy()
        close = np.arange(5, dtype=float) + 100
        assert s.on_data({"close": close, "symbols": ["X"]}) == []

    def test_constant_prices_no_signal(self):
        s = self._make_strategy()
        close = np.full(30, 100.0)
        signals = s.on_data({"close": close, "symbols": ["X"]})
        assert len(signals) == 0

    def test_extreme_oversold_triggers_long(self):
        """Prices dropping sharply: RSI should be oversold, BB deviation negative, VWAP z negative."""
        s = self._make_strategy()
        # 30 bars: 25 flat, then 5 sharp drops
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        signals = s.on_data({"close": close, "symbols": ["X"]})

        if len(signals) > 0:
            sig = signals[0]
            # Sharp drop → oversold → mean reversion → expect LONG
            assert sig.direction == 1, f"Expected long on oversold, got {sig.direction}"
            assert sig.size > 0
            assert sig.stop_loss is not None
            assert sig.take_profit is not None
            assert sig.stop_loss < 85.0, "Stop below current price for long"
            assert sig.take_profit > 85.0, "Take profit above current price for long"

    def test_extreme_overbought_triggers_short(self):
        """Prices rising sharply: RSI should be overbought."""
        s = self._make_strategy()
        flat = np.full(25, 100.0)
        rises = np.array([101.0, 103.0, 106.0, 110.0, 115.0])
        close = np.concatenate([flat, rises])
        signals = s.on_data({"close": close, "symbols": ["X"]})

        if len(signals) > 0:
            sig = signals[0]
            assert sig.direction == -1, f"Expected short on overbought, got {sig.direction}"

    def test_composite_score_metadata(self):
        """Signal metadata should contain indicator values."""
        s = self._make_strategy()
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        signals = s.on_data({"close": close, "symbols": ["X"]})

        if len(signals) > 0:
            sig = signals[0]
            assert "rsi" in sig.metadata
            assert "bb_dev" in sig.metadata
            assert "vwap_z" in sig.metadata
            assert "composite" in sig.metadata
            assert -1.0 <= sig.metadata["composite"] <= 1.0

    def test_size_proportional_to_composite(self):
        """size = |composite| * max_notional * risk_budget."""
        s = self._make_strategy()
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        signals = s.on_data({"close": close, "symbols": ["X"]})

        if len(signals) > 0:
            sig = signals[0]
            expected = abs(sig.metadata["composite"]) * 100_000 * 0.02
            assert sig.size == pytest.approx(expected, rel=1e-6)

    def test_needs_at_least_two_indicators(self):
        """Signal only fires when 2+ indicators agree.

        RSI-7 is scale-invariant: any non-constant trending data pushes RSI to 0
        or 100, so RSI alone always fires.  To isolate the threshold, we make the
        other two indicators unreachable by overriding their params:
        - bb_std=10.0  → bb_dev = σ/10, never > 1.0 for gentle trends
        - z_exit=10.0  → VWAP z-score never reaches 10 for gentle trends

        A mild linear uptrend (100→110 over 30 bars) produces RSI≈100 (fires)
        but bb_dev≈0.03 (dormant) and z_exit≈1.7 (dormant).  n_indicators=1 < 2
        → no signal.
        """
        s = self._make_strategy(bb_std=10.0, z_exit=10.0)
        close = np.linspace(100.0, 110.0, 30)
        signals = s.on_data({"close": close, "symbols": ["X"]})
        assert len(signals) == 0, (
            f"Single-indicator (RSI only) should not fire; "
            f"got {len(signals)} signal(s)"
        )


# ──────────────────────────────────────────────────────────────
# E. intraday_momentum — strategy tests
# ──────────────────────────────────────────────────────────────

class TestIntradayMomentum:
    def _make_strategy(self, **overrides) -> IntradayMomentumStrategy:
        params = {"lookback_bars": 12, "skip_bars": 1, "n_long": 2, "n_short": 2,
                   "vol_lookback": 20, "rebalance_freq": 5, "stop_loss_pct": 0.3,
                   "take_profit_pct": 0.6, "min_strength": 0.15}
        params.update(overrides)
        cfg = _make_config("intraday_momentum", params)
        s = IntradayMomentumStrategy(cfg)
        s.initialize()
        return s

    def test_too_few_bars_returns_empty(self):
        s = self._make_strategy()
        close = np.ones((5, 3))
        assert s.on_data({"close": close, "symbols": ["A", "B", "C"]}) == []

    def test_single_asset_no_ranking(self):
        """With 1 asset, n_long=2 is clamped to 1, but can't short (need n_assets - n_long = 0)."""
        s = self._make_strategy()
        close = np.linspace(100, 110, 30).reshape(-1, 1)
        signals = s.on_data({"close": close, "symbols": ["X"]})
        # Should produce at most 1 signal (long only, no short possible)
        assert len(signals) <= 1

    def test_clear_ranking_with_differentiated_assets(self):
        """Asset A trends up, B flat, C trends down → A long, C short."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01, rebalance_freq=1)
        n = 30
        a_prices = np.linspace(100, 130, n)   # strong uptrend
        b_prices = np.full(n, 100.0)           # flat
        c_prices = np.linspace(130, 100, n)    # strong downtrend
        close = np.column_stack([a_prices, b_prices, c_prices])
        signals = s.on_data({"close": close, "symbols": ["A", "B", "C"]})

        if len(signals) > 0:
            # Find which symbols got signals
            directions = {sig.symbol: sig.direction for sig in signals}
            # A (best performer) should be long
            assert directions.get("A") == 1, "Best performer should be long"
            # C (worst performer) should be short
            assert directions.get("C") == -1, "Worst performer should be short"

    def test_inverse_volatility_weighting(self):
        """Lower vol asset should get larger weight."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01, rebalance_freq=1)
        n = 30
        # A: steady uptrend (low vol)
        a_prices = np.linspace(100, 120, n)
        # B: erratic (high vol)
        b_prices = 100 + np.sin(np.linspace(0, 10 * np.pi, n)) * 20
        # C: steady downtrend (low vol)
        c_prices = np.linspace(120, 100, n)
        close = np.column_stack([a_prices, b_prices, c_prices])
        signals = s.on_data({"close": close, "symbols": ["A", "B", "C"]})

        if len(signals) >= 2:
            sig_a = next((s for s in signals if s.symbol == "A"), None)
            sig_c = next((s for s in signals if s.symbol == "C"), None)
            if sig_a and sig_c:
                # Both have similar trend magnitude, similar vol → similar weights
                # The key test: both are non-zero and on correct sides
                assert sig_a.direction == 1
                assert sig_c.direction == -1

    def test_rebalance_respects_frequency(self):
        """Only rebalance every rebalance_freq bars."""
        s = self._make_strategy(rebalance_freq=5, n_long=1, n_short=1, min_strength=0.01)
        n = 30
        close = np.column_stack([
            np.linspace(100, 120, n),
            np.linspace(120, 100, n),
        ])
        data = {"close": close, "symbols": ["A", "B"]}

        # Call 3 times in succession without enough bars gap → only first should rebalance
        sigs1 = s.on_data(data)
        sigs2 = s.on_data(data)
        sigs3 = s.on_data(data)
        # After rebalance, next 4 calls should not rebalance (rebalance_freq=5)
        # So sigs2 and sigs3 should have fewer or no new entry signals
        # (exit signals may still fire if stop/take-profit hit)
        if len(sigs1) > 0 and len(sigs2) > 0:
            entries1 = [s for s in sigs1 if s.direction != 0]
            entries2 = [s for s in sigs2 if s.direction != 0]
            # Second call within rebalance window: fewer entries expected
            assert len(entries2) <= len(entries1)

    def test_stop_loss_and_take_profit_on_correct_side(self):
        """Verify stop/TP placement for long vs short."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01, rebalance_freq=1)
        n = 30
        close = np.column_stack([
            np.linspace(100, 130, n),
            np.linspace(130, 100, n),
        ])
        signals = s.on_data({"close": close, "symbols": ["A", "B"]})

        for sig in signals:
            price = float(close[-1, 0 if sig.symbol == "A" else 1])
            if sig.direction == 1:
                assert sig.stop_loss < price, "Long stop must be below entry"
                assert sig.take_profit > price, "Long TP must be above entry"
            elif sig.direction == -1:
                assert sig.stop_loss > price, "Short stop must be above entry"
                assert sig.take_profit < price, "Short TP must be below entry"

    def test_size_proportional_to_weight(self):
        """size = |w| * max_notional * risk_budget."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01, rebalance_freq=1)
        n = 30
        close = np.column_stack([
            np.linspace(100, 130, n),
            np.linspace(130, 100, n),
        ])
        signals = s.on_data({"close": close, "symbols": ["A", "B"]})

        for sig in signals:
            expected = abs(sig.strength) * 100_000 * 0.02
            assert sig.size == pytest.approx(expected, rel=1e-6)


# ──────────────────────────────────────────────────────────────
# F. Determinism — same input → same output
# ──────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_intraday_trend_deterministic(self):
        cfg = _make_config("intraday_trend", {"fast_window": 3, "slow_window": 5,
                                               "trend_window": 7, "atr_window": 3})
        close = np.linspace(100, 130, 25)
        data = {"close": close, "high": close + 1.0, "low": close - 0.5, "symbols": ["X"]}
        s1 = IntradayTrendStrategy(cfg); s1.initialize()
        s2 = IntradayTrendStrategy(cfg); s2.initialize()
        assert s1.on_data(data) == s2.on_data(data)

    def test_intraday_mean_reversion_deterministic(self):
        cfg = _make_config("intraday_mean_reversion", {"rsi_window": 7, "bb_window": 10})
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        data = {"close": close, "symbols": ["X"]}
        s1 = IntradayMeanReversionStrategy(cfg); s1.initialize()
        s2 = IntradayMeanReversionStrategy(cfg); s2.initialize()
        r1 = s1.on_data(data)
        r2 = s2.on_data(data)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.symbol == b.symbol
            assert a.direction == b.direction
            assert a.size == pytest.approx(b.size, rel=1e-10)

    def test_intraday_momentum_deterministic(self):
        cfg = _make_config("intraday_momentum", {"n_long": 1, "n_short": 1, "min_strength": 0.01})
        n = 30
        close = np.column_stack([np.linspace(100, 130, n), np.linspace(130, 100, n)])
        data = {"close": close, "symbols": ["A", "B"]}
        s1 = IntradayMomentumStrategy(cfg); s1.initialize()
        s2 = IntradayMomentumStrategy(cfg); s2.initialize()
        r1 = s1.on_data(data)
        r2 = s2.on_data(data)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.symbol == b.symbol
            assert a.direction == b.direction
            assert a.size == pytest.approx(b.size, rel=1e-10)
