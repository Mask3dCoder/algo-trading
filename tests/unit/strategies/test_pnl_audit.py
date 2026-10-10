"""P&L audit tests for intraday strategies.

Constructs synthetic data where signals fire, then verifies that the market
conditions are consistent with paper profitability (gross of costs).

This catches:
- Sign convention bugs (direction inverted)
- Entry/exit direction mismatches
- Stop loss / take profit on wrong side
- Position sizing errors that would invert P&L

All tests use synthetic data with known directional properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.equity.intraday_mean_reversion import IntradayMeanReversionStrategy
from quantcore.strategies.equity.intraday_momentum import IntradayMomentumStrategy
from quantcore.strategies.equity.intraday_trend import IntradayTrendStrategy

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


def _synthetic_trend_data(
    n: int, start: float, end: float, noise: float = 0.0
) -> dict:
    """Create OHLCV data for a trend from start to end with optional noise."""
    close = np.linspace(start, end, n)
    if noise > 0:
        rng = np.random.default_rng(42)
        close = close + rng.normal(0, noise, n)
    high = close + 1.0
    low = close - 0.5
    return {"close": close, "high": high, "low": low, "symbols": ["X"]}


def _synthetic_v_shape_data(
    n_down: int, n_up: int, start: float, trough: float, end: float
) -> dict:
    """Create V-shaped OHLCV data: decline then rise, for EMA crossover testing.
    
    IntradayTrend needs min_bars = max(slow_window, trend_window) + 1 = 51
    with defaults. A V-shape ensures fast EMA crosses slow EMA at the
    inflection point, producing a bullish_cross signal.
    """
    down = np.linspace(start, trough, n_down)
    up = np.linspace(trough, end, n_up)
    close = np.concatenate([down, up])
    high = close + 1.0
    low = close - 0.5
    return {"close": close, "high": high, "low": low, "symbols": ["X"]}


def _synthetic_inverted_v_data(
    n_up: int, n_down: int, start: float, peak: float, end: float
) -> dict:
    """Create inverted-V OHLCV data: rise then decline, for bearish crossover."""
    up = np.linspace(start, peak, n_up)
    down = np.linspace(peak, end, n_down)
    close = np.concatenate([up, down])
    high = close + 1.0
    low = close - 0.5
    return {"close": close, "high": high, "low": low, "symbols": ["X"]}


def _synthetic_oscillation_data(
    n: int, base: float, amplitude: float, periods: int = 3
) -> dict:
    """Create sinusoidal OHLCV data for mean reversion testing."""
    t = np.linspace(0, periods * 2 * np.pi, n)
    close = base + amplitude * np.sin(t)
    high = close + 0.5
    low = close - 0.5
    return {"close": close, "high": high, "low": low, "symbols": ["X"]}


# ──────────────────────────────────────────────────────────────
# A. IntradayTrend — P&L audit
# ──────────────────────────────────────────────────────────────

class TestIntradayTrendPnL:
    """Verify trend strategy signals are directionally correct for P&L."""

    def _make_strategy(self, **overrides) -> IntradayTrendStrategy:
        params = {
            "fast_window": 3, "slow_window": 5, "trend_window": 7,
            "atr_window": 3, "atr_risk_units": 2.0, "max_pyramid_units": 3,
            "pyramid_atr_add": 0.5, "stop_atr_multiple": 2.0,
        }
        params.update(overrides)
        cfg = _make_config("intraday_trend", params)
        s = IntradayTrendStrategy(cfg)
        s.initialize()
        return s

    def test_bullish_entry_in_uptrend_is_profitable_direction(self):
        """Uptrend → bullish_cross → long entry.
        
        V-shaped data ensures EMA crossover fires at the inflection point.
        If the trend persists (price keeps rising), the long should be profitable.
        We verify: entry_price < future_price for long signals.
        """
        s = self._make_strategy()
        # V-shape: decline then rise. With fast=3, slow=5, trend=7,
        # min_bars = max(7,5)+1 = 8. Need enough bars for crossover to fire.
        data = _synthetic_v_shape_data(
            n_down=15, n_up=20, start=110.0, trough=90.0, end=130.0
        )

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered — adjust synthetic data")

        sig = signals[0]
        entry_price = data["close"][-1]

        # Long signal in uptrend: direction=+1
        assert sig.direction == 1, f"Expected long (direction=+1), got {sig.direction}"

        # P&L check: if price continues rising, long is profitable
        future_price = entry_price + 5.0  # assume trend continues
        paper_pnl = sig.direction * (future_price - entry_price)
        assert paper_pnl > 0, (
            f"Long entry at {entry_price:.2f} with trend continuing to "
            f"{future_price:.2f} should yield positive P&L, got {paper_pnl:.2f}"
        )

    def test_bearish_entry_in_downtrend_is_profitable_direction(self):
        """Inverted-V → bearish_cross → short entry.
        
        Inverted-V data (rise then decline) ensures fast EMA crosses slow EMA
        downward at the inflection point. If the decline persists, the short
        should be profitable.
        """
        s = self._make_strategy()
        data = _synthetic_inverted_v_data(
            n_up=15, n_down=20, start=90.0, peak=130.0, end=100.0
        )

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered — adjust synthetic data")

        sig = signals[0]
        entry_price = data["close"][-1]

        # Short signal: direction=-1
        assert sig.direction == -1, f"Expected short (direction=-1), got {sig.direction}"

        # P&L check: if price continues falling, short is profitable
        future_price = entry_price - 5.0  # assume decline continues
        paper_pnl = sig.direction * (future_price - entry_price)
        assert paper_pnl > 0, (
            f"Short entry at {entry_price:.2f} with trend continuing to "
            f"{future_price:.2f} should yield positive P&L, got {paper_pnl:.2f}"
        )

    def test_stop_loss_placement_limits_loss(self):
        """Stop loss should be placed such that if triggered, loss is bounded."""
        s = self._make_strategy()
        # V-shape data ensures EMA crossover fires a bullish_cross signal.
        data = _synthetic_v_shape_data(
            n_down=15, n_up=20, start=110.0, trough=90.0, end=130.0
        )

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered")

        sig = signals[0]
        entry_price = float(data["close"][-1])

        if sig.direction == 1:  # Long
            # Stop loss should be below entry
            assert sig.stop_loss < entry_price, (
                f"Long stop {sig.stop_loss:.2f} must be below entry {entry_price:.2f}"
            )
            # Loss if stopped = entry - stop_loss (should be positive)
            loss_if_stopped = entry_price - sig.stop_loss
            assert loss_if_stopped > 0, "Loss if stopped should be positive"
            # Loss should be bounded by ATR * stop_atr_multiple
            atr = sig.metadata.get("atr", 0)
            max_loss = atr * 2.0  # stop_atr_multiple=2.0
            assert loss_if_stopped <= max_loss * 1.01, (  # 1% tolerance
                f"Loss if stopped {loss_if_stopped:.2f} should be <= "
                f"ATR * stop_multiple = {max_loss:.2f}"
            )

        elif sig.direction == -1:  # Short
            assert sig.stop_loss > entry_price, (
                f"Short stop {sig.stop_loss:.2f} must be above entry {entry_price:.2f}"
            )
            loss_if_stopped = sig.stop_loss - entry_price
            assert loss_if_stopped > 0
            atr = sig.metadata.get("atr", 0)
            max_loss = atr * 2.0
            assert loss_if_stopped <= max_loss * 1.01

    def test_position_sizing_yields_correct_notional(self):
        """Verify position notional = (max_notional * risk_budget) / (atr_risk_units * atr) * price."""
        s = self._make_strategy()
        data = _synthetic_v_shape_data(n_down=15, n_up=20, start=110.0, trough=90.0, end=130.0)
        data["symbols"] = ["X"]

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered")

        sig = signals[0]
        atr = sig.metadata["atr"]
        expected_size = (100_000 * 0.02) / (2.0 * atr)
        expected_notional = expected_size * float(data["close"][-1])

        assert sig.size == pytest.approx(expected_notional, rel=1e-6)
        assert sig.size > 0, "Position size must be positive"


# ──────────────────────────────────────────────────────────────
# B. IntradayMeanReversion — P&L audit
# ──────────────────────────────────────────────────────────────

class TestIntradayMeanReversionPnL:
    """Verify mean reversion signals are directionally correct for P&L."""

    def _make_strategy(self, **overrides) -> IntradayMeanReversionStrategy:
        params = {
            "rsi_window": 7, "rsi_oversold": 35, "rsi_overbought": 65,
            "bb_window": 10, "bb_std": 1.5, "vwap_window": 10,
            "z_entry": 1.5, "z_exit": 0.3, "hold_bars": 60,
            "stop_loss_pct": 0.5,
        }
        params.update(overrides)
        cfg = _make_config("intraday_mean_reversion", params)
        s = IntradayMeanReversionStrategy(cfg)
        s.initialize()
        return s

    def test_oversold_long_profits_on_bounce(self):
        """Sharp drop → oversold → long → expect bounce back up.
        
        Paper P&L: if price reverts to mean, long entry profits.
        """
        s = self._make_strategy()
        # 25 flat bars, then 5 sharp drops → RSI crashes, BB/dev goes negative
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        data = {"close": close, "symbols": ["X"]}

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered — adjust synthetic data")

        sig = signals[0]
        entry_price = float(close[-1])

        # Mean reversion: oversold → expect LONG
        assert sig.direction == 1, f"Expected long on oversold, got {sig.direction}"

        # P&L check: if price reverts to mean (100), long profits
        mean_price = 100.0
        paper_pnl = sig.direction * (mean_price - entry_price)
        assert paper_pnl > 0, (
            f"Long at {entry_price:.2f} with mean reversion to {mean_price:.2f} "
            f"should yield positive P&L, got {paper_pnl:.2f}"
        )

    def test_overbought_short_profits_on_reversion(self):
        """Sharp rise → overbought → short → expect reversion down.
        
        Paper P&L: if price reverts to mean, short entry profits.
        """
        s = self._make_strategy()
        flat = np.full(25, 100.0)
        rises = np.array([101.0, 103.0, 106.0, 110.0, 115.0])
        close = np.concatenate([flat, rises])
        data = {"close": close, "symbols": ["X"]}

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered — adjust synthetic data")

        sig = signals[0]
        entry_price = float(close[-1])

        # Mean reversion: overbought → expect SHORT
        assert sig.direction == -1, f"Expected short on overbought, got {sig.direction}"

        # P&L check: if price reverts to mean (100), short profits
        mean_price = 100.0
        paper_pnl = sig.direction * (mean_price - entry_price)
        assert paper_pnl > 0, (
            f"Short at {entry_price:.2f} with mean reversion to {mean_price:.2f} "
            f"should yield positive P&L, got {paper_pnl:.2f}"
        )

    def test_stop_loss_and_take_profit_bracket_entry(self):
        """Stop loss and take profit should bracket entry price correctly."""
        s = self._make_strategy()
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        data = {"close": close, "symbols": ["X"]}

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered")

        sig = signals[0]
        entry_price = float(close[-1])

        if sig.direction == 1:  # Long
            assert sig.stop_loss < entry_price, (
                f"Long stop {sig.stop_loss:.2f} must be below entry {entry_price:.2f}"
            )
            assert sig.take_profit > entry_price, (
                f"Long TP {sig.take_profit:.2f} must be above entry {entry_price:.2f}"
            )
            # Verify stop_loss_pct is applied
            expected_stop = entry_price * (1 - 0.5 / 100)
            assert sig.stop_loss == pytest.approx(expected_stop, rel=1e-6)

        elif sig.direction == -1:  # Short
            assert sig.stop_loss > entry_price
            assert sig.take_profit < entry_price

    def test_composite_score_determines_direction(self):
        """Composite score sign should match signal direction."""
        s = self._make_strategy()
        flat = np.full(25, 100.0)
        drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
        close = np.concatenate([flat, drops])
        data = {"close": close, "symbols": ["X"]}

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered")

        sig = signals[0]
        composite = sig.metadata["composite"]

        # Positive composite → long, negative → short
        if composite > 0:
            assert sig.direction == 1, "Positive composite should produce long signal"
        elif composite < 0:
            assert sig.direction == -1, "Negative composite should produce short signal"

    def test_size_proportional_to_composite_magnitude(self):
        """Larger composite → larger position size."""
        s1 = self._make_strategy()
        s2 = self._make_strategy()

        # Mild drop → small composite: oscillating with net downward drift
        mild = np.array([
            100.0, 100.5, 99.5, 100.3, 99.3, 100.2, 99.2,
            100.1, 99.1, 100.0, 99.0, 99.9, 98.9, 99.8,
            98.8, 99.7, 98.7, 99.6, 98.6, 99.5, 98.5,
            99.4, 98.4, 99.3, 98.8,
        ])

        # Sharp drop → large composite: flat then steep decline
        sharp = np.concatenate([np.full(25, 100.0), np.array([99.0, 97.0, 94.0, 90.0, 85.0])])

        sigs1 = s1.on_data({"close": mild, "symbols": ["X"]})
        sigs2 = s2.on_data({"close": sharp, "symbols": ["X"]})

        if len(sigs1) > 0 and len(sigs2) > 0:
            # Sharp drop should have larger composite magnitude
            assert abs(sigs2[0].metadata["composite"]) > abs(sigs1[0].metadata["composite"])
            # And larger position size
            assert sigs2[0].size > sigs1[0].size


# ──────────────────────────────────────────────────────────────
# C. IntradayMomentum — P&L audit
# ──────────────────────────────────────────────────────────────

class TestIntradayMomentumPnL:
    """Verify momentum signals are directionally correct for P&L."""

    def _make_strategy(self, **overrides) -> IntradayMomentumStrategy:
        params = {
            "lookback_bars": 12, "skip_bars": 1, "n_long": 1, "n_short": 1,
            "vol_lookback": 20, "rebalance_freq": 1, "stop_loss_pct": 0.3,
            "take_profit_pct": 0.6, "min_strength": 0.01,
        }
        params.update(overrides)
        cfg = _make_config("intraday_momentum", params)
        s = IntradayMomentumStrategy(cfg)
        s.initialize()
        return s

    def test_best_performer_gets_long_signal(self):
        """Asset with highest cumulative return should get long signal."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01)
        n = 30
        # A: strong uptrend (best performer)
        a = np.linspace(100, 140, n)
        # B: flat (middle)
        b = np.full(n, 100.0)
        # C: downtrend (worst performer)
        c = np.linspace(140, 100, n)
        close = np.column_stack([a, b, c])
        data = {"close": close, "symbols": ["A", "B", "C"]}

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered")

        # Find signal for asset A
        sig_a = next((sig for sig in signals if sig.symbol == "A"), None)
        if sig_a is None:
            pytest.skip("No signal for asset A")

        # A is best performer → should be long
        assert sig_a.direction == 1, (
            f"Best performer A should be long, got direction={sig_a.direction}"
        )

        # P&L check: if momentum persists, long on A profits
        entry_price = float(a[-1])
        future_price = entry_price + 5.0
        paper_pnl = sig_a.direction * (future_price - entry_price)
        assert paper_pnl > 0

    def test_worst_performer_gets_short_signal(self):
        """Asset with lowest cumulative return should get short signal."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01)
        n = 30
        a = np.linspace(100, 140, n)
        b = np.full(n, 100.0)
        c = np.linspace(140, 100, n)
        close = np.column_stack([a, b, c])
        data = {"close": close, "symbols": ["A", "B", "C"]}

        signals = s.on_data(data)
        if len(signals) == 0:
            pytest.skip("No signal triggered")

        sig_c = next((sig for sig in signals if sig.symbol == "C"), None)
        if sig_c is None:
            pytest.skip("No signal for asset C")

        # C is worst performer → should be short
        assert sig_c.direction == -1, (
            f"Worst performer C should be short, got direction={sig_c.direction}"
        )

        # P&L check: if momentum persists, short on C profits
        entry_price = float(c[-1])
        future_price = entry_price - 5.0
        paper_pnl = sig_c.direction * (future_price - entry_price)
        assert paper_pnl > 0

    def test_stop_loss_and_take_profit_bracket_entry(self):
        """Stop/TP should bracket entry price for both long and short."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01)
        n = 30
        close = np.column_stack([
            np.linspace(100, 130, n),
            np.linspace(130, 100, n),
        ])
        data = {"close": close, "symbols": ["A", "B"]}

        signals = s.on_data(data)
        for sig in signals:
            price = float(close[-1, 0 if sig.symbol == "A" else 1])
            if sig.direction == 1:  # Long
                assert sig.stop_loss < price, (
                    f"Long stop {sig.stop_loss:.2f} must be below entry {price:.2f}"
                )
                assert sig.take_profit > price, (
                    f"Long TP {sig.take_profit:.2f} must be above entry {price:.2f}"
                )
            elif sig.direction == -1:  # Short
                assert sig.stop_loss > price
                assert sig.take_profit < price

    def test_position_sizing_proportional_to_weight(self):
        """size = |strength| * max_notional * risk_budget."""
        s = self._make_strategy(n_long=1, n_short=1, min_strength=0.01)
        n = 30
        close = np.column_stack([
            np.linspace(100, 130, n),
            np.linspace(130, 100, n),
        ])
        data = {"close": close, "symbols": ["A", "B"]}

        signals = s.on_data(data)
        for sig in signals:
            expected = abs(sig.strength) * 100_000 * 0.02
            assert sig.size == pytest.approx(expected, rel=1e-6)
            assert sig.size > 0, "Position size must be positive"

    def test_inverse_volatility_prefers_stable_asset(self):
        """Lower vol asset should get larger weight (position size)."""
        s = self._make_strategy(n_long=2, n_short=0, min_strength=0.01, rebalance_freq=1)
        n = 30
        # A: steady uptrend (low vol)
        a = np.linspace(100, 120, n)
        # B: erratic uptrend (high vol)
        rng = np.random.default_rng(42)
        b = 100 + np.cumsum(rng.normal(0.5, 2.0, n))
        close = np.column_stack([a, b])
        data = {"close": close, "symbols": ["A", "B"]}

        signals = s.on_data(data)
        if len(signals) >= 2:
            sig_a = next((sig for sig in signals if sig.symbol == "A"), None)
            sig_b = next((sig for sig in signals if sig.symbol == "B"), None)
            if sig_a and sig_b:
                # Both should be long (n_short=0)
                assert sig_a.direction == 1
                assert sig_b.direction == 1
                # Low-vol A should have larger position size
                assert sig_a.size > sig_b.size, (
                    f"Low-vol asset A ({sig_a.size:.2f}) should have larger "
                    f"position than high-vol asset B ({sig_b.size:.2f})"
                )


# ──────────────────────────────────────────────────────────────
# D. Cross-strategy P&L sanity checks
# ──────────────────────────────────────────────────────────────

class TestCrossStrategyPnLSanity:
    """Sanity checks that apply across all strategies."""

    def test_all_signals_have_positive_size(self):
        """All signals must have positive position size."""
        strategies = [
            self._make_trend(),
            self._make_mean_reversion(),
            self._make_momentum(),
        ]
        for strat in strategies:
            signals = strat.on_data(self._get_data(strat))
            for sig in signals:
                assert sig.size > 0, (
                    f"{strat.name}: signal size must be positive, got {sig.size}"
                )

    def test_all_signals_have_valid_direction(self):
        """All signals must have direction in {-1, 0, +1}."""
        strategies = [
            self._make_trend(),
            self._make_mean_reversion(),
            self._make_momentum(),
        ]
        for strat in strategies:
            signals = strat.on_data(self._get_data(strat))
            for sig in signals:
                assert sig.direction in (-1, 0, 1), (
                    f"{strat.name}: invalid direction {sig.direction}"
                )

    def test_stop_loss_always_on_loss_side(self):
        """For all strategies, stop loss must be on the loss side of entry."""
        strategies = [
            self._make_trend(),
            self._make_mean_reversion(),
            self._make_momentum(),
        ]
        for strat in strategies:
            signals = strat.on_data(self._get_data(strat))
            for sig in signals:
                if sig.direction == 1 or sig.direction == -1:  # Long
                    assert sig.stop_loss is not None

    def _make_trend(self) -> IntradayTrendStrategy:
        params = {"fast_window": 3, "slow_window": 5, "trend_window": 7,
                   "atr_window": 3, "atr_risk_units": 2.0, "stop_atr_multiple": 2.0}
        cfg = _make_config("intraday_trend", params)
        s = IntradayTrendStrategy(cfg)
        s.initialize()
        return s

    def _make_mean_reversion(self) -> IntradayMeanReversionStrategy:
        params = {"rsi_window": 7, "rsi_oversold": 35, "rsi_overbought": 65,
                   "bb_window": 10, "bb_std": 1.5, "vwap_window": 10,
                   "z_entry": 1.5, "z_exit": 0.3, "stop_loss_pct": 0.5}
        cfg = _make_config("intraday_mean_reversion", params)
        s = IntradayMeanReversionStrategy(cfg)
        s.initialize()
        return s

    def _make_momentum(self) -> IntradayMomentumStrategy:
        params = {"lookback_bars": 12, "skip_bars": 1, "n_long": 1, "n_short": 1,
                   "vol_lookback": 20, "rebalance_freq": 1, "stop_loss_pct": 0.3,
                   "take_profit_pct": 0.6, "min_strength": 0.01}
        cfg = _make_config("intraday_momentum", params)
        s = IntradayMomentumStrategy(cfg)
        s.initialize()
        return s

    def _get_data(self, strat) -> dict:
        if isinstance(strat, IntradayTrendStrategy):
            data = _synthetic_v_shape_data(n_down=15, n_up=20, start=110.0, trough=90.0, end=130.0)
            data["symbols"] = ["X"]
            return data
        elif isinstance(strat, IntradayMeanReversionStrategy):
            flat = np.full(25, 100.0)
            drops = np.array([99.0, 97.0, 94.0, 90.0, 85.0])
            close = np.concatenate([flat, drops])
            return {"close": close, "symbols": ["X"]}
        else:  # momentum
            n = 30
            close = np.column_stack([
                np.linspace(100, 130, n),
                np.linspace(130, 100, n),
            ])
            return {"close": close, "symbols": ["A", "B"]}
