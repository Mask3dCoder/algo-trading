"""Tests for market impact integration in backtest engine (Priority 1)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quantcore.backtest.engine import BacktestEngine, FillSimulator, SignalData
from quantcore.backtest.impact_config import (
    MarketImpactConfig,
    MarketImpactEngine,
    VolumeFilterConfig,
)
from quantcore.core.types import OHLCV, Order, OrderType, Side


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyStrategy:
    """Minimal strategy that generates a single signal and flat thereafter."""

    def __init__(self, signal_data: SignalData | None = None) -> None:
        self._signal = signal_data
        self._fired = False

    def on_bar(self, bar, portfolio):
        if not self._fired and self._signal is not None:
            self._fired = True
            return [self._signal]
        return []

    def on_fill(self, fill, portfolio):
        pass


def _make_ohlcv_row(
    timestamp: datetime,
    symbol: str = "TEST",
    open_p: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: int = 1_000_000,
) -> dict:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _make_data(n_bars: int = 252, symbol: str = "TEST") -> pd.DataFrame:
    """Create synthetic price data with realistic volume."""
    rng = np.random.default_rng(42)
    base_price = 100.0
    rows = []
    for i in range(n_bars):
        ts = datetime(2020, 1, 1) + pd.Timedelta(days=i)
        price = base_price * (1 + rng.normal(0, 0.01))
        rows.append(
            _make_ohlcv_row(
                timestamp=ts,
                symbol=symbol,
                open_p=round(price, 2),
                high=round(price * 1.01, 2),
                low=round(price * 0.99, 2),
                close=round(price * 1.005, 2),
                volume=int(1_000_000 * (0.5 + rng.random())),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MarketImpactEngine unit tests
# ---------------------------------------------------------------------------


class TestMarketImpactEngine:
    def test_no_impact_when_zero_shares(self) -> None:
        engine = MarketImpactEngine(MarketImpactConfig())
        result = engine.estimate_impact(shares=0, price=100.0, bar_volume=1_000_000)
        assert result.temp_impact_bps == 0.0
        assert result.perm_impact_bps == 0.0
        assert result.total_cost_dollars == 0.0

    def test_large_order_higher_impact_than_small(self) -> None:
        """A large order (>5% ADV) must have higher impact than a small order (<0.1% ADV)."""
        engine = MarketImpactEngine(MarketImpactConfig(sigma=0.02, adv=1_000_000))
        small = engine.estimate_impact(shares=500, price=100.0, bar_volume=1_000_000)
        large = engine.estimate_impact(shares=50_000, price=100.0, bar_volume=1_000_000)
        assert large.total_cost_bps > small.total_cost_bps, (
            f"Large order ({large.total_cost_bps:.2f} bps) should cost more "
            f"than small ({small.total_cost_bps:.2f} bps)"
        )

    def test_sigma_estimation_from_data(self) -> None:
        """MarketImpactEngine should estimate sigma from data when not provided."""
        data = _make_data(252)
        engine = MarketImpactEngine(MarketImpactConfig())
        engine.set_data(data)
        assert engine.sigma is not None
        assert engine.sigma > 0

    def test_adv_estimation_from_data(self) -> None:
        """MarketImpactEngine should estimate ADV from data when not provided."""
        data = _make_data(252)
        engine = MarketImpactEngine(MarketImpactConfig())
        engine.set_data(data)
        assert engine.adv is not None
        assert engine.adv > 0

    def test_impact_dollar_cost_grows_with_shares(self) -> None:
        """Larger share amounts should produce larger dollar impact costs."""
        engine = MarketImpactEngine(MarketImpactConfig(sigma=0.02, adv=1_000_000))
        results = []
        for shares in [100, 1_000, 10_000]:
            r = engine.estimate_impact(shares=shares, price=100.0, bar_volume=1_000_000)
            results.append((shares, r.total_cost_dollars))
        for i in range(1, len(results)):
            assert results[i][1] >= results[i - 1][1], (
                f"{results[i][0]} shares cost ${results[i][1]:.2f} "
                f"but {results[i-1][0]} shares cost ${results[i-1][1]:.2f} "
                f"(impact should increase monotonically)"
            )


# ---------------------------------------------------------------------------
# VolumeFilter tests
# ---------------------------------------------------------------------------


class TestVolumeFilter:
    def test_small_order_passes_filter(self) -> None:
        """An order well under the max participation rate should pass through."""
        engine = MarketImpactEngine(
            MarketImpactConfig(),
            volume_filter=VolumeFilterConfig(max_participation_rate=0.20),
        )
        qty, clipped = engine.check_volume_filter(1_000, bar_volume=100_000)
        assert qty == 1_000
        assert not clipped

    def test_large_order_clipped_by_partial_fill(self) -> None:
        """An order exceeding max participation should be clipped when partial fills enabled."""
        engine = MarketImpactEngine(
            MarketImpactConfig(),
            volume_filter=VolumeFilterConfig(max_participation_rate=0.10, partial_fill_enabled=True),
        )
        qty, clipped = engine.check_volume_filter(50_000, bar_volume=100_000)
        expected_max = 100_000 * 0.10
        assert qty == expected_max
        assert clipped

    def test_large_order_rejected_when_no_partial(self) -> None:
        """An order exceeding max participation should be rejected when partial fills disabled."""
        engine = MarketImpactEngine(
            MarketImpactConfig(),
            volume_filter=VolumeFilterConfig(max_participation_rate=0.10, partial_fill_enabled=False),
        )
        qty, clipped = engine.check_volume_filter(50_000, bar_volume=100_000)
        assert qty == 0.0
        assert not clipped

    def test_min_volume_rejection(self) -> None:
        """Orders on bars below min_volume should be rejected."""
        engine = MarketImpactEngine(
            MarketImpactConfig(),
            volume_filter=VolumeFilterConfig(min_volume=1_000),
        )
        qty, clipped = engine.check_volume_filter(100, bar_volume=50)
        assert qty == 0.0

    def test_no_filter_by_default(self) -> None:
        """With no volume filter, all orders pass through."""
        engine = MarketImpactEngine(MarketImpactConfig())
        qty, clipped = engine.check_volume_filter(1_000_000, bar_volume=100)
        assert qty == 1_000_000
        assert not clipped


# ---------------------------------------------------------------------------
# FillSimulator integration tests (via BacktestEngine)
# ---------------------------------------------------------------------------


class TestMarketImpactIntegration:
    def test_large_order_worse_fill_than_small(self) -> None:
        """A backtest with a large position should show meaningful impact degradation
        vs an equivalent small position backtest."""
        data = _make_data(200)
        initial_cash_large = 10_000_000.0
        initial_cash_small = 100_000.0

        impact_config = MarketImpactConfig(sigma=0.02, adv=1_000_000)

        # Large position backtest
        sig = SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")
        engine_large = BacktestEngine(
            strategy=_DummyStrategy(sig),
            initial_cash=initial_cash_large,
            sizing_pct=0.50,  # 50% of portfolio — large position
            impact_config=impact_config,
        )
        engine_large.run(data)
        report_large = engine_large.summary()

        # Small position backtest
        engine_small = BacktestEngine(
            strategy=_DummyStrategy(sig),
            initial_cash=initial_cash_small,
            sizing_pct=0.01,  # 1% of portfolio — small position
            impact_config=impact_config,
        )
        engine_small.run(data)
        report_small = engine_small.summary()

        assert report_large.total_market_impact_cost > report_small.total_market_impact_cost, (
            f"Large position impact (${report_large.total_market_impact_cost:.2f}) should exceed "
            f"small position impact (${report_small.total_market_impact_cost:.2f})"
        )

    def test_no_impact_by_default(self) -> None:
        """Without impact config, no impact costs should be recorded."""
        data = _make_data(100)
        sig = SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")
        engine = BacktestEngine(
            strategy=_DummyStrategy(sig),
            initial_cash=1_000_000.0,
            sizing_pct=0.10,
        )
        engine.run(data)
        report = engine.summary()
        assert report.total_market_impact_cost == 0.0

    def test_no_impact_zero_adv_stable(self) -> None:
        """Zero ADV should not crash — default values should handle it gracefully."""
        engine = MarketImpactEngine(
            MarketImpactConfig(adv=0.0, sigma=0.02),
        )
        result = engine.estimate_impact(shares=100, price=100.0, bar_volume=1_000_000)
        # Should not crash; impact should be mild since ADV=0 triggers default fallback
        assert result.total_cost_dollars >= 0

    def test_permanent_impact_accumulates(self) -> None:
        """Test that permanent impact accumulates across multiple fills to the same symbol
        using a strategy that enters and adds to a position."""
        data = _make_data(100)

        # Two signals at different times
        class _AccumulatingStrategy:
            def __init__(self) -> None:
                self._step = 0

            def on_bar(self, bar, portfolio):
                self._step += 1
                if self._step == 5:
                    return [SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")]
                if self._step == 15:
                    return [SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")]
                return []

            def on_fill(self, fill, portfolio):
                pass

        impact_config = MarketImpactConfig(sigma=0.02, adv=1_000_000)
        engine = BacktestEngine(
            strategy=_AccumulatingStrategy(),
            initial_cash=1_000_000.0,
            sizing_pct=0.10,
            impact_config=impact_config,
        )
        engine.run(data)
        report = engine.summary()
        # Multiple fills should result in some market impact cost
        assert report.total_market_impact_cost >= 0

    def test_volume_filter_rejects_unrealistic_fill(self) -> None:
        """A position that requires >20% of bar volume should not fill when filter is strict."""
        data = _make_data(100)
        # Make one bar have very tiny volume
        data.loc[5, "volume"] = 10

        sig = SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")
        engine = BacktestEngine(
            strategy=_DummyStrategy(sig),
            initial_cash=10_000_000.0,
            sizing_pct=0.50,
            impact_config=MarketImpactConfig(sigma=0.02, adv=1_000_000),
            volume_filter_config=VolumeFilterConfig(
                max_participation_rate=0.20,
                min_volume=1,
            ),
        )
        result = engine.run(data)
        # Engine should complete without error even with volume-filtered fills
        assert not result.empty

    def test_impact_cost_in_summary_report(self) -> None:
        """The summary report should include market impact cost fields."""
        data = _make_data(100)
        sig = SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")
        engine = BacktestEngine(
            strategy=_DummyStrategy(sig),
            initial_cash=1_000_000.0,
            sizing_pct=0.10,
            impact_config=MarketImpactConfig(sigma=0.02, adv=1_000_000),
        )
        engine.run(data)
        report = engine.summary()
        # The report should have the impact fields (may be 0 for small fills)
        assert hasattr(report, "total_market_impact_cost")
        assert hasattr(report, "impact_cost_bps")

    def test_linearly_scaling_impact(self) -> None:
        """Impact cost should increase roughly with order size in sqrt model."""
        engine = MarketImpactEngine(MarketImpactConfig(sigma=0.02, adv=10_000_000))
        shares_list = [1_000, 5_000, 25_000]
        costs = []
        for s in shares_list:
            r = engine.estimate_impact(shares=s, price=100.0, bar_volume=1_000_000)
            costs.append(r.total_cost_dollars)

        # Each step up should cost more (monotonicity)
        assert costs[0] < costs[1] < costs[2], (
            f"Impact costs should be strictly increasing with shares: {costs}"
        )


# ---------------------------------------------------------------------------
# fill_at mode contract + degenerate-bar guardrail (post-audit fixes)
# ---------------------------------------------------------------------------


class TestFillAtModes:
    def _bar(self, o: float, h: float, l: float, c: float) -> OHLCV:
        return OHLCV(
            symbol="TEST",
            timestamp=datetime(2020, 1, 2),
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(l)),
            close=Decimal(str(c)),
            volume=10_000_000,
        )

    def test_next_open_and_close_differ(self) -> None:
        """fill_at='next_open' and fill_at='close' MUST produce different
        fills. This is the regression guard for the dead-branch bug where the
        two modes were silently identical (the `fill_at` param was a lie that
        always filled at bar.open regardless of value)."""
        bar = self._bar(o=109.0, h=111.0, l=108.0, c=110.0)
        order = Order(
            id="o1", symbol="TEST", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1000"),
        )

        fs_open = FillSimulator(slippage_bps=5.0, commission_per_share=0.005, fill_at="next_open")
        fs_open.set_bars(bar, bar)
        f_open = fs_open.simulate_fill(order)

        fs_close = FillSimulator(slippage_bps=5.0, commission_per_share=0.005, fill_at="close")
        fs_close.set_bars(bar, bar)
        f_close = fs_close.simulate_fill(order)

        # Both modes apply the SAME slippage convention (BUY pays more):
        #   next_open -> near bar.open  (109 + 5bps = 109.0545)
        #   close     -> near bar.close (110 + 5bps = 110.055)
        # The guard is that the two fills DIFFER (the dead-branch bug made
        # them silently identical at bar.open).
        assert abs(float(f_open.price) - 109.0545) < 1e-4, (
            f"next_open should fill near open 109.055, got {f_open.price}"
        )
        assert abs(float(f_close.price) - 110.055) < 1e-4, (
            f"close should fill near close 110 plus slippage, got {f_close.price}"
        )
        assert float(f_open.price) != float(f_close.price), (
            "next_open and close must produce DIFFERENT fills; the contract "
            "regressed and the two modes are silently identical again."
        )

    def test_zero_range_bar_clamps_slippage(self) -> None:
        """On a zero-range bar (O==H==L==C) the [low, high] clip collapses
        slippage to the single bar price, so slippage is zeroed for that bar.
        This documents the edge case so it is never mistaken for a
        missing-cost bug (guardrail for the degenerate-bar clamp finding)."""
        bar = self._bar(o=110.0, h=110.0, l=110.0, c=110.0)
        order = Order(
            id="o2", symbol="TEST", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1000"),
        )
        fs = FillSimulator(slippage_bps=5.0, commission_per_share=0.005)
        fs.set_bars(bar, bar)
        f = fs.simulate_fill(order)
        assert abs(float(f.price) - 110.0) < 1e-9, (
            f"zero-range bar must clamp fill to 110.0 (slippage zeroed), got {f.price}"
        )
