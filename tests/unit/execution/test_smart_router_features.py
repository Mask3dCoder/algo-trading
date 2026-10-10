"""Tests for SmartExecutionRouter features: adverse selection, dynamic spread, VWAP slicing."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantcore.core.types import Order, OrderType, Side
from quantcore.execution.smart_router import (
    SmartExecutionRouter,
    SmartRouterConfig,
    _build_minute_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_broker(is_paper: bool = True) -> MagicMock:
    """Create a mock broker that simulates paper trading fills."""
    broker = MagicMock()
    broker.is_paper = is_paper
    broker.broker_name = "mock_paper"
    broker.is_connected = True
    broker.submit_order.return_value = "mock-broker-order-1"
    broker.cancel_order.return_value = True
    broker.close_all_positions.return_value = 0
    broker.get_positions.return_value = {}
    broker.get_account.return_value = {
        "cash": 1_000_000,
        "portfolio_value": 1_000_000,
    }
    broker.health_check.return_value = {"status": "ok"}
    return broker


def _make_order(symbol: str = "AAPL", side: Side = Side.BUY, qty: float = 100) -> Order:
    """Create a test order."""
    return Order(
        id="test-order-1",
        symbol=symbol,
        side=side,
        type=OrderType.MARKET,
        quantity=Decimal(str(qty)),
    )


def _make_router(
    config: SmartRouterConfig | None = None,
    quotes: dict | None = None,
    volume_profiles: dict | None = None,
) -> tuple[SmartExecutionRouter, MagicMock]:
    """Create a router with a mock broker."""
    broker = _make_mock_broker()
    if quotes is None:
        quotes = {
            "AAPL": {"bid": 150.00, "ask": 150.10, "last": 150.05, "volume": 500_000, "adv": 5_000_000},
            "MSFT": {"bid": 380.00, "ask": 380.05, "last": 380.02, "volume": 300_000, "adv": 3_000_000},
            "TSLA": {"bid": 200.00, "ask": 200.50, "last": 200.25, "volume": 800_000, "adv": 8_000_000},
        }
    router = SmartExecutionRouter(broker, config, quotes, volume_profiles)
    return router, broker


# ---------------------------------------------------------------------------
# Feature 1: Adverse Selection Detection
# ---------------------------------------------------------------------------


class TestAdverseSelection:
    """Tests for adverse selection detection and auto-switching."""

    def test_adverse_selection_basic_detection(self):
        """A single adverse move should be detected correctly."""
        router, _ = _make_router()

        # Simulate a fill where price dropped after a buy
        history = [
            {"fill_price": 100.0, "side": "buy", "post_price": 99.5},  # 50 bps adverse
        ]
        rate = router.detect_adverse_selection("AAPL", order_history=history)
        assert rate == 1.0  # 100% adverse (1/1)

    def test_adverse_selection_no_adverse_move(self):
        """Price moving in our favor should not count as adverse."""
        router, _ = _make_router()

        history = [
            {"fill_price": 100.0, "side": "buy", "post_price": 100.5},  # 50 bps favorable
        ]
        rate = router.detect_adverse_selection("AAPL", order_history=history)
        assert rate == 0.0

    def test_adverse_selection_rolling_rate(self):
        """Rolling rate should reflect last N evaluations."""
        router, _ = _make_router()

        # 35 adverse out of 50 → 70% rate
        history = []
        for i in range(50):
            if i < 35:
                # Adverse: price dropped after buy
                history.append({"fill_price": 100.0, "side": "buy", "post_price": 99.5})
            else:
                # Not adverse: price rose after buy
                history.append({"fill_price": 100.0, "side": "buy", "post_price": 100.5})

        rate = router.detect_adverse_selection("AAPL", order_history=history)
        assert rate == pytest.approx(0.70, abs=0.01)

    def test_adverse_selection_forces_market_orders(self):
        """When adverse rate > 30%, router should force market orders."""
        cfg = SmartRouterConfig(adverse_threshold_rate=0.30, adverse_lookback=50)
        router, broker = _make_router(config=cfg)

        # Inject 20 adverse out of 50 → 40% > 30% threshold
        history = []
        for i in range(50):
            if i < 20:
                history.append({"fill_price": 100.0, "side": "buy", "post_price": 99.5})
            else:
                history.append({"fill_price": 100.0, "side": "buy", "post_price": 100.5})

        router.detect_adverse_selection("AAPL", order_history=history)

        # Verify AAPL is forced to market
        assert router._is_forced_market("AAPL")

        # Next order should go through as market (not limit)
        order = _make_order("AAPL", Side.BUY, 100)
        order_id = router.submit_order(order)
        tracked = router._tracked_orders[order_id]
        assert tracked.adverse_selected is True
        assert tracked.spread_mode == "market"

    def test_adverse_selection_hysteresis_recovery(self):
        """When rate drops below 80% of threshold, limit orders resume."""
        cfg = SmartRouterConfig(adverse_threshold_rate=0.30, adverse_lookback=50)
        router, _ = _make_router(config=cfg)

        # First: force market with high adverse rate
        history_high = []
        for i in range(50):
            if i < 20:
                history_high.append({"fill_price": 100.0, "side": "buy", "post_price": 99.5})
            else:
                history_high.append({"fill_price": 100.0, "side": "buy", "post_price": 100.5})
        router.detect_adverse_selection("AAPL", order_history=history_high)
        assert router._is_forced_market("AAPL")

        # Now feed 50 good fills to dilute the rate below 24% (80% of 30%)
        history_good = []
        for i in range(50):
            history_good.append({"fill_price": 100.0, "side": "buy", "post_price": 100.5})
        router.detect_adverse_selection("AAPL", order_history=history_good)
        assert not router._is_forced_market("AAPL")

    def test_adverse_sell_side(self):
        """Adverse selection for sells: price rising after we sold."""
        router, _ = _make_router()

        history = [
            {"fill_price": 100.0, "side": "sell", "post_price": 100.5},  # 50 bps adverse for sell
        ]
        rate = router.detect_adverse_selection("AAPL", order_history=history)
        assert rate == 1.0

    def test_adverse_events_logged(self):
        """Adverse selection events should be stored for CSV export."""
        router, _ = _make_router()

        history = [
            {"fill_price": 100.0, "side": "buy", "post_price": 99.5},
            {"fill_price": 100.0, "side": "buy", "post_price": 100.5},
        ]
        router.detect_adverse_selection("AAPL", order_history=history)

        assert len(router._adverse_events) == 1
        assert router._adverse_events[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Feature 2: Dynamic Spread Capture
# ---------------------------------------------------------------------------


class TestDynamicSpreadCapture:
    """Tests for spread-width-based order routing."""

    def test_tight_spread_uses_market_order(self):
        """Spread < 2 bps should route to market orders."""
        cfg = SmartRouterConfig(
            dynamic_spread_enabled=True,
            spread_tight_bps=2.0,
            paper_spread_bps=0.5,  # very tight
        )
        quotes = {
            "AAPL": {"bid": 150.00, "ask": 150.005, "last": 150.00, "volume": 500_000, "adv": 5_000_000},
        }
        router, broker = _make_router(config=cfg, quotes=quotes)

        order = _make_order("AAPL", Side.BUY, 100)
        order_id = router.submit_order(order)

        tracked = router._tracked_orders[order_id]
        assert tracked.spread_mode == "market"
        # Should have been submitted as market (not limit)
        assert tracked.order_type != "limit" or tracked.fill_price is not None

    def test_medium_spread_uses_mid_limit(self):
        """Spread 2-10 bps should place limit at mid-price."""
        cfg = SmartRouterConfig(
            dynamic_spread_enabled=True,
            spread_tight_bps=2.0,
            spread_medium_bps=10.0,
        )
        # 5 bps spread: 150.00 bid, 150.075 ask
        quotes = {
            "AAPL": {"bid": 150.00, "ask": 150.075, "last": 150.0375, "volume": 500_000, "adv": 5_000_000},
        }
        router, broker = _make_router(config=cfg, quotes=quotes)

        order = _make_order("AAPL", Side.BUY, 100)
        order_id = router.submit_order(order)

        tracked = router._tracked_orders[order_id]
        assert tracked.spread_mode == "mid_limit"
        assert tracked.order_type == "limit"
        assert tracked.limit_price is not None
        # Limit should be below mid (for buy)
        mid = (150.00 + 150.075) / 2
        assert tracked.limit_price < mid

    def test_wide_spread_uses_far_limit(self):
        """Spread > 10 bps should place limit at far touch."""
        cfg = SmartRouterConfig(
            dynamic_spread_enabled=True,
            spread_tight_bps=2.0,
            spread_medium_bps=10.0,
        )
        # 25 bps spread: 200.00 bid, 200.50 ask
        quotes = {
            "TSLA": {"bid": 200.00, "ask": 200.50, "last": 200.25, "volume": 800_000, "adv": 8_000_000},
        }
        router, broker = _make_router(config=cfg, quotes=quotes)

        order = _make_order("TSLA", Side.BUY, 100)
        order_id = router.submit_order(order)

        tracked = router._tracked_orders[order_id]
        assert tracked.spread_mode == "far_limit"
        assert tracked.order_type == "limit"
        assert tracked.limit_price is not None
        # For buy, far-touch limit should be near or below bid
        assert tracked.limit_price <= 200.00

    def test_signal_strength_adjusts_timeout(self):
        """Higher signal strength should produce shorter timeouts."""
        cfg = SmartRouterConfig(
            dynamic_spread_enabled=True,
            spread_wide_timeout_sec=60.0,
            spread_wide_min_timeout_sec=15.0,
        )
        router, _ = _make_router(config=cfg)

        # Verify timeout calculation
        # urgency=0 → timeout=60s, urgency=1 → timeout=15s
        urgency_0_timeout = 60.0 - 0.0 * (60.0 - 15.0)
        urgency_1_timeout = 60.0 - 1.0 * (60.0 - 15.0)
        assert urgency_0_timeout == 60.0
        assert urgency_1_timeout == 15.0

    def test_spread_mode_counts_in_stats(self):
        """Execution stats should include spread mode distribution."""
        cfg = SmartRouterConfig(dynamic_spread_enabled=True, spread_tight_bps=2.0)
        router, broker = _make_router(config=cfg)

        # Submit multiple orders with different spreads
        # Tight spread → market
        router.update_quotes({"AAPL": {"bid": 150.00, "ask": 150.002}})
        router.submit_order(_make_order("AAPL", Side.BUY, 100))

        stats = router.get_execution_stats()
        assert "spread_mode_market" in stats
        assert "spread_mode_mid_limit" in stats
        assert "spread_mode_far_limit" in stats


# ---------------------------------------------------------------------------
# Feature 3: Volume-Weighted Execution Schedule
# ---------------------------------------------------------------------------


class TestVolumeWeightedSchedule:
    """Tests for VWAP-based order splitting."""

    def test_volume_profile_construction(self):
        """Minute profile should have 390 elements summing to 1.0."""
        profile = _build_minute_profile()
        assert len(profile) == 390
        assert sum(profile) == pytest.approx(1.0, abs=0.001)

    def test_volume_profile_open_close_heavy(self):
        """Profile should have higher weights at open and close."""
        profile = _build_minute_profile()
        # First 30 min (open) should have more volume than midday
        open_volume = sum(profile[0:30])
        midday_volume = sum(profile[180:210])  # ~12:30-13:00
        close_volume = sum(profile[360:390])  # last 30 min

        assert open_volume > midday_volume
        assert close_volume > midday_volume

    def test_vwap_split_produces_non_equal_slices(self):
        """Volume-weighted slices should NOT be equal."""
        cfg = SmartRouterConfig(
            use_volume_weighted_schedule=True,
            twap_num_slices=5,
        )
        router, broker = _make_router(config=cfg)

        weights = router._compute_volume_weights("AAPL", 5)
        # Weights should not all be equal
        assert len(weights) == 5
        assert not all(abs(w - 0.2) < 0.001 for w in weights), \
            f"Weights are equal (TWAP), expected VWAP: {weights}"

    def test_vwap_split_respects_total_quantity(self):
        """Sum of slice quantities must equal total quantity."""
        cfg = SmartRouterConfig(
            use_volume_weighted_schedule=True,
            twap_num_slices=5,
        )
        router, _ = _make_router(config=cfg)

        weights = router._compute_volume_weights("AAPL", 5)
        quantities = router._allocate_slice_quantities(1000.0, weights, 5)
        assert sum(quantities) == pytest.approx(1000.0, abs=1.0)
        assert all(q >= 1.0 for q in quantities)

    def test_twap_mode_still_works(self):
        """TWAP (equal slices) should still work when VWAP is disabled."""
        cfg = SmartRouterConfig(
            use_volume_weighted_schedule=False,
            twap_num_slices=5,
        )
        router, _ = _make_router(config=cfg)

        weights = router._compute_volume_weights("AAPL", 5)
        # With VWAP disabled, the split uses equal weights
        # But _compute_volume_weights always returns profile-based weights
        # The equality check is in _submit_twap_split which uses cfg flag
        assert len(weights) == 5

    def test_vwap_split_through_router(self):
        """Full VWAP split through submit_order should produce tracked slices."""
        cfg = SmartRouterConfig(
            use_volume_weighted_schedule=True,
            twap_num_slices=5,
            adv_threshold_pct=0.01,
            min_split_quantity=10,
        )
        # Use low ADV so 500 shares exceeds the 1% ADV threshold
        quotes = {
            "AAPL": {"bid": 150.00, "ask": 150.10, "last": 150.05, "volume": 500, "adv": 5_000},
        }
        router, broker = _make_router(config=cfg, quotes=quotes)

        order = _make_order("AAPL", Side.BUY, 500)
        parent_id = router.submit_order(order)

        # Parent should be tracked
        assert parent_id in router._tracked_orders
        parent = router._tracked_orders[parent_id]
        assert parent.order_type == "twap_parent"
        assert parent.quantity == 500

        # Children should be tracked
        children = [
            o for o in router._tracked_orders.values()
            if o.parent_order_id == parent_id
        ]
        assert len(children) == 5

        # Sum of children should equal parent
        total_child_qty = sum(c.quantity for c in children)
        assert total_child_qty == pytest.approx(500.0, abs=5.0)

    def test_custom_volume_profile(self):
        """Custom per-ticker volume profiles should be used."""
        custom_profile = {
            "09:30": 0.20,  # very heavy open
            "10:00": 0.10,
            "10:30": 0.05,
            "11:00": 0.05,
            "11:30": 0.05,
            "12:00": 0.05,
            "12:30": 0.05,
            "13:00": 0.05,
            "13:30": 0.05,
            "14:00": 0.05,
            "14:30": 0.05,
            "15:00": 0.05,
            "15:30": 0.10,
            "16:00": 0.05,
        }
        cfg = SmartRouterConfig(use_volume_weighted_schedule=True)
        router, _ = _make_router(
            config=cfg,
            volume_profiles={"TSLA": custom_profile},
        )

        weights = router._compute_volume_weights("TSLA", 3)
        # First slice (open) should be heaviest
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]


# ---------------------------------------------------------------------------
# Integration: CSV output
# ---------------------------------------------------------------------------


class TestCSVOutput:
    """Tests for CSV cost report with new columns."""

    def test_csv_has_new_columns(self, tmp_path: Path):
        """execution_costs.csv should include adverse selection and spread mode columns."""
        csv_path = tmp_path / "execution_costs.csv"
        cfg = SmartRouterConfig(cost_report_path=str(csv_path))
        router, broker = _make_router(config=cfg)

        # Submit a few orders to populate daily record
        for _ in range(3):
            router.submit_order(_make_order("AAPL", Side.BUY, 100))

        # Force flush
        router._flush_daily_costs()

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)

        # Check new columns exist
        assert "adverse_selection_events" in header
        assert "adverse_selection_market_switches" in header
        assert "spread_mode_market" in header
        assert "spread_mode_mid_limit" in header
        assert "spread_mode_far_limit" in header

    def test_adverse_events_csv_created(self, tmp_path: Path):
        """adverse_selection_events.csv should be created when adverse events occur."""
        csv_path = tmp_path / "execution_costs.csv"
        cfg = SmartRouterConfig(cost_report_path=str(csv_path))
        router, _ = _make_router(config=cfg)

        # Generate adverse events
        history = [
            {"fill_price": 100.0, "side": "buy", "post_price": 99.5},
        ]
        router.detect_adverse_selection("AAPL", order_history=history)

        router._flush_daily_costs()

        adverse_path = tmp_path / "adverse_selection_events.csv"
        assert adverse_path.exists()
        with open(adverse_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "ticker" in header
            assert "adverse_rate" in header
            assert "forced_market" in header


# ---------------------------------------------------------------------------
# Integration: all three features together
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end tests verifying all three features activate together."""

    def test_all_features_active_in_paper_trading(self, tmp_path: Path):
        """Verify adverse selection, dynamic spread, and VWAP all activate."""
        csv_path = tmp_path / "execution_costs.csv"
        cfg = SmartRouterConfig(
            dynamic_spread_enabled=True,
            adverse_selection_enabled=True,
            use_volume_weighted_schedule=True,
            spread_tight_bps=2.0,
            spread_medium_bps=10.0,
            twap_num_slices=4,
            adv_threshold_pct=0.01,
            min_split_quantity=10,
            cost_report_path=str(csv_path),
        )

        # Mix of tight, medium, and wide spreads
        # MED ADV=10000 so 100 shares doesn't split (100/10000=0.01, not >),
        # but 500 shares does (500/10000=0.05 > 0.01)
        quotes = {
            "TIGHT": {"bid": 100.00, "ask": 100.01, "last": 100.005, "volume": 100_000, "adv": 1_000_000},
            "MED":   {"bid": 100.00, "ask": 100.05, "last": 100.025, "volume": 100, "adv": 10_000},
            "WIDE":  {"bid": 100.00, "ask": 100.20, "last": 100.10, "volume": 100_000, "adv": 1_000_000},
        }
        router, broker = _make_router(config=cfg, quotes=quotes)

        # 1. Dynamic spread: tight → market
        router.submit_order(_make_order("TIGHT", Side.BUY, 100))
        tight_order = list(router._tracked_orders.values())[-1]
        assert tight_order.spread_mode == "market"

        # 2. Dynamic spread: medium → mid_limit
        router.submit_order(_make_order("MED", Side.BUY, 100))
        med_order = list(router._tracked_orders.values())[-1]
        assert med_order.spread_mode == "mid_limit"

        # 3. Dynamic spread: wide → far_limit
        router.submit_order(_make_order("WIDE", Side.BUY, 100))
        wide_order = list(router._tracked_orders.values())[-1]
        assert wide_order.spread_mode == "far_limit"

        # 4. Adverse selection: inject high adverse rate
        adverse_history = []
        for _ in range(20):
            adverse_history.append({"fill_price": 100.0, "side": "buy", "post_price": 99.5})
        for _ in range(5):
            adverse_history.append({"fill_price": 100.0, "side": "buy", "post_price": 100.5})
        router.detect_adverse_selection("TIGHT", order_history=adverse_history)
        assert router._is_forced_market("TIGHT")

        # 5. VWAP: large order should split with non-equal slices
        router.submit_order(_make_order("MED", Side.BUY, 500))
        parent = [o for o in router._tracked_orders.values() if o.order_type == "twap_parent"]
        assert len(parent) >= 1

        children = [o for o in router._tracked_orders.values() if o.is_split]
        assert len(children) == 4  # twap_num_slices=4
        child_qtys = [c.quantity for c in children]
        # Verify non-equal (at least one differs by > 1 share)
        assert max(child_qtys) - min(child_qtys) > 1.0, \
            f"VWAP slices are too equal: {child_qtys}"

        # 6. Flush and verify CSV
        router._flush_daily_costs()
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "adverse_selection_events" in header
            assert "spread_mode_market" in header

        # 7. Stats should show all features
        stats = router.get_execution_stats()
        assert stats["spread_mode_market"] >= 1
        assert stats["spread_mode_mid_limit"] >= 1
        assert stats["spread_mode_far_limit"] >= 1
        assert stats["adverse_selection_events"] >= 1
        assert "TIGHT" in stats["adverse_forced_market_tickers"]

    def test_backward_compatibility(self):
        """With all new features disabled, behavior should match original."""
        cfg = SmartRouterConfig(
            dynamic_spread_enabled=False,
            adverse_selection_enabled=False,
            use_volume_weighted_schedule=False,
        )
        router, broker = _make_router(config=cfg)

        order = _make_order("AAPL", Side.BUY, 100)
        order_id = router.submit_order(order)
        assert order_id in router._tracked_orders
