"""Tests for live trading additions: paper, trader, watchdog, reconciliation."""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest

from quantcore.live.paper import (
    PaperConfig, PaperFill, PaperAccount, PaperBroker, compute_graduation_metrics,
)
from quantcore.live.trader import (
    TraderState, TraderConfig, TraderSnapshot, LiveTrader,
)
from quantcore.live.watchdog import (
    WatchdogState, WatchdogConfig, WatchdogSnapshot, Watchdog,
)
from quantcore.live.reconciliation import (
    ReconciliationStatus, MismatchSeverity,
    PositionReconciliation, ReconciliationReport,
    ReconciliationConfig, ReconciliationEngine,
    compute_pnl_difference,
)
from quantcore.core.types import Position, Order, Side, OrderType


# ---------------------------------------------------------------------------
# Paper Broker tests
# ---------------------------------------------------------------------------

class TestPaperConfig:
    def test_default(self):
        c = PaperConfig()
        assert c.initial_capital == 100_000.0
        assert c.slippage_bps == 2.0

    def test_custom(self):
        c = PaperConfig(initial_capital=50000, slippage_bps=5.0)
        assert c.initial_capital == 50000
        assert c.slippage_bps == 5.0


class TestPaperBroker:
    @pytest.fixture
    def broker(self):
        return PaperBroker()

    def test_init(self, broker):
        assert broker.account.cash == 100_000.0
        assert broker.portfolio_value == 100_000.0

    def test_submit_order(self, broker):
        order = Order(
            id="pb-001", symbol="AAPL", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        oid = broker.submit_order(order)
        assert oid == "pb-001"

    def test_update_market_data(self, broker):
        broker.update_market_data({"AAPL": 200.0, "MSFT": 400.0})
        assert broker._current_prices["AAPL"] == 200.0

    def test_process_fills(self, broker):
        broker.update_market_data({"AAPL": 200.0})
        order = Order(
            id="pb-002", symbol="AAPL", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        broker.submit_order(order)
        fills = broker.process_fills()
        assert len(fills) >= 1
        fill = fills[0]
        assert fill.symbol == "AAPL"
        assert fill.quantity > 0
        assert fill.fill_price > 0

    def test_get_account(self, broker):
        broker.update_market_data({"AAPL": 200.0})
        order = Order(
            id="pb-003", symbol="AAPL", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        broker.submit_order(order)
        broker.process_fills()
        acct = broker.get_account()
        assert "cash" in acct
        assert "equity" in acct

    def test_get_positions(self, broker):
        broker.update_market_data({"AAPL": 200.0})
        order = Order(
            id="pb-004", symbol="AAPL", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        broker.submit_order(order)
        broker.process_fills()
        positions = broker.get_positions()
        assert "AAPL" in positions

    def test_equity_curve(self, broker):
        broker.update_market_data({"AAPL": 200.0})
        broker.update_equity()
        dates, values = broker.get_equity_curve()
        assert len(dates) == 1
        assert values[0] > 0

    def test_reset(self, broker):
        broker.update_market_data({"AAPL": 200.0})
        order = Order(
            id="pb-005", symbol="AAPL", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        broker.submit_order(order)
        broker.process_fills()
        broker.reset()
        assert broker.account.cash == 100_000.0
        assert len(broker.account.positions) == 0


class TestComputeGraduationMetrics:
    def test_returns_dict(self):
        broker = PaperBroker()
        broker.update_market_data({"AAPL": 200.0})
        for i in range(10):
            broker.update_equity()
        metrics = compute_graduation_metrics(broker)
        assert "sharpe" in metrics
        assert "max_dd" in metrics
        assert "ready" in metrics


# ---------------------------------------------------------------------------
# Live Trader tests
# ---------------------------------------------------------------------------

class TestTraderConfig:
    def test_default(self):
        c = TraderConfig()
        assert c.startup_checks is True

    def test_custom(self):
        c = TraderConfig(reconnect_delay_sec=10.0, max_startup_retries=5)
        assert c.reconnect_delay_sec == 10.0
        assert c.max_startup_retries == 5


class TestLiveTrader:
    def test_init(self):
        trader = LiveTrader()
        assert trader.state == TraderState.INIT
        assert trader.is_healthy is True

    def test_snapshot_idle(self):
        trader = LiveTrader()
        snap = trader._snapshot()
        assert isinstance(snap, TraderSnapshot)
        assert snap.trader_state == TraderState.INIT

    def test_startup_no_deps(self):
        trader = LiveTrader()
        result = trader.startup()
        assert result is True

    def test_shutdown(self):
        trader = LiveTrader()
        trader.shutdown()
        assert trader.state == TraderState.SHUTDOWN


# ---------------------------------------------------------------------------
# Watchdog tests
# ---------------------------------------------------------------------------

class TestWatchdog:
    def test_init(self):
        wd = Watchdog()
        assert wd.state == WatchdogState.HEALTHY
        assert wd.is_alive is True

    def test_heartbeat(self):
        wd = Watchdog()
        wd.heartbeat()
        assert wd._heartbeat_count == 1
        assert wd.state == WatchdogState.HEALTHY

    def test_check_with_heartbeat(self):
        wd = Watchdog()
        wd.heartbeat()
        snap = wd.check()
        assert snap.state == WatchdogState.HEALTHY
        assert snap.heartbeats_received == 1

    def test_disable(self):
        wd = Watchdog()
        wd.disable()
        assert wd.is_alive is False

    def test_enable_after_disable(self):
        wd = Watchdog()
        wd.disable()
        wd.enable()
        assert wd.is_alive is True

    def test_on_trigger(self):
        callback_called = []
        def cb(state):
            callback_called.append(state)
        wd = Watchdog()
        wd.on_trigger(cb)
        wd._trigger_actions()
        assert len(callback_called) >= 1


# ---------------------------------------------------------------------------
# Reconciliation tests
# ---------------------------------------------------------------------------

class TestReconciliationEngine:
    def test_init(self):
        engine = ReconciliationEngine()
        assert engine.config.position_tolerance_pct == 0.01

    def test_matched_positions(self):
        engine = ReconciliationEngine()
        internal = {
            "AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200),
        }
        broker = {
            "AAPL": {"symbol": "AAPL", "quantity": 100, "market_value": 20000},
        }
        report = engine.reconcile(internal, broker, 80000.0, {"cash": 80000, "equity": 100000})
        assert report.status == ReconciliationStatus.MATCHED

    def test_mismatch_detected(self):
        engine = ReconciliationEngine()
        internal = {
            "AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200),
        }
        broker = {
            "AAPL": {"symbol": "AAPL", "quantity": 90, "market_value": 18000},
        }
        report = engine.reconcile(internal, broker, 80000.0, {"cash": 80000, "equity": 100000})
        assert report.total_mismatches >= 1

    def test_missing_internal(self):
        engine = ReconciliationEngine()
        internal = {}
        broker = {
            "AAPL": {"symbol": "AAPL", "quantity": 100, "market_value": 20000},
        }
        report = engine.reconcile(internal, broker, 80000.0, {"cash": 80000, "equity": 100000})
        assert report.status == ReconciliationStatus.MISMATCHED


class TestReconciliationStatus:
    def test_enum_values(self):
        assert ReconciliationStatus.MATCHED == "matched"
        assert ReconciliationStatus.MISMATCHED == "mismatched"


class TestComputePnlDifference:
    def test_equal(self):
        internal_trades = [{"realized_pnl": 500}, {"realized_pnl": -200}]
        broker_trades = [{"realized_pnl": 500}, {"realized_pnl": -200}]
        result = compute_pnl_difference(internal_trades, broker_trades)
        assert result["delta"] == 0.0

    def test_different(self):
        internal_trades = [{"realized_pnl": 500}]
        broker_trades = [{"realized_pnl": 400}]
        result = compute_pnl_difference(internal_trades, broker_trades)
        assert result["delta"] == 100.0
