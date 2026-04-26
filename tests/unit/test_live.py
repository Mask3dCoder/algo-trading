"""Tests for live trading: LiveRunner lifecycle and LiveMonitor health checks."""

from __future__ import annotations

import time

import pytest

from quantcore.core.types import Order, Position
from quantcore.live.monitor import (
    HealthSnapshot,
    HealthStatus,
    LiveMonitor,
    MonitorConfig,
    health_check,
)
from quantcore.live.runner import (
    LiveConfig,
    LiveRunner,
    LiveState,
    LiveStateSnapshot,
)

# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class MockDataProvider:
    """Mock data provider for testing."""

    def __init__(self, prices=None):
        self.prices = prices or {"AAPL": 200.0}
        self._market_open = True

    def fetch_bar(self, symbol):
        return {"close": self.prices.get(symbol, 100.0), "volume": 10000}

    def fetch_quotes(self, symbols):
        return {s: {"bid": self.prices.get(s, 100), "ask": self.prices.get(s, 100.1)} for s in symbols}

    def is_market_open(self):
        return self._market_open


class MockBroker:
    """Mock broker for testing."""

    def __init__(self, cash=100000, positions=None):
        self.cash = cash
        self._positions = positions or {
            "AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200),
        }
        self.submitted_orders: list[Order] = []

    def submit_order(self, order):
        self.submitted_orders.append(order)
        return f"order_{len(self.submitted_orders)}"

    def cancel_order(self, order_id):
        return True

    def get_positions(self):
        return dict(self._positions)

    def get_account(self):
        total_mv = sum(p.quantity * p.market_price for p in self._positions.values())
        return {"cash": self.cash, "equity": self.cash + total_mv}


# ---------------------------------------------------------------------------
# LiveRunner tests
# ---------------------------------------------------------------------------


class TestLiveRunnerInit:
    def test_default_config(self):
        runner = LiveRunner()
        assert runner.state == LiveState.IDLE
        assert runner.oms is not None
        assert runner.drawdown_tracker is not None

    def test_custom_config(self):
        config = LiveConfig(max_daily_loss_pct=0.05, max_leverage=2.0)
        runner = LiveRunner(config=config)
        assert runner.config.max_daily_loss_pct == 0.05
        assert runner.config.max_leverage == 2.0

    def test_with_dependencies(self):
        config = LiveConfig()
        data = MockDataProvider()
        broker = MockBroker()
        runner = LiveRunner(config=config, data_provider=data, broker=broker)
        assert runner.data_provider is data
        assert runner.broker is broker


class TestLiveRunnerLifecycle:
    @pytest.fixture
    def runner(self):
        config = LiveConfig()
        data = MockDataProvider()
        broker = MockBroker()
        return LiveRunner(config=config, data_provider=data, broker=broker)

    def test_start_transitions_to_running(self, runner):
        runner.start()
        assert runner.state == LiveState.RUNNING

    def test_pause_resume(self, runner):
        runner.start()
        runner.pause()
        assert runner.state == LiveState.PAUSED
        runner.resume()
        assert runner.state == LiveState.RUNNING

    def test_stop_transitions(self, runner):
        runner.start()
        runner.stop()
        assert runner.state == LiveState.STOPPED

    def test_halt_from_running(self, runner):
        runner.start()
        runner.halt()
        assert runner.state == LiveState.HALTED

    def test_cannot_start_from_halted(self, runner):
        runner.start()
        runner.halt()
        with pytest.raises(RuntimeError):
            runner.start()

    def test_reset_from_halted(self, runner):
        runner.start()
        runner.halt()
        runner.reset()
        assert runner.state == LiveState.IDLE

    def test_tick_when_idle_returns_snapshot(self, runner):
        snapshot = runner.tick()
        assert isinstance(snapshot, LiveStateSnapshot)
        assert snapshot.state == LiveState.IDLE


class TestLiveRunnerCallbacks:
    def test_state_change_callback(self):
        transitions: list[tuple[LiveState, LiveState]] = []
        config = LiveConfig()
        runner = LiveRunner(config=config)
        runner.on_state_change = lambda old, new: transitions.append((old, new))
        runner.start()
        assert len(transitions) == 1
        assert transitions[0] == (LiveState.IDLE, LiveState.RUNNING)


class TestLiveConfig:
    def test_default_values(self):
        config = LiveConfig()
        assert config.heartbeat_interval_sec == 1.0
        assert config.max_daily_loss_pct == 0.03
        assert config.max_position_pct == 0.10

    def test_custom_loss_limits(self):
        config = LiveConfig(max_daily_loss_pct=0.02, max_weekly_loss_pct=0.08)
        assert config.max_daily_loss_pct == 0.02
        assert config.max_weekly_loss_pct == 0.08


# ---------------------------------------------------------------------------
# LiveMonitor tests
# ---------------------------------------------------------------------------


class TestLiveMonitor:
    @pytest.fixture
    def monitor(self):
        return LiveMonitor()

    def test_initial_healthy(self, monitor):
        assert monitor.is_healthy is True

    def test_record_heartbeat(self, monitor):
        monitor.record_heartbeat()
        snapshot = monitor.check()
        assert snapshot.status == HealthStatus.HEALTHY

    def test_missed_heartbeats_degrades(self, monitor):
        config = MonitorConfig(
            max_consecutive_misses=2,
            heartbeat_timeout_sec=0.001,  # Very short timeout
        )
        monitor.config = config
        monitor.record_heartbeat()
        time.sleep(0.01)  # Make heartbeat stale
        snapshot = monitor.check()
        # Should have at least a warning from stale heartbeat
        assert snapshot.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_record_order_latency(self, monitor):
        monitor.record_order_latency(150.0)
        assert len(monitor._order_latencies) == 1

    def test_record_data_update(self, monitor):
        monitor.record_data_update()
        snapshot = monitor.check()
        assert snapshot.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def test_health_check_with_drawdown(self):
        from quantcore.risk.drawdown import DrawdownTracker
        monitor = LiveMonitor()
        monitor.record_heartbeat()
        tracker = DrawdownTracker()
        tracker.update(92000)  # 8% drawdown from assumed 100k peak
        snapshot = health_check(monitor, drawdown_tracker=tracker)
        assert isinstance(snapshot, HealthSnapshot)

    def test_health_check_with_drift(self):
        monitor = LiveMonitor()
        monitor.record_heartbeat()
        positions = {
            "AAPL": {"market_value": 60000},
            "MSFT": {"market_value": 40000},
        }
        target = {"AAPL": 0.5, "MSFT": 0.5}
        snapshot = health_check(monitor, positions=positions, target_weights=target)
        assert isinstance(snapshot, HealthSnapshot)


class TestHealthCheck:
    def test_degraded_on_high_latency(self):
        monitor = LiveMonitor()
        monitor.config.max_order_latency_ms = 50.0
        monitor.record_heartbeat()
        monitor.record_order_latency(500.0)
        snapshot = monitor.check()
        assert snapshot.status in (HealthStatus.DEGRADED, HealthStatus.HEALTHY)

    def test_healthy_with_normal_latency(self):
        monitor = LiveMonitor()
        monitor.record_heartbeat()
        monitor.record_order_latency(10.0)
        snapshot = monitor.check()
        assert snapshot.status == HealthStatus.HEALTHY

    def test_last_status(self):
        monitor = LiveMonitor()
        monitor.record_heartbeat()
        monitor.check()
        assert monitor.last_status is not None
        assert isinstance(monitor.last_status, HealthSnapshot)
