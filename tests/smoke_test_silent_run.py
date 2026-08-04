"""Smoke test for Silent Run modules."""
from quantcore.live.paper_trading.live_connector import (
    LiveStreamConnector, PassiveExecutionProxy, SilentRunOrchestrator, Tick
)
from quantcore.live.paper_trading.connection_monitor import (
    ConnectionMonitor, ConnectionState
)


def test_passive_proxy():
    proxy = PassiveExecutionProxy()
    record = proxy.log_bypassed_order("SPY", "buy", 10, 450.0, 0.02)
    assert record.ticker == "SPY"
    assert record.mode == "passive"
    assert proxy.trade_count == 1
    print("PassiveExecutionProxy: OK (trades logged=%d)" % proxy.trade_count)


def test_connection_monitor():
    mon = ConnectionMonitor(alert_threshold_sec=30.0)
    mon.register_feed("test_ws")
    mon.heartbeat("test_ws")
    assert mon.is_healthy("test_ws")
    print("ConnectionMonitor: OK (feed healthy=%s)" % mon.is_healthy("test_ws"))
    mon.stop()


def test_live_connector():
    conn = LiveStreamConnector(tickers=["SPY", "QQQ"])
    # Simulate a tick dispatch
    tick = conn._simulate_tick("SPY")
    assert tick.ticker == "SPY"
    assert tick.mid > 0
    print("LiveStreamConnector: OK (tick=%s @ %.2f)" % (tick.ticker, tick.mid))


if __name__ == "__main__":
    test_passive_proxy()
    test_connection_monitor()
    test_live_connector()
    print("All Silent Run modules verified.")
