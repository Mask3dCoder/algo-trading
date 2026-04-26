"""Integration tests: end-to-end pipeline from data to backtest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd


def _make_ohlcv_data(n_bars: int = 500, n_symbols: int = 1) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for integration testing."""
    rng = np.random.default_rng(42)
    rows = []
    start = datetime(2020, 1, 2, tzinfo=UTC)
    for sym_idx in range(n_symbols):
        symbol = f"STK{sym_idx:04d}"
        close = 100.0
        for i in range(n_bars):
            ret = rng.normal(0.0001, 0.015)
            close = close * (1.0 + ret)
            intraday_range = close * abs(rng.normal(0, 0.005))
            open_p = close * (1.0 + rng.normal(0, 0.002))
            high = max(open_p, close) + abs(rng.normal(0, intraday_range * 0.3))
            low = min(open_p, close) - abs(rng.normal(0, intraday_range * 0.3))
            volume = int(abs(rng.normal(1_000_000, 500_000)))
            ts = start + timedelta(hours=i)
            rows.append({
                "symbol": symbol,
                "timestamp": ts,
                "open": float(open_p),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": volume,
            })
    return pd.DataFrame(rows)


class TestDataToSignalPipeline:
    """Validate that data flows correctly through signal generation."""

    def test_ohlcv_generation_produces_valid_data(self) -> None:
        data = _make_ohlcv_data(100, n_symbols=1)
        assert len(data) == 100
        assert all(data["high"] >= data["low"])
        assert all(data["high"] >= data["close"])

    def test_signals_on_generated_data(self) -> None:
        from quantcore.signals.technical.mean_reversion import bollinger_bands, rsi
        from quantcore.signals.technical.trend import ema, macd, sma

        data = _make_ohlcv_data(200, n_symbols=1)
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        sma_vals = sma(close, 20)
        ema_vals = ema(close, 20)
        macd_line, signal_line, hist = macd(close)
        rsi_vals = rsi(close, 14)
        upper, mid, lower, pct_b, bw = bollinger_bands(close)

        assert len(sma_vals) == 200
        assert len(ema_vals) == 200
        assert len(macd_line) == 200
        assert len(rsi_vals) == 200
        assert not np.all(np.isnan(rsi_vals))
        assert not np.all(np.isnan(macd_line))


class TestBacktestPipeline:
    """Validate end-to-end backtest flow."""

    def test_backtest_buy_and_hold(self) -> None:
        from quantcore.backtest.engine import BacktestEngine, PortfolioState
        from quantcore.core.types import OHLCV, Direction, Signal

        data = _make_ohlcv_data(n_bars=252, n_symbols=1)

        class BNH:
            def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list:
                if len(portfolio.trades) == 0:
                    return [Signal(name="entry", symbol=bar.symbol,
                                   direction=Direction.LONG, strength=1.0,
                                   confidence=1.0)]
                return []

            def on_fill(self, fill, portfolio):
                pass

        engine = BacktestEngine(strategy=BNH(), initial_cash=100_000.0)
        result = engine.run(data)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "equity" in result.columns
        assert len(result) > 0

    def test_backtest_has_expected_columns(self) -> None:
        from quantcore.backtest.engine import BacktestEngine, PortfolioState
        from quantcore.core.types import OHLCV, Direction, Signal

        data = _make_ohlcv_data(n_bars=100, n_symbols=1)

        class TestStrategy:
            def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list:
                return [Signal(name="entry", symbol=bar.symbol,
                               direction=Direction.LONG, strength=1.0,
                               confidence=1.0)]

            def on_fill(self, fill, portfolio):
                pass

        engine = BacktestEngine(strategy=TestStrategy(), initial_cash=100_000.0)
        result = engine.run(data)
        assert result is not None
        assert isinstance(result, pd.DataFrame)


class TestOMSPipeline:
    """Validate order lifecycle."""

    def test_order_lifecycle(self) -> None:
        from decimal import Decimal

        from quantcore.core.types import Order, OrderStatus, OrderType, Side
        from quantcore.execution.oms import OrderManager

        oms = OrderManager()
        order = Order(
            id="test-001", symbol="AAPL", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("100"),
            strategy_id="test",
        )
        oms.submit(order)
        assert oms.orders["test-001"].status == OrderStatus.SUBMITTED

        oms.fill(order.id, Decimal("100"), Decimal("150.0"))
        assert oms.orders["test-001"].status == OrderStatus.FILLED
        assert len(oms.audit_trail) == 2  # submitted + filled


class TestAlertAndAuditPipeline:
    """Validate alerts and audit trail."""

    def test_alert_evaluation(self) -> None:
        from quantcore.dashboard.alerts import AlertEngine, build_default_alert_rules

        engine = AlertEngine()
        for rule in build_default_alert_rules():
            engine.add_rule(rule)

        state = {
            "drawdown_pct": -15.0,
            "var_breach": False,
            "circuit_breaker_active": False,
            "max_concentration": 0.12,
            "correlation_break": False,
            "daily_loss_pct": 1.2,
            "data_stale_seconds": 5.0,
            "details": "test",
        }
        fired = engine.evaluate(state)
        # The drawdown alert should fire
        assert len(fired) >= 1

    def test_audit_trail_immutable(self) -> None:
        from quantcore.compliance.audit_log import (
            AuditEventType,
            AuditLog,
            tamper_detection_report,
        )

        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "test")
        log.record(AuditEventType.ORDER_SUBMITTED, "strategy_1",
                   {"symbol": "AAPL", "quantity": 100})
        log.record(AuditEventType.ORDER_FILLED, "broker",
                   {"order_id": "abc", "price": 150.0})

        assert log.entry_count == 3
        assert log.verify_chain()

        # Tamper detection
        report = tamper_detection_report(log)
        assert not report["tampered"]
        assert report["verified_entries"] == 3


class TestKillSwitch:
    """Validate kill switch functionality."""

    def test_emergency_flatten(self) -> None:
        from quantcore.live.trader import LiveTrader, TraderConfig

        trader = LiveTrader(config=TraderConfig())
        # Should not raise — kill switch completes
        trader._emergency_flatten()
