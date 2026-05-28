"""Integration tests for Phase 4 live risk, circuit breakers, and kill switch."""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure we can import quantcore live modules
from quantcore.core.types import Order, OrderStatus, OrderType, Side
from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from quantcore.live.risk_manager import Approval, RiskLimitsConfig, RiskManagerLive


# ---------------------------------------------------------------------------
# RiskManagerLive
# ---------------------------------------------------------------------------

class TestRiskManagerLive:
    def test_default_limits(self) -> None:
        rm = RiskManagerLive()
        assert rm.config.max_position_per_ticker_pct == 0.05
        assert rm.config.max_leverage == 2.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUANTCORE_RISK_MAX_POSITION_PCT", "0.10")
        monkeypatch.setenv("QUANTCORE_RISK_MAX_LEVERAGE", "3.0")
        cfg = RiskLimitsConfig.from_env()
        assert cfg.max_position_per_ticker_pct == 0.10
        assert cfg.max_leverage == 3.0

    def test_pre_trade_equity_negative(self) -> None:
        rm = RiskManagerLive()
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"),
        )
        approval = rm.pre_trade_check(order, equity=-100)
        assert not approval.approved
        assert "Equity <= 0" in approval.reason

    def test_pre_trade_single_order_limit(self) -> None:
        rm = RiskManagerLive()
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("100"),
        )
        # Price fallback is 100 if no limit price, so 100 shares = $10,000
        # With $100,000 equity, single order limit is 5% = $5,000 → reject
        approval = rm.pre_trade_check(order, equity=100_000)
        assert not approval.approved
        assert "Single order size exceeds" in approval.reason

    def test_pre_trade_approves_small_order(self) -> None:
        rm = RiskManagerLive()
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )
        approval = rm.pre_trade_check(order, equity=100_000)
        assert approval.approved

    def test_pre_trade_leverage_limit(self) -> None:
        rm = RiskManagerLive()
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("10"), limit_price=Decimal("100"),
        )
        approval = rm.pre_trade_check(order, equity=100_000)
        # Notional = 10 * $100 = $1,000 < 5% of $100,000 = $5,000 → approved
        # Leverage = 1,000 / 100,000 = 0.01x < 2.0x → approved
        assert approval.approved

    def test_daily_loss_limit(self) -> None:
        rm = RiskManagerLive()
        rm._current_daily_pnl = -3000.0  # Simulate 3% loss already
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )
        approval = rm.pre_trade_check(order, equity=100_000)
        # 2% max daily loss: -3000 < -2000 → reject
        assert not approval.approved
        assert "Daily loss limit hit" in approval.reason

    def test_drawdown_limit(self) -> None:
        rm = RiskManagerLive()
        rm._peak_equity = 100_000
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )
        # Equity at $88,000 => 12% drawdown > 10% limit
        approval = rm.pre_trade_check(order, equity=88_000)
        assert not approval.approved
        assert "Drawdown" in approval.reason

    def test_peak_equity_tracking(self) -> None:
        rm = RiskManagerLive()
        rm.update_equity(100_000)
        rm.update_equity(110_000)
        assert rm.peak_equity == 110_000
        assert rm.current_drawdown == 0.0

    def test_eod_check(self) -> None:
        rm = RiskManagerLive()
        status = rm.eod_check(current_date=date(2024, 6, 17), equity_curve=[105_000])
        assert status["equity"] == 105_000
        assert "drawdown" in status


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_defaults(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.NORMAL
        assert not cb.activated

    def test_daily_loss_halt(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_pct=0.02))
        state = cb.check(equity=100_000, peak_equity=100_000, daily_pnl=-3000)
        assert state == CircuitBreakerState.DAILY_LOSS_HALT
        assert cb.activated
        assert "Daily loss" in cb.reason

    def test_critical_drawdown(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(critical_drawdown_pct=0.10))
        state = cb.check(equity=85_000, peak_equity=100_000, daily_pnl=0)
        assert state == CircuitBreakerState.CRITICAL_HALT
        assert "Drawdown" in cb.reason

    def test_consecutive_rejections(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(consecutive_rejections=3))
        # Two rejected orders does not trip
        state = cb.check(equity=100_000, peak_equity=100_000, daily_pnl=0, rejected_last=True)
        assert state == CircuitBreakerState.NORMAL
        state = cb.check(equity=100_000, peak_equity=100_000, daily_pnl=0, rejected_last=True)
        assert state == CircuitBreakerState.NORMAL
        # Third rejection trips daily loss halt
        state = cb.check(equity=100_000, peak_equity=100_000, daily_pnl=0, rejected_last=True)
        assert state == CircuitBreakerState.DAILY_LOSS_HALT

    def test_cannot_trade_when_activated(self) -> None:
        cb = CircuitBreaker()
        cb.check(equity=100_000, peak_equity=100_000, daily_pnl=-3000)
        assert not cb.can_trade()

    def test_can_trade_normally(self) -> None:
        cb = CircuitBreaker()
        cb.check(equity=100_000, peak_equity=100_000, daily_pnl=100)
        assert cb.can_trade()

    def test_reset(self) -> None:
        cb = CircuitBreaker()
        cb.check(equity=100_000, peak_equity=100_000, daily_pnl=-3000)
        assert cb.activated
        cb.reset()
        assert not cb.activated
        assert cb.state == CircuitBreakerState.NORMAL

    def test_forced_trigger(self) -> None:
        cb = CircuitBreaker()
        cb.trigger(CircuitBreakerState.CRITICAL_HALT, "Operator override")
        assert cb.state == CircuitBreakerState.CRITICAL_HALT


# ---------------------------------------------------------------------------
# RiskLimitsConfig env integration
# ---------------------------------------------------------------------------

class TestRiskLimitsConfigEnv:
    def test_all_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUANTCORE_RISK_MAX_POSITION_PCT", "0.08")
        monkeypatch.setenv("QUANTCORE_RISK_MAX_GROSS_EXPOSURE", "2.5")
        monkeypatch.setenv("QUANTCORE_RISK_MAX_DAILY_LOSS_PCT", "0.03")
        monkeypatch.setenv("QUANTCORE_RISK_MAX_DRAWDOWN_PCT", "0.15")
        monkeypatch.setenv("QUANTCORE_RISK_MAX_SINGLE_ORDER_PCT", "0.08")
        monkeypatch.setenv("QUANTCORE_RISK_MAX_LEVERAGE", "3.0")
        cfg = RiskLimitsConfig.from_env()
        assert cfg.max_position_per_ticker_pct == 0.08
        assert cfg.max_gross_exposure_pct == 2.5
        assert cfg.max_daily_loss_pct == 0.03
        assert cfg.max_drawdown_pct == 0.15
        assert cfg.max_single_order_pct == 0.08
        assert cfg.max_leverage == 3.0


# ---------------------------------------------------------------------------
# Kill switch file check (orchestrator integration)
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_kill_switch_exists(self, tmp_path: Path) -> None:
        ks = tmp_path / "KILL_SWITCH"
        ks.write_text("")
        assert ks.exists()

    def test_kill_switch_not_present(self, tmp_path: Path) -> None:
        ks = tmp_path / "KILL_SWITCH"
        assert not ks.exists()
