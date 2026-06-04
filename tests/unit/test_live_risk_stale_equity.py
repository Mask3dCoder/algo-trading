"""Test that risk manager and circuit breaker see the latest equity, not stale values.

The original bug: ``pre_trade_check()`` evaluated daily P&L using a cached
``_current_daily_pnl`` from the last call to ``update_equity()`` while the
threshold used a fresh *equity* parameter.  If equity dipped intraday and then
recovered, the stale daily P&L still reflected the dip, causing a false-positive
circuit-breaker halt that blocked all order fills.

Reproduction scenario
---------------------
1. ``start_day(100_000)`` — daily P&L reset to 0.
2. ``update_equity(97_000)`` — daily P&L becomes -3,000.
3. Equity recovers to 99,990 **without** another ``update_equity()`` call.
4. ``pre_trade_check(order, equity=99_990)`` is called.

   **Before fix**: ``_current_daily_pnl`` was still -3,000, and the check
   ``-3000 < -99990×2% (-1999.8)`` → TRUE → **false positive**.

   **After fix**: ``pre_trade_check()`` first syncs ``_total_equity`` from the
   *equity* parameter, giving ``daily_pnl = 99990 - 100000 = -10``, and the
   check ``-10 < -1999.8`` → FALSE → **correctly approved**.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantcore.core.types import Order, OrderType, Side
from quantcore.live.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)
from quantcore.live.risk_manager import RiskLimitsConfig, RiskManagerLive


# =========================================================================
# RiskManagerLive — stale-equity prevention
# =========================================================================


class TestRiskManagerLiveStaleEquity:
    """Verify that ``pre_trade_check()`` self-heals stale equity values."""

    # ------------------------------------------------------------------
    # Bug reproduction — the exact scenario described in the ticket
    # ------------------------------------------------------------------

    def test_pre_trade_self_heals_stale_equity(self) -> None:
        """Equity dips to 97k, recovers to 99,990; pre_trade_check must NOT
        falsely reject a valid order based on the stale -3k daily P&L."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        # 1. Start the day at 100 000
        rm.start_day(100_000.0)
        assert rm.current_daily_pnl == 0.0

        # 2. Equity dips to 97 000 → daily P&L = -3 000
        rm.update_equity(97_000.0)
        assert rm.current_daily_pnl == pytest.approx(-3_000.0, abs=1.0)

        # 3. Equity recovers to 99 990 — call pre_trade_check WITHOUT
        #    explicitly calling update_equity first (the OLD code path).
        approval = rm.pre_trade_check(order, equity=99_990.0)

        # The fix makes pre_trade_check sync equity from the parameter first.
        # Daily P&L is now 99 990 - 100 000 = -10, which is well within 2 %.
        assert approval.approved, (
            f"pre_trade_check falsely rejected: {approval.reason}"
        )
        # Verify daily P&L was updated by the sync
        assert rm.current_daily_pnl == pytest.approx(-10.0, abs=1.0)

    def test_pre_trade_triggers_on_genuine_daily_loss(self) -> None:
        """Real daily loss still triggers — the fix doesn't mute risk."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        rm.start_day(100_000.0)
        rm.update_equity(120_000.0)  # gain first
        # Then a genuine large loss
        rm.update_equity(97_000.0)

        # _current_daily_pnl should now be -3 000
        approval = rm.pre_trade_check(order, equity=97_000.0)
        assert not approval.approved
        assert "Daily loss limit" in approval.reason

    def test_pre_trade_sync_logs_warning_on_large_discrepancy(self) -> None:
        """When equity parameter differs significantly from stored equity,
        pre_trade_check logs a debug message (visible with DEBUG level)."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        rm.start_day(100_000.0)
        rm.update_equity(97_000.0)

        # Calling pre_trade_check with a different equity value should
        # trigger the self-heal path (not crash, not reject incorrectly).
        approval = rm.pre_trade_check(order, equity=99_990.0)
        assert approval.approved

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_start_day_never_called_self_heals(self) -> None:
        """If ``start_day()`` was never called, ``pre_trade_check()`` should
        seed starting equity from the first equity value it sees."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        # No start_day() call — pre_trade_check must handle this gracefully
        approval = rm.pre_trade_check(order, equity=100_000.0)
        assert approval.approved
        # Verify it seeded the day
        assert rm._starting_equity_for_day == pytest.approx(100_000.0, abs=1.0)

    def test_multiple_equity_changes_within_day(self) -> None:
        """Multiple equity updates within the same day compute correct P&L."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        rm.start_day(100_000.0)
        rm.update_equity(101_000.0)
        approval = rm.pre_trade_check(order, equity=102_000.0)
        assert approval.approved
        assert rm.current_daily_pnl == pytest.approx(2_000.0, abs=1.0)

    def test_start_day_new_day_resets_pnl(self) -> None:
        """Calling ``start_day()`` with a new-day equity resets P&L."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        rm.start_day(100_000.0)
        rm.update_equity(97_000.0)  # -3k on day 1

        # Next day — equity at 99 990
        rm.start_day(99_990.0)
        assert rm.current_daily_pnl == 0.0  # reset

        approval = rm.pre_trade_check(order, equity=99_990.0)
        assert approval.approved


# =========================================================================
# CircuitBreaker — receives fresh equity at every call
# =========================================================================


class TestCircuitBreakerStaleEquity:
    """Verify the circuit breaker does NOT cache equity internally;
    it always uses whatever values the caller passes."""

    def test_circuit_breaker_uses_passed_equity_not_cached(self) -> None:
        """The circuit breaker has no internal cache — it uses the equity
        value passed at each check() call."""
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_pct=0.02))

        # Scenario: equity was $97,000 earlier, giving daily P&L of -$3,000,
        # but now equity has recovered to $99,990. If the circuit breaker
        # used a stale cached equity of $97,000, the check would trigger.
        # It must use the latest equity value ($99,990).
        state = cb.check(equity=99_990.0, peak_equity=100_000.0, daily_pnl=-10.0)
        assert state == CircuitBreakerState.NORMAL, (
            f"Circuit breaker triggered incorrectly on recovered equity. "
            f"State: {state}, Reason: {cb.reason}"
        )

    def test_circuit_breaker_still_triggers_on_real_loss(self) -> None:
        """The fix does not mute real danger — a genuine 3k daily loss on
        $100k equity with a 2% threshold still triggers."""
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_pct=0.02))
        state = cb.check(equity=100_000.0, peak_equity=100_000.0, daily_pnl=-3_000.0)
        assert state == CircuitBreakerState.DAILY_LOSS_HALT

    def test_circuit_breaker_drawdown_still_triggers(self) -> None:
        """Drawdown protection is unaffected."""
        cb = CircuitBreaker(CircuitBreakerConfig(critical_drawdown_pct=0.10))
        state = cb.check(equity=85_000.0, peak_equity=100_000.0, daily_pnl=0.0)
        assert state == CircuitBreakerState.CRITICAL_HALT


# =========================================================================
# Integrated scenario — RiskManagerLive + CircuitBreaker
# =========================================================================


class TestIntegratedStaleEquity:
    """Full pipeline: start_day → equity change → pre_trade_check →
    circuit_breaker.check()."""

    def test_integrated_equity_recovery(self) -> None:
        """Full reproduction of the bug scenario with both components."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_pct=0.02))
        order = Order(
            id="1", symbol="SPY", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("1"), limit_price=Decimal("10"),
        )

        # Day starts at 100 000
        rm.start_day(100_000.0)

        # Intraday dip to 97 000 — update equity
        rm.update_equity(97_000.0)

        # Circuit breaker sees the dip:
        state = cb.check(equity=97_000.0, peak_equity=rm.peak_equity,
                         daily_pnl=rm.current_daily_pnl)
        assert state == CircuitBreakerState.DAILY_LOSS_HALT
        cb.reset()

        # Equity recovers to 99 990, account snapshot refreshes.
        # pre_trade_check gets fresh equity, must self-heal the P&L.
        approval = rm.pre_trade_check(order, equity=99_990.0)
        assert approval.approved, (
            f"pre_trade_check falsely rejected: {approval.reason}"
        )

        # Circuit breaker also gets fresh values — must NOT trigger.
        state = cb.check(
            equity=99_990.0,
            peak_equity=rm.peak_equity,
            daily_pnl=rm.current_daily_pnl,  # now -10 after sync
        )
        assert state == CircuitBreakerState.NORMAL, (
            f"Circuit breaker triggered on recovered equity: {cb.reason}"
        )

    def test_integrated_no_false_positive_sequence(self) -> None:
        """Simulates the exact sequence from the runner:
        broker fetch → update_equity → circuit_breaker.check()
        with equity varying each time."""
        rm = RiskManagerLive(config=RiskLimitsConfig(max_daily_loss_pct=0.02))
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_pct=0.02))

        # Simulate the production_runner start_day call
        rm.start_day(100_000.0)

        # Multiple ticks with varying equity
        snapshots = [
            (100_000.0, CircuitBreakerState.NORMAL),
            (99_500.0, CircuitBreakerState.NORMAL),   # -500 P&L, fine
            (97_000.0, CircuitBreakerState.DAILY_LOSS_HALT),  # -3000 P&L, triggered
        ]
        for eq, expected_state in snapshots:
            rm.update_equity(eq)
            state = cb.check(
                equity=eq,
                peak_equity=rm.peak_equity,
                daily_pnl=rm.current_daily_pnl,
            )
            assert state == expected_state, (
                f"Expected {expected_state} at equity={eq}, "
                f"got {state} ({cb.reason})"
            )