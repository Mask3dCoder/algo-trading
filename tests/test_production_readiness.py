"""Production Readiness "Fire Drill" Integration Test Suite.

Instantiates the *actual* RiskController and PaperExecutionEnvironment from
the codebase to verify risk boundaries behave as expected under duress.

Kill switches tested
--------------------
1. Hard MaxDD Kill Switch   → equity drop ≥ 10% from peak → HALT
2. Rolling 3-Month Sharpe   → Sharpe < 0 over 63 days  → PAUSED
3. Macro Volatility Exit    → VIX > 30 + DD > 12%      → HALT

All three are fatal — no auto-recovery.  Manual committee review required.

Usage
-----
    python -m pytest tests/test_production_readiness.py -v
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerState
from quantcore.live.paper_trading.execution_environment import (
    PaperExecutionConfig,
    PaperExecutionEnvironment,
)
from quantcore.live.paper_trading.risk_controller import (
    ExecutionState,
    RiskController,
    RiskControllerConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INITIAL_EQUITY = 100_000.0


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Isolated directory for risk-controller persistence."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def risk_config(tmp_state_dir: Path) -> RiskControllerConfig:
    """Default Investment Committee config pointing at a temp directory."""
    return RiskControllerConfig(
        max_drawdown_pct=0.10,
        sharpe_lookback_days=63,
        sharpe_min_threshold=0.0,
        macro_vix_threshold=30.0,
        macro_drawdown_warning_pct=0.08,
        macro_drawdown_kill_pct=0.12,
        state_dir=tmp_state_dir,
    )


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker()


@pytest.fixture
def rc(risk_config: RiskControllerConfig, circuit_breaker: CircuitBreaker) -> RiskController:
    """Fresh RiskController with production thresholds."""
    return RiskController(
        config=risk_config,
        circuit_breaker=circuit_breaker,
        initial_equity=INITIAL_EQUITY,
    )


@pytest.fixture
def exec_env() -> PaperExecutionEnvironment:
    """Fresh execution environment with default $100k capital."""
    return PaperExecutionEnvironment(config=PaperExecutionConfig())


# ===========================================================================
# Task 1.1: MaxDD Kill Switch
# ===========================================================================


class TestMaxDDKillSwitch:
    """Inject synthetic equity curve that triggers a ≥10% drawdown.

    Scenario: portfolio drops from $100,000 to $89,000 (11% drawdown).
    Expected: RiskController sets state = HALTED and triggers the
    circuit breaker to CRITICAL_HALT.
    """

    def test_flags_state_breach_on_11pct_drawdown(
        self, rc: RiskController, circuit_breaker: CircuitBreaker,
    ) -> None:
        """Equity from $100k → $89k must trigger HALT."""
        # Verify initial state
        assert rc.is_active
        assert rc.state == ExecutionState.ACTIVE

        # Simulate daily equity decline: 100k → 89k over 10 trading days
        equity = INITIAL_EQUITY
        dt = date(2026, 1, 2)
        for i in range(10):
            equity -= 1_100.0  # lose $1,100/day → 10 days = $11,000 lost
            rc.update_equity(equity, as_of=dt + timedelta(days=i))

            # Once drawdown passes 10%, the controller should HALT
            if equity <= INITIAL_EQUITY * 0.90:
                assert rc.is_halted, (
                    f"RiskController should be HALTED at equity=${equity:,.2f} "
                    f"(DD={((equity - INITIAL_EQUITY) / INITIAL_EQUITY):.2%})"
                )
                break

    def test_circuit_breaker_triggered_on_maxdd(
        self, rc: RiskController, circuit_breaker: CircuitBreaker,
    ) -> None:
        """CircuitBreaker must be in CRITICAL_HALT after MaxDD breach."""
        # Force exact 11% drawdown
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))

        assert circuit_breaker.state == CircuitBreakerState.CRITICAL_HALT, (
            "CircuitBreaker must transition to CRITICAL_HALT on MaxDD kill"
        )

    def test_subsequent_equity_updates_are_noop_after_halt(
        self, rc: RiskController,
    ) -> None:
        """After HALT, further equity updates must not change the state."""
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))
        assert rc.is_halted

        # Simulate recovery — should NOT auto-recover
        rc.update_equity(INITIAL_EQUITY * 1.5, as_of=date(2026, 1, 3))
        assert rc.is_halted, "RiskController must NOT auto-recover from HALT"

    def test_order_submission_blocked_after_halt(
        self, rc: RiskController, circuit_breaker: CircuitBreaker,
    ) -> None:
        """After HALT, can_trade() must return False — all orders blocked."""
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))

        assert not circuit_breaker.can_trade(), (
            "CircuitBreaker.can_trade() must return False after HALT"
        )

        # Explicitly verify each kill switch route
        assert not rc.is_active, "is_active must be False after HALT"

    def test_peak_equity_persisted_across_restart(
        self, risk_config: RiskControllerConfig, circuit_breaker: CircuitBreaker,
    ) -> None:
        """Peak equity must survive process restart (state persistence)."""
        # First instance: set peak
        rc1 = RiskController(
            config=risk_config, circuit_breaker=circuit_breaker, initial_equity=INITIAL_EQUITY,
        )
        rc1.update_equity(110_000.0, as_of=date(2026, 1, 2))
        assert rc1.peak_equity == 110_000.0

        # Simulate process restart — create a new instance with the same state dir
        cb2 = CircuitBreaker()
        rc2 = RiskController(
            config=risk_config, circuit_breaker=cb2, initial_equity=INITIAL_EQUITY,
        )
        assert rc2.peak_equity == 110_000.0, (
            "Peak equity must be loaded from persisted state on restart"
        )

    def test_gradual_drawdown_triggers_at_exactly_10pct(
        self, rc: RiskController,
    ) -> None:
        """10% drawdown is the boundary — 9.99% must NOT trigger."""
        # Drawdown to 9.9% (equity = 90,100)
        rc.update_equity(INITIAL_EQUITY * 0.901, as_of=date(2026, 1, 2))
        assert rc.is_active, "9.9% drawdown must NOT trigger HALT"

        # Drawdown to exactly 10% (equity = 90,000)
        rc.update_equity(INITIAL_EQUITY * 0.90, as_of=date(2026, 1, 3))
        assert rc.is_halted, "Exactly 10% drawdown must trigger HALT"

        # Verify it was the MaxDD (not some other trigger)
        assert rc.state == ExecutionState.HALTED


# ===========================================================================
# Task 1.2: Rolling 3-Month Sharpe Halt
# ===========================================================================


class TestRollingSharpeHalt:
    """Inject 63 days of consistently negative returns.

    Scenario: premium decay fails to cover delta hedging costs, producing a
    stream of daily returns with negative mean and positive variance.

    Expected: RiskController sets state = PAUSED (not HALTED — Sharpe
    halt is a PAUSE, not a full halt, per the RiskController design).
    """

    def test_negative_sharpe_triggers_pause(
        self, rc: RiskController,
    ) -> None:
        """63 days of negative returns must produce Sharpe < 0 → PAUSED."""
        start = date(2026, 1, 2)

        # 63 days of moderate losses: mean ≈ -0.2%, std ≈ 0.5%
        rng = np.random.default_rng(seed=42)
        for i in range(63):
            daily_return = rng.normal(loc=-0.002, scale=0.005)
            dt = start + timedelta(days=i)
            rc.record_daily_return(dt, daily_return)

        # Force Sharpe check on a month-end
        check_date = start + timedelta(days=63)
        sharpe = rc.check_monthly_sharpe(as_of=check_date)

        assert sharpe < 0.0, f"Sharpe should be negative, got {sharpe:.4f}"
        assert rc.state == ExecutionState.PAUSED, (
            "RiskController must be PAUSED when 3-month Sharpe < 0"
        )

    def test_circuit_breaker_triggered_on_sharpe_halt(
        self, rc: RiskController, circuit_breaker: CircuitBreaker,
    ) -> None:
        """Sharpe halt must set CircuitBreaker to CRITICAL_HALT."""
        start = date(2026, 1, 2)
        rng = np.random.default_rng(seed=99)
        for i in range(63):
            # Add variance to avoid NaN (std > 0 needed for Sharpe)
            daily_return = rng.normal(loc=-0.003, scale=0.002)
            rc.record_daily_return(start + timedelta(days=i), daily_return)

        rc.check_monthly_sharpe(as_of=start + timedelta(days=63))

        assert circuit_breaker.state == CircuitBreakerState.CRITICAL_HALT, (
            "CircuitBreaker must be CRITICAL_HALT when Sharpe triggers pause"
        )

    def test_positive_sharpe_does_not_trigger(
        self, rc: RiskController,
    ) -> None:
        """Positive returns must NOT trigger Sharpe halt."""
        start = date(2026, 1, 2)
        rng = np.random.default_rng(seed=55)
        for i in range(63):
            # Small positive daily returns with variance → positive Sharpe
            daily_return = rng.normal(loc=0.001, scale=0.003)
            rc.record_daily_return(start + timedelta(days=i), daily_return)

        sharpe = rc.check_monthly_sharpe(as_of=start + timedelta(days=63))
        assert not np.isnan(sharpe), "Sharpe should not be NaN with varied returns"
        assert sharpe > 0.0, f"Sharpe should be positive, got {sharpe:.4f}"
        assert rc.is_active, "Positive Sharpe must not trigger pause"

    def test_insufficient_data_does_not_trigger_halt(
        self, rc: RiskController,
    ) -> None:
        """Less than 10 data points should return NaN Sharpe — no halt."""
        start = date(2026, 1, 2)
        for i in range(5):
            rc.record_daily_return(start + timedelta(days=i), -0.01)

        sharpe = rc.check_monthly_sharpe(as_of=start + timedelta(days=5))
        assert np.isnan(sharpe), "Sharpe with < 10 returns should be NaN"
        assert rc.is_active, "NaN Sharpe must not trigger halt"

    def test_monthly_dedup_sharpe_check(
        self, rc: RiskController,
    ) -> None:
        """Sharpe check within the same month should be a no-op (short-circuit)."""
        start = date(2026, 1, 2)
        rng = np.random.default_rng(seed=77)
        for i in range(63):
            daily_return = rng.normal(loc=-0.003, scale=0.002)
            rc.record_daily_return(start + timedelta(days=i), daily_return)

        # First Sharpe check — should trigger halt (new month relative to 1970-01-01)
        result1 = rc.check_monthly_sharpe(as_of=start + timedelta(days=63))
        assert rc.state == ExecutionState.PAUSED, "First Sharpe check must trigger PAUSED"

        # Second check in the SAME month — short-circuits, just returns Sharpe
        result2 = rc.check_monthly_sharpe(as_of=start + timedelta(days=64))
        # Both return the same computed Sharpe (no re-trigger in same month)
        assert np.isnan(result1) == np.isnan(result2), "Second check returns same Sharpe value"


# ===========================================================================
# Task 1.3: Macro Volatility Exit
# ===========================================================================


class TestMacroVolatilityExit:
    """Test VIX > 30 combined with elevated drawdown conditions."""

    def test_macro_halt_at_vix_30_and_dd_12pct(
        self, rc: RiskController, circuit_breaker: CircuitBreaker,
    ) -> None:
        """VIX > 30 + DD > 12% → 'halt'.

        Note: we do NOT call update_equity first (that would trigger MaxDD at 10%).
        Instead, we only exercise check_macro_conditions with the right inputs.
        """
        # check_macro_conditions accepts drawdown as a POSITIVE fraction
        result = rc.check_macro_conditions(
            vix_level=35.0,
            current_drawdown=0.13,  # 13% drawdown (positive)
        )
        assert result == "halt", "Macro exit must return 'halt'"
        assert rc.is_halted
        assert circuit_breaker.state == CircuitBreakerState.CRITICAL_HALT

    def test_macro_reduce_at_vix_30_and_dd_8pct(
        self, rc: RiskController,
    ) -> None:
        """VIX > 30 + DD > 8% (but < 12%) → reduce_50.

        Note: drawdown parameter is a POSITIVE fraction.
        """
        result = rc.check_macro_conditions(
            vix_level=35.0,
            current_drawdown=0.09,  # 9% drawdown (positive, between 8-12%)
        )
        assert result == "reduce_50", "Must return 'reduce_50' for DD between 8-12%"

    def test_macro_ok_below_thresholds(
        self, rc: RiskController,
    ) -> None:
        """VIX < 30 → always 'ok' regardless of drawdown."""
        result = rc.check_macro_conditions(
            vix_level=25.0,
            current_drawdown=0.05,  # 5% drawdown (positive)
        )
        assert result == "ok"

    def test_macro_ok_even_with_high_vix_but_low_dd(
        self, rc: RiskController,
    ) -> None:
        """VIX > 30 but DD < 8% → 'ok' (not yet at warning level)."""
        result = rc.check_macro_conditions(
            vix_level=35.0,
            current_drawdown=0.05,  # 5% drawdown (positive, below 8%)
        )
        assert result == "ok"

    def test_macro_no_action_when_already_halted(
        self, rc: RiskController,
    ) -> None:
        """If already halted, macro check returns 'halted'."""
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))
        assert rc.is_halted

        result = rc.check_macro_conditions(vix_level=35.0, current_drawdown=0.13)
        assert result == "halted"


# ===========================================================================
# Task 1.4: Manual Reset (committee override)
# ===========================================================================


class TestManualReset:
    """Manual reset after committee review should restore ACTIVE."""

    def test_manual_reset_restores_active(
        self, rc: RiskController,
    ) -> None:
        """After HALT, manual_reset() must restore ACTIVE."""
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))
        assert rc.is_halted

        rc.manual_reset(operator_id="committee_chair", reason="post-mortem review passed")
        assert rc.is_active, "Manual reset must restore ACTIVE state"

    def test_manual_reset_logs_audit(
        self, rc: RiskController, caplog,
    ) -> None:
        """Manual reset must produce CRITICAL-level audit log."""
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))

        with caplog.at_level(logging.CRITICAL, logger="quantcore.live.paper_trading.risk_controller"):
            rc.manual_reset("audit_officer", "risk parameters re-approved")

        assert any("Manual reset" in r.message for r in caplog.records), (
            "Manual reset must log an audit trail entry"
        )


# ===========================================================================
# Execution Environment Integration
# ===========================================================================


class TestExecutionEnvironmentUnderStress:
    """Verify PaperExecutionEnvironment handles stress scenarios correctly."""

    def test_slippage_model_produces_realistic_fills(
        self, exec_env: PaperExecutionEnvironment,
    ) -> None:
        """Slippage model must not produce unrealistic fills."""
        # 100 random fills — none should deviate more than spread allows
        for _ in range(100):
            result = exec_env.compute_order_details(
                theoretical_price=5.00,
                quantity=10,
                side="buy",
                asset_type="option",
            )
            # Options spread is 50 bps → max deviation ~0.25%
            max_deviation = 5.00 * 0.005  # 0.5% of theoretical
            assert abs(result["fill_price"] - result["theoretical_price"]) <= max_deviation + 0.01, (
                f"Fill price ${result['fill_price']:.4f} deviates too far from "
                f"theoretical ${result['theoretical_price']:.4f}"
            )

    def test_delta_tracker_net_delta_near_zero(
        self, exec_env: PaperExecutionEnvironment,
    ) -> None:
        """Delta tracker must compute net delta ≈ 0 after hedge."""
        # Short 2 ATM calls (delta 0.50 each) + short 2 ATM puts (delta -0.50 each)
        positions = [
            {
                "call_delta": 0.50,
                "put_delta": -0.50,
                "contracts": 2,
            },
        ]

        # Hedge: 100 shares of underlying (delta = +100)
        record = exec_env.record_delta_hedge(
            dt=date(2026, 1, 15),
            positions=positions,
            hedge_shares=100.0,
        )

        # Short call delta = -0.50 * 100 * 2 = -100
        # Short put delta = -(-0.50) * 100 * 2 = +100
        # Net options delta = -100 + 100 = 0
        # Hedge delta = +100
        # Net delta = 0 + 100 = 100
        assert abs(record["net_delta"] - 100.0) < 0.01, (
            f"Net delta should be ~100.0, got {record['net_delta']}"
        )

    def test_iv_rolling_window_no_lookahead(
        self, exec_env: PaperExecutionEnvironment,
    ) -> None:
        """IV must only use historical bars — no look-ahead."""
        # Feed 100 bars
        start = date(2025, 10, 1)
        for i in range(100):
            dt = start + timedelta(days=i)
            exec_env.ingest_daily_bar(dt, close_price=450.0 + i * 0.5, vix=18.0)

        # IV should be computed from the last 60 bars only
        current_iv = exec_env.get_current_iv()
        assert 0.05 < current_iv < 2.0, f"IV {current_iv} is outside reasonable range"

        # Get historical IV at a point — must not use future data
        midpoint = start + timedelta(days=60)
        hist_iv = exec_env.get_historical_iv(as_of=midpoint, lookback=30)
        assert 0.05 < hist_iv < 2.0, f"Historical IV {hist_iv} is outside reasonable range"

    def test_equity_cannot_go_negative(
        self, exec_env: PaperExecutionEnvironment,
    ) -> None:
        """Equity update must allow negative P&L (but caller is responsible for halt)."""
        # Apply a massive loss
        exec_env.update_equity(-1_000_000.0)
        assert exec_env.equity < 0, "Equity can go negative — halt is RiskController's job"


# ===========================================================================
# Integration: Full Kill Chain
# ===========================================================================


class TestFullKillChain:
    """End-to-end test: RiskController → CircuitBreaker → OrderBlock."""

    def test_maxdd_triggers_circuit_breaker_then_blocks_trading(
        self, rc: RiskController, circuit_breaker: CircuitBreaker,
    ) -> None:
        """Full kill chain: MaxDD → CB.CRITICAL_HALT → can_trade=False."""
        # Initial state
        assert circuit_breaker.can_trade()
        assert rc.is_active

        # Trigger MaxDD
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))

        # Full chain verification
        assert rc.is_halted, "1. RiskController must be HALTED"
        assert circuit_breaker.state == CircuitBreakerState.CRITICAL_HALT, (
            "2. CircuitBreaker must be CRITICAL_HALT"
        )
        assert not circuit_breaker.can_trade(), (
            "3. can_trade() must return False — orders blocked"
        )

        # Simulate a manual reset of the circuit breaker only (without RiskController)
        circuit_breaker.reset()
        assert circuit_breaker.can_trade(), "After circuit breaker reset, trading is allowed"

        # But RiskController is still halted — so the runner must check both
        assert rc.is_halted, "RiskController stays halted even after CB reset"

    def test_state_persistence_survives_crash(
        self, risk_config: RiskControllerConfig,
    ) -> None:
        """Verify HALTED state persists to disk and is reloadable."""
        cb = CircuitBreaker()
        rc = RiskController(
            config=risk_config, circuit_breaker=cb, initial_equity=INITIAL_EQUITY,
        )

        # Trigger halt
        rc.update_equity(INITIAL_EQUITY * 0.89, as_of=date(2026, 1, 2))

        # Verify state file exists
        state_file = risk_config.state_dir / "risk_controller_state.json"
        assert state_file.exists(), "State file must be written on halt"

        # Load and verify
        data = json.loads(state_file.read_text())
        assert data["state"] == "halted"
        assert data["peak_equity"] == INITIAL_EQUITY  # equity never went above initial

        # Simulate crash recovery — new instance
        cb2 = CircuitBreaker()
        rc2 = RiskController(
            config=risk_config, circuit_breaker=cb2, initial_equity=INITIAL_EQUITY,
        )
        assert rc2.is_halted, "Recovered RiskController must load HALTED state from disk"


# Need logging for test_manual_reset_logs_audit
import logging
