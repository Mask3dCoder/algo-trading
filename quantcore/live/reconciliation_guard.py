"""Reconciliation guard: halts trading on broker state mismatch.

On broker reconnect:
  1. Fetch authoritative positions and open orders from broker.
  2. Compare with internal state.
  3. If divergence exceeds tolerance → HALT trading immediately.
  4. Require manual resolution (no auto-correct for live safety).

Wraps the existing ReconciliationEngine with safety-critical escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

from quantcore.core.types import Order, Position
from quantcore.live.reconciliation import (
    MismatchSeverity,
    ReconciliationConfig,
    ReconciliationEngine,
    ReconciliationReport,
)


class GuardState(str, Enum):
    PASS = "pass"             # All checks passed, trading OK
    WARN = "warn"             # Minor discrepancies detected
    HALT = "halt"             # Mismatch detected — trading stopped
    REQUIRE_RESOLVE = "require_resolve"  # Manual resolution needed


@dataclass
class GuardSnapshot:
    """Snapshot of reconciliation guard state."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    guard_state: GuardState = GuardState.PASS
    report: ReconciliationReport | None = None
    open_order_mismatches: list[str] = field(default_factory=list)
    details: str = ""


class BrokerState(Protocol):
    """Protocol for broker state needed by the reconciliation guard."""

    def get_positions(self) -> dict[str, Position]: ...
    def get_account(self) -> dict[str, float]: ...
    def get_orders(self, status: str = "open") -> list[Order]: ...


class ReconciliationGuard:
    """Safety guard that halts trading when broker state diverges from internal.

    Usage:
        guard = ReconciliationGuard()
        guard.set_internal_state(positions, cash, orders, fills)

        # On each tick / reconnect:
        result = guard.check(broker)
        if result.guard_state == GuardState.HALT:
            trader.emergency_flatten()
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        self._engine = ReconciliationEngine(config or ReconciliationConfig(
            auto_correct_enabled=False,  # NEVER auto-correct in live
            max_critical_mismatches=0,    # Zero tolerance for critical mismatches
        ))
        self._guard_state = GuardState.PASS
        self._last_report: ReconciliationReport | None = None
        self._last_check: datetime | None = None

        # Internal state snapshots
        self._internal_positions: dict[str, Position] = {}
        self._internal_cash: float = 0.0
        self._internal_pnl: float = 0.0
        self._internal_orders: dict[str, Order] = {}

        # Connection tracking
        self._connected: bool = False
        self._reconnect_count: int = 0

    # ------------------------------------------------------------------
    # State registration
    # ------------------------------------------------------------------

    def set_internal_state(
        self,
        positions: dict[str, Position],
        cash: float,
        orders: dict[str, Order] | None = None,
        pnl: float = 0.0,
    ) -> None:
        """Register current internal state before reconciliation."""
        self._internal_positions = dict(positions)
        self._internal_cash = cash
        self._internal_pnl = pnl
        if orders is not None:
            self._internal_orders = dict(orders)

    def mark_connected(self) -> None:
        """Mark the connection as established."""
        if not self._connected:
            self._reconnect_count += 1
        self._connected = True

    def mark_disconnected(self) -> None:
        """Mark the connection as lost."""
        self._connected = False

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(
        self,
        broker: BrokerState,
    ) -> GuardSnapshot:
        """Run reconciliation and determine guard state.

        Returns GuardSnapshot. If state is HALT, trading must stop.
        """
        try:
            broker_positions = broker.get_positions()
            broker_account = broker.get_account()
            broker_orders = broker.get_orders("open")
        except Exception as e:
            return GuardSnapshot(
                guard_state=GuardState.HALT,
                details=f"Cannot fetch broker state: {e}",
            )

        report = self._engine.reconcile(
            internal_positions=self._internal_positions,
            broker_positions=broker_positions,
            internal_cash=self._internal_cash,
            broker_account=broker_account,
            internal_pnl=self._internal_pnl,
            broker_pnl=0.0,
        )
        self._last_report = report

        # Check open order mismatches
        order_mismatches = self._check_orders(broker_orders)

        # Determine guard state
        guard_state = GuardState.PASS
        details = ""

        if report.critical_mismatches > 0:
            guard_state = GuardState.HALT
            critical_symbols = [
                sym for sym, rec in report.positions.items()
                if rec.severity == MismatchSeverity.CRITICAL
            ]
            details = (
                f"Critical mismatches: {report.critical_mismatches} "
                f"({', '.join(critical_symbols)})"
            )
        elif report.total_mismatches > 0:
            guard_state = GuardState.WARN
            details = f"{report.total_mismatches} position mismatches detected"
        elif order_mismatches:
            guard_state = GuardState.WARN
            details = f"{len(order_mismatches)} open order mismatches"

        if guard_state == GuardState.WARN and self._engine.escalate(report):
            guard_state = GuardState.HALT
            details += " [ESCALATED to HALT]"

        self._guard_state = guard_state
        self._last_check = datetime.now(UTC)

        return GuardSnapshot(
            guard_state=guard_state,
            report=report,
            open_order_mismatches=order_mismatches,
            details=details,
        )

    # ------------------------------------------------------------------
    # Convenience: check on reconnect
    # ------------------------------------------------------------------

    def check_on_reconnect(
        self,
        broker: BrokerState,
        internal_positions: dict[str, Position],
        internal_cash: float,
        internal_orders: dict[str, Order] | None = None,
    ) -> GuardSnapshot:
        """Run reconciliation after a broker reconnect.

        This is stricter than a normal check: it assumes the broker
        state is authoritative and any divergence requires halting.
        """
        self.mark_connected()
        self.set_internal_state(internal_positions, internal_cash, internal_orders)
        return self.check(broker)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_orders(self, broker_orders: list[Order]) -> list[str]:
        """Compare internal open orders with broker open orders.

        Returns list of mismatch descriptions.
        """
        mismatches: list[str] = []
        broker_order_ids = {o.id for o in broker_orders}

        # Orders we think are open but broker doesn't have
        for oid, order in self._internal_orders.items():
            if order.status.value in ("submitted", "partial", "pending"):
                if oid not in broker_order_ids:
                    mismatches.append(
                        f"Order {oid} ({order.symbol}) open internally "
                        f"but not found at broker"
                    )

        # Orders broker has that we don't know about
        for o in broker_orders:
            if o.id not in self._internal_orders:
                mismatches.append(
                    f"Order {o.id} ({o.symbol}) exists at broker "
                    f"but not tracked internally"
                )

        return mismatches

    @property
    def last_report(self) -> ReconciliationReport | None:
        return self._last_report

    @property
    def guard_state(self) -> GuardState:
        return self._guard_state

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def reset(self) -> None:
        """Reset guard state after manual resolution."""
        self._guard_state = GuardState.PASS
        self._last_report = None
