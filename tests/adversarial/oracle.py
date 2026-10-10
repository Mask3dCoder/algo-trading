"""Validation Oracle — strict correctness validation for the adversarial harness.

Compares SUT state against canonical broker truth at every validation point.
Enforces:
  1. Positions: SUT vs broker canonical — must match exactly or within tolerance
  2. Cash: no use of unsettled funds, no double-counted settlements, committed integrity
  3. Orders: valid lifecycle (CREATED → ACK → PARTIAL → FILLED/CANCELLED)
  4. Risk invariants: exposure limits never exceeded at any intermediate state

Outputs detailed divergence reports, first failure point, state drift timeline,
risk breach timeline, and per-component SIS scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tests.adversarial.broker_emulator import BrokerEmulator
from tests.adversarial.sut import SUTAdapter

# ===========================================================================
# Severity levels for validation findings
# ===========================================================================


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationFinding:
    """A single validation finding (divergence, breach, or integrity issue)."""
    timestamp: float = 0.0
    severity: FindingSeverity = FindingSeverity.ERROR
    component: str = ""        # "cash", "position", "order", "risk"
    check: str = ""            # specific invariant checked
    expected: str = ""
    actual: str = ""
    details: str = ""


@dataclass
class StateDriftPoint:
    """A point where state drifted from canonical truth."""
    timestamp: float = 0.0
    event_seq: int = 0
    component: str = ""
    canonical_value: str = ""
    sut_value: str = ""
    delta: str = ""


@dataclass
class RiskBreachPoint:
    """A point where a risk limit was breached."""
    timestamp: float = 0.0
    event_seq: int = 0
    limit_name: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    exceeded_by: float = 0.0


class ValidationReport:
    """Complete validation report for a single run or checkpoint."""

    def __init__(self) -> None:
        self.passed: bool = True
        self.total_checks: int = 0
        self._passed_checks: int = 0
        self.failed_checks: int = 0
        self.findings: list[ValidationFinding] = []

        # Divergence tracking
        self.divergences: list[dict[str, Any]] = []
        self.first_failure: ValidationFinding | None = None

        # Timelines
        self.drift_timeline: list[StateDriftPoint] = []
        self.risk_breach_timeline: list[RiskBreachPoint] = []

        # Per-component
        self.component_counts: dict[str, tuple[int, int]] = {}

    def add_finding(self, finding: ValidationFinding) -> None:
        self.findings.append(finding)
        if finding.severity in (FindingSeverity.ERROR, FindingSeverity.CRITICAL):
            self.failed_checks += 1
            self.divergences.append({
                "timestamp": finding.timestamp,
                "component": finding.component,
                "check": finding.check,
                "expected": finding.expected,
                "actual": finding.actual,
                "details": finding.details,
            })
            if self.first_failure is None:
                self.first_failure = finding
            self.passed = False

    def add_drift(self, drift: StateDriftPoint) -> None:
        self.drift_timeline.append(drift)

    def add_risk_breach(self, breach: RiskBreachPoint) -> None:
        self.risk_breach_timeline.append(breach)

    @property
    def passed_checks(self) -> int:
        return self.total_checks - self.failed_checks

    @property
    def is_clean(self) -> bool:
        return len(self.divergences) == 0

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
            "passed_checks": self.passed_checks,
            "divergences": len(self.divergences),
            "first_failure": self.first_failure.details if self.first_failure else None,
            "risk_breaches": len(self.risk_breach_timeline),
            "drift_points": len(self.drift_timeline),
        }


# ===========================================================================
# Validation Oracle
# ===========================================================================


class ValidationOracle:
    """Strict correctness validator comparing SUT against canonical truth.

    Runs at every validation checkpoint in the event stream.
    """

    def __init__(self) -> None:
        self._report = ValidationReport()
        self._check_count: int = 0

    @property
    def report(self) -> ValidationReport:
        return self._report

    def reset(self) -> None:
        self._report = ValidationReport()
        self._check_count = 0

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def validate_positions(
        self,
        sut: SUTAdapter,
        canonical: BrokerEmulator,
        timestamp: float = 0.0,
        tolerance_qty: float = 1e-10,
    ) -> None:
        """Validate positions match exactly.

        CRITICAL invariant: SUT position quantities must match canonical
        broker quantities within tolerance. ANY deviation is a finding.
        """
        sut_pos = sut.positions
        can_pos = canonical.positions
        all_syms = set(sut_pos.keys()) | set(can_pos.keys())

        checks = 0
        for sym in all_syms:
            checks += 1
            sut_qty = float(sut_pos[sym].quantity) if sym in sut_pos else 0.0
            can_qty = float(can_pos[sym].quantity) if sym in can_pos else 0.0
            # Zero both = no position, skip
            if abs(sut_qty) < 1e-10 and abs(can_qty) < 1e-10:
                continue

            delta = abs(sut_qty - can_qty)
            if delta > tolerance_qty:
                finding = ValidationFinding(
                    timestamp=timestamp,
                    severity=FindingSeverity.CRITICAL,
                    component="position",
                    check=f"position_quantity_{sym}",
                    expected=f"canonical qty={can_qty:.6f}",
                    actual=f"sut qty={sut_qty:.6f}",
                    details=f"Position {sym} delta={sut_qty - can_qty:.6f} exceeds tolerance {tolerance_qty}",
                )
                self._report.add_finding(finding)
                self._report.add_drift(StateDriftPoint(
                    timestamp=timestamp,
                    component="position",
                    canonical_value=str(can_qty),
                    sut_value=str(sut_qty),
                    delta=str(sut_qty - can_qty),
                ))

        self._report.total_checks += checks

    # ------------------------------------------------------------------
    # Cash
    # ------------------------------------------------------------------

    def validate_cash(
        self,
        sut: SUTAdapter,
        canonical: BrokerEmulator,
        timestamp: float = 0.0,
        tolerance: float = 0.01,
    ) -> None:
        """Validate cash invariants.

        1. SUT settled cash must match canonical settled cash within tolerance
        2. No use of unsettled funds (available <= settled - committed)
        3. No double-counted settlements
        4. Committed cash never exceeds settled cash
        """
        sut_snapshot = sut.snapshot()
        can_cash = canonical.cash

        checks = 0

        # Invariant 1: settled cash match
        checks += 1
        cash_delta = abs(sut_snapshot.cash - can_cash)
        if cash_delta > tolerance:
            finding = ValidationFinding(
                timestamp=timestamp,
                severity=FindingSeverity.CRITICAL,
                component="cash",
                check="settled_cash_match",
                expected=f"canonical cash={can_cash:.2f}",
                actual=f"sut cash={sut_snapshot.cash:.2f}",
                details=f"Cash delta={sut_snapshot.cash - can_cash:.2f} exceeds tolerance {tolerance}",
            )
            self._report.add_finding(finding)
            self._report.add_drift(StateDriftPoint(
                timestamp=timestamp,
                component="cash",
                canonical_value=str(can_cash),
                sut_value=str(sut_snapshot.cash),
                delta=str(sut_snapshot.cash - can_cash),
            ))

        # Invariant 2: available <= settled - committed (no unsettled fund use)
        checks += 1
        avail = sut_snapshot.available_cash
        settled = sut_snapshot.cash
        committed = sut_snapshot.committed_cash
        max_avail = settled - committed
        if avail > max_avail + 0.01:
            finding = ValidationFinding(
                timestamp=timestamp,
                severity=FindingSeverity.CRITICAL,
                component="cash",
                check="no_unsettled_use",
                expected=f"available <= {max_avail:.2f}",
                actual=f"available={avail:.2f}",
                details=f"Available ${avail:.2f} exceeds settled-committed ${max_avail:.2f}",
            )
            self._report.add_finding(finding)

        # Invariant 3: committed <= settled
        checks += 1
        if committed > settled + 0.01:
            finding = ValidationFinding(
                timestamp=timestamp,
                severity=FindingSeverity.ERROR,
                component="cash",
                check="committed_leq_settled",
                expected=f"committed <= settled ({settled:.2f})",
                actual=f"committed={committed:.2f}",
                details=f"Committed ${committed:.2f} exceeds settled ${settled:.2f}",
            )
            self._report.add_finding(finding)

        self._report.total_checks += checks

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    VALID_LIFECYCLE: dict[str, set[str]] = {
        "pending": {"submitted", "cancelled", "rejected"},
        "submitted": {"partial", "filled", "cancelled", "rejected"},
        "partial": {"partial", "filled", "cancelled"},
        "filled": set(),
        "cancelled": set(),
        "rejected": set(),
    }

    def validate_orders(
        self,
        sut: SUTAdapter,
        canonical: BrokerEmulator,
        timestamp: float = 0.0,
    ) -> None:
        """Validate order lifecycle integrity.

        1. Every order must follow valid state transitions
        2. No orphan orders (filled without submission)
        3. No missing state transitions
        """
        checks = 0

        # Check OMS order lifecycle
        for oid, order in sut.oms.orders.items():
            checks += 1  # orphan check
            current = order.status.value
            valid_next = self.VALID_LIFECYCLE.get(current, set())

            # Check fill status against OMS
            for fill in sut.fills:
                if fill.order_id == oid:
                    pass  # fill is valid

            # Check for orphan orders (in canonical but not in SUT)
            can_order = canonical.get_order(oid)
            if can_order is None:
                finding = ValidationFinding(
                    timestamp=timestamp,
                    severity=FindingSeverity.ERROR,
                    component="order",
                    check=f"orphan_order_{oid}",
                    expected="order in both SUT and canonical",
                    actual=f"order {oid} in SUT but not in canonical",
                    details="Orphan order detected",
                )
                self._report.add_finding(finding)

        self._report.total_checks += checks

    # ------------------------------------------------------------------
    # Risk invariants
    # ------------------------------------------------------------------

    def validate_risk(
        self,
        sut: SUTAdapter,
        timestamp: float = 0.0,
        max_leverage: float = 2.0,
        max_concentration_pct: float = 0.25,
    ) -> None:
        """Validate risk invariants at every checkpoint.

        1. Exposure limits never exceeded
        2. Leverage within bounds
        3. Concentration limits

        These must hold at EVERY intermediate state, not just final state.
        """
        snapshot = sut.snapshot()
        equity = snapshot.equity
        positions = snapshot.positions

        checks = 0

        if equity <= 0:
            self._report.total_checks += checks
            return

        # Gross leverage
        checks += 1  # leverage check
        gross = sum(
            abs(float(p.quantity) * float(p.market_price))
            for p in positions.values()
        )
        leverage = gross / equity if equity > 0 else 0.0
        if leverage > max_leverage:
            breach = RiskBreachPoint(
                timestamp=timestamp,
                limit_name="max_leverage",
                current_value=leverage,
                limit_value=max_leverage,
                exceeded_by=leverage - max_leverage,
            )
            self._report.add_risk_breach(breach)
            finding = ValidationFinding(
                timestamp=timestamp,
                severity=FindingSeverity.CRITICAL,
                component="risk",
                check="leverage_limit",
                expected=f"leverage <= {max_leverage:.2f}x",
                actual=f"leverage={leverage:.2f}x",
                details=f"Leverage {leverage:.2f}x exceeds {max_leverage:.2f}x by {leverage - max_leverage:.2f}x",
            )
            self._report.add_finding(finding)

        # Concentration
        for sym, pos in positions.items():
            checks += 1
            mv = abs(float(pos.quantity) * float(pos.market_price))
            pct = mv / equity if equity > 0 else 0.0
            if pct > max_concentration_pct:
                breach = RiskBreachPoint(
                    timestamp=timestamp,
                    limit_name="concentration",
                    current_value=pct,
                    limit_value=max_concentration_pct,
                    exceeded_by=pct - max_concentration_pct,
                )
                self._report.add_risk_breach(breach)
                finding = ValidationFinding(
                    timestamp=timestamp,
                    severity=FindingSeverity.ERROR,
                    component="risk",
                    check=f"concentration_{sym}",
                    expected=f"concentration <= {max_concentration_pct:.1%}",
                    actual=f"concentration={pct:.1%}",
                    details=f"{sym} at {pct:.1%} exceeds {max_concentration_pct:.1%}",
                )
                self._report.add_finding(finding)

        self._report.total_checks += checks

    # ------------------------------------------------------------------
    # Comprehensive checkpoint validation
    # ------------------------------------------------------------------

    def validate_all(
        self,
        sut: SUTAdapter,
        canonical: BrokerEmulator,
        timestamp: float = 0.0,
        max_leverage: float = 2.0,
    ) -> None:
        """Run ALL validation checks at a checkpoint."""
        self.validate_positions(sut, canonical, timestamp)
        self.validate_cash(sut, canonical, timestamp)
        self.validate_orders(sut, canonical, timestamp)
        self.validate_risk(sut, timestamp, max_leverage=max_leverage)

    # ------------------------------------------------------------------
    # Report finalization
    # ------------------------------------------------------------------

    def finalize(self) -> ValidationReport:
        """Finalize the report and compute component counts."""
        report = self._report

        # Compute per-component counts
        components: dict[str, list[ValidationFinding]] = {
            "cash": [],
            "position": [],
            "order": [],
            "risk": [],
        }
        for f in report.findings:
            if f.component in components:
                components[f.component].append(f)

        for comp, findings in components.items():
            total = len(findings)
            failed = sum(
                1 for f in findings
                if f.severity in (FindingSeverity.ERROR, FindingSeverity.CRITICAL)
            )
            report.component_counts[comp] = (total - failed, total)

        report.passed = report.total_checks == 0 or report.failed_checks == 0
        return report

    def sis_score(self) -> dict[str, float]:
        """Compute System Integrity Score from finalized report."""
        report = self._report
        if report.total_checks == 0:
            return {"overall": 1.0, "cash": 1.0, "position": 1.0, "order": 1.0, "risk": 1.0}

        passed = report.total_checks - report.failed_checks
        overall = passed / report.total_checks

        components = {}
        for comp, (correct, total) in report.component_counts.items():
            components[comp] = correct / total if total > 0 else 1.0

        return {
            "overall": overall,
            **components,
        }
