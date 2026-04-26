"""Position, P&L, and cash reconciliation between internal state and broker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

import numpy as np

from quantcore.core.types import Position, Order, Side


class ReconciliationStatus(str, Enum):
    MATCHED = "matched"
    MISMATCHED = "mismatched"
    MISSING_INTERNAL = "missing_internal"
    MISSING_BROKER = "missing_broker"
    ERROR = "error"


class MismatchSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PositionReconciliation:
    """Result of reconciling a single position."""
    symbol: str = ""
    status: ReconciliationStatus = ReconciliationStatus.MATCHED
    internal_qty: float = 0.0
    broker_qty: float = 0.0
    delta_qty: float = 0.0
    internal_value: float = 0.0
    broker_value: float = 0.0
    delta_value: float = 0.0
    severity: MismatchSeverity = MismatchSeverity.INFO
    details: str = ""


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    positions: dict[str, PositionReconciliation] = field(default_factory=dict)
    cash_internal: float = 0.0
    cash_broker: float = 0.0
    cash_delta: float = 0.0
    equity_internal: float = 0.0
    equity_broker: float = 0.0
    equity_delta: float = 0.0
    pnl_internal: float = 0.0
    pnl_broker: float = 0.0
    pnl_delta: float = 0.0
    status: ReconciliationStatus = ReconciliationStatus.MATCHED
    total_mismatches: int = 0
    critical_mismatches: int = 0
    resolved: bool = False


@dataclass
class ReconciliationConfig:
    """Thresholds and settings for reconciliation."""
    position_tolerance_pct: float = 0.01  # 1% quantity tolerance
    position_tolerance_abs: float = 1.0  # absolute share tolerance
    cash_tolerance_abs: float = 10.0  # $10 cash tolerance
    pnl_tolerance_pct: float = 0.05  # 5% P&L tolerance
    max_critical_mismatches: int = 1
    auto_correct_enabled: bool = False
    auto_correct_max_amount: float = 1000.0
    reconciliation_interval_sec: float = 300.0
    escalation_threshold: int = 3  # mismatches before escalation


class ReconciliationEngine:
    """Compare internal portfolio state with broker state and resolve differences."""

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        self.config = config or ReconciliationConfig()
        self._history: list[ReconciliationReport] = []
        self._mismatch_counts: dict[str, int] = {}
        self._corrections: list[dict] = []

    def reconcile(
        self,
        internal_positions: dict[str, Position],
        broker_positions: dict[str, Any],
        internal_cash: float,
        broker_account: dict[str, float],
        internal_pnl: float = 0.0,
        broker_pnl: float = 0.0,
    ) -> ReconciliationReport:
        """Run full reconciliation between internal state and broker state."""
        report = ReconciliationReport(
            cash_internal=internal_cash,
            cash_broker=float(broker_account.get("cash", 0)),
            equity_internal=internal_cash + sum(
                float(p.market_value)
                for p in internal_positions.values()
            ),
            equity_broker=float(broker_account.get("equity", broker_account.get("portfolio_value", 0))),
            pnl_internal=internal_pnl,
            pnl_broker=broker_pnl,
        )
        report.cash_delta = report.cash_internal - report.cash_broker
        report.equity_delta = report.equity_internal - report.equity_broker
        report.pnl_delta = report.pnl_internal - report.pnl_broker

        # Reconcile positions
        all_symbols = set(internal_positions.keys()) | set(
            str(p.get("symbol", k)) if isinstance(p, dict) else k
            for k, p in broker_positions.items()
        )

        for symbol in all_symbols:
            pos_rec = self._reconcile_position(
                symbol, internal_positions.get(symbol), broker_positions,
            )
            report.positions[symbol] = pos_rec
            if pos_rec.status != ReconciliationStatus.MATCHED:
                report.total_mismatches += 1
                if pos_rec.severity == MismatchSeverity.CRITICAL:
                    report.critical_mismatches += 1

        # Determine overall status
        if report.critical_mismatches > 0:
            report.status = ReconciliationStatus.MISMATCHED
        elif report.total_mismatches > 0:
            report.status = ReconciliationStatus.MISMATCHED
        else:
            report.status = ReconciliationStatus.MATCHED

        # Auto-correct if enabled
        if self.config.auto_correct_enabled and report.total_mismatches > 0:
            self._auto_correct(report)

        self._history.append(report)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return report

    def _reconcile_position(
        self, symbol: str, internal: Position | None, broker: dict[str, Any],
    ) -> PositionReconciliation:
        broker_pos = None
        if isinstance(broker, dict):
            broker_pos = broker.get(symbol)
            if broker_pos is None:
                for v in broker.values():
                    if isinstance(v, dict) and v.get("symbol") == symbol:
                        broker_pos = v
                        break

        if internal is None and broker_pos is None:
            return PositionReconciliation(symbol=symbol)

        if internal is None:
            b_qty = float(broker_pos.get("quantity", broker_pos.get("qty", 0))) if isinstance(broker_pos, dict) else float(getattr(broker_pos, "quantity", 0))
            return PositionReconciliation(
                symbol=symbol, status=ReconciliationStatus.MISSING_INTERNAL,
                broker_qty=b_qty, delta_qty=-b_qty,
                severity=MismatchSeverity.WARNING,
                details="Position exists at broker but not internally",
            )

        if broker_pos is None:
            return PositionReconciliation(
                symbol=symbol, status=ReconciliationStatus.MISSING_BROKER,
                internal_qty=float(internal.quantity), delta_qty=float(internal.quantity),
                severity=MismatchSeverity.WARNING,
                details="Position exists internally but not at broker",
            )

        i_qty = float(internal.quantity)
        b_qty = float(broker_pos.get("quantity", broker_pos.get("qty", 0))) if isinstance(broker_pos, dict) else float(getattr(broker_pos, "quantity", 0))
        delta = i_qty - b_qty
        i_val = float(internal.market_value)
        b_val = float(broker_pos.get("market_value", b_qty * float(internal.market_price))) if isinstance(broker_pos, dict) else float(getattr(broker_pos, "market_value", 0))

        severity = MismatchSeverity.INFO
        status = ReconciliationStatus.MATCHED

        if abs(delta) > self.config.position_tolerance_abs:
            pct_diff = abs(delta) / max(abs(i_qty), abs(b_qty), 1)
            if pct_diff > self.config.position_tolerance_pct:
                status = ReconciliationStatus.MISMATCHED
                if pct_diff > 0.05 or abs(i_val - b_val) > 5000:
                    severity = MismatchSeverity.CRITICAL
                elif pct_diff > 0.02:
                    severity = MismatchSeverity.WARNING

        return PositionReconciliation(
            symbol=symbol, status=status,
            internal_qty=i_qty, broker_qty=b_qty, delta_qty=delta,
            internal_value=i_val, broker_value=b_val, delta_value=i_val - b_val,
            severity=severity,
            details="" if status == ReconciliationStatus.MATCHED else f"Quantity mismatch: {delta:+.1f} shares",
        )

    def _auto_correct(self, report: ReconciliationReport) -> None:
        """Attempt automatic correction of minor mismatches."""
        corrections: list[dict] = []
        for sym, rec in report.positions.items():
            if rec.severity == MismatchSeverity.INFO and rec.status == ReconciliationStatus.MISMATCHED:
                if abs(rec.broker_value - rec.internal_value) < self.config.auto_correct_max_amount:
                    corrections.append({
                        "symbol": sym, "action": "update_to_broker",
                        "old_qty": rec.internal_qty, "new_qty": rec.broker_qty,
                    })
        if corrections:
            self._corrections.extend(corrections)
            report.resolved = any(
                c for c in corrections
                if abs(c["new_qty"] - c["old_qty"]) <= self.config.position_tolerance_abs
            )

    def get_history(self, n: int = 10) -> list[ReconciliationReport]:
        return self._history[-n:]

    def is_stable(self, n: int = 5) -> bool:
        recent = self._history[-n:]
        if len(recent) < n:
            return False
        return all(r.status == ReconciliationStatus.MATCHED for r in recent)

    def escalate(self, report: ReconciliationReport) -> bool:
        """Check if reconciliation failures warrant escalation."""
        if report.critical_mismatches >= self.config.max_critical_mismatches:
            return True
        for sym in report.positions:
            self._mismatch_counts[sym] = self._mismatch_counts.get(sym, 0) + 1
            if self._mismatch_counts[sym] >= self.config.escalation_threshold:
                return True
        return False


def compute_pnl_difference(
    internal_trades: list[dict], broker_trades: list[dict],
) -> dict[str, float]:
    """Compute P&L attribution between internal and broker trade records."""
    total_internal = sum(float(t.get("pnl", t.get("realized_pnl", 0))) for t in internal_trades)
    total_broker = sum(float(t.get("pnl", t.get("realized_pnl", 0))) for t in broker_trades)
    return {
        "internal_pnl": total_internal,
        "broker_pnl": total_broker,
        "delta": total_internal - total_broker,
        "delta_pct": (total_internal - total_broker) / max(abs(total_broker), 1) * 100,
    }
