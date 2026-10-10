"""Broker Emulator — canonical truth layer that can intentionally diverge from SUT.

Acts as authoritative external broker state:
  - Maintains canonical position ledger independent of SUT
  - Maintains canonical order state machine with fill sequence numbers
  - Tracks settlement lifecycle independently
  - Supports intentional divergence: dropped fills, duplicates, delays, reorders

The emulator is the "what actually happened" — the SUT may or may not
see the same events depending on injected failures.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from quantcore.core.cash import AssetClass, CashLedger
from quantcore.core.types import Fill, Order, Position, Side


class FillSeqStatus(str, Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    DROPPED = "dropped"  # intentionally lost
    DUPLICATE = "duplicate"


@dataclass
class CanonicalFillRecord:
    """A fill as recorded by the canonical broker emulator."""
    seq: int
    fill: Fill
    symbol: str
    timestamp: datetime
    status: FillSeqStatus = FillSeqStatus.PENDING
    ack_to_sut: bool = True  # whether SUT receives this fill


@dataclass
class CanonicalOrderRecord:
    """An order as tracked by the canonical broker."""
    order: Order
    submitted_at: datetime
    fills: list[CanonicalFillRecord] = field(default_factory=list)
    acked: bool = True
    cancelled: bool = False
    reject_reason: str = ""


@dataclass
class BrokerStateSnapshot:
    """Point-in-time snapshot of canonical broker state."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    open_orders: list[str] = field(default_factory=list)
    fills_generated: int = 0
    fills_acked_to_sut: int = 0
    fills_dropped: int = 0
    seq_num: int = 0


class BrokerEmulator:
    """Canonical broker truth layer.

    This is the authoritative source of "what happened" in the market.
    The SUT may receive a subset of these events depending on injected failures.

    Usage:
        broker = BrokerEmulator(initial_cash=1_000_000.0)

        # SUT submits order
        order = Order(id="o1", symbol="AAPL", ...)
        broker.submit_order(order)

        # Generate fills (canonical)
        fill = broker.generate_fill("o1", price=150.0, qty=100)

        # SUT receives a fill (may be dropped, duplicated, etc.)
        if broker.should_ack_fill(fill_record):
            sut.receive_fill(fill_record.fill)

        # Reconcile SUT state against canonical state
        report = broker.reconcile(sut_positions, sut_cash)
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        initial_positions: dict[str, Position] | None = None,
    ) -> None:
        self._initial_cash = initial_cash
        self._seq: int = 0

        # Canonical state
        self._ledger = CashLedger(initial_cash=initial_cash)
        self._positions: dict[str, Position] = dict(initial_positions or {})

        # Order tracking
        self._orders: dict[str, CanonicalOrderRecord] = {}

        # Fill tracking
        self._fill_history: list[CanonicalFillRecord] = []
        self._fill_keys: set[tuple[str, Decimal, Decimal]] = set()  # idempotency

        # Divergence control
        self._dropped_fill_ids: set[str] = set()
        self._duplicate_fill_ids: set[str] = set()
        self._delayed_fills: dict[str, float] = {}  # fill_id -> delay_seconds
        self._silent_disconnect: bool = False
        self._disconnect_until: datetime | None = None
        self._reorder_queue: list[CanonicalFillRecord] = []

        # Settlement
        self._settlement_cycle_overrides: dict[AssetClass, int] = {}

        # Concurrency
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> bool:
        """Submit an order to the canonical broker. Returns ack success."""
        with self._lock:
            if self._silent_disconnect:
                return False
            record = CanonicalOrderRecord(
                order=copy.deepcopy(order),
                submitted_at=datetime.utcnow(),
                acked=not self._silent_disconnect,
            )
            self._orders[order.id] = record
            return record.acked

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if successful."""
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].cancelled = True
                return True
            return False

    def reject_order(self, order_id: str, reason: str = "") -> None:
        """Reject an order on the canonical side."""
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].reject_reason = reason or "rejected"

    # ------------------------------------------------------------------
    # Fill generation (canonical)
    # ------------------------------------------------------------------

    def generate_fill(
        self,
        order_id: str,
        price: float,
        quantity: float,
        symbol: str | None = None,
        commission: float = 0.0,
        timestamp: datetime | None = None,
    ) -> CanonicalFillRecord | None:
        """Generate a canonical fill for an order. Returns None if order unknown.

        This records the fill in the canonical ledger regardless of whether
        the SUT will receive it (divergence is intentional).
        """
        with self._lock:
            if order_id not in self._orders:
                return None

            order = self._orders[order_id].order
            sym = symbol or order.symbol
            self._seq += 1

            fill = Fill(
                order_id=order_id,
                price=Decimal(str(price)),
                quantity=Decimal(str(quantity)),
                commission=Decimal(str(commission)),
                timestamp=timestamp or datetime.utcnow(),
                side=order.side,
            )

            record = CanonicalFillRecord(
                seq=self._seq,
                fill=fill,
                symbol=sym,
                timestamp=fill.timestamp,
                status=FillSeqStatus.PENDING,
                ack_to_sut=True,
            )

            # Apply to canonical state (broker always executes)
            self._apply_fill_to_canonical(fill, sym)
            self._fill_history.append(record)
            self._orders[order_id].fills.append(record)

            return record

    def _apply_fill_to_canonical(self, fill: Fill, symbol: str) -> None:
        """Apply a fill to the canonical broker's own ledger."""
        qty = float(fill.quantity)
        price = float(fill.price)
        commission = float(fill.commission)
        is_buy = fill.side == Side.BUY if fill.side else True

        # Update canonical position
        current = self._positions.get(symbol)
        if current is None or abs(float(current.quantity)) < 1e-10:
            init_qty = qty if is_buy else -qty
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(init_qty)),
                avg_cost=Decimal(str(price)),
                market_price=Decimal(str(price)),
            )
        else:
            old_qty = float(current.quantity)
            old_cost = float(current.avg_cost)
            if is_buy:
                new_qty = old_qty + qty
                if old_qty >= 0:
                    total = old_cost * old_qty + price * qty
                    new_avg = total / new_qty if new_qty > 0 else old_cost
                else:
                    new_avg = old_cost  # covering short
            else:
                new_qty = old_qty - qty
                if old_qty <= 0:
                    total = old_cost * abs(old_qty) + price * qty
                    new_avg = total / abs(new_qty) if new_qty != 0 else old_cost
                else:
                    new_avg = old_cost  # reducing long

            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(new_qty)),
                avg_cost=Decimal(str(new_avg)),
                market_price=Decimal(str(price)),
            )

        # Update canonical cash
        cash_delta = -(qty * price) if is_buy else (qty * price)
        self._ledger.add_unsettled(
            amount=cash_delta - commission,
            asset_class="equity",
            trade_date=datetime.utcnow(),
            description=f"Canonical fill {fill.order_id}",
        )

    # ------------------------------------------------------------------
    # Divergence controls
    # ------------------------------------------------------------------

    def drop_fill(self, fill_record: CanonicalFillRecord) -> None:
        """Mark a fill as dropped — SUT never receives it."""
        fill_record.status = FillSeqStatus.DROPPED
        fill_record.ack_to_sut = False
        self._dropped_fill_ids.add(fill_record.fill.order_id)

    def duplicate_fill(self, fill_record: CanonicalFillRecord) -> CanonicalFillRecord | None:
        """Create a duplicate of a fill (same order_id, price, qty)."""
        if not fill_record:
            return None
        dup = copy.deepcopy(fill_record)
        dup.seq = self._seq + 1
        self._seq += 1
        dup.status = FillSeqStatus.DUPLICATE
        dup.timestamp = datetime.utcnow()
        dup.fill.timestamp = dup.timestamp
        self._fill_history.append(dup)
        self._duplicate_fill_ids.add(dup.fill.order_id)
        return dup

    def delay_fill(self, fill_record: CanonicalFillRecord, delay_seconds: float) -> None:
        """Delay acknowledgement of a fill to the SUT."""
        self._delayed_fills[str(fill_record.seq)] = delay_seconds

    def queue_fill(self, fill_record: CanonicalFillRecord) -> None:
        """Queue a fill for reordered delivery."""
        self._reorder_queue.append(fill_record)

    def flush_reorder_queue(self) -> list[CanonicalFillRecord]:
        """Flush the reorder queue (returns in potentially different order)."""
        results = list(self._reorder_queue)
        self._reorder_queue.clear()
        return results

    def set_disconnect(self, duration_seconds: float = 30.0) -> None:
        """Simulate broker disconnect. All orders/fills during this period fail."""
        self._silent_disconnect = True
        self._disconnect_until = datetime.utcnow() + timedelta(seconds=duration_seconds)

    def reconnect(self) -> None:
        """Reconnect broker."""
        self._silent_disconnect = False
        self._disconnect_until = None

    def set_settlement_cycle(self, asset_class: AssetClass, days: int) -> None:
        """Override settlement cycle for an asset class."""
        self._settlement_cycle_overrides[asset_class] = days

    def should_ack_fill(self, fill_record: CanonicalFillRecord, current_time: datetime | None = None) -> bool:
        """Determine if a fill should be acknowledged to the SUT."""
        if not fill_record.ack_to_sut:
            return False
        if self._silent_disconnect:
            return False
        seq_str = str(fill_record.seq)
        if seq_str in self._delayed_fills:
            delay = self._delayed_fills[seq_str]
            age = (datetime.utcnow() - fill_record.timestamp).total_seconds()
            if age < delay:
                return False
        return True

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, Position]:
        """Canonical positions (the truth)."""
        return dict(self._positions)

    @property
    def cash(self) -> float:
        """Canonical settled cash."""
        return self._ledger.settled_cash

    @property
    def unsettled_cash(self) -> float:
        return self._ledger.unsettled_cash

    @property
    def total_cash(self) -> float:
        return self._ledger.total_cash

    @property
    def seq_num(self) -> int:
        return self._seq

    @property
    def fill_history(self) -> list[CanonicalFillRecord]:
        return list(self._fill_history)

    def get_order(self, order_id: str) -> CanonicalOrderRecord | None:
        return self._orders.get(order_id)

    def snapshot(self) -> BrokerStateSnapshot:
        return BrokerStateSnapshot(
            positions=dict(self._positions),
            cash=self._ledger.settled_cash,
            open_orders=[oid for oid, o in self._orders.items() if not o.cancelled],
            fills_generated=len(self._fill_history),
            fills_acked_to_sut=sum(1 for f in self._fill_history if f.ack_to_sut),
            fills_dropped=len(self._dropped_fill_ids),
            seq_num=self._seq,
        )

    def equity(self) -> float:
        """Total canonical equity (cash + position market value)."""
        pos_value = sum(
            float(p.quantity) * float(p.market_price)
            for p in self._positions.values()
        )
        return self._ledger.settled_cash + pos_value

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        sut_positions: dict[str, Position],
        sut_cash: float,
        tolerance_qty: float = 1e-10,
        tolerance_cash: float = 0.01,
    ) -> dict[str, Any]:
        """Compare SUT state against canonical truth.

        Returns reconciliation report with deltas for all divergences.
        """
        report: dict[str, Any] = {
            "matched": True,
            "position_deltas": {},
            "cash_delta": 0.0,
            "divergences": [],
        }

        # Position reconciliation
        all_symbols = set(self._positions.keys()) | set(sut_positions.keys())
        for sym in all_symbols:
            can_qty = float(self._positions[sym].quantity) if sym in self._positions else 0.0
            sut_qty = float(sut_positions[sym].quantity) if sym in sut_positions else 0.0
            delta = abs(can_qty - sut_qty)
            if delta > tolerance_qty:
                report["matched"] = False
                report["position_deltas"][sym] = {
                    "canonical": can_qty,
                    "sut": sut_qty,
                    "delta": can_qty - sut_qty,
                }
                report["divergences"].append(
                    f"Position {sym}: canonical={can_qty:.4f} sut={sut_qty:.4f}"
                )

        # Cash reconciliation
        cash_delta = self._ledger.settled_cash - sut_cash
        if abs(cash_delta) > tolerance_cash:
            report["matched"] = False
            report["cash_delta"] = cash_delta
            report["divergences"].append(
                f"Cash: canonical={self._ledger.settled_cash:.2f} "
                f"sut={sut_cash:.2f} delta={cash_delta:.2f}"
            )

        return report

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._seq = 0
        self._ledger = CashLedger(initial_cash=self._initial_cash)
        self._positions = {}
        self._orders.clear()
        self._fill_history.clear()
        self._fill_keys.clear()
        self._dropped_fill_ids.clear()
        self._duplicate_fill_ids.clear()
        self._delayed_fills.clear()
        self._silent_disconnect = False
        self._disconnect_until = None
        self._reorder_queue.clear()
        self._settlement_cycle_overrides.clear()
