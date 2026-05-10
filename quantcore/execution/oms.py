"""Order Management System: lifecycle, state machine, idempotency, audit trail."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from quantcore.core.exceptions import OrderError
from quantcore.core.types import Order, OrderStatus, OrderType, Side


class OrderState(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Valid state transitions
TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.NEW: {OrderState.SUBMITTED, OrderState.REJECTED},
    OrderState.SUBMITTED: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED},
    OrderState.PARTIAL: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED},
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
}


@dataclass
class AuditEntry:
    timestamp: datetime
    order_id: str
    from_state: str
    to_state: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """Manages order lifecycle with state machine enforcement and audit trail."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._audit: list[AuditEntry] = []
        self._submitted_ids: set[str] = set()
        self._fill_keys: set[tuple[str, Decimal, Decimal]] = set()  # (order_id, qty, price)
        self._on_state_change: Callable[[Order, OrderState, OrderState], None] | None = None
        self._lock = threading.Lock()

    @property
    def orders(self) -> dict[str, Order]:
        return self._orders

    @property
    def audit_trail(self) -> list[AuditEntry]:
        return self._audit

    def on_state_change(self, callback: Callable[[Order, OrderState, OrderState], None]) -> None:
        """Register a callback for state transitions."""
        self._on_state_change = callback

    def submit(self, order: Order) -> str:
        """Submit a new order. Performs idempotency check.

        Returns order_id.
        """
        with self._lock:
            # Idempotency check
            if order.id in self._submitted_ids:
                raise OrderError(f"Duplicate order: {order.id}")

            self._orders[order.id] = order
            self._submitted_ids.add(order.id)

            # Transition: NEW → SUBMITTED
            self._transition(order.id, OrderState.NEW, OrderState.SUBMITTED)
            order.status = OrderStatus("submitted")
            order.updated_at = datetime.utcnow()

            # Update health state for monitoring
            from quantcore.core.health_state import record_order_submitted
            record_order_submitted()

        return order.id

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        order = self._get_order(order_id)
        current_state = self._current_state(order)

        if current_state not in (OrderState.SUBMITTED, OrderState.PARTIAL):
            raise OrderError(f"Cannot cancel order {order_id} in state {current_state}")

        self._transition(order_id, current_state, OrderState.CANCELLED)
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()

    def modify(self, order_id: str, new_params: dict[str, Any]) -> None:
        """Modify an open order (cancel-replace pattern)."""
        order = self._get_order(order_id)
        old_state = self._current_state(order)

        if old_state not in (OrderState.NEW, OrderState.SUBMITTED, OrderState.PARTIAL):
            raise OrderError(f"Cannot modify order {order_id} in state {old_state}")

        for key, value in new_params.items():
            if hasattr(order, key):
                setattr(order, key, value)

        order.updated_at = datetime.utcnow()

    def fill(self, order_id: str, quantity: Decimal, price: Decimal) -> OrderState:
        """Report a fill for an order. Thread-safe with idempotency check.

        Returns the new state.
        """
        with self._lock:
            # Fill idempotency: reject duplicate fills
            fill_key = (order_id, quantity, price)
            if fill_key in self._fill_keys:
                return self._current_state(self._orders.get(order_id))  # type: ignore[return-value]
            self._fill_keys.add(fill_key)

            order = self._get_order(order_id)
            current_state = self._current_state(order)

            if current_state not in (OrderState.SUBMITTED, OrderState.PARTIAL):
                raise OrderError(f"Cannot fill order {order_id} in state {current_state}")

            order.filled_qty += quantity
            if order.avg_fill_price is None:
                order.avg_fill_price = price
            else:
                total_qty = order.filled_qty
                order.avg_fill_price = (
                    (order.avg_fill_price * (total_qty - quantity) + price * quantity) / total_qty
                )

            if order.filled_qty >= order.quantity:
                self._transition(order_id, current_state, OrderState.FILLED)
                order.status = OrderStatus.FILLED
                new_state = OrderState.FILLED
            else:
                if current_state != OrderState.PARTIAL:
                    self._transition(order_id, current_state, OrderState.PARTIAL)
                order.status = OrderStatus.PARTIAL
                new_state = OrderState.PARTIAL

            order.updated_at = datetime.utcnow()
        return new_state

    def reject(self, order_id: str, reason: str = "") -> None:
        """Reject an order."""
        order = self._get_order(order_id)
        current_state = self._current_state(order)
        self._transition(order_id, current_state, OrderState.REJECTED, metadata={"reason": reason})
        order.status = OrderStatus.REJECTED
        order.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise OrderError(f"Order not found: {order_id}")
        return self._orders[order_id]

    @staticmethod
    def _current_state(order: Order) -> OrderState:
        """Map Order.status to internal OrderState."""
        mapping = {
            OrderStatus.PENDING: OrderState.NEW,
            OrderStatus.SUBMITTED: OrderState.SUBMITTED,
            OrderStatus.PARTIAL: OrderState.PARTIAL,
            OrderStatus.FILLED: OrderState.FILLED,
            OrderStatus.CANCELLED: OrderState.CANCELLED,
            OrderStatus.REJECTED: OrderState.REJECTED,
        }
        return mapping.get(order.status, OrderState.NEW)

    def _transition(
        self,
        order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Validate and record a state transition."""
        valid_targets = TRANSITIONS.get(from_state, set())
        if to_state not in valid_targets:
            raise OrderError(
                f"Invalid transition: {from_state.value} → {to_state.value} for order {order_id}"
            )

        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            order_id=order_id,
            from_state=from_state.value,
            to_state=to_state.value,
            metadata=metadata or {},
        )
        self._audit.append(entry)

        if self._on_state_change and order_id in self._orders:
            self._on_state_change(self._orders[order_id], from_state, to_state)


# ---------------------------------------------------------------------------
# Convenience factory functions for order types
# ---------------------------------------------------------------------------


def market_order(
    symbol: str,
    side: Side,
    quantity: Decimal,
    strategy_id: str = "",
    order_id: str | None = None,
) -> Order:
    """Create a market order."""
    oid = order_id or f"mkt-{symbol}-{datetime.utcnow().timestamp():.0f}"
    return Order(
        id=oid,
        symbol=symbol,
        side=side,
        type=OrderType.MARKET,
        quantity=quantity,
        strategy_id=strategy_id,
    )


def limit_order(
    symbol: str,
    side: Side,
    quantity: Decimal,
    limit_price: Decimal,
    strategy_id: str = "",
    order_id: str | None = None,
) -> Order:
    """Create a limit order."""
    oid = order_id or f"lmt-{symbol}-{datetime.utcnow().timestamp():.0f}"
    return Order(
        id=oid,
        symbol=symbol,
        side=side,
        type=OrderType.LIMIT,
        quantity=quantity,
        limit_price=limit_price,
        strategy_id=strategy_id,
    )


def stop_order(
    symbol: str,
    side: Side,
    quantity: Decimal,
    stop_price: Decimal,
    strategy_id: str = "",
    order_id: str | None = None,
) -> Order:
    """Create a stop order."""
    oid = order_id or f"stp-{symbol}-{datetime.utcnow().timestamp():.0f}"
    return Order(
        id=oid,
        symbol=symbol,
        side=side,
        type=OrderType.STOP,
        quantity=quantity,
        stop_price=stop_price,
        strategy_id=strategy_id,
    )


def bracket_order(
    symbol: str,
    side: Side,
    quantity: Decimal,
    entry_price: Decimal,
    take_profit: Decimal,
    stop_loss: Decimal,
    strategy_id: str = "",
) -> tuple[Order, Order, Order]:
    """Create a bracket order (entry + take-profit + stop-loss).

    Returns (entry_order, tp_order, sl_order).
    """
    ts = datetime.utcnow().timestamp()
    oid_entry = f"bracket-entry-{symbol}-{ts:.0f}"
    oid_tp = f"bracket-tp-{symbol}-{ts:.0f}"
    oid_sl = f"bracket-sl-{symbol}-{ts:.0f}"

    close_side = Side.SELL if side == Side.BUY else Side.BUY

    entry = limit_order(symbol, side, quantity, entry_price, strategy_id, oid_entry)
    tp = limit_order(symbol, close_side, quantity, take_profit, strategy_id, oid_tp)
    sl = stop_order(symbol, close_side, quantity, stop_loss, strategy_id, oid_sl)

    return entry, tp, sl
