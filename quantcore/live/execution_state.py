"""Thread-safe execution state monitor for live broker event streams.

The monitor is intentionally broker-neutral. Broker adapters should translate
native callbacks into :class:`ExecutionEvent` objects and enqueue them. The
background worker is the only component that mutates order, inventory, cash, and
reservation state.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from quantcore.core.types import Side

log = logging.getLogger(__name__)

_FILL_STATES = {"partially_filled", "filled"}
_TERMINAL_STATES = {"cancelled", "rejected", "expired"}
_PENDING_STATES = {"pending_submit", "submitted", "pending_cancel"}


class ExecutionEventState(str, Enum):
    """Broker-neutral order lifecycle states normalized by adapters."""

    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


_STATE_ALIASES: dict[str, ExecutionEventState] = {
    "pending_submit": ExecutionEventState.PENDING_SUBMIT,
    "pending": ExecutionEventState.PENDING_SUBMIT,
    "new": ExecutionEventState.SUBMITTED,
    "accepted": ExecutionEventState.SUBMITTED,
    "submitted": ExecutionEventState.SUBMITTED,
    "open": ExecutionEventState.SUBMITTED,
    "partial": ExecutionEventState.PARTIALLY_FILLED,
    "partially_filled": ExecutionEventState.PARTIALLY_FILLED,
    "partiallyfilled": ExecutionEventState.PARTIALLY_FILLED,
    "filled": ExecutionEventState.FILLED,
    "complete": ExecutionEventState.FILLED,
    "pending_cancel": ExecutionEventState.PENDING_CANCEL,
    "pendingcancel": ExecutionEventState.PENDING_CANCEL,
    "cancelled": ExecutionEventState.CANCELLED,
    "canceled": ExecutionEventState.CANCELLED,
    "cancel": ExecutionEventState.CANCELLED,
    "rejected": ExecutionEventState.REJECTED,
    "expire": ExecutionEventState.EXPIRED,
    "expired": ExecutionEventState.EXPIRED,
}


@dataclass(frozen=True)
class ExecutionEvent:
    """Normalized execution event emitted by broker callbacks.

    ``quantity`` is the quantity represented by this event. For fill events it is
    the incremental fill quantity. ``filled_quantity`` is optional cumulative
    filled quantity and is used to derive incremental fills when ``quantity`` is
    not supplied.
    """

    broker_name: str
    internal_order_id: str
    state: ExecutionEventState
    symbol: str
    side: Side
    event_id: str = ""
    broker_order_id: str | None = None
    quantity: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal | None = None
    price: Decimal | None = None
    commission: Decimal = Decimal("0")
    estimated_price: Decimal | None = None
    execution_hash: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class InFlightReservation:
    """Estimated cash/margin reserved while an order is in flight."""

    order_id: str
    internal_order_id: str
    broker_order_id: str | None
    symbol: str
    side: Side
    quantity: Decimal
    reserved_cash: Decimal
    reserved_margin: Decimal
    estimated_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderExecutionState:
    """Internal per-order state maintained by the monitor."""

    internal_order_id: str
    broker_order_id: str | None
    symbol: str
    side: Side
    quantity: Decimal
    state: ExecutionEventState
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    reserved_cash: Decimal = Decimal("0")
    reserved_margin: Decimal = Decimal("0")
    execution_hashes: set[str] = field(default_factory=set)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InventoryPosition:
    """Incremental inventory projection built from execution events."""

    symbol: str
    quantity: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    cash_delta: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of anti-race broker snapshot reconciliation."""

    timestamp: datetime
    broker_order_ids: set[str]
    in_flight_order_ids: set[str]
    skipped_order_ids: set[str]
    skipped_symbols: set[str]
    reconciled_symbols: set[str]
    should_sync_account: bool
    details: str = ""


def _as_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _as_side(value: Any) -> Side:
    if isinstance(value, Side):
        return value
    text = str(value).strip().upper()
    if text in {"BUY", "B"}:
        return Side.BUY
    if text in {"SELL", "S"}:
        return Side.SELL
    raise ValueError(f"Unsupported execution side: {value!r}")


def _as_event_state(value: str | ExecutionEventState) -> ExecutionEventState:
    if isinstance(value, ExecutionEventState):
        return value
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return ExecutionEventState(normalized)
    except ValueError:
        try:
            return _STATE_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported execution state: {value!r}") from exc


def _stable_hash(parts: Iterable[Any]) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ExecutionStateMonitor:
    """Async worker that turns broker callbacks into durable execution state.

    Broker callbacks should call :meth:`enqueue` and return immediately. The
    worker processes events sequentially, deduplicates execution reports, updates
    inventory, and maintains cash/margin reservations for open orders.
    """

    def __init__(
        self,
        *,
        margin_requirement: Decimal | float = Decimal("1"),
        max_processed_exec_ids: int = 100_000,
    ) -> None:
        self.margin_requirement = _as_decimal(margin_requirement, Decimal("1"))
        self.max_processed_exec_ids = max_processed_exec_ids

        self._queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = threading.RLock()
        self._closed = False
        self._accepting_events = True

        self._processed_exec_ids: set[str] = set()
        self._orders: dict[str, OrderExecutionState] = {}
        self._in_flight_reservations: dict[str, InFlightReservation] = {}
        self._positions: dict[str, InventoryPosition] = {}
        self._cash = Decimal("0")

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the asynchronous worker on the provided or current event loop."""
        if self._task and not self._task.done():
            return
        if self._closed:
            raise RuntimeError("ExecutionStateMonitor has been stopped")

        active_loop = loop or asyncio.get_running_loop()
        self._loop = active_loop
        self._queue = asyncio.Queue()
        self._task = active_loop.create_task(self._worker_loop(), name="ExecutionStateMonitor")

    async def stop(self) -> None:
        """Cancel the worker and wait for queue shutdown."""
        self._closed = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def hold_events(self) -> None:
        """Stop accepting new broker events immediately.

        This is safe to call from synchronous broker disconnect callbacks. Use
        :meth:`pause` when reconnect code needs to drain and clear the queue too.
        """
        with self._state_lock:
            self._accepting_events = False

    async def pause(self) -> None:
        """Pause event ingestion, drain the in-flight event, and clear backlog."""
        self.hold_events()
        await self.drain()
        await self.clear_pending_events()

    def resume(self) -> None:
        """Resume accepting broker events after reconciliation."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("ExecutionStateMonitor has been stopped")
            self._accepting_events = True

    async def clear_pending_events(self) -> None:
        """Drop queued events that arrived before a broker reconciliation pass."""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()

    @property
    def accepting_events(self) -> bool:
        with self._state_lock:
            return self._accepting_events

    def enqueue(self, event: ExecutionEvent) -> asyncio.Future[None] | None:
        """Enqueue an event from a synchronous broker callback.

        The method never performs state mutation. If the callback runs on the
        monitor event loop, the event is queued directly. If it runs on another
        thread, the put is scheduled with ``run_coroutine_threadsafe`` and the
        callback returns immediately.
        """
        self._ensure_started()
        assert self._loop is not None

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        with self._state_lock:
            if not self._accepting_events:
                return None
            if current_loop is self._loop:
                self._queue.put_nowait(event)
                return None
            return asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)

    async def enqueue_async(self, event: ExecutionEvent) -> None:
        """Enqueue an event from async broker streaming code."""
        with self._state_lock:
            if not self._accepting_events:
                return
        if self._loop is None:
            self.start()
        await self._queue.put(event)

    async def drain(self) -> None:
        """Wait until all currently queued events have been processed."""
        await self._queue.join()

    def process_event(self, event: ExecutionEvent) -> bool:
        """Synchronously process one event.

        This is primarily useful for deterministic unit tests. Normal live usage
        should call :meth:`enqueue`.
        """
        event_id = self._event_identity(event)
        with self._state_lock:
            if event_id in self._processed_exec_ids:
                return False

            self._processed_exec_ids.add(event_id)
            self._trim_processed_ids()

            state = event.state
            if state.value in _PENDING_STATES:
                self._handle_submitted(event)
            elif state.value in _FILL_STATES:
                self._handle_fill(event)
            elif state.value in _TERMINAL_STATES:
                self._handle_terminal(event)
            else:  # pragma: no cover - defensive guard for future states
                log.warning("Unhandled execution state for %s: %s", event.internal_order_id, state)

        return True

    async def _worker_loop(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                self.process_event(event)
            except Exception:
                log.exception("Failed to process execution event: %s", event)
            finally:
                self._queue.task_done()

    def _ensure_started(self) -> None:
        if self._task and not self._task.done():
            return
        try:
            self.start()
        except RuntimeError as exc:
            raise RuntimeError(
                "ExecutionStateMonitor must be started from a running event loop "
                "before enqueueing events from another thread"
            ) from exc

    def _event_identity(self, event: ExecutionEvent) -> str:
        if event.event_id:
            return event.event_id
        if event.execution_hash:
            return f"{event.broker_name}:execution:{event.execution_hash}"

        raw = dict(event.raw or {})
        for key in (
            "execution_id",
            "exec_id",
            "fill_id",
            "trade_id",
            "transaction_id",
            "order_update_id",
        ):
            value = raw.get(key)
            if value not in (None, ""):
                return f"{event.broker_name}:{key}:{value}"

        if event.state.value in _FILL_STATES:
            parts = (
                event.broker_name,
                event.broker_order_id or event.internal_order_id,
                event.symbol,
                event.side.value,
                event.quantity,
                event.price,
                event.commission,
                event.timestamp.isoformat(),
            )
        else:
            parts = (
                event.broker_name,
                event.broker_order_id or event.internal_order_id,
                event.state.value,
                event.timestamp.isoformat(),
            )
        return _stable_hash(parts)

    def _trim_processed_ids(self) -> None:
        while len(self._processed_exec_ids) > self.max_processed_exec_ids:
            self._processed_exec_ids.pop()

    def _reservation_key(self, event: ExecutionEvent) -> str:
        return str(event.broker_order_id or event.internal_order_id)

    def _get_or_create_order_state(self, event: ExecutionEvent) -> OrderExecutionState:
        state = self._orders.get(event.internal_order_id)
        if state is None:
            state = OrderExecutionState(
                internal_order_id=event.internal_order_id,
                broker_order_id=event.broker_order_id,
                symbol=event.symbol,
                side=event.side,
                quantity=event.quantity or (event.remaining_quantity or Decimal("0")),
                state=event.state,
            )
            self._orders[event.internal_order_id] = state
        else:
            state.broker_order_id = event.broker_order_id or state.broker_order_id
            state.symbol = event.symbol
            state.side = event.side
            if event.quantity:
                state.quantity = event.quantity
            state.state = event.state
        state.updated_at = event.timestamp
        return state

    def _handle_submitted(self, event: ExecutionEvent) -> None:
        state = self._get_or_create_order_state(event)
        reservation = self._estimate_reservation(event)
        existing = self._in_flight_reservations.get(self._reservation_key(event))
        if existing is not None:
            reservation.filled_quantity = existing.filled_quantity
            reservation.remaining_quantity = max(
                Decimal("0"), reservation.quantity - reservation.filled_quantity
            )

        self._in_flight_reservations[self._reservation_key(event)] = reservation
        state.reserved_cash = reservation.reserved_cash
        state.reserved_margin = reservation.reserved_margin
        state.remaining_quantity = reservation.remaining_quantity

    def _incremental_fill_quantity(
        self,
        event: ExecutionEvent,
        state: OrderExecutionState,
    ) -> Decimal:
        if event.quantity > 0:
            return event.quantity
        if event.filled_quantity > 0:
            return max(Decimal("0"), event.filled_quantity - state.filled_quantity)
        return Decimal("0")

    def _handle_fill(self, event: ExecutionEvent) -> None:
        state = self._get_or_create_order_state(event)
        fill_qty = self._incremental_fill_quantity(event, state)
        price = event.price or state.avg_fill_price or Decimal("0")

        if fill_qty > 0 and price > 0:
            self._apply_fill_to_inventory(
                symbol=event.symbol,
                side=event.side,
                quantity=fill_qty,
                price=price,
                commission=event.commission,
                timestamp=event.timestamp,
            )
            old_filled = state.filled_quantity
            old_avg = state.avg_fill_price
            state.filled_quantity += fill_qty
            if state.filled_quantity > 0 and price > 0:
                if old_filled > 0 and old_avg > 0:
                    state.avg_fill_price = (
                        (old_avg * old_filled) + (price * fill_qty)
                    ) / state.filled_quantity
                else:
                    state.avg_fill_price = price

        key = self._reservation_key(event)
        reservation = self._in_flight_reservations.get(key)
        if reservation is None:
            reservation = self._estimate_reservation(event)
            self._in_flight_reservations[key] = reservation

        filled_notional = fill_qty * (price or reservation.estimated_price or Decimal("0"))
        reservation.filled_quantity += fill_qty
        reservation.remaining_quantity = max(
            Decimal("0"), reservation.quantity - reservation.filled_quantity
        )
        if event.side == Side.BUY:
            reservation.reserved_cash = max(
                Decimal("0"), reservation.reserved_cash - filled_notional
            )
        else:
            reservation.reserved_margin = max(
                Decimal("0"),
                reservation.reserved_margin - (filled_notional * self.margin_requirement),
            )
        reservation.updated_at = event.timestamp

        state.reserved_cash = reservation.reserved_cash
        state.reserved_margin = reservation.reserved_margin
        state.remaining_quantity = reservation.remaining_quantity

        if event.execution_hash:
            state.execution_hashes.add(event.execution_hash)

        if event.state == ExecutionEventState.FILLED:
            self._in_flight_reservations.pop(key, None)
            state.reserved_cash = Decimal("0")
            state.reserved_margin = Decimal("0")

    def _handle_terminal(self, event: ExecutionEvent) -> None:
        key = self._reservation_key(event)
        self._in_flight_reservations.pop(key, None)

        state = self._orders.get(event.internal_order_id)
        if state is not None:
            state.state = event.state
            state.remaining_quantity = Decimal("0")
            state.reserved_cash = Decimal("0")
            state.reserved_margin = Decimal("0")
            state.updated_at = event.timestamp

    def _estimate_reservation(self, event: ExecutionEvent) -> InFlightReservation:
        quantity = event.remaining_quantity if event.remaining_quantity is not None else event.quantity
        quantity = abs(quantity)
        price = event.price or event.estimated_price or Decimal("0")
        notional = quantity * price

        if event.side == Side.BUY:
            reserved_cash = notional
            reserved_margin = Decimal("0")
        else:
            reserved_cash = Decimal("0")
            reserved_margin = notional * self.margin_requirement

        return InFlightReservation(
            order_id=self._reservation_key(event),
            internal_order_id=event.internal_order_id,
            broker_order_id=event.broker_order_id,
            symbol=event.symbol,
            side=event.side,
            quantity=quantity,
            reserved_cash=reserved_cash,
            reserved_margin=reserved_margin,
            estimated_price=price or None,
            remaining_quantity=quantity,
            created_at=event.timestamp,
            updated_at=event.timestamp,
        )

    def _apply_fill_to_inventory(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        timestamp: datetime,
    ) -> None:
        position = self._positions.get(symbol)
        if position is None:
            position = InventoryPosition(symbol=symbol)
            self._positions[symbol] = position

        signed_qty = quantity if side == Side.BUY else -quantity
        old_qty = position.quantity
        old_avg = position.avg_cost

        if old_qty == 0:
            position.quantity = signed_qty
            position.avg_cost = price
        elif (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
            new_abs_qty = abs(old_qty) + quantity
            position.avg_cost = ((abs(old_qty) * old_avg) + (quantity * price)) / new_abs_qty
            position.quantity += signed_qty
        else:
            closing_qty = min(quantity, abs(old_qty))
            if old_qty > 0:
                position.realized_pnl += closing_qty * (price - old_avg)
            else:
                position.realized_pnl += closing_qty * (old_avg - price)

            remaining_qty = old_qty + signed_qty
            if remaining_qty == 0:
                position.quantity = Decimal("0")
                position.avg_cost = Decimal("0")
            elif (remaining_qty > 0 and signed_qty < 0) or (remaining_qty < 0 and signed_qty > 0):
                position.quantity = remaining_qty
                # Existing average cost remains valid for the remaining leg.
            else:
                position.quantity = remaining_qty
                position.avg_cost = price

        cash_delta = (-signed_qty * price) - commission
        position.cash_delta += cash_delta
        position.commission += commission
        self._cash += cash_delta
        position.updated_at = timestamp

    def reconcile_with_broker_snapshot(
        self,
        broker_orders: Mapping[str, Any] | Iterable[Any],
        broker_positions: Mapping[str, Any] | None = None,
        broker_account: Mapping[str, float] | None = None,
    ) -> ReconciliationResult:
        """Return a race-safe reconciliation plan for a broker snapshot.

        If an open broker order ID matches a key in ``in_flight_reservations``,
        that symbol is skipped. This prevents polling reconciliation from
        overwriting a position while a streaming fill for the same symbol is
        being applied.
        """
        skipped_order_ids: set[str] = set()
        skipped_symbols: set[str] = set()
        broker_order_ids: set[str] = set()

        with self._state_lock:
            reservations = dict(self._in_flight_reservations)

        if isinstance(broker_orders, Mapping):
            order_items = broker_orders.items()
        else:
            order_items = ((None, order) for order in broker_orders)

        for maybe_id, order in order_items:
            order_id = self._extract_order_id(order) or str(maybe_id)
            if not order_id:
                continue
            broker_order_ids.add(order_id)
            symbol = self._extract_symbol(order)
            if order_id in reservations:
                skipped_order_ids.add(order_id)
                if symbol:
                    skipped_symbols.add(symbol)

        reconciled_symbols: set[str] = set()
        if broker_positions:
            for symbol in broker_positions:
                if symbol not in skipped_symbols:
                    reconciled_symbols.add(str(symbol))

        return ReconciliationResult(
            timestamp=datetime.utcnow(),
            broker_order_ids=broker_order_ids,
            in_flight_order_ids=set(reservations),
            skipped_order_ids=skipped_order_ids,
            skipped_symbols=skipped_symbols,
            reconciled_symbols=reconciled_symbols,
            should_sync_account=not skipped_symbols,
            details=(
                f"Skipped {len(skipped_symbols)} symbol(s) with in-flight broker "
                f"orders: {sorted(skipped_symbols)}"
                if skipped_symbols
                else "No in-flight order collision detected"
            ),
        )

    def _extract_order_id(self, order: Any) -> str | None:
        if isinstance(order, str):
            return order
        if isinstance(order, Mapping):
            for key in ("broker_order_id", "order_id", "id", "client_order_id"):
                value = order.get(key)
                if value not in (None, ""):
                    return str(value)
        for attr in ("broker_order_id", "order_id", "id", "client_order_id"):
            value = getattr(order, attr, None)
            if value not in (None, ""):
                return str(value)
        return None

    def _extract_symbol(self, order: Any) -> str | None:
        if isinstance(order, Mapping):
            for key in ("symbol", "contract_symbol"):
                value = order.get(key)
                if value not in (None, ""):
                    return str(value)
        for attr in ("symbol", "contract_symbol"):
            value = getattr(order, attr, None)
            if value not in (None, ""):
                return str(value)
        return None

    @property
    def processed_exec_ids(self) -> set[str]:
        with self._state_lock:
            return set(self._processed_exec_ids)

    @property
    def in_flight_reservations(self) -> dict[str, InFlightReservation]:
        with self._state_lock:
            return dict(self._in_flight_reservations)

    @property
    def orders(self) -> dict[str, OrderExecutionState]:
        with self._state_lock:
            return {
                order_id: OrderExecutionState(
                    internal_order_id=state.internal_order_id,
                    broker_order_id=state.broker_order_id,
                    symbol=state.symbol,
                    side=state.side,
                    quantity=state.quantity,
                    state=state.state,
                    filled_quantity=state.filled_quantity,
                    remaining_quantity=state.remaining_quantity,
                    avg_fill_price=state.avg_fill_price,
                    reserved_cash=state.reserved_cash,
                    reserved_margin=state.reserved_margin,
                    execution_hashes=set(state.execution_hashes),
                    updated_at=state.updated_at,
                )
                for order_id, state in self._orders.items()
            }

    @property
    def positions(self) -> dict[str, InventoryPosition]:
        with self._state_lock:
            return {
                symbol: InventoryPosition(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_cost=position.avg_cost,
                    realized_pnl=position.realized_pnl,
                    cash_delta=position.cash_delta,
                    commission=position.commission,
                    updated_at=position.updated_at,
                )
                for symbol, position in self._positions.items()
            }

    @property
    def cash(self) -> Decimal:
        with self._state_lock:
            return self._cash
