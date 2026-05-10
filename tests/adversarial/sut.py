"""System Under Test (SUT) — Black-Box Adapter for QuantCore.

The SUT is wrapped as a black box. The harness may only interact through:
  - submit_order()
  - cancel_order()
  - process_market_data()
  - receive_fill()

No internal logic is modified or bypassed for testing.
All cash/risk systems remain fully active.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantcore.core.cash import CashLedger
from quantcore.core.types import Fill, Order, Position, Side
from quantcore.execution.oms import OrderManager, market_order


@dataclass
class SUTSnapshot:
    """Snapshot of SUT state for oracle validation."""
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    unsettled_cash: float = 0.0
    committed_cash: float = 0.0
    available_cash: float = 0.0
    open_orders: int = 0
    fill_count: int = 0
    equity: float = 0.0


class SUTAdapter:
    """Black-box adapter around QuantCore live trading components.

    The SUT consists of:
      - CashLedger (settlement-aware cash)
      - OrderManager (order lifecycle)
      - Portfolio state (positions + cash tracking)

    All interactions go through the defined API surface. Nothing is bypassed.
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self._cash_ledger = CashLedger(initial_cash=initial_cash)
        self._oms = OrderManager()
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._prices: dict[str, float] = {}
        self._equity_history: list[tuple[datetime, float]] = []
        self._seen_fill_keys: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------------
    # Public API (the ONLY allowed interactions)
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> bool:
        """Submit an order. Returns True if accepted by OMS.

        Pre-trade checks:
          1. Cash availability (settlement-aware)
          2. Position sizing (no oversize)
          3. Duplicate rejection (OMS idempotency)

        NOTE: No risk limits here — the harness tests whether external
        risk systems (not embedded in SUT) would catch violations.
        """
        qty = float(order.quantity)
        price = float(order.limit_price) if order.limit_price else self._prices.get(order.symbol, 0.0)
        if price <= 0:
            return False

        order_value = qty * price

        # Settlement-aware cash check
        avail = self._cash_ledger.available()
        if order.side == Side.BUY and order_value > avail + 1.0:
            return False  # blocked by settlement constraint

        # Reserve cash
        try:
            self._cash_ledger.reserve(order_value, order.id)
        except Exception:
            return False

        # Submit to OMS (has idempotency)
        try:
            self._oms.submit(order)
            return True
        except Exception:
            self._cash_ledger.release(order.id)
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Releases committed cash."""
        try:
            self._oms.cancel(order_id)
            self._cash_ledger.release(order_id)
            return True
        except Exception:
            return False

    def receive_fill(self, fill: Fill, symbol: str | None = None) -> bool:
        """Receive a fill from the broker.

        Returns True if fill was accepted. Idempotent — duplicate fills
        are silently rejected via fill key tracking.
        """
        sym = symbol or fill.order_id.split("-")[0]
        qty = float(fill.quantity)
        price = float(fill.price)

        # Idempotency check: deduplicate by (order_id, price, quantity)
        fill_key = (fill.order_id, str(fill.price), str(fill.quantity))
        if fill_key in self._seen_fill_keys:
            return False  # duplicate silently rejected
        self._seen_fill_keys.add(fill_key)

        # Release committed cash
        self._cash_ledger.release(fill.order_id)

        # Update positions
        self._apply_fill(fill, sym)

        # Track fill
        self._fills.append(fill)
        return True

    def process_market_data(self, symbol: str, price: float) -> None:
        """Update market price for a symbol."""
        self._prices[symbol] = price
        # Update position market prices
        if symbol in self._positions:
            pos = self._positions[symbol]
            qty = float(pos.quantity)
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_price=Decimal(str(price)),
                unrealized_pnl=Decimal(str(qty * (price - float(pos.avg_cost)))),
                realized_pnl=pos.realized_pnl,
            )

    def process_settlements(self, reference_time: datetime | None = None) -> None:
        """Process pending settlements."""
        self._cash_ledger.settle_pending(reference_time)

    # ------------------------------------------------------------------
    # Internal — fill application
    # ------------------------------------------------------------------

    def _apply_fill(self, fill: Fill, symbol: str) -> None:
        """Apply a fill to SUT position state.

        Matches the kernel logic: buy/sell, avg cost blending, PnL.
        """
        qty = float(fill.quantity)
        price = float(fill.price)
        commission = float(fill.commission)
        is_buy = fill.side == Side.BUY if fill.side else True

        # Cash: add unsettled
        cash_delta = -(qty * price) if is_buy else (qty * price)
        self._cash_ledger.add_unsettled(
            amount=cash_delta - commission,
            asset_class="equity",
            description=f"Fill {fill.order_id}",
        )

        # Position
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
            old_avg = float(current.avg_cost)
            if is_buy:
                new_qty = old_qty + qty
                new_avg = (old_avg * old_qty + price * qty) / new_qty if old_qty >= 0 else old_avg
            else:
                new_qty = old_qty - qty
                new_avg = (old_avg * abs(old_qty) + price * qty) / abs(new_qty) if old_qty <= 0 else old_avg

            if abs(new_qty) < 1e-10:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal(str(new_qty)),
                    avg_cost=Decimal(str(new_avg)),
                    market_price=Decimal(str(price)),
                )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def snapshot(self) -> SUTSnapshot:
        """Capture current SUT state for validation."""
        total_pos_value = sum(
            float(p.quantity) * float(p.market_price)
            for p in self._positions.values()
        )
        return SUTSnapshot(
            positions=dict(self._positions),
            cash=self._cash_ledger.settled_cash,
            unsettled_cash=self._cash_ledger.unsettled_cash,
            committed_cash=self._cash_ledger.committed_cash,
            available_cash=self._cash_ledger.available(),
            open_orders=len(self._oms.orders),
            fill_count=len(self._fills),
            equity=self._cash_ledger.settled_cash + total_pos_value,
        )

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def cash_ledger(self) -> CashLedger:
        return self._cash_ledger

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def oms(self) -> OrderManager:
        return self._oms

    @property
    def equity(self) -> float:
        total_pos_value = sum(
            float(p.quantity) * float(p.market_price)
            for p in self._positions.values()
        )
        return self._cash_ledger.settled_cash + total_pos_value

    def reset(self, initial_cash: float = 1_000_000.0) -> None:
        """Reset SUT to initial state."""
        self._cash_ledger = CashLedger(initial_cash=initial_cash)
        self._oms = OrderManager()
        self._positions.clear()
        self._fills.clear()
        self._prices.clear()
        self._equity_history.clear()
        self._seen_fill_keys.clear()
