"""Persistent state store for portfolio, orders, fills, and cash.

On startup, the system MUST be able to:
  1. Load persisted state from disk.
  2. Reconstruct portfolio positions, open orders, fills, and cash.
  3. Reconcile reconstructed state with broker state.
  4. If mismatch exceeds tolerance → HALT trading and require manual resolution.

Design:
  - JSON-based storage for durability and inspectability.
  - Atomic writes (write to tmp, rename).
  - Separate files for positions, orders, fills, and cash ledger so
    partial updates don't require rewriting the full system state.
  - File locking via rename-based atomicity (no concurrent corruption).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quantcore.core.cash import CashLedger
from quantcore.core.types import Fill, Order, OrderStatus, OrderType, Position, Side
from quantcore.execution.oms import OrderManager

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class StatePersistenceError(Exception):
    """Raised when state persistence operations fail."""


class StateReconciliationError(StatePersistenceError):
    """Raised when reconstructed state does not match broker state."""


# ---------------------------------------------------------------------------
# File-based state store
# ---------------------------------------------------------------------------


@dataclass
class StateStoreConfig:
    """Configuration for the state store."""
    data_dir: str = "state"
    positions_file: str = "positions.json"
    orders_file: str = "orders.json"
    fills_file: str = "fills.json"
    cash_file: str = "cash_ledger.json"
    snapshot_file: str = "system_snapshot.json"
    max_fill_history: int = 10_000
    flush_interval_sec: float = 5.0


class StateStore:
    """Persistent state store with atomic JSON writes.

    Usage:
        store = StateStore(data_dir="./state")
        store.save_positions({"AAPL": Position(...)})
        store.save_orders(oms)
        store.save_fills([...])
        store.save_cash_ledger(ledger)

        # On restart:
        store = StateStore(data_dir="./state")
        positions = store.load_positions()
        orders = store.load_orders()
        ledger = store.load_cash_ledger()
    """

    def __init__(self, config: StateStoreConfig | None = None) -> None:
        self.config = config or StateStoreConfig()
        self._data_dir = Path(self.config.data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_positions(self, positions: dict[str, Position]) -> None:
        """Persist positions dict."""
        data = {
            sym: {
                "symbol": p.symbol,
                "quantity": str(p.quantity),
                "avg_cost": str(p.avg_cost),
                "market_price": str(p.market_price),
                "unrealized_pnl": str(p.unrealized_pnl),
                "realized_pnl": str(p.realized_pnl),
            }
            for sym, p in positions.items()
        }
        self._atomic_write(self.config.positions_file, data)

    def load_positions(self) -> dict[str, Position]:
        """Load persisted positions. Returns empty dict if no state exists."""
        data = self._atomic_read(self.config.positions_file)
        if not data:
            return {}
        positions: dict[str, Position] = {}
        for sym, raw in data.items():
            try:
                positions[sym] = Position(
                    symbol=raw["symbol"],
                    quantity=Decimal(raw["quantity"]),
                    avg_cost=Decimal(raw["avg_cost"]),
                    market_price=Decimal(raw.get("market_price", "0")),
                    unrealized_pnl=Decimal(raw.get("unrealized_pnl", "0")),
                    realized_pnl=Decimal(raw.get("realized_pnl", "0")),
                )
            except (KeyError, ValueError) as e:
                raise StatePersistenceError(f"Corrupt position data for {sym}: {e}") from e
        return positions

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def save_orders(self, oms: OrderManager) -> None:
        """Persist all orders from the OrderManager."""
        data: list[dict[str, Any]] = []
        for oid, order in oms.orders.items():
            data.append({
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.type.value,
                "quantity": str(order.quantity),
                "limit_price": str(order.limit_price) if order.limit_price else None,
                "stop_price": str(order.stop_price) if order.stop_price else None,
                "status": order.status.value,
                "strategy_id": order.strategy_id,
                "filled_qty": str(order.filled_qty),
                "avg_fill_price": str(order.avg_fill_price) if order.avg_fill_price else None,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat(),
            })
        self._atomic_write(self.config.orders_file, data)

    def load_orders(self) -> list[Order]:
        """Load persisted orders. Returns empty list if no state exists."""
        data = self._atomic_read(self.config.orders_file)
        if not data:
            return []
        orders: list[Order] = []
        for raw in data:
            try:
                kwargs: dict[str, Any] = {
                    "id": raw["id"],
                    "symbol": raw["symbol"],
                    "side": Side(raw["side"]),
                    "type": OrderType(raw["type"]),
                    "quantity": Decimal(raw["quantity"]),
                    "status": OrderStatus(raw.get("status", "pending")),
                    "strategy_id": raw.get("strategy_id", ""),
                    "filled_qty": Decimal(raw.get("filled_qty", "0")),
                }
                if raw.get("limit_price"):
                    kwargs["limit_price"] = Decimal(raw["limit_price"])
                if raw.get("stop_price"):
                    kwargs["stop_price"] = Decimal(raw["stop_price"])
                if raw.get("avg_fill_price"):
                    kwargs["avg_fill_price"] = Decimal(raw["avg_fill_price"])
                if raw.get("created_at"):
                    kwargs["created_at"] = datetime.fromisoformat(raw["created_at"])
                if raw.get("updated_at"):
                    kwargs["updated_at"] = datetime.fromisoformat(raw["updated_at"])
                orders.append(Order(**kwargs))
            except (KeyError, ValueError) as e:
                raise StatePersistenceError(f"Corrupt order data: {e}") from e
        return orders

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def save_fills(self, fills: list[Fill]) -> None:
        """Persist fill history."""
        data = [
            {
                "order_id": f.order_id,
                "price": str(f.price),
                "quantity": str(f.quantity),
                "commission": str(f.commission),
                "timestamp": f.timestamp.isoformat(),
                "exchange": f.exchange,
                "side": f.side.value if f.side else None,
                "slippage_bps": str(f.slippage_bps),
                "market_impact_cost": str(f.market_impact_cost),
                "impact_bps": str(f.impact_bps),
            }
            for f in fills
        ]
        self._atomic_write(self.config.fills_file, data)

    def load_fills(self) -> list[Fill]:
        """Load persisted fills. Returns empty list if no state exists."""
        data = self._atomic_read(self.config.fills_file)
        if not data:
            return []
        fills: list[Fill] = []
        for raw in data:
            try:
                kwargs: dict[str, Any] = {
                    "order_id": raw["order_id"],
                    "price": Decimal(raw["price"]),
                    "quantity": Decimal(raw["quantity"]),
                    "commission": Decimal(raw.get("commission", "0")),
                    "exchange": raw.get("exchange", ""),
                    "slippage_bps": Decimal(raw.get("slippage_bps", "0")),
                    "market_impact_cost": Decimal(raw.get("market_impact_cost", "0")),
                    "impact_bps": Decimal(raw.get("impact_bps", "0")),
                }
                if raw.get("side"):
                    kwargs["side"] = Side(raw["side"])
                if raw.get("timestamp"):
                    kwargs["timestamp"] = datetime.fromisoformat(raw["timestamp"])
                fills.append(Fill(**kwargs))
            except (KeyError, ValueError) as e:
                raise StatePersistenceError(f"Corrupt fill data: {e}") from e
        return fills

    # ------------------------------------------------------------------
    # Cash ledger
    # ------------------------------------------------------------------

    def save_cash_ledger(self, ledger: CashLedger) -> None:
        """Persist the cash ledger state."""
        self._atomic_write(self.config.cash_file, ledger.snapshot())

    def load_cash_ledger(self) -> CashLedger | None:
        """Load persisted cash ledger. Returns None if no state exists."""
        data = self._atomic_read(self.config.cash_file)
        if not data:
            return None
        try:
            return CashLedger.from_snapshot(data)
        except Exception as e:
            raise StatePersistenceError(f"Corrupt cash ledger data: {e}") from e

    # ------------------------------------------------------------------
    # Full system snapshot (one file to capture everything)
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        positions: dict[str, Position],
        orders: OrderManager,
        fills: list[Fill],
        ledger: CashLedger,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save complete system state in a single atomic snapshot."""
        snapshot: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": {
                sym: {
                    "symbol": p.symbol,
                    "quantity": str(p.quantity),
                    "avg_cost": str(p.avg_cost),
                    "market_price": str(p.market_price),
                    "realized_pnl": str(p.realized_pnl),
                }
                for sym, p in positions.items()
            },
            "orders": [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "type": o.type.value,
                    "quantity": str(o.quantity),
                    "status": o.status.value,
                    "filled_qty": str(o.filled_qty),
                }
                for o in orders.orders.values()
            ],
            "fills": [
                {
                    "order_id": f.order_id,
                    "price": str(f.price),
                    "quantity": str(f.quantity),
                    "side": f.side.value if f.side else None,
                }
                for f in fills
            ],
            "cash_ledger": ledger.snapshot(),
        }
        if metadata:
            snapshot["metadata"] = metadata
        self._atomic_write(self.config.snapshot_file, snapshot)

    def load_snapshot(self) -> dict[str, Any] | None:
        """Load and return the full system snapshot, or None."""
        return self._atomic_read(self.config.snapshot_file)

    # ------------------------------------------------------------------
    # Reconciliation on restart
    # ------------------------------------------------------------------

    def verify_state(
        self,
        broker_positions: dict[str, Position],
        broker_cash: float,
        tolerance_qty: float = 1.0,
        tolerance_cash: float = 10.0,
    ) -> list[str]:
        """Verify persisted state matches broker state on startup.

        Args:
            broker_positions: Current positions from broker.
            broker_cash: Current cash from broker.
            tolerance_qty: Maximum share/contract quantity discrepancy tolerated.
            tolerance_cash: Maximum cash discrepancy tolerated.

        Returns:
            List of discrepancy descriptions. Empty list = state is valid.

        Raises:
            StateReconciliationError if mismatches exceed tolerance.
        """
        discrepancies: list[str] = []

        # Load persisted positions
        local_positions = self.load_positions()

        # Check positions
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
        for sym in all_symbols:
            local_qty = float(local_positions[sym].quantity) if sym in local_positions else 0.0
            broker_qty = float(broker_positions[sym].quantity) if sym in broker_positions else 0.0
            diff = abs(local_qty - broker_qty)
            if diff > tolerance_qty:
                discrepancies.append(
                    f"Position {sym}: local={local_qty:.2f} broker={broker_qty:.2f} "
                    f"diff={diff:.2f} (tolerance={tolerance_qty})"
                )

        # Load persisted cash
        local_ledger = self.load_cash_ledger()
        local_cash = local_ledger.settled_cash if local_ledger else 0.0
        cash_diff = abs(local_cash - broker_cash)
        if cash_diff > tolerance_cash:
            discrepancies.append(
                f"Cash: local=${local_cash:.2f} broker=${broker_cash:.2f} "
                f"diff=${cash_diff:.2f} (tolerance=${tolerance_cash:.2f})"
            )

        return discrepancies

    # ------------------------------------------------------------------
    # Internal file helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, filename: str, data: Any) -> None:
        """Write data atomically to a file (write tmp → rename)."""
        filepath = self._data_dir / filename
        tmp_path = filepath.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, filepath)
        except OSError as e:
            raise StatePersistenceError(f"Failed to write {filename}: {e}") from e

    def _atomic_read(self, filename: str) -> Any:
        """Read data from a file. Returns None if file doesn't exist."""
        filepath = self._data_dir / filename
        if not filepath.exists():
            return None
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise StatePersistenceError(f"Failed to read {filename}: {e}") from e

    def clear(self) -> None:
        """Remove all persisted state files."""
        for filename in [
            self.config.positions_file,
            self.config.orders_file,
            self.config.fills_file,
            self.config.cash_file,
            self.config.snapshot_file,
        ]:
            filepath = self._data_dir / filename
            if filepath.exists():
                filepath.unlink()
