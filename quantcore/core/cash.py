"""Settlement-aware cash ledger with three-tier cash tracking.

Provides canonical settlement logic for the entire system:
  - settled_cash:  Fully cleared cash, available for trading.
  - unsettled_cash: Cash from trades pending settlement (T+1/T+2).
  - committed_cash: Cash reserved by open, unfilled orders.

Order sizing MUST only use: settled_cash - committed_cash.

Settlement rules (Reg T / industry standard):
  - Equities: T+2 settlement
  - Options:   T+1 settlement
  - FX:        T+0 settlement (spot)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal


AssetClass = Literal["equity", "option", "fx"]


SETTLEMENT_LAG: dict[AssetClass, int] = {
    "equity": 2,  # T+2
    "option": 1,  # T+1
    "fx": 0,       # T+0
}


class CashError(Exception):
    """Raised when a cash operation violates settlement or sufficiency rules."""


@dataclass
class SettlementEntry:
    """A single unsettled cash amount with its expected settlement date."""
    amount: float
    asset_class: AssetClass
    trade_date: datetime
    settle_date: datetime
    description: str = ""


class CashLedger:
    """Three-tier cash ledger with settlement-aware availability checks.

    Usage:
        ledger = CashLedger(initial_cash=100_000.0)
        avail = ledger.available()          # settled - committed
        ledger.reserve(10_000.0, "ord-1")   # commit for an order
        ledger.release("ord-1")             # release committed
        ledger.add_unsettled(-50_000.0, "equity", "Buy AAPL")
        ledger.settle_pending()             # move due entries → settled
    """

    def __init__(self, initial_cash: float = 0.0) -> None:
        self._settled: float = initial_cash
        self._unsettled: list[SettlementEntry] = []
        self._committed: dict[str, float] = {}  # order_id -> amount
        self._total_committed: float = 0.0

    # ------------------------------------------------------------------
    # Public read properties
    # ------------------------------------------------------------------

    @property
    def settled_cash(self) -> float:
        return self._settled

    @property
    def unsettled_cash(self) -> float:
        return sum(e.amount for e in self._unsettled)

    @property
    def committed_cash(self) -> float:
        return self._total_committed

    @property
    def total_cash(self) -> float:
        """Total cash (settled + unsettled)."""
        return self._settled + self.unsettled_cash

    def available(self) -> float:
        """Cash available for trading: settled - committed.

        This is the ONLY method that should be used for order sizing.
        """
        return self._settled - self._total_committed

    # ------------------------------------------------------------------
    # Reserve / release (committed cash for open orders)
    # ------------------------------------------------------------------

    def reserve(self, amount: float, order_id: str) -> None:
        """Reserve cash for an open order.

        Raises CashError if insufficient settled cash (after existing commitments).
        """
        if amount <= 0:
            return
        avail = self._settled - self._total_committed
        if amount > avail + 1e-10:  # small tolerance for fp
            raise CashError(
                f"Insufficient available cash: need ${amount:.2f}, "
                f"have ${avail:.2f} (settled=${self._settled:.2f}, "
                f"committed=${self._total_committed:.2f})"
            )
        self._committed[order_id] = self._committed.get(order_id, 0.0) + amount
        self._total_committed += amount

    def release(self, order_id: str, amount: float | None = None) -> None:
        """Release committed cash for an order (on fill or cancel).

        Args:
            order_id: The order whose commitment to release.
            amount: Specific amount to release. If None, releases all.
        """
        if order_id not in self._committed:
            return
        existing = self._committed[order_id]
        rel = min(existing, amount) if amount is not None else existing
        self._total_committed -= rel
        self._committed[order_id] -= rel
        if self._committed[order_id] <= 1e-10:
            del self._committed[order_id]

    # ------------------------------------------------------------------
    # Settlement (unsettled → settled)
    # ------------------------------------------------------------------

    def add_unsettled(
        self,
        amount: float,
        asset_class: AssetClass = "equity",
        trade_date: datetime | None = None,
        description: str = "",
    ) -> None:
        """Record an unsettled cash entry from a trade.

        A buy (negative amount) means cash owed — it reduces available
        cash when it settles. A sell (positive amount) adds cash when it settles.
        """
        td = trade_date or datetime.utcnow()
        lag = SETTLEMENT_LAG.get(asset_class, 2)
        settle_date = td + timedelta(days=lag)
        self._unsettled.append(SettlementEntry(
            amount=amount,
            asset_class=asset_class,
            trade_date=td,
            settle_date=settle_date,
            description=description,
        ))

    def settle_pending(self, reference_time: datetime | None = None) -> list[SettlementEntry]:
        """Move all due unsettled entries into settled cash.

        Args:
            reference_time: Current time (defaults to utcnow).

        Returns:
            List of entries that were settled this call.
        """
        now = reference_time or datetime.utcnow()
        due = [e for e in self._unsettled if e.settle_date <= now]
        self._unsettled = [e for e in self._unsettled if e.settle_date > now]
        for entry in due:
            self._settled += entry.amount
        return due

    # ------------------------------------------------------------------
    # Direct cash operations (for non-trade cash movements)
    # ------------------------------------------------------------------

    def add_cash(self, amount: float) -> None:
        """Add cash directly to settled balance (e.g., deposit, dividend)."""
        self._settled += amount

    def withdraw_cash(self, amount: float) -> None:
        """Withdraw cash from settled balance.

        Raises CashError if insufficient settled cash.
        """
        avail = self._settled - self._total_committed
        if amount > avail + 1e-10:
            raise CashError(
                f"Insufficient settled cash for withdrawal: "
                f"need ${amount:.2f}, available ${avail:.2f}"
            )
        self._settled -= amount

    # ------------------------------------------------------------------
    # Settlement-lag-aware order sizing helper
    # ------------------------------------------------------------------

    def max_buying_power(self) -> float:
        """Maximum buying power based on settled, uncommitted cash.

        This is the strict Reg T interpretation: you cannot trade
        with unsettled proceeds (no free-riding).
        """
        return max(0.0, self._settled - self._total_committed)

    # ------------------------------------------------------------------
    # Snapshot / state capture
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Capture full cash state for persistence."""
        return {
            "settled": self._settled,
            "unsettled": [
                {
                    "amount": e.amount,
                    "asset_class": e.asset_class,
                    "trade_date": e.trade_date.isoformat(),
                    "settle_date": e.settle_date.isoformat(),
                    "description": e.description,
                }
                for e in self._unsettled
            ],
            "committed": dict(self._committed),
            "total_committed": self._total_committed,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> CashLedger:
        """Restore a CashLedger from a snapshot dict."""
        ledger = cls(initial_cash=data.get("settled", 0.0))
        for entry_data in data.get("unsettled", []):
            ledger._unsettled.append(SettlementEntry(
                amount=entry_data["amount"],
                asset_class=entry_data["asset_class"],
                trade_date=datetime.fromisoformat(entry_data["trade_date"]),
                settle_date=datetime.fromisoformat(entry_data["settle_date"]),
                description=entry_data.get("description", ""),
            ))
        ledger._committed = dict(data.get("committed", {}))
        ledger._total_committed = data.get("total_committed", 0.0)
        return ledger
