"""Abstract broker adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from quantcore.core.types import Fill, Order, Position


@dataclass
class AccountInfo:
    """Standardized account information."""
    account_id: str = ""
    buying_power: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    margin_multiplier: float = 2.0  # Reg T = 2:1
    currency: str = "USD"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BrokerOrder:
    """Standardized order representation for broker adapters."""
    symbol: str
    side: str  # buy/sell
    order_type: str  # market/limit/stop/stop_limit
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "day"
    client_order_id: str = ""
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    market_price: Decimal | None = None  # Current market price for fill simulation


class BrokerAdapter(ABC):
    """Abstract broker adapter.

    All broker implementations must implement these methods.
    """

    broker_name: str = "abstract"

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    @abstractmethod
    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order. Returns broker order ID."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if successful."""
        ...

    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get order status by ID."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        ...

    @abstractmethod
    async def get_orders(self, status: str = "open") -> list[Order]:
        """Get orders by status."""
        ...

    def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Stream fills via WebSocket. Override in implementations."""
        raise NotImplementedError(f"{self.broker_name} does not support fill streaming")

    def stream_positions(self, callback: Callable[[list[Position]], None]) -> None:
        """Stream position updates via WebSocket. Override in implementations."""
        raise NotImplementedError(f"{self.broker_name} does not support position streaming")
