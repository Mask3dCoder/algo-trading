"""Paper trading broker: in-memory simulation with realistic fills."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from quantcore.core.types import Fill, Order, OrderStatus, OrderType, Position, Side
from quantcore.execution.brokers.base import AccountInfo, BrokerAdapter, BrokerOrder


@dataclass
class PaperConfig:
    """Paper broker configuration."""
    initial_cash: float = 1_000_000.0
    latency_ms: float = 50.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    margin_rate: float = 0.5  # 50% margin requirement
    short_rate: float = 0.02  # Annual borrow cost for shorts
    slippage_model: str = "bps_1"  # 1 bps slippage
    fill_probability: float = 0.99  # Probability of fill under normal conditions


class PaperBroker(BrokerAdapter):
    """In-memory simulated broker for testing and paper trading.

    Uses backtest fill simulation models for realistic behavior.
    """

    broker_name = "paper"

    def __init__(self, config: PaperConfig | None = None) -> None:
        self.config = config or PaperConfig()
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._positions: dict[str, Position] = {}
        self._cash = Decimal(str(self.config.initial_cash))
        self._connected = False
        self._fill_callbacks: list[Callable[[Fill], None]] = []

        # Commission tracking
        self._total_commission = Decimal("0")

    # ------------------------------------------------------------------
    # Connection (simulated)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def submit_order(self, order: BrokerOrder) -> str:
        if not self._connected:
            await self.connect()

        broker_id = f"paper-{uuid.uuid4().hex[:12]}"

        # Convert to internal Order
        internal = Order(
            id=broker_id,
            symbol=order.symbol,
            side=Side(order.side),
            type=OrderType(order.order_type),
            quantity=order.quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            status=OrderStatus.SUBMITTED,
        )

        self._orders[broker_id] = internal

        # Simulate execution latency
        await asyncio.sleep(self.config.latency_ms / 1000.0)

        # Fill simulation
        fill_price = Decimal(str(self._get_fill_price(order)))
        fill_qty = order.quantity

        # Compute commission
        commission = Decimal(str(max(
            self.config.commission_per_share * float(fill_qty),
            self.config.min_commission,
        )))

        fill = Fill(
            order_id=broker_id,
            price=fill_price,
            quantity=fill_qty,
            commission=commission,
            exchange="PAPER",
        )

        self._fills.append(fill)
        self._total_commission += commission

        # Update position
        side_mult = Decimal("1") if order.side == "buy" else Decimal("-1")
        if order.symbol in self._positions:
            pos = self._positions[order.symbol]
            total_qty = pos.quantity + side_mult * fill_qty
            if total_qty == 0:
                # Liquidated
                pos.quantity = Decimal("0")
                pos.avg_cost = Decimal("0")
            else:
                total_cost = pos.avg_cost * pos.quantity + fill_price * fill_qty
                pos.quantity = total_qty
                pos.avg_cost = total_cost / pos.quantity if pos.quantity != 0 else Decimal("0")
        else:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=side_mult * fill_qty,
                avg_cost=fill_price,
                market_price=fill_price,
            )

        # Update cash
        trade_value = fill_price * fill_qty
        self._cash -= side_mult * trade_value + commission

        # Update order status
        internal.status = OrderStatus.FILLED
        internal.filled_qty = fill_qty
        internal.avg_fill_price = fill_price

        # Notify callbacks
        for cb in self._fill_callbacks:
            cb(fill)

        return broker_id

    async def cancel_order(self, order_id: str) -> bool:
        if order_id not in self._orders:
            return False
        order = self._orders[order_id]
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        order.status = OrderStatus.CANCELLED
        return True

    async def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        return self._orders[order_id]

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_account(self) -> AccountInfo:
        total_pv = float(self._cash)
        for pos in self._positions.values():
            total_pv += float(pos.market_value)

        return AccountInfo(
            account_id="paper",
            buying_power=float(self._cash) * self.config.margin_rate * 2,
            cash=float(self._cash),
            portfolio_value=total_pv,
            equity=total_pv,
            margin_used=total_pv - float(self._cash),
        )

    async def get_orders(self, status: str = "open") -> list[Order]:
        if status == "open":
            return [o for o in self._orders.values()
                    if o.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)]
        elif status == "filled":
            return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        return list(self._orders.values())

    def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        self._fill_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fill_price(self, order: BrokerOrder) -> float:
        """Simulate fill price with simple slippage."""
        base_price = 100.0  # Default reference price
        if order.limit_price is not None:
            base_price = float(order.limit_price)

        # Simple 1 bps slippage
        slippage = 0.0001  # 1 bps
        if order.side == "buy":
            return base_price * (1.0 + slippage)
        else:
            return base_price * (1.0 - slippage)
