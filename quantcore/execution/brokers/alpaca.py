"""Alpaca Markets broker adapter — REST API + WebSocket streaming."""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from quantcore.core.types import Fill, Order, OrderStatus, OrderType, Position, Side
from quantcore.execution.brokers.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerOrder,
)

log = logging.getLogger(__name__)


class AlpacaBroker(BrokerAdapter):
    """Alpaca Trading API broker adapter.

    Uses the alpaca-py SDK for order management and WebSocket streaming.
    Supports paper and live trading.
    """

    broker_name = "alpaca"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        self._connected = False
        self._client = None
        self._stream = None

        self._base_url = (
            "https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets"
        )
        self._stream_url = (
            "wss://paper-api.alpaca.markets/stream" if paper
            else "wss://stream.data.alpaca.markets/v2/iex"
        )

        self._fill_callbacks: list[Callable[[Fill], None]] = []
        self._trade_sub: dict[str, Any] = {}

    async def connect(self) -> None:
        """Initialize Alpaca client and connect WebSocket."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.stream import TradingStream
        except ImportError as e:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            ) from e

        self._client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        self._stream = TradingStream(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        # Subscribe to trade updates
        self._stream.subscribe_trade_updates(self._on_trade_update)

        await self._stream.connect()
        self._connected = True
        log.info(f"Connected to Alpaca {'paper' if self.paper else 'live'} API")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        if self._stream:
            await self._stream.disconnect()
            self._stream = None
        self._connected = False
        self._client = None
        log.info("Disconnected from Alpaca")

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order to Alpaca.

        Returns the Alpaca order ID.
        """
        if not self._client:
            await self.connect()

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest

        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL

        if order.order_type == "market":
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        elif order.order_type == "limit":
            req = LimitOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                limit_price=float(order.limit_price or 0),
                time_in_force=TimeInForce.DAY,
            )
        elif order.order_type in ("stop", "stop_loss"):
            req = StopOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                stop_price=float(order.stop_price or 0),
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
            )

        try:
            alpaca_order = self._client.submit_order(req)
            return str(alpaca_order.id)
        except Exception as e:
            log.error(f"Alpaca order submission failed: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in Alpaca."""
        if not self._client:
            return False

        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            log.warning(f"Failed to cancel Alpaca order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Order:
        """Get order details from Alpaca."""
        if not self._client:
            raise RuntimeError("Alpaca client not connected")

        try:
            ao = self._client.get_order_by_id(order_id)
            return _alpaca_order_to_internal(ao)
        except Exception as e:
            log.error(f"Failed to get Alpaca order {order_id}: {e}")
            raise

    async def get_positions(self) -> list[Position]:
        """Get all open positions from Alpaca."""
        if not self._client:
            return []

        try:
            positions = self._client.get_all_positions()
            return [
                Position(
                    symbol=p.symbol,
                    quantity=Decimal(str(p.qty)),
                    avg_cost=Decimal(str(p.avg_entry_price)),
                    market_price=Decimal(str(p.current_price or 0)),
                    unrealized_pnl=Decimal(str(p.unrealized_pl or 0)),
                )
                for p in positions
            ]
        except Exception as e:
            log.error(f"Failed to get Alpaca positions: {e}")
            return []

    async def get_account(self) -> AccountInfo:
        """Get account info from Alpaca."""
        if not self._client:
            raise RuntimeError("Alpaca client not connected")

        try:
            acct = self._client.get_account()
            return AccountInfo(
                account_id=str(acct.id),
                buying_power=float(acct.buying_power or 0),
                cash=float(acct.cash or 0),
                portfolio_value=float(acct.portfolio_value or 0),
                equity=float(acct.equity or 0),
                margin_used=float(acct.portfolio_value or 0) - float(acct.cash or 0),
                margin_multiplier=float(acct.multiplier or 2),
            )
        except Exception as e:
            log.error(f"Failed to get Alpaca account: {e}")
            raise

    async def get_orders(self, status: str = "open") -> list[Order]:
        """Get all orders from Alpaca."""
        if not self._client:
            return []

        try:
            from alpaca.trading.enums import QueryOrderStatus

            status_map = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            orders = self._client.get_orders(
                status=status_map.get(status, QueryOrderStatus.OPEN)
            )
            return [_alpaca_order_to_internal(o) for o in orders]
        except Exception as e:
            log.error(f"Failed to get Alpaca orders: {e}")
            return []

    def _on_trade_update(self, update) -> None:
        """Handle trade update from Alpaca WebSocket."""
        try:
            fill = Fill(
                order_id=str(update.order.get("id", "")),
                price=Decimal(str(update.price or 0)),
                quantity=Decimal(str(update.qty or 0)),
                commission=Decimal("0"),
                exchange="alpaca",
            )
            for cb in self._fill_callbacks:
                cb(fill)
        except Exception as e:
            log.error(f"Error handling Alpaca trade update: {e}")

    def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Register callback for fill updates."""
        self._fill_callbacks.append(callback)


def _alpaca_order_to_internal(ao: Any) -> Order:
    """Convert Alpaca order object to internal Order model."""
    from alpaca.trading.enums import OrderSide as AlpacaSide

    return Order(
        id=str(ao.id),
        symbol=str(ao.symbol),
        side=Side.BUY if ao.side == AlpacaSide.BUY else Side.SELL,
        type=OrderType(ao.type.lower() if hasattr(ao, 'type') else 'market'),
        quantity=Decimal(str(ao.qty or 0)),
        limit_price=Decimal(str(ao.limit_price)) if ao.limit_price else None,
        stop_price=Decimal(str(ao.stop_price)) if ao.stop_price else None,
        status=OrderStatus(ao.status.lower() if ao.status else 'pending'),
        filled_qty=Decimal(str(ao.filled_qty or 0)),
        avg_fill_price=Decimal(str(ao.filled_avg_price)) if ao.filled_avg_price else None,
    )
