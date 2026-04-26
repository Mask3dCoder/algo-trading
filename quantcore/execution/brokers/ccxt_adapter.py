"""CCXT crypto exchange adapter — unified interface for 100+ exchanges."""

from __future__ import annotations

import asyncio
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


# Supported exchanges
SUPPORTED_EXCHANGES = ["binance", "coinbase", "kraken", "bybit", "okx"]


class CCXTBroker(BrokerAdapter):
    """CCXT crypto broker adapter.

    Supports spot and futures trading on major exchanges.
    """

    broker_name = "ccxt"

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        secret: str = "",
        password: str = "",
        sandbox: bool = True,
    ) -> None:
        if exchange_id.lower() not in SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Unsupported exchange: {exchange_id}. "
                f"Supported: {SUPPORTED_EXCHANGES}"
            )

        self.exchange_id = exchange_id.lower()
        self.api_key = api_key
        self.secret = secret
        self.password = password
        self.sandbox = sandbox

        self._connected = False
        self._exchange = None
        self._fill_callbacks: list[Callable[[Fill], None]] = []
        self._ws_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Initialize exchange and connect."""
        try:
            import ccxt.async_support as ccxt_async
        except ImportError:
            raise ImportError("ccxt not installed. Run: pip install ccxt")

        exchange_class = getattr(ccxt_async, self.exchange_id.lower())

        config: dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
        }
        if self.password:
            config["password"] = self.password

        # Sandbox/testnet configuration
        if self.sandbox:
            if self.exchange_id == "binance":
                config["urls"] = {"api": "https://testnet.binance.vision/api"}
            elif self.exchange_id == "coinbase":
                config["urls"] = {"api": "https://api-public.sandbox.pro.coinbase.com"}
            elif self.exchange_id == "kraken":
                config["urls"] = {"api": "https://api.demo.kraken.com"}

        self._exchange = exchange_class(config)
        await self._exchange.load_markets()
        self._connected = True
        log.info(f"Connected to {self.exchange_id} {'sandbox' if self.sandbox else 'live'}")

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._exchange:
            await self._exchange.close()
        self._connected = False
        self._exchange = None
        log.info(f"Disconnected from {self.exchange_id}")

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit order to the exchange.

        Returns exchange order ID.
        """
        if not self._exchange:
            await self.connect()

        symbol = self._normalize_symbol(order.symbol)
        order_type = order.order_type.lower()

        try:
            if order_type == "market":
                result = await self._exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order.side.lower(),
                    amount=float(order.quantity),
                )
            elif order_type == "limit":
                result = await self._exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=order.side.lower(),
                    amount=float(order.quantity),
                    price=float(order.limit_price or 0),
                )
            else:
                result = await self._exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=order.side.lower(),
                    amount=float(order.quantity),
                    price=float(order.stop_price or order.limit_price or 0),
                )

            return str(result.get("id", ""))

        except Exception as e:
            log.error(f"Order submission failed on {self.exchange_id}: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._exchange:
            return False

        try:
            await self._exchange.cancel_order(order_id)
            return True
        except Exception as e:
            log.warning(f"Cancel order failed on {self.exchange_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Order:
        """Get order details from exchange."""
        if not self._exchange:
            raise RuntimeError(f"{self.exchange_id} not connected")

        try:
            result = await self._exchange.fetch_order(order_id)
            return _ccxt_order_to_internal(result)
        except Exception as e:
            log.error(f"Failed to fetch order {order_id}: {e}")
            raise

    async def get_positions(self) -> list[Position]:
        """Get open positions."""
        if not self._exchange:
            return []

        try:
            if self._exchange.has["fetchPositions"]:
                result = await self._exchange.fetch_positions()
                return [
                    Position(
                        symbol=p.get("symbol", ""),
                        quantity=Decimal(str(p.get("contracts", 0))),
                        avg_cost=Decimal(str(p.get("entryPrice", 0))),
                        market_price=Decimal(str(p.get("markPrice", 0))),
                        unrealized_pnl=Decimal(str(p.get("unrealizedPnl", 0))),
                    )
                    for p in result
                ]
            else:
                # Spot: positions via balance
                balance = await self._exchange.fetch_balance()
                positions = []
                for currency, data in balance.get("free", {}).items():
                    if data > 0:
                        positions.append(Position(
                            symbol=currency,
                            quantity=Decimal(str(data)),
                            avg_cost=Decimal("0"),
                        ))
                return positions
        except Exception as e:
            log.error(f"Failed to fetch positions: {e}")
            return []

    async def get_account(self) -> AccountInfo:
        """Get account info."""
        if not self._exchange:
            raise RuntimeError(f"{self.exchange_id} not connected")

        try:
            balance = await self._exchange.fetch_balance()
            return AccountInfo(
                account_id=f"{self.exchange_id}-account",
                cash=float(balance.get("free", {}).get("USDT", 0)),
                portfolio_value=float(balance.get("total", {}).get("USDT", 0)),
                equity=float(balance.get("total", {}).get("USDT", 0)),
            )
        except Exception as e:
            log.error(f"Failed to fetch account: {e}")
            raise

    async def get_orders(self, status: str = "open") -> list[Order]:
        """Get orders."""
        if not self._exchange:
            return []

        try:
            result = await self._exchange.fetch_orders()
            return [_ccxt_order_to_internal(o) for o in result]
        except Exception as e:
            log.error(f"Failed to fetch orders: {e}")
            return []

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to exchange format (e.g., BTC/USDT)."""
        # Already in CCXT format
        if "/" in symbol:
            return symbol
        # Add USDT pair if just base
        return f"{symbol.upper()}/USDT"


def _ccxt_order_to_internal(o: dict) -> Order:
    """Convert CCXT order dict to internal Order model."""
    status_map = {
        "open": OrderStatus.SUBMITTED,
        "closed": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
    }

    return Order(
        id=str(o.get("id", "")),
        symbol=str(o.get("symbol", "")),
        side=Side.BUY if o.get("side") == "buy" else Side.SELL,
        type=OrderType(o.get("type", "market")),
        quantity=Decimal(str(o.get("amount", 0))),
        limit_price=Decimal(str(o.get("price", 0))) if o.get("price") else None,
        status=status_map.get(o.get("status", ""), OrderStatus.PENDING),
        filled_qty=Decimal(str(o.get("filled", 0))),
        avg_fill_price=Decimal(str(o.get("average", 0))) if o.get("average") else None,
    )
