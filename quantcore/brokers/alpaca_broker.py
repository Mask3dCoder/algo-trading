"""Alpaca broker adapter — implements BrokerAdapter with alpaca-py SDK.

Used by paper_trade.py and the live runner for order submission, position
management, and account queries through the Alpaca Trading API.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantcore.core.types import Fill, Order, OrderStatus, OrderType, Position, Side
from quantcore.execution.brokers.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerOrder,
)

log = logging.getLogger(__name__)

# Alpaca → Internal status mapping
_ALPACA_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.PENDING,
    "held": OrderStatus.PENDING,
    "accepted": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "stopped": OrderStatus.PENDING,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "calculated": OrderStatus.FILLED,
}


class AlpacaBrokerError(Exception):
    """Base class for Alpaca broker errors."""
    pass


class BrokerAuthError(AlpacaBrokerError):
    """Authentication or permission error (HTTP 403)."""
    pass


class OrderRejectedError(AlpacaBrokerError):
    """Order was rejected by Alpaca (HTTP 422)."""
    pass


class AlpacaBroker(BrokerAdapter):
    """Alpaca Trading API broker adapter.

    Uses the alpaca-py SDK (TradingClient) for order operations.
    Supports paper and live trading driven by ALPACA_BASE_URL.
    """

    broker_name = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        mode: str = "paper",
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET")
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not self.api_key or not self.secret_key:
            raise BrokerAuthError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set as environment "
                "variables or passed to AlpacaBroker constructor"
            )

        # Mode parameter takes priority; URL is secondary signal
        self._paper = mode == "paper" or (mode != "live" and "paper-api" in base_url)
        self._base_url = base_url

        self._connected = False
        self._client: Any = None

        if self._paper:
            log.info("Alpaca broker initialized in PAPER mode — no real money at risk")
        else:
            log.warning("WARNING: Alpaca broker initialized in LIVE mode — real money at risk")

    @property
    def is_paper(self) -> bool:
        return self._paper

    # ---- BrokerAdapter interface ---------------------------------------------

    async def connect(self) -> None:
        """Initialize the Alpaca TradingClient."""
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            ) from e

        self._client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self._paper,
        )
        self._connected = True
        log.info("Alpaca broker connected (%s)", "paper" if self._paper else "live")

    async def disconnect(self) -> None:
        """Disconnect — no-op for REST client but clears state."""
        self._connected = False
        self._client = None
        log.info("Alpaca broker disconnected")

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order to Alpaca and return the broker order ID."""
        if not self._client:
            await self.connect()

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest

        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL

        try:
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
            elif order.order_type in ("stop", "stop_limit"):
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

            alpaca_order = self._client.submit_order(req)
            return str(alpaca_order.id)

        except Exception as e:
            status_code = getattr(e, 'status_code', 0) if hasattr(e, 'status_code') else getattr(getattr(e, 'response', None), 'status_code', 0)
            msg = str(e)

            if status_code == 403:
                raise BrokerAuthError(
                    f"Alpaca API returned 403 Forbidden. Check API key permissions. "
                    f"Details: {msg}"
                ) from e
            elif status_code == 429:
                # Exponential backoff retry
                for attempt in range(1, 4):
                    wait = 2 ** attempt
                    log.warning("Alpaca rate limited (429). Retrying in %ds (attempt %d/3)", wait, attempt)
                    time.sleep(wait)
                    try:
                        alpaca_order = self._client.submit_order(req)
                        return str(alpaca_order.id)
                    except Exception:
                        continue
                raise AlpacaBrokerError(
                    f"Alpaca rate limit (429) — retried 3 times, still failing"
                ) from e
            elif status_code == 422:
                log.error("Alpaca rejected order (422): %s", msg)
                raise OrderRejectedError(
                    f"Alpaca rejected order {order.symbol}: {msg}"
                ) from e
            else:
                log.error("Alpaca order submission failed: %s", msg)
                raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by broker ID. Returns True on success."""
        if not self._client:
            return False

        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            log.warning("Failed to cancel Alpaca order %s: %s", order_id, e)
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        if not self._client:
            return 0

        try:
            result = self._client.cancel_all_orders()
            count = len(result) if isinstance(result, list) else 1
            log.info("Cancelled %d open orders", count)
            return count
        except Exception as e:
            log.warning("Failed to cancel all orders: %s", e)
            return 0

    async def get_order(self, order_id: str) -> Order:
        """Get order details from Alpaca."""
        if not self._client:
            raise RuntimeError("Alpaca client not connected")

        ao = self._client.get_order_by_id(order_id)
        return _alpaca_order_to_internal(ao)

    async def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        return await self.get_orders(status="open")

    async def get_all_orders(self) -> list[Order]:
        """Get all orders (open + closed)."""
        return await self.get_orders(status="all")

    async def get_orders(self, status: str = "open") -> list[Order]:
        """Get orders from Alpaca filtered by status."""
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
            log.error("Failed to get Alpaca orders: %s", e)
            return []

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a single symbol."""
        if not self._client:
            return None

        try:
            p = self._client.get_open_position(symbol)
            return Position(
                symbol=symbol,
                quantity=Decimal(str(p.qty)),
                avg_cost=Decimal(str(p.avg_entry_price)),
                market_price=Decimal(str(p.current_price or 0)),
                unrealized_pnl=Decimal(str(p.unrealized_pl or 0)),
            )
        except Exception:
            return None

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
            log.error("Failed to get Alpaca positions: %s", e)
            return []

    async def close_position(self, symbol: str) -> bool:
        """Close the entire position for a symbol at market.

        Returns True on success.
        """
        if not self._client:
            return False

        try:
            self._client.close_position(symbol)
            log.info("Closed position: %s", symbol)
            return True
        except Exception as e:
            log.warning("Failed to close position %s: %s", symbol, e)
            return False

    async def close_all_positions(self) -> int:
        """Close all positions at market. Returns count of positions closed."""
        if not self._client:
            return 0

        try:
            self._client.close_all_positions()
            log.info("Closed all positions (kill switch)")
            return 1
        except Exception as e:
            # Fallback: close one-by-one
            log.warning("Bulk close failed, closing individually: %s", e)
            positions = await self.get_positions()
            count = 0
            for p in positions:
                try:
                    self._client.close_position(p.symbol)
                    count += 1
                except Exception:
                    pass
            return count

    async def get_account(self) -> AccountInfo:
        """Get account info from Alpaca."""
        if not self._client:
            raise RuntimeError("Alpaca client not connected")

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

    # ---- Non-async helper methods for paper_trade.py compatibility ------------

    def get_account_sync(self) -> dict[str, float]:
        """Synchronous account info returned as dict (paper_trade.py format).

        Returns a dict with cash, equity, portfolio_value, total_commission keys.
        """
        import asyncio

        async def _get() -> AccountInfo:
            return await self.get_account()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            info = _run_async_in_new_loop(_get())
        else:
            info = asyncio.run(_get())

        return {
            "cash": info.cash,
            "equity": info.equity,
            "portfolio_value": info.portfolio_value,
            "total_commission": 0.0,
        }

    def get_positions_sync(self) -> dict[str, Position]:
        """Synchronous positions returned as dict keyed by symbol."""
        import asyncio

        async def _get() -> list[Position]:
            return await self.get_positions()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            pos_list = _run_async_in_new_loop(_get())
        else:
            pos_list = asyncio.run(_get())

        return {p.symbol: p for p in pos_list}


def _run_async_in_new_loop(coro) -> Any:
    """Run a coroutine in a new event loop (for sync wrapper use)."""
    import asyncio
    import concurrent.futures

    def _runner() -> Any:
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner)
        return future.result(timeout=30)


def _alpaca_order_to_internal(ao: Any) -> Order:
    """Convert Alpaca order object to internal Order model.

    Maps every Alpaca order status to the corresponding internal OrderStatus.
    """
    from alpaca.trading.enums import OrderSide as AlpacaSide

    raw_status = str(ao.status).lower() if getattr(ao, 'status', None) else "new"
    internal_status = _ALPACA_STATUS_MAP.get(raw_status, OrderStatus.PENDING)

    # Handle quantity: Alpaca stores qty as string
    qty = Decimal(str(getattr(ao, 'qty', 0) or 0))
    filled_qty = Decimal(str(getattr(ao, 'filled_qty', 0) or 0))

    return Order(
        id=str(ao.id),
        symbol=str(ao.symbol),
        side=Side.BUY if ao.side == AlpacaSide.BUY else Side.SELL,
        type=OrderType(getattr(ao, 'type', 'market').lower()),
        quantity=qty,
        limit_price=Decimal(str(ao.limit_price)) if getattr(ao, 'limit_price', None) else None,
        stop_price=Decimal(str(ao.stop_price)) if getattr(ao, 'stop_price', None) else None,
        status=internal_status,
        filled_qty=filled_qty,
        avg_fill_price=Decimal(str(ao.filled_avg_price)) if getattr(ao, 'filled_avg_price', None) else None,
    )
