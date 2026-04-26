"""Interactive Brokers adapter via TWS/Gateway API (ibapi)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from decimal import Decimal

from quantcore.core.types import Fill, Order, Position
from quantcore.execution.brokers.base import (
    AccountInfo,
    BrokerAdapter,
    BrokerOrder,
)

log = logging.getLogger(__name__)


class IBKRBroker(BrokerAdapter):
    """Interactive Brokers adapter using IB's TWS API.

    Requires: TWS or IB Gateway running with API enabled.
    Default port: 7497 (TWS paper), 7496 (TWS live), 4002 (Gateway paper).

    Connection is synchronous (ibapi uses a client thread model).
    """

    broker_name = "ibkr"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # TWS paper trading default
        client_id: int = 1,
        account_id: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_id = account_id

        self._connected = False
        self._client = None
        self._wrapper = None
        self._thread: threading.Thread | None = None

        # State tracking
        self._next_order_id: int = 0
        self._orders: dict[int, Order] = {}
        self._positions: list[Position] = []
        self._account_info: AccountInfo | None = None
        self._fill_callbacks: list[Callable[[Fill], None]] = []
        self._order_id_map: dict[str, int] = {}  # internal → IB order ID

        self._account_ready = threading.Event()

    async def connect(self) -> None:
        """Connect to TWS/Gateway."""
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper as IBWrapper
        except ImportError:
            raise ImportError(
                "ibapi not installed. Download from Interactive Brokers: "
                "https://interactivebrokers.github.io/#"
            )

        # Use a custom wrapper class
        class IBWrapperImpl(IBWrapper):
            def __init__(self, broker: IBKRBroker):
                super().__init__()
                self.broker = broker

            def nextValidId(self, orderId: int):
                self.broker._next_order_id = orderId

            def openOrder(self, orderId, contract, order, orderState):
                pass

            def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, *args):
                pass

            def execDetails(self, reqId, contract, execution):
                fill = Fill(
                    order_id=str(execution.orderId),
                    price=Decimal(str(execution.price)),
                    quantity=Decimal(str(execution.shares)),
                    commission=Decimal(str(execution.commission)) if execution.commission else Decimal("0"),
                    exchange=execution.exchange or "IB",
                )
                for cb in self.broker._fill_callbacks:
                    cb(fill)

            def position(self, account, contract, position, avgCost):
                pos = Position(
                    symbol=contract.symbol,
                    quantity=Decimal(str(position)),
                    avg_cost=Decimal(str(avgCost)) if avgCost > 0 else Decimal("0"),
                    market_price=Decimal("0"),
                )
                self.broker._positions.append(pos)

            def updateAccountValue(self, key, val, currency, accountName):
                if accountName == self.broker.account_id:
                    if self.broker._account_info is None:
                        self.broker._account_info = AccountInfo(account_id=accountName)
                    info = self.broker._account_info
                    val_f = float(val)
                    if key == "NetLiquidation":
                        info.portfolio_value = val_f
                    elif key == "CashBalance":
                        info.cash = val_f
                    elif key == "BuyingPower":
                        info.buying_power = val_f
                    elif key == "GrossPositionValue":
                        info.equity = val_f

            def accountDownloadEnd(self, accountName):
                if accountName == self.broker.account_id:
                    self.broker._account_ready.set()

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                log.warning(f"IB Error {errorCode}: {errorString}")

        self._wrapper = IBWrapperImpl(self)
        self._client = EClient(self._wrapper)
        self._client.connect(self.host, self.port, self.client_id)

        self._thread = threading.Thread(target=self._client.run, daemon=True)
        self._thread.start()

        # Wait for next valid order ID
        time.sleep(0.5)
        self._connected = True

        # Request account updates
        self._client.reqAccountUpdates(True, self.account_id)
        self._account_ready.wait(timeout=5)

        log.info(f"Connected to IB {self.host}:{self.port}")

    async def disconnect(self) -> None:
        """Disconnect from TWS."""
        if self._client:
            self._client.reqAccountUpdates(False, self.account_id)
            self._client.disconnect()
        self._connected = False
        self._client = None
        log.info("Disconnected from IB")

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit order to IB."""
        if not self._client or not self._next_order_id:
            await self.connect()

        try:
            from ibapi.contract import Contract
            from ibapi.order import Order as IBOrder
        except ImportError:
            raise ImportError("ibapi not installed")

        # Build contract
        contract = Contract()
        contract.symbol = order.symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        # Build order
        ib_order = IBOrder()
        ib_order.action = order.side.upper()
        ib_order.totalQuantity = float(order.quantity)
        ib_order.orderType = order.order_type.upper()
        if order.limit_price is not None:
            ib_order.lmtPrice = float(order.limit_price)
        if order.stop_price is not None:
            ib_order.auxPrice = float(order.stop_price)
        ib_order.tif = order.time_in_force.upper()

        oid = self._next_order_id
        self._next_order_id += 1

        self._client.placeOrder(oid, contract, ib_order)

        broker_id = f"ib-{oid}"
        self._order_id_map[broker_id] = oid

        return broker_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an IB order."""
        if not self._client:
            return False

        oid = self._order_id_map.get(order_id)
        if oid is None:
            return False

        self._client.cancelOrder(oid)
        return True

    async def get_order(self, order_id: str) -> Order:
        """Get order status from IB."""
        # IB doesn't have a simple get_order; tracked internally
        if order_id in self._orders:
            return self._orders[order_id]
        raise KeyError(f"Order {order_id} not found")

    async def get_positions(self) -> list[Position]:
        """Get positions from IB (cached from streaming)."""
        if self._client:
            self._client.reqPositions()
            time.sleep(0.5)
        return self._positions

    async def get_account(self) -> AccountInfo:
        """Get account info."""
        if self._account_info:
            return self._account_info
        raise RuntimeError("Account info not available")

    async def get_orders(self, status: str = "open") -> list[Order]:
        """Get all open orders."""
        if self._client:
            self._client.reqOpenOrders()
            time.sleep(0.5)
        return list(self._orders.values())

    def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Register fill callback."""
        self._fill_callbacks.append(callback)
