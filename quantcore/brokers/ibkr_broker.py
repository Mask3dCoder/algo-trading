"""Interactive Brokers broker adapter using ib_insync.

Prerequisites:
- TWS or IB Gateway must be running and logged in
- Paper trading account must be selected in TWS/Gateway
- API connections must be enabled in TWS: File > Global Configuration > API > Settings
  - Enable ActiveX and Socket Clients: checked
  - Socket port: 7497 (TWS paper) or 4002 (Gateway paper)
  - Allow connections from localhost only: checked (for security)
  - Read-Only API: unchecked (we need to submit orders)

Connection ports:
- TWS Paper Trading: 7497
- TWS Live Trading:  7496
- IB Gateway Paper:  4002
- IB Gateway Live:   4001
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal
from typing import Any

from quantcore.core.types import Fill, Order, OrderStatus, OrderType, Position, Side
from quantcore.execution.brokers.base import AccountInfo, BrokerAdapter, BrokerOrder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBKR → Internal status mapping
# ---------------------------------------------------------------------------

IBKR_STATUS_MAP: dict[str, str] = {
    "PendingSubmit": "pending",
    "PendingCancel": "pending_cancel",
    "PreSubmitted": "pending",
    "Submitted": "open",
    "ApiPending": "pending",
    "ApiCancelled": "cancelled",
    "Cancelled": "cancelled",
    "Filled": "filled",
    "Inactive": "cancelled",
}

_INTERNAL_TO_IB_STATUS: dict[str, str] = {v: k for k, v in IBKR_STATUS_MAP.items()}

PORT_MODES: dict[int, str] = {
    7497: "TWS Paper",
    7496: "TWS Live",
    4002: "Gateway Paper",
    4001: "Gateway Live",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ibkr_action(side: Side) -> str:
    return "BUY" if side == Side.BUY else "SELL"


def _internal_side(action: str) -> Side:
    return Side.BUY if action.upper() == "BUY" else Side.SELL


def _internal_order_type(ibkr_order: Any) -> OrderType:
    otype = getattr(ibkr_order, "orderType", "MKT") or "MKT"
    mapping = {
        "MKT": OrderType.MARKET,
        "LMT": OrderType.LIMIT,
        "STP": OrderType.STOP,
        "STP LMT": OrderType.STOP_LIMIT,
    }
    return mapping.get(otype.upper(), OrderType.MARKET)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IBKRBrokerError(Exception):
    """Base exception for IBKR broker errors."""


class IBKRConnectionError(IBKRBrokerError):
    """Cannot connect to TWS/Gateway."""


class IBKROrderRejectedError(IBKRBrokerError):
    """Order was rejected by IBKR."""


class IBKRMarketClosedError(IBKRBrokerError):
    """Market is closed — order cannot be submitted."""


# ---------------------------------------------------------------------------
# IBKR Broker Adapter
# ---------------------------------------------------------------------------


class IBKRBroker(BrokerAdapter):
    """Interactive Brokers adapter using ib_insync.

    Provides connectivity to TWS/IB Gateway for order execution, position
    tracking, and account queries. Uses ib_insync's event-driven model
    internally with a synchronous facade for the QuantCore OMS.

    Usage:
        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        await broker.connect()
        orders = await broker.get_orders()
        ...
        await broker.disconnect()
    """

    broker_name = "ibkr"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        account_id: str = "",
        timeout: float = 10.0,
        readonly: bool = False,
    ) -> None:
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("IBKR_PORT", "7497"))
        self.client_id = client_id or int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self.account_id = account_id
        self.timeout = timeout
        self.readonly = readonly

        self._ib: Any = None  # ib_insync.IB instance
        self._connected = False
        self._order_map: dict[str, Any] = {}  # internal_id → ib_insync Trade
        self._pending_orders: dict[int, str] = {}  # ib_id → internal_id

        # Validate port
        if self.port not in PORT_MODES:
            log.warning(
                "Unknown IBKR port %d — expected one of %s. "
                "Check TWS API settings.",
                self.port, list(PORT_MODES),
            )

        self._is_paper = self.port in (7497, 4002)
        mode = PORT_MODES.get(self.port, "Unknown")
        if self._is_paper:
            log.info("IBKR broker initialized in PAPER mode (%s) — no real money at risk", mode)
        else:
            log.warning("IBKR broker initialized in LIVE mode (%s) — REAL MONEY AT RISK", mode)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to TWS/Gateway and verify account access."""
        try:
            import ib_insync
        except ImportError:
            raise ImportError(
                "ib_insync not installed. Run: pip install ib_insync"
            )

        self._ib = ib_insync.IB()
        self._ib.RequestTimeout = self.timeout

        try:
            self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        except ConnectionRefusedError:
            raise IBKRConnectionError(
                f"Cannot connect to TWS/Gateway at {self.host}:{self.port}. "
                f"Is TWS running and logged in? "
                f"Is API enabled? (Edit > Global Configuration > API > Settings)"
            )
        except OSError as e:
            raise IBKRConnectionError(
                f"Connection to {self.host}:{self.port} failed: {e}. "
                f"Check that TWS/Gateway is running and API is enabled."
            )

        self._connected = True

        # Verify account access
        accounts = self._ib.managedAccounts()
        if not accounts:
            self._ib.disconnect()
            self._connected = False
            raise IBKRConnectionError("Connected but no managed accounts found")

        # Detect paper vs live account
        for acct in accounts:
            prefix = acct[:2] if len(acct) >= 2 else ""
            if prefix == "DU":
                if not self._is_paper:
                    log.warning(
                        "Paper account %s detected but port %d suggests LIVE mode. "
                        "Verify TWS trading mode matches.",
                        acct, self.port,
                    )
                else:
                    log.info("Paper trading account: %s", acct)
            elif prefix == "U":
                if self._is_paper:
                    log.warning(
                        "LIVE account %s detected on paper port %d! "
                        "Check TWS login mode — switch to Paper Trading if this is simulation.",
                        acct, self.port,
                    )
                else:
                    log.warning("LIVE trading account: %s — real money at risk", acct)
            else:
                log.info("Account: %s (type unknown from prefix)", acct)

        # Store the first account if none specified
        if not self.account_id and accounts:
            self.account_id = accounts[0]

        # Subscribe to order status updates
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details

        # Set up disconnect handler
        self._ib.disconnectedEvent += lambda: self._on_disconnected()

        log.info(
            "Connected to IBKR %s:%d (clientId=%d, account=%s, mode=%s)",
            self.host, self.port, self.client_id, self.account_id,
            "PAPER" if self._is_paper else "LIVE",
        )

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway cleanly."""
        if self._ib and self._ib.isConnected():
            # Cancel market data subscriptions
            try:
                self._ib.cancelMktData()
            except Exception:
                pass
            self._ib.disconnect()
        self._connected = False
        self._ib = None
        log.info("Disconnected from IBKR")

    def _on_disconnected(self) -> None:
        """Handle unexpected disconnection."""
        self._connected = False
        log.warning("IBKR connection lost unexpectedly")

    # ------------------------------------------------------------------
    # Order status / fill callbacks
    # ------------------------------------------------------------------

    _fill_callbacks: list = []

    def _on_order_status(self, trade: Any) -> None:
        """ib_insync orderStatusEvent handler."""
        ib_status = trade.orderStatus.status
        internal_status = IBKR_STATUS_MAP.get(ib_status, "pending")
        log.debug("Order %s status: %s → %s", trade.order.orderId, ib_status, internal_status)

    def _on_exec_details(self, trade: Any, fill_obj: Any) -> None:
        """ib_insync execDetailsEvent handler."""
        try:
            fill = Fill(
                order_id=str(trade.order.orderId),
                price=Decimal(str(fill_obj.execution.price)),
                quantity=Decimal(str(fill_obj.execution.shares)),
                commission=Decimal("0"),
                exchange=fill_obj.execution.exchange or "IB",
            )
            for cb in self._fill_callbacks:
                cb(fill)
        except Exception as e:
            log.error("Error processing IBKR execution: %s", e)

    def stream_fills(self, callback) -> None:
        """Register a callback for fill events."""
        self._fill_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order to IBKR. Returns broker order ID string."""
        if not self._connected or not self._ib or not self._ib.isConnected():
            await self.connect()

        import ib_insync

        # Build contract
        contract = ib_insync.Stock(order.symbol, "SMART", "USD")
        try:
            self._ib.qualifyContracts(contract)
        except Exception as e:
            raise IBKROrderRejectedError(
                f"Contract qualification failed for {order.symbol}: {e}"
            )

        # Build IBKR order
        action = order.side.upper()
        total_qty = float(order.quantity)

        if order.order_type == "market":
            ib_order = ib_insync.MarketOrder(action, total_qty)
        elif order.order_type == "limit":
            price = float(order.limit_price or 0)
            ib_order = ib_insync.LimitOrder(action, total_qty, price)
        elif order.order_type in ("stop", "stop_loss"):
            price = float(order.stop_price or 0)
            ib_order = ib_insync.StopOrder(action, total_qty, price)
        else:
            ib_order = ib_insync.MarketOrder(action, total_qty)

        ib_order.tif = order.time_in_force.upper() if order.time_in_force else "DAY"

        # Place the order
        trade = self._ib.placeOrder(contract, ib_order)
        ib_order_id = trade.order.orderId

        broker_id = f"ibkr-{ib_order_id}"
        self._order_map[broker_id] = trade
        self._pending_orders[ib_order_id] = broker_id

        # Wait for acknowledgement
        start = time.time()
        while time.time() - start < self.timeout:
            self._ib.sleep(0.1)
            status = trade.orderStatus.status
            if status in ("Submitted", "PreSubmitted", "Filled"):
                log.info(
                    "Order %s (%s %s %s @ %s): %s",
                    broker_id, order.symbol, action, total_qty,
                    order.order_type, status,
                )
                return broker_id
            if status in ("Cancelled", "ApiCancelled", "Inactive"):
                reason = getattr(trade.orderStatus, "whyHeld", "") or status
                raise IBKROrderRejectedError(
                    f"Order {broker_id} rejected: {reason}"
                )

        raise IBKRBrokerError(
            f"Order {broker_id} timed out waiting for acknowledgement after {self.timeout}s"
        )

    async def submit_bracket_order(self, order: BrokerOrder) -> str:
        """Submit a bracket order with optional stop-loss and take-profit children.

        Creates a parent MARKET order with attached STOP (stop-loss) and
        LIMIT (take-profit) child orders linked via OCA group. When one
        child fills, the other is automatically cancelled.

        Returns the parent broker order ID.
        """
        if not self._connected or not self._ib or not self._ib.isConnected():
            await self.connect()

        import ib_insync

        contract = ib_insync.Stock(order.symbol, "SMART", "USD")
        try:
            self._ib.qualifyContracts(contract)
        except Exception as e:
            raise IBKROrderRejectedError(
                f"Contract qualification failed for {order.symbol}: {e}"
            )

        action = order.side.upper()
        total_qty = float(order.quantity)
        opposite = "SELL" if action == "BUY" else "BUY"
        oca_group = f"bracket_{int(time.time() * 1e6)}"

        # Parent MARKET order
        parent = ib_insync.MarketOrder(action, total_qty)
        parent.tif = order.time_in_force.upper() if order.time_in_force else "DAY"
        parent.ocaGroup = oca_group
        parent.ocaType = 1  # Cancel with block

        # Place parent first
        parent_trade = self._ib.placeOrder(contract, parent)
        parent_id = parent_trade.order.orderId
        broker_id = f"ibkr-{parent_id}"
        self._order_map[broker_id] = parent_trade
        self._pending_orders[parent_id] = broker_id

        # Child orders
        children: list[ib_insync.Order] = []

        if order.take_profit_price and float(order.take_profit_price) > 0:
            tp = ib_insync.LimitOrder(
                opposite, total_qty, float(order.take_profit_price),
            )
            tp.tif = order.time_in_force.upper() if order.time_in_force else "DAY"
            tp.ocaGroup = oca_group
            tp.ocaType = 1
            tp.parentId = parent_id
            tp.transmit = True
            children.append(tp)

        if order.stop_loss_price and float(order.stop_loss_price) > 0:
            sl = ib_insync.StopOrder(
                opposite, total_qty, float(order.stop_loss_price),
            )
            sl.tif = order.time_in_force.upper() if order.time_in_force else "DAY"
            sl.ocaGroup = oca_group
            sl.ocaType = 1
            sl.parentId = parent_id
            sl.transmit = True
            children.append(sl)

        # Submit children
        for child in children:
            child_trade = self._ib.placeOrder(contract, child)
            child_id = child_trade.order.orderId
            child_broker_id = f"ibkr-{child_id}"
            self._order_map[child_broker_id] = child_trade
            self._pending_orders[child_id] = child_broker_id

        # Wait for parent acknowledgement
        start = time.time()
        while time.time() - start < self.timeout:
            self._ib.sleep(0.1)
            status = parent_trade.orderStatus.status
            if status in ("Submitted", "PreSubmitted", "Filled"):
                log.info(
                    "Bracket order %s (%s %s %s) with %d children placed",
                    broker_id, order.symbol, action, total_qty, len(children),
                )
                return broker_id
            if status in ("Cancelled", "ApiCancelled", "Inactive"):
                reason = getattr(parent_trade.orderStatus, "whyHeld", "") or status
                raise IBKROrderRejectedError(
                    f"Bracket order {broker_id} rejected: {reason}"
                )

        raise IBKRBrokerError(
            f"Bracket order {broker_id} timed out after {self.timeout}s"
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        trade = self._order_map.get(order_id)
        if not trade:
            log.warning("Order %s not found for cancellation", order_id)
            return False

        status = trade.orderStatus.status
        if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
            log.info("Order %s already in terminal state: %s", order_id, status)
            return False

        try:
            self._ib.cancelOrder(trade.order)
            self._ib.sleep(0.5)
            log.info("Order %s cancelled", order_id)
            return True
        except Exception as e:
            log.error("Failed to cancel order %s: %s", order_id, e)
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel ALL open orders across all sessions (kill switch)."""
        if not self._ib or not self._ib.isConnected():
            return 0

        open_count = len(self._ib.openTrades())
        try:
            self._ib.reqGlobalCancel()
            self._ib.sleep(2.0)  # wait for cancellations to propagate
            log.warning("Global cancel: %d open orders cancelled", open_count)
            return open_count
        except Exception as e:
            log.error("Global cancel failed: %s", e)
            return 0

    # ------------------------------------------------------------------
    # Order queries
    # ------------------------------------------------------------------

    async def get_order(self, order_id: str) -> Order:
        """Get an order by its internal ID."""
        trade = self._order_map.get(order_id)
        if not trade:
            raise KeyError(f"Order {order_id} not found")

        return _trade_to_internal_order(trade)

    async def get_orders(self, status: str = "open") -> list[Order]:
        """Get all orders matching status filter."""
        if not self._ib or not self._ib.isConnected():
            return []

        if status in ("open",):
            trades = self._ib.openTrades()
        else:
            trades = self._ib.trades()

        orders = []
        for trade in trades:
            ib_status = trade.orderStatus.status
            internal_status = IBKR_STATUS_MAP.get(ib_status, "pending")
            if status == "open" and internal_status not in ("open", "pending", "pending_cancel"):
                continue
            orders.append(_trade_to_internal_order(trade))

        return orders

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Get all open positions from IBKR."""
        if not self._ib or not self._ib.isConnected():
            return []

        positions = []
        for pos in self._ib.positions():
            positions.append(Position(
                symbol=pos.contract.symbol,
                quantity=Decimal(str(pos.position)),
                avg_cost=Decimal(str(pos.avgCost)) if pos.avgCost > 0 else Decimal("0"),
                market_price=(
                    Decimal(str(pos.marketPrice))
                    if hasattr(pos, "marketPrice") and pos.marketPrice
                    else Decimal("0")
                ),
            ))
        return positions

    async def close_position(self, symbol: str) -> bool:
        """Close a position for a single symbol."""
        if not self._ib or not self._ib.isConnected():
            return False

        positions = self._ib.positions()
        for pos in positions:
            if pos.contract.symbol == symbol and float(pos.position) != 0:
                import ib_insync

                contract = ib_insync.Stock(symbol, "SMART", "USD")
                self._ib.qualifyContracts(contract)

                quantity = abs(float(pos.position))
                action = "SELL" if float(pos.position) > 0 else "BUY"
                order = ib_insync.MarketOrder(action, quantity)

                trade = self._ib.placeOrder(contract, order)
                broker_id = f"ibkr-close-{trade.order.orderId}"
                self._order_map[broker_id] = trade

                log.info("Closing position %s: %s %s", symbol, action, quantity)
                return True

        return True  # no position to close

    async def close_all_positions(self) -> int:
        """Close all open positions (kill switch action)."""
        if not self._ib or not self._ib.isConnected():
            return 0

        closed = 0
        for pos in self._ib.positions():
            if float(pos.position) == 0:
                continue

            import ib_insync

            contract = ib_insync.Stock(pos.contract.symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            quantity = abs(float(pos.position))
            action = "SELL" if float(pos.position) > 0 else "BUY"
            order = ib_insync.MarketOrder(action, quantity)

            trade = self._ib.placeOrder(contract, order)
            broker_id = f"ibkr-flatten-{trade.order.orderId}"
            self._order_map[broker_id] = trade
            closed += 1

        log.warning("Closing all positions: %d total", closed)
        return closed

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    async def get_account(self) -> AccountInfo:
        """Get account summary from IBKR."""
        if not self._ib or not self._ib.isConnected():
            raise RuntimeError("IBKR not connected")

        summary = self._ib.accountSummary()
        info = AccountInfo(account_id=self.account_id)

        # Collect account-specific values (in base currency) and
        # 'All' aggregate values for per-currency breakdowns.
        base_nlv = 0.0
        base_cash = 0.0
        base_equity = 0.0
        base_avail = 0.0
        base_gpv = 0.0
        usd_rate = 1.0
        acct = self.account_id

        for item in summary:
            try:
                val = float(item.value)
            except (ValueError, TypeError):
                continue

            # Exchange rate for base→USD conversion
            if item.tag == "ExchangeRate" and item.currency == "USD":
                usd_rate = max(val, 0.001)

            # Account-specific entries (in base currency)
            if item.account == acct:
                if item.tag == "NetLiquidation":
                    base_nlv = val
                elif item.tag == "TotalCashValue":
                    base_cash = val
                elif item.tag == "AvailableFunds":
                    base_avail = val
                elif item.tag == "GrossPositionValue":
                    base_gpv = val

            # Stock market value only appears under 'All'
            if item.account == "All" and item.tag == "StockMarketValue" and item.currency == "BASE":
                base_equity = val

        # Fallback: use NetLiquidationByCurrency BASE if NetLiquidation not found
        if base_nlv == 0.0:
            for item in summary:
                if item.tag == "NetLiquidationByCurrency" and item.currency == "BASE":
                    try:
                        base_nlv = float(item.value)
                    except (ValueError, TypeError):
                        pass
                    break

        # Convert base-currency values to USD
        if usd_rate > 0:
            info.portfolio_value = base_nlv / usd_rate
            info.cash = base_cash / usd_rate
            info.equity = base_equity / usd_rate
            info.buying_power = base_avail / usd_rate
            info.margin_used = base_gpv / usd_rate

        if info.portfolio_value > 0 and info.cash > 0:
            info.margin_used = max(0.0, info.portfolio_value - info.cash)

        return info

    async def get_positions_by_symbol(self) -> dict[str, Position]:
        """Get positions keyed by symbol for quick lookup."""
        positions = await self.get_positions()
        return {p.symbol: p for p in positions}

    # ------------------------------------------------------------------
    # Convenience: sync wrappers for paper_trade.py compatibility
    # ------------------------------------------------------------------

    def connect_sync(self) -> None:
        """Synchronous connect for scripts that don't use asyncio.

        Calls ib_insync directly instead of asyncio.run(self.connect())
        because ib_insync's IB.connect() uses asyncio.timeout() internally
        which fails with 'Timeout should be used inside a task' in Python 3.14.
        """
        try:
            import ib_insync
        except ImportError:
            raise ImportError("ib_insync not installed. Run: pip install ib_insync")

        self._ib = ib_insync.IB()
        self._ib.RequestTimeout = self.timeout

        try:
            self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        except ConnectionRefusedError:
            raise IBKRConnectionError(
                f"Cannot connect to TWS/Gateway at {self.host}:{self.port}. "
                f"Is TWS running and logged in? "
                f"Is API enabled? (Edit > Global Configuration > API > Settings)"
            )
        except OSError as e:
            raise IBKRConnectionError(
                f"Connection to {self.host}:{self.port} failed: {e}. "
                f"Check that TWS/Gateway is running and API is enabled."
            )

        self._connected = True

        accounts = self._ib.managedAccounts()
        if not accounts:
            self._ib.disconnect()
            self._connected = False
            raise IBKRConnectionError("Connected but no managed accounts found")

        for acct in accounts:
            prefix = acct[:2] if len(acct) >= 2 else ""
            if prefix == "DU":
                self.account_id = acct
                if not self._is_paper:
                    log.warning(
                        "Paper account %s detected but port %d suggests LIVE mode.",
                        acct, self.port,
                    )
                else:
                    log.info("Paper trading account: %s", acct)
            elif prefix == "U":
                self.account_id = acct
                if self._is_paper:
                    log.warning(
                        "LIVE account %s detected on paper port %d!",
                        acct, self.port,
                    )
                else:
                    log.info("Live trading account: %s", acct)

        if not self.account_id and accounts:
            self.account_id = accounts[0]

        log.info(
            "IBKR connected: %s:%d (clientId=%d, account=%s)",
            self.host, self.port, self.client_id, self.account_id,
        )

    def submit_order_sync(self, order: BrokerOrder) -> str:
        """Synchronous order submission — calls ib_insync directly."""
        if not self._connected or not self._ib or not self._ib.isConnected():
            self.connect_sync()

        import ib_insync

        contract = ib_insync.Stock(order.symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        action = order.side.upper()
        total_qty = float(order.quantity)

        if order.order_type == "market":
            ib_order = ib_insync.MarketOrder(action, total_qty)
        elif order.order_type == "limit":
            price = float(order.limit_price or 0)
            ib_order = ib_insync.LimitOrder(action, total_qty, price)
        elif order.order_type in ("stop", "stop_loss"):
            price = float(order.stop_price or 0)
            ib_order = ib_insync.StopOrder(action, total_qty, price)
        else:
            ib_order = ib_insync.MarketOrder(action, total_qty)

        ib_order.tif = order.time_in_force.upper() if order.time_in_force else "DAY"

        trade = self._ib.placeOrder(contract, ib_order)
        ib_order_id = trade.order.orderId
        broker_id = f"ibkr-{ib_order_id}"
        self._order_map[broker_id] = trade
        self._pending_orders[ib_order_id] = broker_id

        # Wait for acknowledgement
        start = time.time()
        while time.time() - start < self.timeout:
            self._ib.sleep(0.1)
            status = trade.orderStatus.status
            if status in ("Submitted", "PreSubmitted", "Filled"):
                log.info(
                    "Order %s (%s %s %s @ %s): %s",
                    broker_id, order.symbol, action, total_qty,
                    order.order_type, status,
                )
                return broker_id
            if status in ("Cancelled", "ApiCancelled", "Inactive"):
                reason = getattr(trade.orderStatus, "whyHeld", "") or status
                raise IBKROrderRejectedError(
                    f"Order {broker_id} rejected: {reason}"
                )

        raise IBKRBrokerError(
            f"Order {broker_id} timed out after {self.timeout}s"
        )

    def submit_bracket_order_sync(self, order: BrokerOrder) -> str:
        """Synchronous bracket order submission — calls ib_insync directly."""
        if not self._connected or not self._ib or not self._ib.isConnected():
            self.connect_sync()

        import ib_insync

        contract = ib_insync.Stock(order.symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        action = order.side.upper()
        total_qty = float(order.quantity)
        opposite = "SELL" if action == "BUY" else "BUY"
        oca_group = f"bracket_{int(time.time() * 1e6)}"

        parent = ib_insync.MarketOrder(action, total_qty)
        parent.tif = order.time_in_force.upper() if order.time_in_force else "DAY"
        parent.ocaGroup = oca_group
        parent.ocaType = 1

        parent_trade = self._ib.placeOrder(contract, parent)
        parent_id = parent_trade.order.orderId
        broker_id = f"ibkr-{parent_id}"
        self._order_map[broker_id] = parent_trade
        self._pending_orders[parent_id] = broker_id

        children: list = []
        if order.take_profit_price and float(order.take_profit_price) > 0:
            tp = ib_insync.LimitOrder(opposite, total_qty, float(order.take_profit_price))
            tp.tif = order.time_in_force.upper() if order.time_in_force else "DAY"
            tp.ocaGroup = oca_group
            tp.ocaType = 1
            tp.parentId = parent_id
            tp.transmit = True
            children.append(tp)

        if order.stop_loss_price and float(order.stop_loss_price) > 0:
            sl = ib_insync.StopOrder(opposite, total_qty, float(order.stop_loss_price))
            sl.tif = order.time_in_force.upper() if order.time_in_force else "DAY"
            sl.ocaGroup = oca_group
            sl.ocaType = 1
            sl.parentId = parent_id
            sl.transmit = True
            children.append(sl)

        for child in children:
            child_trade = self._ib.placeOrder(contract, child)
            child_id = child_trade.order.orderId
            child_broker_id = f"ibkr-{child_id}"
            self._order_map[child_broker_id] = child_trade
            self._pending_orders[child_id] = child_broker_id

        start = time.time()
        while time.time() - start < self.timeout:
            self._ib.sleep(0.1)
            status = parent_trade.orderStatus.status
            if status in ("Submitted", "PreSubmitted", "Filled"):
                log.info("Bracket order %s placed with %d children", broker_id, len(children))
                return broker_id
            if status in ("Cancelled", "ApiCancelled", "Inactive"):
                reason = getattr(parent_trade.orderStatus, "whyHeld", "") or status
                raise IBKROrderRejectedError(f"Bracket order {broker_id} rejected: {reason}")

        raise IBKRBrokerError(f"Bracket order {broker_id} timed out after {self.timeout}s")

    def get_account_sync(self) -> AccountInfo:
        """Synchronous account query — calls ib_insync directly."""
        if not self._ib or not self._ib.isConnected():
            raise RuntimeError("IBKR not connected")

        summary = self._ib.accountSummary()
        info = AccountInfo(account_id=self.account_id)

        base_nlv = 0.0
        base_cash = 0.0
        base_equity = 0.0
        base_avail = 0.0
        base_gpv = 0.0
        usd_rate = 1.0
        acct = self.account_id

        for item in summary:
            try:
                val = float(item.value)
            except (ValueError, TypeError):
                continue
            if item.tag == "ExchangeRate" and item.currency == "USD":
                usd_rate = max(val, 0.001)
            if item.account == acct:
                if item.tag == "NetLiquidation":
                    base_nlv = val
                elif item.tag == "TotalCashValue":
                    base_cash = val
                elif item.tag == "AvailableFunds":
                    base_avail = val
                elif item.tag == "GrossPositionValue":
                    base_gpv = val
            if item.account == "All" and item.tag == "StockMarketValue" and item.currency == "BASE":
                base_equity = val

        if base_nlv == 0.0:
            for item in summary:
                if item.tag == "NetLiquidationByCurrency" and item.currency == "BASE":
                    try:
                        base_nlv = float(item.value)
                    except (ValueError, TypeError):
                        pass
                    break

        rate = max(usd_rate, 0.001)
        info.portfolio_value = base_nlv / rate
        info.cash = base_cash / rate
        info.equity = base_equity / rate if base_equity else base_nlv / rate
        info.buying_power = base_avail / rate
        info.margin_used = base_gpv / rate

        if info.portfolio_value > 0 and info.cash > 0:
            info.margin_used = max(0.0, info.portfolio_value - info.cash)
        return info

    def get_positions_sync(self) -> list[Position]:
        """Synchronous position query — calls ib_insync directly."""
        if not self._ib or not self._ib.isConnected():
            return []

        positions = []
        for pos in self._ib.positions():
            positions.append(Position(
                symbol=pos.contract.symbol,
                quantity=Decimal(str(pos.position)),
                avg_cost=Decimal(str(pos.avgCost)) if pos.avgCost > 0 else Decimal("0"),
                market_price=(
                    Decimal(str(pos.marketPrice))
                    if hasattr(pos, "marketPrice") and pos.marketPrice
                    else Decimal("0")
                ),
            ))
        return positions

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    @property
    def is_connected(self) -> bool:
        return self._connected and (self._ib is not None) and self._ib.isConnected()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trade_to_internal_order(trade: Any) -> Order:
    """Convert an ib_insync Trade to an internal Order."""
    o = trade.order
    s = trade.orderStatus

    return Order(
        id=str(o.orderId),
        symbol=getattr(trade.contract, "symbol", ""),
        side=_internal_side(o.action),
        type=_internal_order_type(o),
        quantity=Decimal(str(getattr(o, "totalQuantity", 0))),
        limit_price=Decimal(str(o.lmtPrice)) if getattr(o, "lmtPrice", 0) else None,
        stop_price=Decimal(str(o.auxPrice)) if getattr(o, "auxPrice", 0) else None,
        status=_internal_status(s.status),
        filled_qty=Decimal(str(s.filled)),
        avg_fill_price=Decimal(str(s.avgFillPrice)) if s.avgFillPrice > 0 else None,
    )


def _internal_status(ibkr_status: str) -> OrderStatus:
    """Map IBKR order status to internal OrderStatus enum."""
    mapping = {
        "PendingSubmit": OrderStatus.PENDING,
        "PendingCancel": OrderStatus.PENDING,
        "PreSubmitted": OrderStatus.PENDING,
        "Submitted": OrderStatus.SUBMITTED,
        "ApiPending": OrderStatus.PENDING,
        "ApiCancelled": OrderStatus.CANCELLED,
        "Cancelled": OrderStatus.CANCELLED,
        "Filled": OrderStatus.FILLED,
        "Inactive": OrderStatus.CANCELLED,
    }
    return mapping.get(ibkr_status, OrderStatus.PENDING)
