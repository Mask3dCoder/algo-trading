"""Interactive Brokers execution gateway — synchronous facade over IBKRBroker.

Implements the live/runner.py Broker protocol so it can be passed directly
to LiveTrader / LiveRunner without any adapter boilerplate:

    broker = IBKRExecutionGateway()     # reads IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID
    broker.submit_order(order)          # sync, blocking
    broker.get_positions()              # returns dict[str, Position]
    broker.get_account()                # returns dict[str, float]

Connection ports (set via IBKR_PORT env var):
    TWS Paper Trading: 7497  (default)
    TWS Live Trading:  7496
    IB Gateway Paper:  4002
    IB Gateway Live:   4001
"""

from __future__ import annotations

# Python 3.14 no longer auto-creates event loops.  ib_insync depends on
# eventkit which calls asyncio.get_event_loop() at *import time*
# (eventkit/util.py).  We must ensure a loop exists before ib_insync is
# ever imported, so do it here at module load.
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import os
from decimal import Decimal
from typing import Any

from quantcore.core.types import Order, Position, Side

log = logging.getLogger(__name__)

# Port → mode mapping for display
_PORT_MODES: dict[int, str] = {
    7497: "TWS Paper",
    7496: "TWS Live",
    4002: "Gateway Paper",
    4001: "Gateway Live",
}


class IBKRExecutionGateway:
    """Synchronous IBKR execution gateway implementing the Broker protocol.

    Wraps the existing IBKRBroker (ib_insync-based) with sync wrappers
    and presents the simple 4-method interface that paper_trade.py and
    LiveRunner expect.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        account_id: str = "",
        timeout: float = 10.0,
        readonly: bool = False,
    ) -> None:
        from quantcore.brokers.ibkr_broker import IBKRBroker

        self._host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self._port = port or int(os.environ.get("IBKR_PORT", "7497"))
        self._client_id = client_id or int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self._timeout = timeout

        self._broker = IBKRBroker(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            account_id=account_id,
            timeout=timeout,
            readonly=readonly,
        )

        self._connected = False
        self._is_paper = self._port in (7497, 4002)
        mode = _PORT_MODES.get(self._port, f"Unknown ({self._port})")

        if self._is_paper:
            log.info("IBKR gateway initialized: %s mode (no real money)", mode)
        else:
            log.warning("IBKR gateway initialized: %s mode — REAL MONEY AT RISK", mode)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to TWS/Gateway. Raises on failure."""
        import ib_insync

        self._broker._ib = ib_insync.IB()
        self._broker._ib.RequestTimeout = self._timeout

        try:
            self._broker._ib.connect(
                self._host, self._port,
                clientId=self._client_id, timeout=self._timeout,
            )
        except ConnectionRefusedError as e:
            raise ConnectionError(
                f"Cannot connect to TWS/Gateway at {self._host}:{self._port}. "
                f"Is TWS running and logged in? Is API enabled? "
                f"(Edit > Global Configuration > API > Settings)"
            ) from e
        except OSError as e:
            raise ConnectionError(
                f"Connection to {self._host}:{self._port} failed: {e}"
            ) from e

        self._broker._connected = True
        self._connected = True

        accounts = self._broker._ib.managedAccounts()
        if accounts:
            self._broker.account_id = accounts[0]

        log.info(
            "IBKR gateway connected: %s:%d (clientId=%d, account=%s)",
            self._host, self._port, self._client_id, self._broker.account_id,
        )

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._broker._ib and self._broker._ib.isConnected():
            self._broker._ib.disconnect()
        self._broker._connected = False
        self._connected = False
        log.info("IBKR gateway disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._broker.is_connected

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    @property
    def broker_name(self) -> str:
        return "ibkr_paper" if self._is_paper else "ibkr_live"

    # ------------------------------------------------------------------
    # Broker protocol (sync) — used by paper_trade.py / LiveRunner
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit an order to IBKR. Returns broker order ID string.

        Calls ib_insync directly — the sync wrappers on IBKRBroker use
        asyncio.run() which conflicts with ib_insync's internal event loop.
        """
        import ib_insync

        if not self._broker._ib or not self._broker._ib.isConnected():
            raise RuntimeError("IBKR not connected")

        symbol = str(order.symbol).strip().upper()
        contract = ib_insync.Stock(symbol, "SMART", "USD")
        self._broker._ib.qualifyContracts(contract)

        action = "BUY" if order.side == Side.BUY else "SELL"
        qty = float(order.quantity)

        order_type_str = (
            order.type.value if hasattr(order.type, "value")
            else str(order.type).lower()
        )

        if order_type_str == "limit" and order.limit_price:
            ib_order = ib_insync.LimitOrder(action, qty, float(order.limit_price))
        elif order_type_str in ("stop", "stop_loss") and order.stop_price:
            ib_order = ib_insync.StopOrder(action, qty, float(order.stop_price))
        else:
            ib_order = ib_insync.MarketOrder(action, qty)

        ib_order.tif = "DAY"

        trade = self._broker._ib.placeOrder(contract, ib_order)
        broker_id = f"ibkr-{trade.order.orderId}"
        self._broker._order_map[broker_id] = trade

        # Wait for acknowledgement
        import time

        start = time.time()
        while time.time() - start < self._timeout:
            self._broker._ib.sleep(0.1)
            status = trade.orderStatus.status
            if status in ("Submitted", "PreSubmitted", "Filled", "Cancelled"):
                break
            if status in ("Inactive",):
                raise RuntimeError(f"IBKR order inactive: {trade.orderStatus}")

        log.info("IBKR order submitted: %s → %s", order.id, broker_id)
        return broker_id

    def submit_bracket_order(
        self,
        order: Order,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> str:
        """Submit a bracket order (parent + stop-loss + take-profit children)."""
        import ib_insync

        if not self._broker._ib or not self._broker._ib.isConnected():
            raise RuntimeError("IBKR not connected")

        symbol = str(order.symbol).strip().upper()
        contract = ib_insync.Stock(symbol, "SMART", "USD")
        self._broker._ib.qualifyContracts(contract)

        action = "BUY" if order.side == Side.BUY else "SELL"
        qty = float(order.quantity)
        parent_price = float(order.limit_price or order.stop_price or 0)

        parent = ib_insync.LimitOrder(action, qty, parent_price) if order.limit_price else ib_insync.MarketOrder(action, qty)
        parent.tif = "DAY"
        parent.orderId = self._broker._ib.client.getReqId()
        parent.transmit = False

        children = []
        if stop_loss is not None and stop_loss > 0:
            sl = ib_insync.Order()
            sl.action = "SELL" if action == "BUY" else "BUY"
            sl.orderType = "STP"
            sl.auxPrice = float(stop_loss)
            sl.totalQuantity = qty
            sl.parentId = parent.orderId
            sl.orderId = self._broker._ib.client.getReqId()
            sl.tif = "DAY"
            sl.transmit = False
            children.append(sl)

        if take_profit is not None and take_profit > 0:
            tp = ib_insync.Order()
            tp.action = "SELL" if action == "BUY" else "BUY"
            tp.orderType = "LMT"
            tp.lmtPrice = float(take_profit)
            tp.totalQuantity = qty
            tp.parentId = parent.orderId
            tp.orderId = self._broker._ib.client.getReqId()
            tp.tif = "DAY"
            tp.transmit = False
            children.append(tp)

        if children:
            children[-1].transmit = True

        all_orders = [parent] + children
        for o in all_orders:
            self._broker._ib.placeOrder(o, contract)

        broker_id = f"ibkr-{parent.orderId}"
        log.info("IBKR bracket order submitted: %s → %s", order.id, broker_id)
        return broker_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID. Returns True on success."""
        if not self._broker._ib or not self._broker._ib.isConnected():
            return False

        trade = self._broker._order_map.get(order_id)
        if trade:
            self._broker._ib.cancelOrder(trade.order)
            return True

        # Fallback: try to extract numeric ID from "ibkr-123" format
        try:
            numeric_id = int(order_id.split("-", 1)[1]) if "-" in order_id else int(order_id)
        except (ValueError, IndexError):
            return False

        for t in self._broker._ib.openOrders():
            if t.order.orderId == numeric_id:
                self._broker._ib.cancelOrder(t.order)
                return True
        return False

    def cancel_all_orders(self) -> int:
        """Cancel ALL open orders (kill switch). Returns count cancelled."""
        if not self._broker._ib or not self._broker._ib.isConnected():
            return 0
        self._broker._ib.reqGlobalCancel()
        return len(self._broker._order_map)

    def close_all_positions(self) -> int:
        """Close all open positions (kill switch). Returns count closed."""
        if not self._broker._ib or not self._broker._ib.isConnected():
            return 0

        import ib_insync

        closed = 0
        for pos in self._broker._ib.positions():
            if float(pos.position) == 0:
                continue
            contract = ib_insync.Stock(pos.contract.symbol, "SMART", "USD")
            self._broker._ib.qualifyContracts(contract)
            qty = abs(float(pos.position))
            action = "SELL" if float(pos.position) > 0 else "BUY"
            order = ib_insync.MarketOrder(action, qty)
            self._broker._ib.placeOrder(contract, order)
            closed += 1

        log.warning("Closing all positions: %d total", closed)
        return closed

    def get_positions(self) -> dict[str, Position]:
        """Return current open positions as {symbol: Position}.

        Calls ib_insync directly — the sync wrappers on IBKRBroker use
        asyncio.run() which conflicts with ib_insync's internal event loop.
        """
        if not self._broker._ib or not self._broker._ib.isConnected():
            return {}

        result: dict[str, Position] = {}
        for pos in self._broker._ib.positions():
            sym = pos.contract.symbol
            result[sym] = Position(
                symbol=sym,
                quantity=Decimal(str(pos.position)),
                avg_cost=Decimal(str(pos.avgCost)) if pos.avgCost > 0 else Decimal("0"),
                market_price=(
                    Decimal(str(pos.marketPrice))
                    if hasattr(pos, "marketPrice") and pos.marketPrice
                    else Decimal("0")
                ),
            )
        return result

    def get_account(self) -> dict[str, float]:
        """Return account info dict with cash, equity, portfolio_value.

        Calls ib_insync directly — the sync wrappers on IBKRBroker use
        asyncio.run() which conflicts with ib_insync's internal event loop.
        """
        if not self._broker._ib or not self._broker._ib.isConnected():
            return {}

        summary = self._broker._ib.accountSummary()
        base_nlv = 0.0
        base_cash = 0.0
        base_equity = 0.0
        base_avail = 0.0
        base_gpv = 0.0
        usd_rate = 1.0
        acct = self._broker.account_id

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

        # Fallback: NetLiquidationByCurrency BASE
        if base_nlv == 0.0:
            for item in summary:
                if item.tag == "NetLiquidationByCurrency" and item.currency == "BASE":
                    try:
                        base_nlv = float(item.value)
                    except (ValueError, TypeError):
                        pass
                    break

        rate = max(usd_rate, 0.001)
        return {
            "cash": base_cash / rate,
            "equity": base_equity / rate if base_equity else base_nlv / rate,
            "portfolio_value": base_nlv / rate,
            "buying_power": base_avail / rate,
            "total_commission": base_gpv / rate,
        }

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current market prices for a list of symbols.

        Uses IBKR's reqMktData for real-time snapshots.
        Returns {symbol: price} — missing symbols are omitted.
        """
        import ib_insync

        prices: dict[str, float] = {}
        if not self._broker._ib or not self._broker._ib.isConnected():
            return prices

        for sym in symbols:
            try:
                contract = ib_insync.Stock(sym, "SMART", "USD")
                self._broker._ib.qualifyContracts(contract)
                ticker = self._broker._ib.reqMktData(contract, "", True, False)
                self._broker._ib.sleep(1.0)
                price = ticker.marketPrice()
                if price and price > 0:
                    prices[sym] = float(price)
                self._broker._ib.cancelMktData(contract)
            except Exception as e:
                log.warning("Could not fetch price for %s: %s", sym, e)

        return prices

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return broker health status dict."""
        status: dict[str, Any] = {
            "connected": self.is_connected,
            "paper": self.is_paper,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
            "account_id": self._broker.account_id,
            "broker_name": self.broker_name,
        }

        if self.is_connected:
            try:
                acct = self.get_account()
                status.update({
                    "cash": acct.get("cash"),
                    "equity": acct.get("equity"),
                    "portfolio_value": acct.get("portfolio_value"),
                    "buying_power": acct.get("buying_power"),
                })
                positions = self.get_positions()
                status["position_count"] = len(positions)
                status["healthy"] = True
            except Exception as exc:
                status["healthy"] = False
                status["error"] = str(exc)
        else:
            status["healthy"] = False
            status["error"] = "Not connected to TWS/Gateway"

        return status

    # ------------------------------------------------------------------
    # Drawdown tracker compatibility (attached by paper_trade.py)
    # ------------------------------------------------------------------

    @property
    def drawdown_tracker(self) -> Any:
        return getattr(self._broker, "drawdown_tracker", None)

    @drawdown_tracker.setter
    def drawdown_tracker(self, value: Any) -> None:
        self._broker.drawdown_tracker = value
