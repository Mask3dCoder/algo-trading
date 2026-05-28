"""Live execution gateway — synchronous paper trading via Alpaca Broker.

Implements the live/runner.py Broker protocol so it can be passed directly
to LiveTrader / LiveRunner without any adapter boilerplate:

    broker = LiveExecutionGateway()   # reads ALPACA_API_KEY / ALPACA_API_SECRET
    broker.submit_order(order)         # sync, blocking (paper fills are immediate)
    broker.get_positions()             # returns dict[str, Position]
    broker.get_account()               # returns dict[str, float]

No hardcoded API keys — credentials come from os.environ or constructor args.

Paper mode is automatic when ALPACA_BASE_URL contains "paper-api"
(default https://paper-api.alpaca.markets).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from decimal import Decimal
from typing import Any

from quantcore.brokers.alpaca_broker import AlpacaBroker
from quantcore.core.types import Order, OrderStatus, OrderType, Position, Side

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal order store (in-memory for paper tracking)
# ---------------------------------------------------------------------------


class _PaperOrderRecord:
    """Manages paper order state with immediate simulated fills."""

    def __init__(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())[:12].upper()
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.status: OrderStatus = OrderStatus.PENDING
        self.filled_qty: Decimal = Decimal("0")
        self.filled_price: float = 0.0
        self._fill_price: float | None = None   # set at construction for immediate fill

    def simulate_fill(self, market_price: float) -> None:
        """Immediately fill the order at market price (paper = instant fills)."""
        self.status = OrderStatus.FILLED
        self.filled_qty = self.quantity
        self.filled_price = market_price
        self._fill_price = market_price

    @property
    def side_str(self) -> str:
        return "buy" if self.side == Side.BUY else "sell"


class LiveExecutionGateway:
    """Synchronous paper execution gateway backed by Alpaca Broker.

    Acts as a facade:
      - Sync interface (no async/await needed — all real work is in threads)
      - Paper fills simulated immediately at current market price
      - Live mode: delegates to AlpacaBroker's sync wrappers
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        self._api_key   = api_key   or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_API_SECRET")
        self._base_url  = base_url  or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        self._paper = "paper-api" in self._base_url

        # Initialise real Alpaca broker for account queries & position sync
        try:
            self._broker = AlpacaBroker(
                api_key=self._api_key,
                secret_key=self._secret_key,
                mode="paper" if self._paper else "live",
            )
            # Connect the async TradingClient before use
            import asyncio
            asyncio.run(self._broker.connect())
            # Use sync wrappers
            self._broker.get_account_sync()
            self._connected = True
            log.info(
                "LiveExecutionGateway: connected to Alpaca %s",
                "PAPER" if self._paper else "LIVE",
            )
        except Exception as exc:
            log.warning(
                "Alpaca broker not available (%s) — running in SIMULATION mode. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET to connect.",
                exc,
            )
            self._broker = None
            self._connected = False

        self._initial_cash = float(initial_cash)
        self._cash: float = initial_cash
        self._positions: dict[str, Position] = {}   # symbol → Position
        self._orders: dict[str, Order] = {}        # order_id → Order
        self._paper_orders: dict[str, _PaperOrderRecord] = {}  # id → record

        # Sync broker positions on startup
        self._sync_from_broker()

    @property
    def broker_name(self) -> str:
        return "alpaca_paper" if self._paper else "alpaca_live"

    @property
    def is_paper(self) -> bool:
        return self._paper

    # ------------------------------------------------------------------
    # Broker protocol (sync)
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit a market order. Returns internal order ID.

        Paper: fills immediately at current market price (slippage modelled).
        Live: delegates to Alpaca and returns broker order ID.
        """
        # Resolve symbol
        symbol = str(order.symbol).strip().upper()

        if self._paper:
            return self._submit_paper_order(order, symbol)
        else:
            return self._submit_live_order(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID. Returns True on success."""
        if self._paper:
            record = self._paper_orders.get(order_id)
            if record and record.status in (
                OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL,
            ):
                record.status = OrderStatus.CANCELLED
                # Release reserved cash
                notional = float(record.filled_qty) * (record._fill_price or 0.0)
                self._cash += notional
                log.info("PAPER: cancelled order %s (%s %s)", order_id, record.side_str, symbol)
                return True
            return False
        else:
            if self._broker:
                import asyncio
                ok = asyncio.run(self._broker.cancel_order(order_id))
                return ok
            return False

    def get_positions(self) -> dict[str, Position]:
        """Return current open positions as {symbol: Position}."""
        return dict(self._positions)

    def get_account(self) -> dict[str, float]:
        """Return account info dict with cash, equity, portfolio_value."""
        if self._broker:
            try:
                info = self._broker.get_account_sync()
                return {
                    "cash":             info.get("cash", self._cash),
                    "equity":           info.get("equity", self._cash),
                    "portfolio_value":  info.get("portfolio_value", self._cash),
                    "buying_power":     info.get("buying_power", self._cash * 2),
                }
            except Exception:
                pass

        # Fallback: compute from local positions
        pos_value = sum(
            float(abs(p.quantity)) * float(p.market_price)
            for p in self._positions.values()
        )
        return {
            "cash":             self._cash,
            "equity":           self._cash + pos_value - sum(
                float(abs(p.quantity)) * float(p.avg_cost)
                for p in self._positions.values()
            ),
            "portfolio_value":  self._cash + pos_value,
            "buying_power":     self._cash * 2,
        }

    # ------------------------------------------------------------------
    # Phase 4 additions: cancel_all_orders, health_check
    # ------------------------------------------------------------------

    def cancel_all_orders(self) -> bool:
        """Cancel every open/paper order. Returns True on success."""
        try:
            if self._paper:
                for oid, rec in list(self._paper_orders.items()):
                    if rec.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                        rec.status = OrderStatus.CANCELLED
                        notional = float(rec.filled_qty) * (rec._fill_price or 0.0)
                        self._cash += notional
                        log.info("PAPER: cancelled order %s", oid)
                return True
            if self._broker:
                import asyncio
                return asyncio.run(self._broker.cancel_all_orders())
            return False
        except Exception as exc:
            log.error("cancel_all_orders failed: %s", exc)
            return False

    def health_check(self) -> dict[str, Any]:
        """Return broker health status."""
        status: dict[str, Any] = {
            "connected": self._connected,
            "paper": self._paper,
            "cash": self._cash,
            "position_count": len(self._positions),
            "open_order_count": len(self._paper_orders) if self._paper else 0,
        }
        if self._broker:
            try:
                acct = self._broker.get_account_sync()
                status.update(
                    {
                        "broker_cash": acct.get("cash"),
                        "broker_equity": acct.get("equity"),
                        "broker_buying_power": acct.get("buying_power"),
                    }
                )
                status["broker_healthy"] = True
            except Exception as exc:
                status["broker_healthy"] = False
                status["broker_error"] = str(exc)
        else:
            status["broker_healthy"] = False
            status["broker_error"] = "Alpaca broker not connected"
        return status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_paper_order(self, order: Order, symbol: str) -> str:
        record = _PaperOrderRecord(
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.type.value if order.type else "market",
            limit_price=float(order.limit_price) if order.limit_price else None,
        )

        # Get current market price from broker or data feed
        market_price = self._get_market_price(symbol)

        # Compute notional — paper fills at market
        notional = float(order.quantity) * market_price
        if record.side == Side.SELL:
            # Short sell: we receive cash on entry
            self._cash += notional
        else:
            # Buy: deduct cash (T+0 in paper)
            if notional > self._cash + 1.0:
                log.warning(
                    "PAPER: insufficient cash to fill %s — rejecting. "
                    "Needed $%.2f, have $%.2f",
                    symbol, notional, self._cash,
                )
                record.status = OrderStatus.REJECTED
                self._paper_orders[record.id] = record
                self._orders[record.id] = self._order_to_typed(record)
                return record.id
            self._cash -= notional

        # Simulate fill
        slippage = market_price * 0.001  # 10 bp slippage model
        fill_price = market_price + slippage if record.side == Side.BUY else market_price - slippage
        record.simulate_fill(fill_price)

        # Update positions
        existing = self._positions.get(symbol)
        if existing:
            qty = float(existing.quantity)
            avg = float(existing.avg_cost)
            if record.side == Side.BUY:
                new_qty  = qty + float(order.quantity)
                new_avg  = (qty * avg + float(order.quantity) * fill_price) / max(new_qty, 1)
            else:
                new_qty  = qty - float(order.quantity)
                new_avg  = avg
            if abs(new_qty) < 1e-8:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal(str(new_qty)),
                    avg_cost=Decimal(str(new_avg)),
                    market_price=Decimal(str(fill_price)),
                    unrealized_pnl=Decimal("0"),
                )
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=order.quantity,
                avg_cost=Decimal(str(fill_price)),
                market_price=Decimal(str(fill_price)),
                unrealized_pnl=Decimal("0"),
            )

        # Persist order
        self._paper_orders[record.id] = record
        self._orders[record.id] = self._order_to_typed(record)
        log.info(
            "PAPER: filled %s %s %s @ %.4f (slippage %.4f)",
            record.id, record.side_str.upper(), symbol, fill_price, slippage,
        )
        return record.id

    def _submit_live_order(self, order: Order) -> str:
        if not self._broker:
            return ""

        import asyncio

        order_side = order.side.name.lower()
        qty = float(order.quantity)

        # Dispatch synchronously
        async def _do() -> str:
            from quantcore.execution.brokers.base import BrokerOrder
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            br_order = BrokerOrder(
                symbol=str(order.symbol).strip().upper(),
                side=order_side,
                quantity=qty,
                order_type="market",
            )
            return await self._broker.submit_order(br_order)

        return asyncio.run(_do())

    def _get_market_price(self, symbol: str) -> float:
        """Fetch latest close price from Alpaca or return $100 fallback."""
        # Try our own cached data feed if available (preferred — uses same source)
        try:
            from quantcore.live.data_feed import LiveDataFeed

            feed = LiveDataFeed(tickers=[symbol])
            price = feed.get_latest_close(symbol)
            if price and price > 0:
                return price
        except Exception:
            pass

        # Try Alpaca broker
        if self._broker:
            try:
                positions = self._broker.get_positions_sync()
                pos = positions.get(symbol)
                if pos:
                    return float(pos.market_price)
            except Exception:
                pass

            try:
                import urllib.request
                headers = {
                    "APCA-API-KEY-ID":     self._api_key or "",
                    "APCA-API-SECRET-KEY": self._secret_key or "",
                }
                data_base = os.environ.get(
                    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
                ).replace("paper-api.", "data.") + f"/v2/stocks/{symbol}/bars"
                from urllib.parse import urlencode
                url = f"{data_base}?{urlencode({'limit': '1', 'adjustment': 'all'})}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    import json
                    bars = json.loads(resp.read().decode()).get("bars", {}).get(symbol, [])
                    if bars:
                        return float(bars[-1].get("c", 100.0))
            except Exception:
                pass

        log.warning("Could not fetch price for %s — falling back to $100.00", symbol)
        return 100.0

    def _order_to_typed(self, record: _PaperOrderRecord) -> Order:
        return Order(
            id=record.id,
            symbol=record.symbol,
            side=record.side,
            type=OrderType(record.order_type or "market"),
            quantity=record.quantity,
            status=record.status,
            filled_qty=record.filled_qty,
            avg_fill_price=Decimal(str(record.filled_price)) if record.status == OrderStatus.FILLED else None,
        )

    def _sync_from_broker(self) -> None:
        """On startup: pull real positions/balance from Alpaca to seed our state."""
        if not self._broker:
            return
        try:
            info = self._broker.get_account_sync()
            self._cash = float(info.get("cash", self._initial_cash))
            positions = self._broker.get_positions_sync()
            self._positions = {str(k): v for k, v in positions.items()}
            log.info(
                "Synced from Alpaca: cash $%.2f, positions %s",
                self._cash, list(self._positions.keys()),
            )
        except Exception as exc:
            log.warning("Could not sync from Alpaca broker: %s", exc)
