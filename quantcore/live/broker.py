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
from datetime import UTC
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
        events_path: str | None = None,
    ) -> None:
        self._api_key   = api_key   or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_API_SECRET")
        self._base_url  = base_url  or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self._events_path = events_path

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

        # Wash-trade cooldown: symbol → last opposite-side order timestamp (epoch)
        self._last_opposite_side: dict[str, tuple[str, float]] = {}
        self._wash_trade_cooldown_sec = 30.0

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

    def submit_order(
        self, order: Order, fill_metadata: dict | None = None
    ) -> str:
        """Submit a market order. Returns internal order ID.

        Paper: fills immediately at current market price (slippage modelled).
        Live: delegates to Alpaca and returns broker order ID.

        fill_metadata: optional dict with keys:
            strat_name, trade_date, weight, filled_at
            Used to record fill events in JSONL.
        """
        # Resolve symbol
        symbol = str(order.symbol).strip().upper()

        log.info(
            "Submitting order: %s %s %s to Alpaca %s",
            symbol, order.side.name.upper() if order.side else "?",
            order.quantity, "PAPER" if self._paper else "LIVE",
        )

        if self._paper:
            result = self._submit_paper_order(order, symbol, fill_metadata=fill_metadata)
        else:
            result = self._submit_live_order(order)

        # Diagnostic logging after submission attempt
        if result:
            record = self._paper_orders.get(result)
            if record:
                log.info(
                    "Order response: %s filled_qty=%s avg_fill_price=%.4f",
                    record.status.value if record.status else "unknown",
                    record.filled_qty,
                    record.filled_price,
                )
            else:
                log.info("Order response: submitted order_id=%s (no paper record)", result)
        else:
            log.warning("Order submission returned empty order_id for %s %s %s",
                        symbol, order.side, order.quantity)

        return result

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
                log.info("PAPER: cancelled order %s (%s %s)", order_id, record.side_str, record.symbol)
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
            except Exception as exc:
                log.debug("Alpaca get_account failed, using local state: %s", exc)

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

    def _submit_paper_order(
        self, order: Order, symbol: str, fill_metadata: dict | None = None
    ) -> str:
        """Submit a paper order, with wash-trade cooldown and option-aware fill."""
        side_str = order.side.name.lower()
        now = time.time()

        # --- Wash-trade cooldown (per-strategy, per-symbol) ---
        strategy_name = fill_metadata.get("strat_name", "") if fill_metadata else ""
        cooldown_key = (strategy_name, symbol)
        prev = self._last_opposite_side.get(cooldown_key)
        if prev is not None:
            prev_side, prev_ts = prev
            if prev_side != side_str and (now - prev_ts) < self._wash_trade_cooldown_sec:
                log.warning(
                    "Wash-trade cooldown for %s (%s): %s rejected (last %s %.1fs ago)",
                    symbol, strategy_name, side_str, prev_side, now - prev_ts,
                )
                record = _PaperOrderRecord(
                    symbol=symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.type.value if order.type else "market",
                    limit_price=float(order.limit_price) if order.limit_price else None,
                )
                record.status = OrderStatus.REJECTED
                self._paper_orders[record.id] = record
                self._orders[record.id] = self._order_to_typed(record)
                return record.id

        # --- Track this side for next order -----------------------------
        self._last_opposite_side[cooldown_key] = (side_str, now)

        # Route to Alpaca paper API so orders appear on the dashboard
        if self._broker and self._connected:
            try:
                alpaca_id = self._submit_live_order(order)
                if alpaca_id:
                    # Create local record tracking the Alpaca order
                    record = _PaperOrderRecord(
                        symbol=symbol,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.type.value if order.type else "market",
                        limit_price=float(order.limit_price) if order.limit_price else None,
                    )
                    record.id = alpaca_id
                    # Fetch fill price from Alpaca
                    try:
                        alpaca_order = self._broker._client.get_order_by_id(alpaca_id)
                        fill_price = float(alpaca_order.filled_avg_price or 0)
                        if fill_price > 0:
                            record.simulate_fill(fill_price)
                            slippage = 0.0
                        else:
                            market_price = self._get_market_price(symbol)
                            record.simulate_fill(market_price)
                            slippage = market_price * 0.001
                    except Exception:
                        market_price = self._get_market_price(symbol)
                        record.simulate_fill(market_price)
                        slippage = market_price * 0.001

                    # Sync positions from Alpaca after fill
                    self._sync_positions_from_alpaca()

                    self._paper_orders[record.id] = record
                    self._orders[record.id] = self._order_to_typed(record)
                    log.info(
                        "PAPER→ALPACA: filled %s %s %s @ %.4f",
                        record.id, record.side_str.upper(), symbol, record.filled_price,
                    )
                    # Write fill event to JSONL
                    fm = fill_metadata or {}
                    self._append_fill_event(
                        order_id=record.id,
                        symbol=symbol,
                        side=record.side_str,
                        qty=float(record.quantity),
                        price=record.filled_price,
                        notional=float(record.quantity) * record.filled_price,
                        strat_name=fm.get("strat_name", ""),
                        trade_date=fm.get("trade_date", ""),
                        weight=fm.get("weight", 1.0),
                        filled_at=fm.get("filled_at", "alpaca_paper"),
                    )
                    return record.id
            except Exception as exc:
                log.warning("Alpaca paper submit failed (%s) — falling back to local sim", exc)

        # Fallback: local simulation if Alpaca is unavailable
        record = _PaperOrderRecord(
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.type.value if order.type else "market",
            limit_price=float(order.limit_price) if order.limit_price else None,
        )

        market_price = self._get_market_price(symbol)
        if market_price is None:
            log.warning("PAPER: no price for %s — rejecting local sim order", symbol)
            record.status = OrderStatus.REJECTED
            self._paper_orders[record.id] = record
            self._orders[record.id] = self._order_to_typed(record)
            return record.id

        # --- Option-aware fill pricing ----------------------------------
        meta = order.metadata or {}
        is_option = meta.get("instrument") in ("OPTION", "option")
        if is_option and meta.get("theoretical_mid"):
            # Option: apply strategy's bid-ask haircut
            theo = float(meta["theoretical_mid"])
            spread_pct = float(meta.get("bid_ask_spread_pct", 0.05))
            commission = float(meta.get("commission_per_contract", 0.65))
            multiplier = int(meta.get("multiplier", 100))
            contracts = int(float(order.quantity))  # for options, quantity = contracts
            if order.side == Side.SELL:
                fill_price = theo * (1.0 - spread_pct)  # receive bid side
            else:
                fill_price = theo * (1.0 + spread_pct)   # pay ask side
            # Commission charged per contract
            total_commission = commission * contracts
            notional = fill_price * multiplier * contracts
            if order.side == Side.SELL:
                self._cash += notional - total_commission
            else:
                if notional + total_commission > self._cash + 1.0:
                    log.warning(
                        "PAPER: insufficient cash for option %s — rejecting. "
                        "Needed premium $%.2f + comm $%.2f, have $%.2f",
                        symbol, notional, total_commission, self._cash,
                    )
                    record.status = OrderStatus.REJECTED
                    self._paper_orders[record.id] = record
                    self._orders[record.id] = self._order_to_typed(record)
                    return record.id
                self._cash -= notional + total_commission
            record.simulate_fill(fill_price)
            # Position key for options: symbol + strike + expiry
            opt_key = meta.get("_position_key", symbol)
            self._update_position(opt_key, order.side, order.quantity, fill_price, metadata=meta)
            log.info(
                "PAPER (local sim OPTION): filled %s %s %s @ %.4f "
                "(spread %.1f%%, comm $%.2f, contracts=%d)",
                record.id, record.side_str.upper(), symbol, fill_price,
                spread_pct * 100, total_commission, contracts,
            )
            self._paper_orders[record.id] = record
            self._orders[record.id] = self._order_to_typed(record)
            fm = fill_metadata or {}
            self._append_fill_event(
                order_id=record.id,
                symbol=symbol,
                side=record.side_str,
                qty=float(record.quantity),
                price=fill_price,
                notional=notional,
                strat_name=fm.get("strat_name", ""),
                trade_date=fm.get("trade_date", ""),
                weight=fm.get("weight", 1.0),
                filled_at=fm.get("filled_at", "local_sim_option"),
            )
            return record.id

        # --- Equity fill pricing ----------------------------------------
        notional = float(order.quantity) * market_price
        if record.side == Side.SELL:
            self._cash += notional
        else:
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

        slippage = market_price * 0.001
        fill_price = market_price + slippage if record.side == Side.BUY else market_price - slippage
        record.simulate_fill(fill_price)

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

        self._paper_orders[record.id] = record
        self._orders[record.id] = self._order_to_typed(record)
        log.info(
            "PAPER (local sim): filled %s %s %s @ %.4f (slippage %.4f)",
            record.id, record.side_str.upper(), symbol, fill_price, slippage,
        )
        # Write fill event to JSONL
        fm = fill_metadata or {}
        self._append_fill_event(
            order_id=record.id,
            symbol=symbol,
            side=record.side_str,
            qty=float(record.quantity),
            price=fill_price,
            notional=float(record.quantity) * fill_price,
            strat_name=fm.get("strat_name", ""),
            trade_date=fm.get("trade_date", ""),
            weight=fm.get("weight", 1.0),
            filled_at=fm.get("filled_at", "local_sim"),
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

            br_order = BrokerOrder(
                symbol=str(order.symbol).strip().upper(),
                side=order_side,
                quantity=qty,
                order_type="market",
            )
            return await self._broker.submit_order(br_order)

        return asyncio.run(_do())

    def _get_market_price(self, symbol: str) -> float | None:
        """Fetch latest close price from Alpaca.  Returns ``None`` when no
        reliable price is available so callers reject the order rather than
        trading at a bogus $100 fallback."""
        # Try our own cached data feed if available (preferred)
        try:
            from quantcore.live.data_feed import LiveDataFeed
            feed = LiveDataFeed(tickers=[symbol])
            price = feed.get_latest_close(symbol)
            if price and price > 0:
                return price
        except Exception as exc:
            log.debug("LiveDataFeed price fetch failed for %s: %s", symbol, exc)

        # Try Alpaca broker position price
        if self._broker:
            try:
                positions = self._broker.get_positions_sync()
                pos = positions.get(symbol)
                if pos and float(pos.market_price) > 0:
                    return float(pos.market_price)
            except Exception as exc:
                log.debug("Broker position price lookup failed for %s: %s", symbol, exc)

            try:
                import urllib.request
                headers = {
                    "APCA-API-KEY-ID":     self._api_key or "",
                    "APCA-API-SECRET-KEY": self._secret_key or "",
                }
                data_base = os.environ.get(
                    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
                ).replace("paper-api.", "data.")
                url = f"{data_base.rstrip('/')}/v2/stocks/{symbol}/bars/latest"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    import json
                    result = json.loads(resp.read().decode())
                    bar = result.get("bar", {})
                    if bar:
                        price = float(bar.get("c", 0))
                        if price > 0:
                            return price
            except Exception as exc:
                log.debug("Alpaca bars/latest failed for %s: %s", symbol, exc)

        log.warning("Could not fetch price for %s — returning None (order will be rejected)", symbol)
        return None

    def _append_fill_event(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        notional: float,
        strat_name: str,
        trade_date: str,
        weight: float,
        filled_at: str,
    ) -> None:
        """Write a fill event to the JSONL log file."""
        if not self._events_path:
            return
        import json
        from datetime import datetime
        try:
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "fill",
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "notional": notional,
                "order_id": order_id,
                "strategy": strat_name,
                "trade_date": trade_date,
                "weight": weight,
                "filled_at": filled_at,
            }
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            pass

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

    def _update_position(
        self,
        key: str,
        side: Side,
        qty: Decimal,
        fill_price: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a position keyed by an arbitrary string (e.g. option composite key).

        For options, ``key`` is a composite distinct from the underlying symbol.
        For equity, ``key`` equals the symbol.
        """
        existing = self._positions.get(key)
        if existing:
            qty_float = float(existing.quantity)
            avg = float(existing.avg_cost)
            if side == Side.BUY:
                new_qty = qty_float + float(qty)
                new_avg = (qty_float * avg + float(qty) * fill_price) / max(new_qty, 1)
            else:
                new_qty = qty_float - float(qty)
                new_avg = avg
            if abs(new_qty) < 1e-8:
                self._positions.pop(key, None)
            else:
                self._positions[key] = Position(
                    symbol=str(existing.symbol),
                    quantity=Decimal(str(new_qty)),
                    avg_cost=Decimal(str(new_avg)),
                    market_price=Decimal(str(fill_price)),
                    unrealized_pnl=Decimal("0"),
                )
        else:
            sym = metadata.get("underlying", key) if metadata else key
            self._positions[key] = Position(
                symbol=str(sym),
                quantity=Decimal(str(qty)) if side == Side.BUY else -Decimal(str(qty)),
                avg_cost=Decimal(str(fill_price)),
                market_price=Decimal(str(fill_price)),
                unrealized_pnl=Decimal("0"),
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

    def _sync_positions_from_alpaca(self) -> None:
        """Pull latest positions and cash from Alpaca after a fill."""
        if not self._broker:
            return
        try:
            info = self._broker.get_account_sync()
            self._cash = float(info.get("cash", self._cash))
            positions = self._broker.get_positions_sync()
            self._positions = {str(k): v for k, v in positions.items()}
        except Exception as exc:
            log.debug("Could not sync positions from Alpaca: %s", exc)
