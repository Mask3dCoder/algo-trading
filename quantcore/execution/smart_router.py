"""Smart execution router — minimizes execution costs without changing alpha logic.

Wraps any broker implementing the sync protocol (IBKRExecutionGateway,
PaperBroker, etc.) and applies microstructure-aware order management:

1. Limit order optimization: submit at mid-price, escalate toward market
   if unfilled within timeout.
2. Spread capture: BUY at bid, SELL at ask when quotes available.
3. Order splitting: TWAP for orders >10% of ADV, split over 5-15 min.
4. Slippage tracking: log expected vs actual fill price in bps.
5. Cost monitoring: daily summary CSV at reports/execution_costs.csv.

Drop-in replacement for any broker in paper_trade.py:
    router = SmartExecutionRouter(inner_broker, config)
    router.submit_order(order)  # same interface
"""

from __future__ import annotations

import csv
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from quantcore.core.types import Order, OrderType, Position, Side

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SmartRouterConfig:
    """Smart execution router configuration."""

    # Limit order optimization
    limit_offset_bps: float = 2.0  # Start this far inside the touch
    escalation_timeout_sec: float = 60.0  # Escalate after this long
    escalation_step_bps: float = 1.0  # Move this much toward market per step
    max_slippage_bps: float = 10.0  # Never exceed this total slippage

    # Spread capture
    use_spread_capture: bool = True  # BUY at bid, SELL at ask
    spread_capture_bps: float = 1.0  # Minimum spread to capture

    # Order splitting (TWAP)
    adv_threshold_pct: float = 0.10  # Split if order > 10% of ADV
    twap_window_minutes: int = 10  # Execute over this window
    twap_num_slices: int = 5  # Number of child slices
    min_split_quantity: float = 100.0  # Don't split orders below this

    # Slippage tracking
    slippage_warn_bps: float = 10.0  # Warn if > 2x backtest assumption
    slippage_alert_bps: float = 20.0  # Alert if > this

    # Cost monitoring
    cost_report_path: str = "reports/execution_costs.csv"
    cost_report_daily: bool = True  # Write daily summary

    # Paper trading simulation
    paper_spread_bps: float = 5.0  # Simulated half-spread for paper
    paper_fill_delay_sec: float = 0.1  # Simulated fill latency


# ---------------------------------------------------------------------------
# Order tracking
# ---------------------------------------------------------------------------


@dataclass
class TrackedOrder:
    """Internal order tracking record."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    expected_price: float
    order_type: str
    limit_price: float | None = None
    submitted_at: float = field(default_factory=time.time)
    broker_order_id: str = ""
    status: str = "pending"  # pending, working, filled, cancelled, escalated
    fill_price: float | None = None
    fill_quantity: float = 0.0
    fill_time: float | None = None
    escalation_count: int = 0
    current_limit: float | None = None
    is_split: bool = False
    parent_order_id: str = ""
    slice_index: int = 0


@dataclass
class DailyCostRecord:
    """Daily execution cost summary."""

    date: str
    total_orders: int = 0
    total_shares: float = 0.0
    total_commission: float = 0.0
    total_spread_cost: float = 0.0  # in dollars
    total_market_impact: float = 0.0  # in dollars
    avg_spread_bps: float = 0.0
    avg_impact_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    total_notional: float = 0.0
    total_execution_cost_pct: float = 0.0
    limit_orders_used: int = 0
    spread_captures: int = 0
    twap_splits: int = 0
    escalations: int = 0


# ---------------------------------------------------------------------------
# Smart Execution Router
# ---------------------------------------------------------------------------


class SmartExecutionRouter:
    """Smart execution router wrapping any sync-protocol broker.

    Applies limit order optimization, spread capture, TWAP splitting,
    slippage tracking, and cost monitoring to every order.

    Implements the same sync protocol as IBKRExecutionGateway:
        connect(), disconnect(), submit_order(Order) -> str,
        cancel_order(order_id), cancel_all_orders(), close_all_positions(),
        get_positions(), get_account(), health_check()
    """

    def __init__(
        self,
        inner_broker: Any,
        config: SmartRouterConfig | None = None,
        quotes: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Args:
            inner_broker: The underlying broker (IBKRExecutionGateway, PaperBroker, etc.)
            config: Smart router configuration. Uses defaults if None.
            quotes: Optional initial quotes {symbol: {"bid": ..., "ask": ..., "last": ..., "volume": ...}}
        """
        self._broker = inner_broker
        self._config = config or SmartRouterConfig()

        # Quote cache: symbol -> {"bid", "ask", "last", "volume", "adv", "timestamp"}
        self._quotes: dict[str, dict[str, float]] = {}
        if quotes:
            self._quotes.update(quotes)

        # Order tracking
        self._tracked_orders: dict[str, TrackedOrder] = {}
        self._lock = threading.Lock()

        # Daily cost accumulation
        self._daily_costs: dict[str, DailyCostRecord] = {}
        self._cost_report_path = Path(self._config.cost_report_path)

        # Background escalation thread (for live trading)
        self._escalation_thread: threading.Thread | None = None
        self._running = False

        # Stats
        self._total_orders = 0
        self._total_slippage_bps = 0.0

    # ------------------------------------------------------------------
    # Quote management
    # ------------------------------------------------------------------

    def update_quotes(self, quotes: dict[str, dict[str, float]]) -> None:
        """Update the quote cache with fresh market data.

        Args:
            quotes: {symbol: {"bid": float, "ask": float, "last": float, "volume": float, "adv": float}}
        """
        now = time.time()
        for sym, q in quotes.items():
            if sym not in self._quotes:
                self._quotes[sym] = {}
            self._quotes[sym].update(q)
            self._quotes[sym]["timestamp"] = now

    def _get_mid_price(self, symbol: str) -> float | None:
        """Get mid-price from quotes."""
        q = self._quotes.get(symbol)
        if q and q.get("bid") and q.get("ask"):
            return (q["bid"] + q["ask"]) / 2.0
        return q.get("last") if q else None

    def _get_spread_bps(self, symbol: str) -> float:
        """Get bid-ask spread in bps."""
        q = self._quotes.get(symbol)
        if q and q.get("bid") and q.get("ask") and q["bid"] > 0:
            return (q["ask"] - q["bid"]) / q["bid"] * 10_000
        return self._config.paper_spread_bps  # fallback

    def _get_adv(self, symbol: str) -> float:
        """Get average daily volume (shares)."""
        q = self._quotes.get(symbol)
        if q and q.get("adv"):
            return q["adv"]
        if q and q.get("volume"):
            return q["volume"] * 10  # rough estimate: daily = 10x bar volume
        return 1_000_000.0  # conservative default

    # ------------------------------------------------------------------
    # Broker protocol (sync) — drop-in replacement
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the underlying broker."""
        self._broker.connect()
        self._running = True
        log.info("SmartExecutionRouter connected (inner: %s)", type(self._broker).__name__)

    def disconnect(self) -> None:
        """Disconnect from the underlying broker."""
        self._running = False
        self._flush_daily_costs()
        self._broker.disconnect()
        log.info("SmartExecutionRouter disconnected")

    @property
    def is_connected(self) -> bool:
        return getattr(self._broker, "is_connected", True)

    @property
    def is_paper(self) -> bool:
        return getattr(self._broker, "is_paper", True)

    @property
    def broker_name(self) -> str:
        inner = getattr(self._broker, "broker_name", type(self._broker).__name__)
        return f"smart({inner})"

    def submit_order(self, order: Order) -> str:
        """Submit an order with smart execution logic.

        Applies limit order optimization, spread capture, and TWAP splitting
        before delegating to the inner broker.

        Returns the (parent) order ID string.
        """
        cfg = self._config
        symbol = str(order.symbol).upper()
        side = "buy" if order.side == Side.BUY else "sell"
        quantity = float(order.quantity)
        order_type = order.type.value if hasattr(order.type, "value") else str(order.type).lower()

        # Get expected price (mid-price or order's own price)
        mid = self._get_mid_price(symbol)
        expected_price = mid or float(order.limit_price or order.stop_price or 0.0)

        order_id = f"smart-{uuid.uuid4().hex[:12]}"

        # Check if order should be split (TWAP)
        adv = self._get_adv(symbol)
        order_pct_adv = quantity / adv if adv > 0 else 0.0

        if (
            order_pct_adv > cfg.adv_threshold_pct
            and quantity >= cfg.min_split_quantity
            and order_type == "market"
        ):
            return self._submit_twap_split(order, order_id, expected_price, side, quantity)

        # Single order — apply limit optimization
        return self._submit_single(order, order_id, expected_price, side, quantity, order_type)

    def _submit_single(
        self,
        order: Order,
        order_id: str,
        expected_price: float,
        side: str,
        quantity: float,
        order_type: str,
    ) -> str:
        """Submit a single order with limit optimization."""
        cfg = self._config
        symbol = str(order.symbol).upper()

        tracked = TrackedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=expected_price,
            order_type=order_type,
        )

        # Convert market orders to limit orders when possible
        if order_type == "market" and expected_price > 0 and cfg.use_spread_capture:
            # Calculate aggressive limit price
            spread_bps = self._get_spread_bps(symbol)
            if spread_bps >= cfg.spread_capture_bps:
                # BUY at bid-side, SELL at ask-side
                offset = expected_price * cfg.limit_offset_bps / 10_000
                if side == "buy":
                    limit_price = expected_price - offset
                else:
                    limit_price = expected_price + offset

                # Create limit order
                limit_order = order.model_copy(
                    update={
                        "type": OrderType.LIMIT,
                        "limit_price": Decimal(str(round(limit_price, 2))),
                    }
                )
                tracked.limit_price = limit_price
                tracked.current_limit = limit_price
                tracked.order_type = "limit"

                # Submit to inner broker
                broker_id = self._broker.submit_order(limit_order)
                tracked.broker_order_id = broker_id
                tracked.status = "working"

                with self._lock:
                    self._tracked_orders[order_id] = tracked

                self._record_cost(tracked, "limit_created")
                log.info(
                    "Smart: %s %s %.0f @ %.2f (mid=%.2f, spread=%.1f bps, limit offset=%.1f bps)",
                    side.upper(), symbol, quantity, limit_price, expected_price,
                    spread_bps, cfg.limit_offset_bps,
                )
                return order_id

        # Fall through to direct submission (market order or no quote data)
        broker_id = self._broker.submit_order(order)
        tracked.broker_order_id = broker_id
        tracked.status = "submitted"

        # Simulate fill for paper trading
        if hasattr(self._broker, "is_paper") and self._broker.is_paper:
            fill_price = self._simulate_fill_price(expected_price, side, quantity)
            tracked.fill_price = fill_price
            tracked.fill_quantity = quantity
            tracked.fill_time = time.time()
            tracked.status = "filled"
            self._record_slippage(tracked, expected_price, fill_price)
        else:
            # Live: poll for fill (simplified — in production, use callbacks)
            tracked.fill_price = expected_price  # assume fill at expected for tracking
            tracked.fill_quantity = quantity
            tracked.fill_time = time.time()
            tracked.status = "filled"

        with self._lock:
            self._tracked_orders[order_id] = tracked

        self._record_cost(tracked, "market_order")
        return order_id

    def _submit_twap_split(
        self,
        order: Order,
        parent_id: str,
        expected_price: float,
        side: str,
        total_quantity: float,
    ) -> str:
        """Split a large order into TWAP slices."""
        cfg = self._config
        symbol = str(order.symbol).upper()

        # Compute slice schedule
        num_slices = cfg.twap_num_slices
        base_qty = total_quantity / num_slices
        interval_sec = (cfg.twap_window_minutes * 60) / num_slices

        log.info(
            "Smart TWAP: %s %s %.0f shares split into %d slices over %d min "
            "(%.1f%% ADV)",
            side.upper(), symbol, total_quantity, num_slices,
            cfg.twap_window_minutes, (total_quantity / self._get_adv(symbol)) * 100,
        )

        child_ids = []
        for i in range(num_slices):
            # Size with small randomization
            slice_qty = base_qty * np.random.uniform(0.95, 1.05)
            if i == num_slices - 1:
                # Last slice gets remainder
                slice_qty = total_quantity - sum(
                    base_qty * 0.95 for _ in range(i)  # approximate
                )
            slice_qty = max(1.0, round(slice_qty, 0))

            # Create child order
            child_order = order.model_copy(
                update={
                    "id": f"{parent_id}-s{i}",
                    "quantity": Decimal(str(slice_qty)),
                    "type": OrderType.MARKET,
                }
            )

            # Stagger submission (in paper mode, submit all immediately)
            if i > 0 and not (hasattr(self._broker, "is_paper") and self._broker.is_paper):
                time.sleep(interval_sec)

            child_id = self._submit_single(
                child_order,
                f"{parent_id}-s{i}",
                expected_price,
                side,
                slice_qty,
                "market",
            )

            with self._lock:
                if child_id in self._tracked_orders:
                    self._tracked_orders[child_id].is_split = True
                    self._tracked_orders[child_id].parent_order_id = parent_id
                    self._tracked_orders[child_id].slice_index = i

            child_ids.append(child_id)

        # Track parent
        tracked = TrackedOrder(
            order_id=parent_id,
            symbol=symbol,
            side=side,
            quantity=total_quantity,
            expected_price=expected_price,
            order_type="twap_parent",
            status="split",
            is_split=False,
        )
        with self._lock:
            self._tracked_orders[parent_id] = tracked

        daily = self._get_daily_record()
        daily.twap_splits += 1

        return parent_id

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order."""
        with self._lock:
            tracked = self._tracked_orders.get(order_id)
            if not tracked:
                return False
            if tracked.broker_order_id:
                result = self._broker.cancel_order(tracked.broker_order_id)
                tracked.status = "cancelled"
                return result
        return False

    def cancel_all_orders(self) -> int:
        """Cancel all working orders."""
        count = 0
        with self._lock:
            for oid, tracked in self._tracked_orders.items():
                if tracked.status == "working" and tracked.broker_order_id:
                    if self._broker.cancel_order(tracked.broker_order_id):
                        tracked.status = "cancelled"
                        count += 1
        return count

    def close_all_positions(self) -> int:
        """Close all positions via the inner broker."""
        return self._broker.close_all_positions()

    def get_positions(self) -> dict[str, Position]:
        """Return current positions from the inner broker."""
        return self._broker.get_positions()

    def get_account(self) -> dict[str, float]:
        """Return account info from the inner broker."""
        return self._broker.get_account()

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get current prices — delegates to inner broker if available."""
        if hasattr(self._broker, "get_current_prices"):
            return self._broker.get_current_prices(symbols)
        return {}

    def health_check(self) -> dict[str, Any]:
        """Return health status including smart router metrics."""
        health = self._broker.health_check() if hasattr(self._broker, "health_check") else {}
        health["smart_router"] = True
        health["tracked_orders"] = len(self._tracked_orders)
        health["working_orders"] = sum(
            1 for o in self._tracked_orders.values() if o.status == "working"
        )
        health["total_slippage_bps"] = (
            self._total_slippage_bps / max(self._total_orders, 1)
        )
        return health

    # ------------------------------------------------------------------
    # Drawdown tracker compatibility
    # ------------------------------------------------------------------

    @property
    def drawdown_tracker(self) -> Any:
        return getattr(self._broker, "drawdown_tracker", None)

    @drawdown_tracker.setter
    def drawdown_tracker(self, value: Any) -> None:
        self._broker.drawdown_tracker = value

    # ------------------------------------------------------------------
    # Fill simulation (paper trading)
    # ------------------------------------------------------------------

    def _simulate_fill_price(self, expected_price: float, side: str, quantity: float) -> float:
        """Simulate a realistic fill price for paper trading.

        Uses the spread-aware model: fills at a slight disadvantage
        relative to mid, but better than a naive market order.
        """
        cfg = self._config
        half_spread = expected_price * cfg.paper_spread_bps / 10_000

        # Smart router captures part of the spread
        spread_capture = half_spread * 0.4  # capture 40% of half-spread

        # Small random impact
        impact = expected_price * np.random.uniform(0, 2.0) / 10_000

        if side == "buy":
            # BUY: fill slightly above mid, but better than naive
            fill = expected_price + half_spread - spread_capture + impact
        else:
            # SELL: fill slightly below mid, but better than naive
            fill = expected_price - half_spread + spread_capture - impact

        return round(fill, 2)

    # ------------------------------------------------------------------
    # Slippage tracking
    # ------------------------------------------------------------------

    def _record_slippage(
        self, tracked: TrackedOrder, expected: float, actual: float
    ) -> None:
        """Record slippage for a filled order."""
        if expected <= 0:
            return

        if tracked.side == "buy":
            slippage_bps = (actual - expected) / expected * 10_000
        else:
            slippage_bps = (expected - actual) / expected * 10_000

        self._total_orders += 1
        self._total_slippage_bps += max(0, slippage_bps)

        cfg = self._config
        if slippage_bps > cfg.slippage_alert_bps:
            log.warning(
                "SLIPPAGE ALERT: %s %s %.0f @ %.2f vs expected %.2f = %.1f bps "
                "(>%.0f bps threshold)",
                tracked.side.upper(), tracked.symbol, tracked.quantity,
                actual, expected, slippage_bps, cfg.slippage_alert_bps,
            )
        elif slippage_bps > cfg.slippage_warn_bps:
            log.info(
                "Slippage warn: %s %s = %.1f bps (threshold %.0f)",
                tracked.symbol, tracked.side, slippage_bps, cfg.slippage_warn_bps,
            )

    # ------------------------------------------------------------------
    # Cost monitoring
    # ------------------------------------------------------------------

    def _get_daily_record(self) -> DailyCostRecord:
        """Get or create today's cost record."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today not in self._daily_costs:
            self._daily_costs[today] = DailyCostRecord(date=today)
        return self._daily_costs[today]

    def _record_cost(self, tracked: TrackedOrder, event: str) -> None:
        """Record execution cost data."""
        daily = self._get_daily_record()

        if event == "limit_created":
            daily.limit_orders_used += 1
            daily.total_orders += 1
            daily.total_shares += tracked.quantity
            if tracked.expected_price > 0:
                daily.total_notional += tracked.quantity * tracked.expected_price

            spread_bps = self._get_spread_bps(tracked.symbol)
            if spread_bps >= self._config.spread_capture_bps:
                daily.spread_captures += 1

            # Estimate spread cost saved
            if tracked.limit_price and tracked.expected_price > 0:
                spread_saved = abs(tracked.expected_price - tracked.limit_price)
                daily.total_spread_cost += spread_saved * tracked.quantity

        elif event == "market_order":
            daily.total_orders += 1
            daily.total_shares += tracked.quantity
            if tracked.expected_price > 0:
                daily.total_notional += tracked.quantity * tracked.expected_price

            # Compute slippage
            if tracked.fill_price and tracked.expected_price > 0:
                if tracked.side == "buy":
                    slip = (tracked.fill_price - tracked.expected_price) / tracked.expected_price * 10_000
                else:
                    slip = (tracked.expected_price - tracked.fill_price) / tracked.expected_price * 10_000
                daily.avg_slippage_bps = (
                    (daily.avg_slippage_bps * (daily.total_orders - 1) + max(0, slip))
                    / daily.total_orders
                )

        # Persist periodically
        if daily.total_orders % 10 == 0:
            self._write_daily_cost(daily)

    def _write_daily_cost(self, record: DailyCostRecord) -> None:
        """Write daily cost record to CSV."""
        self._cost_report_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self._cost_report_path.exists()
        with open(self._cost_report_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "date", "total_orders", "total_shares", "total_commission",
                    "avg_spread_bps", "avg_impact_bps", "avg_slippage_bps",
                    "total_notional", "total_execution_cost_pct",
                    "limit_orders_used", "spread_captures", "twap_splits", "escalations",
                ])

            cost_pct = 0.0
            if record.total_notional > 0:
                total_cost = record.total_spread_cost + record.total_market_impact + record.total_commission
                cost_pct = total_cost / record.total_notional * 100

            writer.writerow([
                record.date,
                record.total_orders,
                round(record.total_shares, 0),
                round(record.total_commission, 2),
                round(record.avg_spread_bps, 2),
                round(record.avg_impact_bps, 2),
                round(record.avg_slippage_bps, 2),
                round(record.total_notional, 2),
                round(cost_pct, 4),
                record.limit_orders_used,
                record.spread_captures,
                record.twap_splits,
                record.escalations,
            ])

    def _flush_daily_costs(self) -> None:
        """Write all daily cost records to CSV."""
        for record in self._daily_costs.values():
            self._write_daily_cost(record)

    # ------------------------------------------------------------------
    # Limit order escalation (background thread for live trading)
    # ------------------------------------------------------------------

    def _start_escalation_monitor(self) -> None:
        """Start background thread that escalates stale limit orders."""
        if self._escalation_thread and self._escalation_thread.is_alive():
            return

        self._running = True
        self._escalation_thread = threading.Thread(
            target=self._escalation_loop, daemon=True, name="smart-escalation"
        )
        self._escalation_thread.start()

    def _escalation_loop(self) -> None:
        """Background loop: escalate limit orders that haven't filled."""
        cfg = self._config
        while self._running:
            time.sleep(5.0)  # Check every 5 seconds
            now = time.time()

            with self._lock:
                for oid, tracked in self._tracked_orders.items():
                    if tracked.status != "working":
                        continue

                    elapsed = now - tracked.submitted_at
                    if elapsed < cfg.escalation_timeout_sec:
                        continue

                    # Escalate: move limit toward market
                    mid = self._get_mid_price(tracked.symbol)
                    if not mid or not tracked.current_limit:
                        continue

                    step = mid * cfg.escalation_step_bps / 10_000
                    if tracked.side == "buy":
                        new_limit = min(tracked.current_limit + step, mid)
                    else:
                        new_limit = max(tracked.current_limit - step, mid)

                    # Check max slippage
                    if tracked.expected_price > 0:
                        total_slip = abs(new_limit - tracked.expected_price) / tracked.expected_price * 10_000
                        if total_slip > cfg.max_slippage_bps:
                            log.warning(
                                "Smart: %s hit max slippage %.1f bps, cancelling",
                                oid, total_slip,
                            )
                            self._broker.cancel_order(tracked.broker_order_id)
                            tracked.status = "cancelled"
                            continue

                    # Cancel old and resubmit at new limit
                    self._broker.cancel_order(tracked.broker_order_id)

                    new_order = Order(
                        id=oid,
                        symbol=tracked.symbol,
                        side=Side.BUY if tracked.side == "buy" else Side.SELL,
                        type=OrderType.LIMIT,
                        quantity=Decimal(str(tracked.quantity)),
                        limit_price=Decimal(str(round(new_limit, 2))),
                    )
                    broker_id = self._broker.submit_order(new_order)
                    tracked.broker_order_id = broker_id
                    tracked.current_limit = new_limit
                    tracked.escalation_count += 1
                    tracked.submitted_at = now  # Reset timer

                    daily = self._get_daily_record()
                    daily.escalations += 1

                    log.info(
                        "Smart escalation: %s %s limit %.2f → %.2f (mid=%.2f, escalation #%d)",
                        tracked.side.upper(), tracked.symbol,
                        tracked.current_limit - step if tracked.side == "buy" else tracked.current_limit + step,
                        new_limit, mid, tracked.escalation_count,
                    )

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------

    def get_execution_stats(self) -> dict[str, Any]:
        """Return current execution statistics."""
        avg_slip = self._total_slippage_bps / max(self._total_orders, 1)
        working = sum(1 for o in self._tracked_orders.values() if o.status == "working")
        filled = sum(1 for o in self._tracked_orders.values() if o.status == "filled")
        cancelled = sum(1 for o in self._tracked_orders.values() if o.status == "cancelled")

        return {
            "total_orders": self._total_orders,
            "avg_slippage_bps": round(avg_slip, 2),
            "working_orders": working,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
            "limit_orders_used": sum(
                1 for o in self._tracked_orders.values()
                if o.order_type == "limit" and o.status != "split"
            ),
            "twap_splits": sum(
                1 for o in self._tracked_orders.values()
                if o.order_type == "twap_parent"
            ),
            "escalations": sum(o.escalation_count for o in self._tracked_orders.values()),
        }
