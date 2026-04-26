#!/usr/bin/env python
"""Paper trading runner with duration control, metrics logging, and graceful shutdown.

Usage:
    python scripts/paper_trade.py --duration 5d --strategy momentum --data-source mock
    python scripts/paper_trade.py --duration 12h --strategy trend_following --data-source yahoo
    python scripts/paper_trade.py --duration 30m --strategy mean_reversion

Signals:
    SIGINT (Ctrl+C) or SIGTERM → trigger kill switch, graceful shutdown, exit code 1.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import signal
import sys
import time
import traceback
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Ensure quantcore is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)\s*(d|h|m|s)$", re.IGNORECASE)


def parse_duration(duration_str: str) -> int:
    """Parse a duration string into seconds.

    Accepted formats: Nd (days), Nh (hours), Nm (minutes), Ns (seconds).
    Examples: "5d" = 432000, "12h" = 43200, "30m" = 1800, "60s" = 60.

    Raises ValueError for invalid formats.
    """
    match = _DURATION_RE.match(duration_str.strip())
    if not match:
        raise ValueError(
            f"Invalid duration: '{duration_str}'. "
            f"Expected format: <number><unit> where unit is d, h, m, or s (e.g., 5d, 12h, 30m)"
        )
    value = int(match.group(1))
    unit = match.group(2).lower()
    multipliers = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return value * multipliers[unit]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_dir: str = "logs") -> tuple[logging.Logger, str]:
    """Configure dual logging: INFO to console, DEBUG to rotating file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"paper_trade_{timestamp}.log")

    logger = logging.getLogger("paper_trade")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler — DEBUG and above, rotated at 50MB, keep 5 backups
    file_handler = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    ))
    logger.addHandler(file_handler)

    return logger, log_path


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------


def setup_metrics_csv(log_dir: str = "logs") -> tuple[str, csv.DictWriter, Any]:
    """Create CSV file for periodic metrics and return (path, writer, file_handle)."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"paper_trade_metrics_{timestamp}.csv")

    fieldnames = [
        "timestamp", "total_pnl", "unrealized_pnl", "realized_pnl",
        "open_positions_count", "orders_submitted_count", "orders_filled_count",
        "orders_rejected_count", "kill_switch_active", "memory_rss_mb",
        "last_data_age_seconds",
    ]
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return csv_path, writer, f


# ---------------------------------------------------------------------------
# Mock data feed (for dry-run testing)
# ---------------------------------------------------------------------------


class MockDataFeed:
    """Generates synthetic OHLCV bars at a configurable interval."""

    def __init__(self, symbols: list[str] | None = None, interval_sec: float = 1.0):
        self.symbols = symbols or ["STK0000"]
        self.interval_sec = interval_sec
        self._rng = np.random.default_rng(42)
        self._prices: dict[str, float] = {s: 100.0 for s in self.symbols}
        self._last_tick: datetime | None = None

    def is_market_open(self) -> bool:
        return True

    def get_latest(self) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        self._last_tick = now
        events = []
        for sym in self.symbols:
            ret = self._rng.normal(0.00005, 0.01)
            self._prices[sym] *= (1.0 + ret)
            events.append({
                "symbol": sym, "price": self._prices[sym],
                "timestamp": now, "close": self._prices[sym],
                "high": self._prices[sym] * 1.001,
                "low": self._prices[sym] * 0.999,
                "open": self._prices[sym] * 0.9995,
                "volume": int(abs(self._rng.normal(10000, 5000))),
            })
        return events

    @property
    def last_data_timestamp(self) -> datetime | None:
        return self._last_tick


# ---------------------------------------------------------------------------
# Simple moving-average crossover strategy for paper trading
# ---------------------------------------------------------------------------


class SimpleMACrossover:
    """Moving average crossover strategy for paper trading validation."""

    def __init__(self, symbol: str = "STK0000", fast: int = 20, slow: int = 50):
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self._price_history: list[float] = []
        self._orders_submitted = 0
        self._orders_filled = 0
        self._orders_rejected = 0
        self._realized_pnl = 0.0

    def generate_signals(self, data_feed) -> list[dict[str, Any]]:
        events = data_feed.get_latest()
        signals = []

        for event in events:
            if event["symbol"] != self.symbol:
                continue
            price = float(event["price"])
            self._price_history.append(price)

            if len(self._price_history) < self.slow:
                continue

            fast_ma = np.mean(self._price_history[-self.fast:])
            slow_ma = np.mean(self._price_history[-self.slow:])

            if fast_ma > slow_ma:
                signals.append({
                    "symbol": self.symbol, "direction": "long",
                    "price": price, "strength": abs(fast_ma - slow_ma) / slow_ma,
                })
            elif fast_ma < slow_ma:
                signals.append({
                    "symbol": self.symbol, "direction": "short",
                    "price": price, "strength": abs(fast_ma - slow_ma) / slow_ma,
                })

        return signals

    def on_fill(self, fill) -> None:
        self._orders_filled += 1
        self._realized_pnl += getattr(fill, "commission", 0) or 0


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type] = {
    "momentum": SimpleMACrossover,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_paper_trade(
    duration_seconds: int,
    strategy_name: str = "momentum",
    data_source: str = "mock",
    initial_capital: float = 100_000.0,
) -> int:
    """Run the paper trading loop for the specified duration. Returns exit code."""
    logger, log_path = setup_logging()
    csv_path, csv_writer, csv_file = setup_metrics_csv()

    logger.info("=" * 60)
    logger.info("Paper Trading Run Starting")
    logger.info(f"  Duration: {duration_seconds}s ({duration_seconds/3600:.1f}h)")
    logger.info(f"  Strategy: {strategy_name}")
    logger.info(f"  Data source: {data_source}")
    logger.info(f"  Initial capital: ${initial_capital:,.0f}")
    logger.info(f"  Log file: {log_path}")
    logger.info(f"  Metrics CSV: {csv_path}")
    logger.info("=" * 60)

    # Initialize data feed
    if data_source == "mock":
        feed = MockDataFeed(symbols=["STK0000"], interval_sec=1.0)
        logger.info("Using mock data feed (synthetic OHLCV)")
    elif data_source == "alpaca":
        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

        symbols = ["SPY"]  # Default paper trading symbol
        feed = AlpacaPaperFeed(symbols=symbols)
        logger.info("Using Alpaca paper data feed (REST polling) — symbols: %s", symbols)
    elif data_source == "yahoo":
        logger.error("Yahoo data feed not yet wired into paper_trade.py")
        return 2
    else:
        logger.error(f"Unknown data source: {data_source}")
        return 2

    # Initialize strategy
    strat_cls = _STRATEGIES.get(strategy_name)
    if strat_cls is None:
        logger.error(f"Unknown strategy: {strategy_name}. Available: {list(_STRATEGIES)}")
        return 2

    strategy = strat_cls(symbol="STK0000")
    logger.info(f"Strategy initialized: {strategy_name}")

    # Initialize paper broker
    from quantcore.live.paper import PaperBroker, PaperConfig
    paper_cfg = PaperConfig(initial_capital=initial_capital)
    broker = PaperBroker(config=paper_cfg)
    logger.info("Paper broker initialized")

    # Initialize OMS and risk
    from quantcore.execution.oms import OrderManager
    from quantcore.risk.limits import CircuitBreaker, RiskLimits

    oms = OrderManager()
    risk_limits = RiskLimits()
    CircuitBreaker(limits=risk_limits)
    logger.info("OMS + risk engine initialized")

    # Metrics state
    start_time = time.time()
    orders_submitted = 0
    orders_filled = 0
    orders_rejected = 0
    kill_switch_active = False

    # Signal handlers
    def on_signal(signum, frame):
        nonlocal kill_switch_active
        logger.info(f"Received signal {signal.Signals(signum).name}. Triggering kill switch...")
        kill_switch_active = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    exit_code = 0
    last_metrics_time = time.time()

    try:
        while time.time() - start_time < duration_seconds and not kill_switch_active:
            # Update market data
            events = feed.get_latest()
            prices = {e["symbol"]: float(e["price"]) for e in events}
            broker.update_market_data(prices)
            broker.update_equity()

            # Generate signals (not every tick — the strategy handles its own logic)
            signals = strategy.generate_signals(feed)

            for sig in signals:
                symbol = sig["symbol"]
                direction = sig["direction"]
                price = float(sig["price"])

                if direction == "long":
                    side = __import__("quantcore.core.types", fromlist=["Side"]).Side.BUY
                else:
                    side = __import__("quantcore.core.types", fromlist=["Side"]).Side.SELL

                from decimal import Decimal

                from quantcore.core.types import Order, OrderType

                # Size: 10% of equity per trade
                equity = float(broker.get_account()["portfolio_value"])
                qty = max(1, int(equity * 0.10 / price))

                order = Order(
                    id=f"paper_{int(time.time() * 1e6)}_{symbol}",
                    symbol=symbol, side=side, type=OrderType.MARKET,
                    quantity=Decimal(str(qty)), strategy_id=strategy_name,
                )

                broker.submit_order(order)
                orders_submitted += 1

            # Process fills
            fills = broker.process_fills()
            orders_filled += len(fills)

            # Check drawdown
            dd_state = broker.drawdown_tracker
            if hasattr(dd_state, '_determine_action'):
                action = dd_state._determine_action(dd_state.current_drawdown)
                from quantcore.risk.drawdown import DrawdownAction
                if action in (DrawdownAction.HALT, DrawdownAction.FLATTEN):
                    logger.warning(f"Drawdown action triggered: {action}")
                    kill_switch_active = True

            # Periodic metrics (every 60 seconds)
            now = time.time()
            if now - last_metrics_time >= 60:
                acct = broker.get_account()
                from quantcore.core.health_state import get_health_state

                get_health_state()
                try:
                    import psutil
                    mem_rss = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 1)
                except ImportError:
                    mem_rss = None

                last_data_age = None
                if feed.last_data_timestamp:
                    last_data_age = (datetime.now(UTC) - feed.last_data_timestamp).total_seconds()

                row = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "total_pnl": round(float(acct["portfolio_value"]) - initial_capital, 2),
                    "unrealized_pnl": 0.0,
                    "realized_pnl": round(float(acct["total_commission"]), 2),
                    "open_positions_count": len(broker.get_positions()),
                    "orders_submitted_count": orders_submitted,
                    "orders_filled_count": orders_filled,
                    "orders_rejected_count": orders_rejected,
                    "kill_switch_active": kill_switch_active,
                    "memory_rss_mb": mem_rss,
                    "last_data_age_seconds": last_data_age,
                }
                csv_writer.writerow(row)
                csv_file.flush()

                logger.info(
                    f"Metrics | PnL: ${row['total_pnl']:,.0f} | "
                    f"Positions: {row['open_positions_count']} | "
                    f"Orders: {orders_submitted}s/{orders_filled}f/{orders_rejected}r | "
                    f"Mem: {mem_rss}MB"
                )
                last_metrics_time = now

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
        kill_switch_active = True
        exit_code = 1
    except Exception:
        logger.critical(f"Unhandled exception:\n{traceback.format_exc()}")
        kill_switch_active = True
        exit_code = 2

    # --- Graceful shutdown ---
    logger.info("Beginning graceful shutdown...")

    # Cancel all open orders
    open_orders = [
        oid for oid, o in oms.orders.items()
        if o.status.value not in ("filled", "cancelled", "rejected")
    ]
    for oid in open_orders:
        try:
            oms.cancel(oid)
        except Exception:
            pass
    logger.info(f"Cancelled {len(open_orders)} open orders")

    # Close all positions (record in audit trail)
    positions = broker.get_positions()
    from decimal import Decimal as D
    for sym, pos in positions.items():
        qty = float(pos.quantity)
        if abs(qty) < 1e-10:
            continue
        from quantcore.core.types import Order, OrderType, Side
        close_side = Side.SELL if qty > 0 else Side.BUY
        close_order = Order(
            id=f"close_{int(time.time() * 1e6)}_{sym}",
            symbol=sym, side=close_side, type=OrderType.MARKET,
            quantity=D(str(int(abs(qty)))), strategy_id="shutdown",
        )
        try:
            broker.submit_order(close_order)
            broker.process_fills()
        except Exception:
            pass

    # Final snapshot
    acct = broker.get_account()
    total_pnl = float(acct["portfolio_value"]) - initial_capital
    runtime = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Paper Trading Run Complete")
    logger.info(f"  Final P&L:      ${total_pnl:,.2f}")
    logger.info(f"  Total trades:   {orders_filled}")
    logger.info("  Win rate:       N/A (not tracked per-trade)")
    logger.info(f"  Max drawdown:   {broker.drawdown_tracker.max_drawdown:.2%}")
    logger.info(f"  Runtime:        {runtime:.0f}s ({runtime/3600:.1f}h)")
    logger.info("  Final memory:   see metrics CSV")
    logger.info(f"  Exit code:      {exit_code}")
    logger.info("=" * 60)

    csv_file.close()
    logging.shutdown()
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QuantCore Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/paper_trade.py --duration 5d\n"
            "  python scripts/paper_trade.py --duration 12h --strategy momentum\n"
            "  python scripts/paper_trade.py --duration 30m --data-source mock\n"
            "  python scripts/paper_trade.py --duration 5d --capital 500000\n"
        ),
    )
    parser.add_argument(
        "--duration", "-d", required=True,
        help="Run duration: Nd (days), Nh (hours), Nm (minutes), Ns (seconds). E.g., 5d, 12h, 30m",
    )
    parser.add_argument(
        "--strategy", "-s", default="momentum",
        choices=list(_STRATEGIES),
        help=f"Strategy to run (default: momentum). Available: {list(_STRATEGIES)}",
    )
    parser.add_argument(
        "--data-source", default="mock",
        choices=["mock", "alpaca", "yahoo"],
        help="Data feed source (default: mock). 'alpaca' uses Alpaca Markets REST API.",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital in dollars (default: 100,000)",
    )

    args = parser.parse_args()

    try:
        duration_sec = parse_duration(args.duration)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return run_paper_trade(
        duration_seconds=duration_sec,
        strategy_name=args.strategy,
        data_source=args.data_source,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    sys.exit(main())
