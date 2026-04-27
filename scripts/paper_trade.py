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

from quantcore.core.time_utils import is_market_open, next_market_open  # noqa: E402
from quantcore.core.types import (  # noqa: E402
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
)
from quantcore.paper.session import SessionSnapshot, SessionStore  # noqa: E402
from quantcore.risk.config_loader import (  # noqa: E402
    RiskConfig,
    load_risk_config,
    run_preflight_checks,
    validate_kill_switch_mechanism,
)
from quantcore.risk.limits import check_order  # noqa: E402

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
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
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
# Account normalization helper
# ---------------------------------------------------------------------------


def _get_account_dict(broker: Any) -> dict[str, float]:
    """Normalize account info to a dict regardless of broker type.

    PaperBroker returns a dict, IBKRBroker returns AccountInfo.
    Uses sync wrapper when available (IBKRBroker).
    """
    try:
        acct = broker.get_account_sync()
    except (AttributeError, TypeError):
        acct = broker.get_account()

    if isinstance(acct, dict):
        return acct
    # AccountInfo or similar dataclass
    return {
        "cash": float(getattr(acct, "cash", 0.0) or 0.0),
        "equity": float(getattr(acct, "equity", 0.0) or 0.0),
        "portfolio_value": float(getattr(acct, "portfolio_value", 0.0) or 0.0),
        "total_commission": float(getattr(acct, "margin_used", 0.0) or 0.0),
    }


def _get_broker_positions(broker: Any) -> dict[str, Any]:
    """Normalize positions to a dict of Position objects.

    PaperBroker returns dict[str, Position], IBKRBroker returns list[Position].
    Uses sync wrapper when available (IBKRBroker).
    """
    try:
        positions = broker.get_positions_sync()
    except (AttributeError, TypeError):
        positions = broker.get_positions()

    if isinstance(positions, dict):
        return positions
    if isinstance(positions, list):
        return {p.symbol: p for p in positions}
    return {}


def _order_to_broker_order(order: Any) -> Any:
    """Convert internal Order to BrokerOrder for IBKR submission."""
    from quantcore.core.types import Side
    from quantcore.execution.brokers.base import BrokerOrder

    side_str = "buy" if order.side == Side.BUY else "sell"
    order_type_str = order.type.value if hasattr(order.type, "value") else str(order.type).lower()
    return BrokerOrder(
        symbol=order.symbol,
        side=side_str,
        order_type=order_type_str,
        quantity=order.quantity,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
        time_in_force="day",
        client_order_id=order.id,
    )


def _submit_order(broker: Any, order: Any, broker_mode: str) -> str:
    """Submit an order through the broker, handling interface differences."""
    if broker_mode == "ibkr":
        bo = _order_to_broker_order(order)
        return broker.submit_order_sync(bo)
    else:
        return broker.submit_order(order)


def _update_broker_state(broker: Any, prices: dict[str, float], broker_mode: str) -> None:
    """Update broker market data and equity. IBKR tracks positions live but still
    updates the drawdown tracker for kill switch protection."""
    if broker_mode == "ibkr":
        # IBKR tracks positions/equity in real-time, but we still need drawdown tracking
        acct = _get_account_dict(broker)
        if hasattr(broker, "drawdown_tracker") and broker.drawdown_tracker is not None:
            broker.drawdown_tracker.update(float(acct["portfolio_value"]))
        return
    if hasattr(broker, "update_market_data"):
        broker.update_market_data(prices)
    if hasattr(broker, "update_equity"):
        broker.update_equity()


def _process_fills(broker: Any, broker_mode: str) -> list:
    """Process pending fills. Returns empty list for IBKR (event-driven)."""
    if broker_mode == "ibkr":
        return []  # fills handled via execDetailsEvent callback
    if hasattr(broker, "process_fills"):
        return broker.process_fills()
    return []

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
# Feed + broker factory
# ---------------------------------------------------------------------------


def create_feed_and_broker(
    data_source: str,
    broker_mode: str,
    symbols: list[str],
    risk_cfg: RiskConfig,
    logger: logging.Logger,
) -> tuple[Any, Any]:
    """Factory that creates the correct feed and broker based on configuration.

    Returns a tuple of (feed, broker).

    Configurations:
      alpaca/alpaca  -> AlpacaPaperFeed + PaperBroker
      alpaca/ibkr    -> AlpacaPaperFeed + IBKRBroker (recommended)
      ibkr/ibkr      -> IBKRFeed + IBKRBroker
      mock/alpaca    -> MockDataFeed + PaperBroker (testing)
      mock/ibkr      -> MockDataFeed + IBKRBroker (testing)
      mock/mock      -> MockDataFeed + PaperBroker (dry-run)
    """
    feed: Any = None
    broker: Any = None

    # ---- Data feed ----
    if data_source == "mock":
        feed = MockDataFeed(symbols=symbols, interval_sec=1.0)
        logger.info("Using mock data feed (synthetic OHLCV)")
    elif data_source == "alpaca":
        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed
        feed = AlpacaPaperFeed(symbols=symbols)
        logger.info("Using Alpaca paper data feed (REST polling) — symbols: %s", symbols)
    elif data_source == "ibkr":
        from quantcore.feeds.ibkr_feed import IBKRFeed
        feed = IBKRFeed(symbols=symbols)
        logger.info("Using IBKR data feed — symbols: %s", symbols)
    elif data_source == "yahoo":
        logger.error("Yahoo data feed not yet wired into paper_trade.py")
        raise ValueError("Yahoo data feed not available")
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # ---- Broker ----
    if broker_mode == "alpaca":
        from quantcore.live.paper import PaperBroker, PaperConfig
        from quantcore.risk.drawdown import DrawdownTracker

        paper_cfg = PaperConfig(initial_capital=100_000.0)  # capital set by caller
        broker = PaperBroker(config=paper_cfg)
        broker.drawdown_tracker = DrawdownTracker(config=risk_cfg.drawdown)
        logger.info("Using Alpaca paper broker")
    elif broker_mode == "ibkr":
        from quantcore.brokers.ibkr_broker import IBKRBroker
        from quantcore.risk.drawdown import DrawdownTracker

        broker = IBKRBroker()
        broker.connect_sync()
        broker.drawdown_tracker = DrawdownTracker(config=risk_cfg.drawdown)
        logger.info("Using IBKR broker — %s mode", "PAPER" if broker.is_paper else "LIVE")
    elif broker_mode == "mock":
        from quantcore.live.paper import PaperBroker, PaperConfig
        from quantcore.risk.drawdown import DrawdownTracker

        paper_cfg = PaperConfig(initial_capital=100_000.0)
        broker = PaperBroker(config=paper_cfg)
        broker.drawdown_tracker = DrawdownTracker(config=risk_cfg.drawdown)
        logger.info("Using mock paper broker")
    else:
        raise ValueError(f"Unknown broker mode: {broker_mode}")

    # Validate combination
    if broker_mode == "ibkr" and data_source == "alpaca":
        logger.info("Hybrid mode: Alpaca data + IBKR execution")

    return feed, broker


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_paper_trade(
    duration_seconds: int,
    strategy_name: str = "momentum",
    data_source: str = "mock",
    broker_mode: str = "mock",
    initial_capital: float = 100_000.0,
    resume: bool = False,
    risk_config: RiskConfig | None = None,
) -> int:
    """Run the paper trading loop for the specified duration. Returns exit code."""
    logger, log_path = setup_logging()
    csv_path, csv_writer, csv_file = setup_metrics_csv()
    session_store = SessionStore()
    risk_cfg = risk_config or load_risk_config()

    # ---- Preflight checks ----
    data_feed_ok = data_source in ("mock", "alpaca")
    resume_age_hours: float | None = None
    if resume and session_store.has_saved_state():
        saved = session_store.load()
        if saved and saved.timestamp:
            try:
                saved_dt = datetime.fromisoformat(saved.timestamp)
                resume_age_hours = (datetime.now(UTC) - saved_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

    preflight_failures = run_preflight_checks(
        risk_cfg,
        initial_capital=initial_capital,
        data_feed_ok=data_feed_ok,
        resume_state_age_hours=resume_age_hours,
    )
    if preflight_failures:
        logger.error("Preflight checks FAILED:")
        for f in preflight_failures:
            logger.error(f"  - {f}")
        logger.error("Aborting run. Fix the issues above or adjust config/risk.yaml")
        return 3
    logger.info("Preflight checks: all passed")

    # Restart recovery
    resume_state: SessionSnapshot | None = None
    if resume and session_store.has_saved_state():
        resume_state = session_store.load()
        if resume_state is not None:
            logger.info("Resuming from saved session state:")
            logger.info(f"  Saved at: {resume_state.timestamp}")
            logger.info(f"  Elapsed:  {resume_state.duration_elapsed_sec:.0f}s")
            logger.info(f"  P&L:      ${resume_state.total_pnl:,.2f}")
            logger.info(f"  PValue:   ${resume_state.portfolio_value:,.2f}")
            if resume_state.max_drawdown <= risk_cfg.preflight.max_startup_drawdown_pct * 100:
                logger.warning(
                    "  Resume drawdown %.2f%% exceeds startup limit %.0f%% — starting fresh",
                    abs(resume_state.max_drawdown),
                    abs(risk_cfg.preflight.max_startup_drawdown_pct * 100),
                )
                resume_state = None
        else:
            logger.warning("Could not parse saved state — starting fresh")

    logger.info("=" * 60)
    logger.info("Paper Trading Run Starting")
    logger.info(f"  Duration: {duration_seconds}s ({duration_seconds/3600:.1f}h)")
    logger.info(f"  Strategy: {strategy_name}")
    logger.info(f"  Data source: {data_source}")
    logger.info(f"  Broker: {broker_mode}")
    logger.info(f"  Initial capital: ${initial_capital:,.0f}")
    logger.info(f"  Resume: {'yes' if resume_state else 'no'}")
    logger.info("  Risk config: config/risk.yaml")
    logger.info(f"  Log file: {log_path}")
    logger.info(f"  Metrics CSV: {csv_path}")
    logger.info("=" * 60)

    # Initialize data feed and broker via factory
    symbols = ["SPY"] if data_source != "mock" else ["STK0000"]
    try:
        feed, broker = create_feed_and_broker(
            data_source=data_source,
            broker_mode=broker_mode,
            symbols=symbols,
            risk_cfg=risk_cfg,
            logger=logger,
        )
    except ValueError as e:
        logger.error(str(e))
        return 2
    except Exception as e:
        logger.error("Failed to initialize feed/broker: %s", e)
        return 2

    # Ensure capital is correct for PaperBroker
    if broker_mode != "ibkr" and hasattr(broker, "config"):
        broker.config.initial_capital = initial_capital

    # Initialize strategy
    strat_cls = _STRATEGIES.get(strategy_name)
    if strat_cls is None:
        logger.error(f"Unknown strategy: {strategy_name}. Available: {list(_STRATEGIES)}")
        return 2

    strategy = strat_cls(symbol=symbols[0])
    logger.info(f"Strategy initialized: {strategy_name}")

    # Initialize OMS and risk with YAML-configured limits
    from quantcore.execution.oms import OrderManager
    from quantcore.risk.limits import CircuitBreaker

    oms = OrderManager()
    circuit_breaker = CircuitBreaker(limits=risk_cfg.limits)
    logger.info("OMS + risk engine + circuit breaker initialized (limits from risk.yaml)")

    # Metrics state — initialize from resume state if available
    resume_offset = resume_state.duration_elapsed_sec if resume_state else 0.0
    start_time = time.time() - resume_offset
    orders_submitted = resume_state.orders_submitted if resume_state else 0
    orders_filled = resume_state.orders_filled if resume_state else 0
    orders_rejected = resume_state.orders_rejected if resume_state else 0
    kill_switch_active = resume_state.kill_switch_active if resume_state else False
    consecutive_rejections = 0
    daily_pnl_start = float(_get_account_dict(broker)["portfolio_value"])

    # Signal handlers
    def on_signal(signum, frame):
        nonlocal kill_switch_active
        logger.info(f"Received signal {signal.Signals(signum).name}. Triggering kill switch...")
        kill_switch_active = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    exit_code = 0
    last_metrics_time = time.time()
    last_state_save_time = time.time()
    market_paused_logged = False

    try:
        while time.time() - start_time < duration_seconds and not kill_switch_active:
            # --- Market hours check ---
            market_open = is_market_open(datetime.now(UTC))
            if data_source == "alpaca" and not market_open:
                if not market_paused_logged:
                    nxt = next_market_open(datetime.now(UTC))
                    logger.info(
                        "Market closed — pausing trading. Next open: %s (in %s)",
                        nxt.strftime("%Y-%m-%d %H:%M ET"),
                        str(nxt - datetime.now(UTC)).split(".")[0],
                    )
                    market_paused_logged = True
                time.sleep(risk_cfg.paper_execution.market_closed_sleep_sec)
                continue
            if market_open:
                market_paused_logged = False

            # --- IBKR reconnection ---
            if broker_mode == "ibkr" and not broker.is_connected:
                logger.warning("IBKR connection lost — attempting reconnect...")
                try:
                    broker.connect_sync()
                    logger.info("IBKR reconnected successfully")
                except Exception as e:
                    logger.error("IBKR reconnection failed: %s", e)
                    kill_switch_active = True
                    continue

            # Update market data
            events = feed.get_latest()
            prices = {e["symbol"]: float(e["price"]) for e in events}
            _update_broker_state(broker, prices, broker_mode)

            # Generate signals (not every tick — the strategy handles its own logic)
            signals = strategy.generate_signals(feed)

            for sig in signals:
                # --- Kill switch: max total rejections ---
                if orders_rejected >= risk_cfg.kill_switch.max_total_rejections:
                    logger.error(
                        "Kill switch: %d total rejections exceeded limit %d",
                        orders_rejected, risk_cfg.kill_switch.max_total_rejections,
                    )
                    kill_switch_active = True
                    break

                if consecutive_rejections >= risk_cfg.kill_switch.max_consecutive_rejections:
                    logger.error(
                        "Kill switch: %d consecutive rejections exceeded limit %d",
                        consecutive_rejections, risk_cfg.kill_switch.max_consecutive_rejections,
                    )
                    kill_switch_active = True
                    break

                # --- Kill switch: daily loss limit ---
                current_pnl = float(_get_account_dict(broker)["portfolio_value"]) - daily_pnl_start
                if current_pnl < risk_cfg.kill_switch.max_daily_loss:
                    logger.error(
                        "Kill switch: daily P&L ${:,.0f} below limit ${:,.0f}",
                        current_pnl, risk_cfg.kill_switch.max_daily_loss,
                    )
                    kill_switch_active = True
                    break

                if kill_switch_active:
                    break

                symbol = sig["symbol"]
                direction = sig["direction"]
                price = float(sig["price"])

                if direction == "long":
                    side = Side.BUY
                else:
                    side = Side.SELL

                from decimal import Decimal

                # Size from risk config
                equity = float(_get_account_dict(broker)["portfolio_value"])
                trade_pct = risk_cfg.paper_execution.trade_size_pct_equity
                min_qty = risk_cfg.paper_execution.min_trade_quantity
                qty = max(min_qty, int(equity * trade_pct / price))

                order = Order(
                    id=f"paper_{int(time.time() * 1e6)}_{symbol}",
                    symbol=symbol, side=side, type=OrderType.MARKET,
                    quantity=Decimal(str(qty)), strategy_id=strategy_name,
                )

                # Pre-trade risk check
                from decimal import Decimal as D

                acct_dict = _get_account_dict(broker)
                ps = PortfolioSnapshot(
                    cash=D(str(acct_dict["cash"])),
                    positions=_get_broker_positions(broker),
                    total_value=D(str(acct_dict["portfolio_value"])),
                )
                check_result = check_order(order, ps, limits=risk_cfg.limits)
                if check_result.status.value == "rejected":
                    logger.warning(
                        "Pre-trade rejected: %s (limit=%s, value=%.3f, threshold=%.3f)",
                        check_result.reason, check_result.limit_name,
                        check_result.current_value, check_result.limit_value,
                    )
                    orders_rejected += 1
                    consecutive_rejections += 1
                    # Check circuit breaker
                    if circuit_breaker.check([check_result]):
                        logger.error(
                            "Circuit breaker activated: %s", circuit_breaker.activation_reason,
                        )
                        kill_switch_active = True
                        break
                    continue

                _submit_order(broker, order, broker_mode)
                orders_submitted += 1
                consecutive_rejections = 0

            # Process fills
            fills = _process_fills(broker, broker_mode)
            orders_filled += len(fills)

            # Check drawdown
            dd_state = broker.drawdown_tracker
            if hasattr(dd_state, '_determine_action'):
                action = dd_state._determine_action(dd_state.current_drawdown)
                from quantcore.risk.drawdown import DrawdownAction
                if action in (DrawdownAction.HALT, DrawdownAction.FLATTEN):
                    logger.warning(f"Drawdown action triggered: {action}")
                    kill_switch_active = True

            # Periodic metrics
            now = time.time()
            metrics_interval = risk_cfg.paper_execution.metrics_interval_sec
            if now - last_metrics_time >= metrics_interval:
                acct = _get_account_dict(broker)
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

                # --- Kill switch: data staleness ---
                if (last_data_age is not None
                        and last_data_age > risk_cfg.kill_switch.max_data_staleness_sec):
                    logger.error(
                        "Kill switch: data stale for %.0fs (limit %ds)",
                        last_data_age, risk_cfg.kill_switch.max_data_staleness_sec,
                    )
                    kill_switch_active = True

                # --- Kill switch: memory usage ---
                if mem_rss is not None and mem_rss > risk_cfg.kill_switch.max_memory_mb:
                    logger.error(
                        "Kill switch: memory %.0fMB exceeds limit %dMB",
                        mem_rss, risk_cfg.kill_switch.max_memory_mb,
                    )
                    kill_switch_active = True

                row = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "total_pnl": round(float(acct["portfolio_value"]) - initial_capital, 2),
                    "unrealized_pnl": 0.0,
                    "realized_pnl": round(float(acct["total_commission"]), 2),
                    "open_positions_count": len(_get_broker_positions(broker)),
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
                    f"Mem: {mem_rss}MB | "
                    f"Market: {'OPEN' if market_open else 'CLOSED'}"
                )
                last_metrics_time = now

                # Save session state for restart recovery
                save_interval = risk_cfg.paper_execution.state_save_interval_sec
                if now - last_state_save_time >= save_interval:
                    broker_positions = _get_broker_positions(broker)
                    pos_dict = {s: float(p.quantity) for s, p in broker_positions.items()}
                    snapshot = SessionSnapshot(
                        total_pnl=float(acct["portfolio_value"]) - initial_capital,
                        portfolio_value=float(acct["portfolio_value"]),
                        cash=float(acct["cash"]),
                        positions=pos_dict,
                        orders_submitted=orders_submitted,
                        orders_filled=orders_filled,
                        orders_rejected=orders_rejected,
                        duration_elapsed_sec=time.time() - start_time,
                        max_drawdown=broker.drawdown_tracker.max_drawdown,
                        kill_switch_active=kill_switch_active,
                    )
                    session_store.save(snapshot)
                    last_state_save_time = now

            time.sleep(risk_cfg.paper_execution.poll_interval_sec)

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
    broker_positions = _get_broker_positions(broker)
    from decimal import Decimal as D
    for sym, pos in broker_positions.items():
        qty = float(pos.quantity)
        if abs(qty) < 1e-10:
            continue
        close_side = Side.SELL if qty > 0 else Side.BUY
        close_order = Order(
            id=f"close_{int(time.time() * 1e6)}_{sym}",
            symbol=sym, side=close_side, type=OrderType.MARKET,
            quantity=D(str(int(abs(qty)))), strategy_id="shutdown",
        )
        try:
            _submit_order(broker, close_order, broker_mode)
            _process_fills(broker, broker_mode)
        except Exception:
            pass

    # Final snapshot
    acct = _get_account_dict(broker)
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

    # Clear session state on clean completion
    if exit_code == 0:
        session_store.clear()
        logger.info("Session state cleared (clean completion)")

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
            "  python scripts/paper_trade.py --duration 30m --data-source mock --broker mock\n"
            "  python scripts/paper_trade.py --duration 5d --capital 500000\n"
            "  python scripts/paper_trade.py --duration 5d --resume\n"
            "  python scripts/paper_trade.py --duration 5d --data-source alpaca --broker ibkr\n"
        ),
    )
    parser.add_argument(
        "--duration", "-d", default=None,
        help="Run duration: Nd (days), Nh (hours), Nm (minutes), Ns (seconds). E.g., 5d, 12h, 30m",
    )
    parser.add_argument(
        "--strategy", "-s", default="momentum",
        choices=list(_STRATEGIES),
        help=f"Strategy to run (default: momentum). Available: {list(_STRATEGIES)}",
    )
    parser.add_argument(
        "--data-source", default="mock",
        choices=["mock", "alpaca", "ibkr", "yahoo"],
        help="Data feed source (default: mock). 'alpaca' = Alpaca REST, 'ibkr' = IBKR TWS.",
    )
    parser.add_argument(
        "--broker", default="mock",
        choices=["mock", "alpaca", "ibkr"],
        help="Broker for execution (default: mock). "
        "'ibkr' = Interactive Brokers, 'alpaca' = Alpaca.",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital in dollars (default: 100,000)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved session state",
    )
    parser.add_argument(
        "--risk-config", default="config/risk.yaml",
        help="Path to risk YAML configuration file (default: config/risk.yaml)",
    )
    parser.add_argument(
        "--verify-kill-switch", action="store_true",
        help="Verify kill switch mechanism and exit without trading",
    )

    args = parser.parse_args()

    # Kill switch verification mode
    if args.verify_kill_switch:
        ok, msg = validate_kill_switch_mechanism()
        print(f"Kill switch verification: {'PASS' if ok else 'FAIL'}")
        print(f"  {msg}")
        # Also check config loads cleanly
        try:
            cfg = load_risk_config(args.risk_config)
            print(f"  Risk config loaded: {args.risk_config}")
            print(f"    Limits: max_concentration={cfg.limits.max_concentration_pct:.0%}")
            print(f"    Circuit breaker: threshold={cfg.circuit_breaker_threshold_pct:.0%}")
            print(f"    Drawdown halt: {cfg.drawdown.halt_threshold:.0%}")
            print(f"    Kill switch triggers: "
                  f"{cfg.kill_switch.max_consecutive_rejections} rejections, "
                  f"{cfg.kill_switch.max_drawdown_pct:.0%} DD, "
                  f"{cfg.kill_switch.max_memory_mb}MB")
            preflight_result = run_preflight_checks(cfg, args.capital, data_feed_ok=True)
            print(f"  Preflight checks: {'PASS' if not preflight_result else 'FAIL'}")
        except Exception as e:
            print(f"  Risk config load FAILED: {e}")
            return 1
        return 0 if ok else 1

    if args.duration is None:
        print("Error: --duration/-d is required for paper trading runs", file=sys.stderr)
        return 2

    try:
        duration_sec = parse_duration(args.duration)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    risk_cfg = load_risk_config(args.risk_config)

    return run_paper_trade(
        duration_seconds=duration_sec,
        strategy_name=args.strategy,
        data_source=args.data_source,
        broker_mode=args.broker,
        initial_capital=args.capital,
        resume=args.resume,
        risk_config=risk_cfg,
    )


if __name__ == "__main__":
    sys.exit(main())
