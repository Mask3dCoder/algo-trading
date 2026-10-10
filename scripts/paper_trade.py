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
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Ensure quantcore is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import asyncio

# nest_asyncio is only needed when an event loop is ALREADY running (e.g., inside
# Jupyter). Applying it unconditionally monkeypatches asyncio process-globally at
# import time, which breaks anyio-based consumers (FastAPI TestClient) in any
# process that merely imports this module. Patch only when required.
try:
    asyncio.get_running_loop()

    import nest_asyncio  # noqa: E402

    nest_asyncio.apply()
except RuntimeError:
    pass  # No running loop — plain asyncio works; do not patch.

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

# Import framework strategy modules so @register decorators fire
import quantcore.strategies.equity.momentum  # noqa: F401, E402
import quantcore.strategies.equity.mean_reversion  # noqa: F401, E402
import quantcore.strategies.equity.trend_following  # noqa: F401, E402
import quantcore.strategies.equity.stat_arb  # noqa: F401, E402
import quantcore.strategies.equity.intraday_mean_reversion  # noqa: F401, E402
import quantcore.strategies.equity.intraday_momentum  # noqa: F401, E402
import quantcore.strategies.equity.intraday_trend  # noqa: F401, E402
import quantcore.strategies.equity.intraday_breakout  # noqa: F401, E402
import quantcore.strategies.macro.cta_trend  # noqa: F401, E402
import quantcore.strategies.macro.carry  # noqa: F401, E402
import quantcore.strategies.macro.risk_parity  # noqa: F401, E402
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401, E402
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401, E402
import quantcore.strategies.macro.state_selective_momentum  # noqa: F401, E402
import quantcore.strategies.macro.factor_residual  # noqa: F401, E402
import quantcore.strategies.macro.factor_residual_qp  # noqa: F401, E402
import quantcore.strategies.macro.factor_residual_strict  # noqa: F401, E402
import quantcore.strategies.macro.factor_residual_v3  # noqa: F401, E402
import quantcore.strategies.macro.carry_trend_filter  # noqa: F401, E402
import quantcore.strategies.macro.regime_adaptive_momentum  # noqa: F401, E402
import quantcore.strategies.macro.true_carry_strategy  # noqa: F401, E402
import quantcore.strategies.macro.simple_test_strategy  # noqa: F401, E402
import quantcore.strategies.equity.earnings_momentum  # noqa: F401, E402
import quantcore.strategies.equity.low_volatility  # noqa: F401, E402
import quantcore.strategies.equity.multifactor_composite  # noqa: F401, E402
import quantcore.strategies.equity.quality_factor  # noqa: F401, E402
import quantcore.strategies.equity.multi_factor  # noqa: F401, E402
import quantcore.strategies.equity.quality_factor_rmw_cma  # noqa: F401, E402
import quantcore.strategies.equity.tsmom  # noqa: F401, E402
import quantcore.strategies.equity.value_factor_hml  # noqa: F401, E402
import quantcore.strategies.equity.cross_sectional_momentum  # noqa: F401, E402
import quantcore.strategies.equity.volume_profile  # noqa: F401, E402
import quantcore.strategies.equity.volatility_breakout  # noqa: F401, E402
import quantcore.strategies.equity.value_factor  # noqa: F401, E402
import quantcore.strategies.options.delta_neutral  # noqa: F401, E402
import quantcore.strategies.options.earnings_vol  # noqa: F401, E402
import quantcore.strategies.options.iron_condor  # noqa: F401, E402
import quantcore.strategies.options.tail_hedge  # noqa: F401, E402
import quantcore.strategies.macro.fred_regime_strategy  # noqa: F401, E402

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
    pv = float(getattr(acct, "portfolio_value", 0.0) or 0.0)
    cash = float(getattr(acct, "cash", 0.0) or 0.0)
    return {
        "cash": cash,
        "equity": float(getattr(acct, "equity", pv) or pv),
        "portfolio_value": pv,
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


def _submit_order(
    broker: Any,
    order: Any,
    broker_mode: str,
    stop_loss: Decimal | None = None,
    take_profit: Decimal | None = None,
) -> str:
    """Submit an order through the broker, handling interface differences.

    If stop_loss or take_profit are provided, uses a bracket order
    (parent MARKET + child STOP/LIMIT linked via OCA group).
    """
    use_bracket = (
        broker_mode == "ibkr"
        and (
            (stop_loss is not None and stop_loss > 0)
            or (take_profit is not None and take_profit > 0)
        )
    )
    if broker_mode == "ibkr":
        bo = _order_to_broker_order(order)
        if stop_loss is not None and stop_loss > 0:
            bo.stop_loss_price = stop_loss
        if take_profit is not None and take_profit > 0:
            bo.take_profit_price = take_profit
        if use_bracket and hasattr(broker, "submit_bracket_order_sync"):
            return broker.submit_bracket_order_sync(bo)
        return broker.submit_order_sync(bo)
    else:
        # Mock/paper broker: bracket orders not supported, submit plain order
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


def _process_fills(broker: Any, broker_mode: str, pending_fills: list | None = None) -> list:
    """Process pending fills. Drains async fill queue for IBKR, sync for others."""
    if broker_mode == "ibkr":
        if pending_fills is not None:
            fills = list(pending_fills)
            pending_fills.clear()
            return fills
        return []
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


class IntradayDataFeed:
    """Provides intraday bars from Alpaca IEX via AlpacaIntradayLoader.

    Uses load_snapshot() to get the latest available bar per ticker.
    Falls back to mock data outside market hours.
    """

    def __init__(
        self,
        symbols: list[str],
        resample_minutes: int = 5,
    ):
        self.symbols = symbols
        self.resample_minutes = resample_minutes
        self._last_tick: datetime | None = None
        self._loader = None
        self._mock_fallback = MockDataFeed(symbols=symbols, interval_sec=1.0)

        try:
            from quantcore.data.intraday_loader import AlpacaIntradayLoader
            self._loader = AlpacaIntradayLoader()
        except Exception:
            pass

    def is_market_open(self) -> bool:
        """Check if US market is open (9:30-16:00 ET, weekdays)."""
        from datetime import timezone
        import pytz
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def get_latest(self) -> list[dict[str, Any]]:
        """Get latest intraday bars for all symbols."""
        if self._loader is None:
            return self._mock_fallback.get_latest()

        now = datetime.now(UTC)
        self._last_tick = now

        try:
            snapshot = self._loader.load_snapshot(self.symbols, now)
        except Exception:
            return self._mock_fallback.get_latest()

        if snapshot is None or snapshot.empty:
            return self._mock_fallback.get_latest()

        events = []
        for _, row in snapshot.iterrows():
            events.append({
                "symbol": str(row["symbol"]),
                "price": float(row["close"]),
                "timestamp": row.get("timestamp", now),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "open": float(row["open"]),
                "volume": int(row.get("volume", 0)),
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
        self._in_position: bool = False
        self._warm: bool = False
        self._prev_fast: float = 0.0
        self._prev_slow: float = 0.0

    def generate_signals(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        signals = []

        for event in events:
            if event["symbol"] != self.symbol:
                continue
            price = float(event["price"])
            self._price_history.append(price)

            if len(self._price_history) < self.slow:
                continue

            fast_ma = float(np.mean(self._price_history[-self.fast:]))
            slow_ma = float(np.mean(self._price_history[-self.slow:]))

            if not self._warm:
                self._prev_fast = fast_ma
                self._prev_slow = slow_ma
                self._warm = True
                if fast_ma > slow_ma and not self._in_position:
                    signals.append({
                        "symbol": self.symbol, "direction": "long",
                        "price": price,
                        "strength": abs(fast_ma - slow_ma) / slow_ma,
                    })
                    self._in_position = True
                continue

            # Golden cross
            if self._prev_fast <= self._prev_slow and fast_ma > slow_ma:
                if not self._in_position:
                    signals.append({
                        "symbol": self.symbol, "direction": "long",
                        "price": price,
                        "strength": abs(fast_ma - slow_ma) / slow_ma,
                    })
                    self._in_position = True
            # Death cross
            elif self._prev_fast >= self._prev_slow and fast_ma < slow_ma:
                if self._in_position:
                    signals.append({
                        "symbol": self.symbol, "direction": "short",
                        "price": price,
                        "strength": abs(fast_ma - slow_ma) / slow_ma,
                    })
                    self._in_position = False

            self._prev_fast = fast_ma
            self._prev_slow = slow_ma

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

# Framework strategies available via --framework flag.
# Keys must match registry names from @register decorators.
_FRAMEWORK_STRATEGIES: set[str] = {
    # Equity
    "equity_mean_reversion",
    "equity_momentum",
    "equity_trend_following",
    "equity_stat_arb",
    "equity_earnings_momentum",
    "equity_cross_sectional_momentum",
    "equity_low_volatility",
    "equity_multifactor_composite",
    "equity_quality_factor",
    "equity_multi_factor",
    "equity_quality_rmw_cma",
    "equity_tsmom",
    "equity_value_hml",
    "equity_volume_profile",
    "equity_volatility_breakout",
    "equity_value_factor",
    # Intraday
    "intraday_mean_reversion",
    "intraday_momentum",
    "intraday_trend",
    "intraday_breakout",
    # Macro
    "macro_cta_trend",
    "macro_carry",
    "macro_carry_trend_filter",
    "macro_risk_parity",
    "macro_cross_sectional_momentum",
    "macro_cross_sectional_factor_neutral",
    "macro_state_selective_momentum",
    "macro_factor_residual",
    "macro_factor_residual_qp",
    "macro_factor_residual_strict",
    "macro_factor_residual_v3",
    "macro_regime_adaptive_momentum",
    "macro_true_carry",
    "macro_fred_regime",
    "simple_buy_hold",
    # Options
    "options_delta_neutral",
    "options_earnings_vol",
    "options_iron_condor",
    "options_tail_hedge",
}


# ---------------------------------------------------------------------------
# Historical data preloading (for strategy warmup)
# ---------------------------------------------------------------------------


def _preload_historical_bars(
    symbols: list[str],
    lookback_minutes: int = 120,
) -> list[dict[str, Any]]:
    """Fetch historical 1-min bars to warm up strategies.

    Tries Alpaca first; falls back to yfinance (free, but may be delayed).
    Returns a flat list of bar dicts sorted by timestamp, or empty list on
    failure.
    """
    # Try Alpaca first
    bars = _preload_alpaca_bars(symbols, lookback_minutes)
    if bars:
        return bars

    # Fall back to yfinance
    bars = _preload_yfinance_bars(symbols, lookback_minutes)
    if bars:
        return bars

    return []


def _preload_alpaca_bars(
    symbols: list[str], lookback_minutes: int,
) -> list[dict[str, Any]]:
    """Fetch historical bars from Alpaca (requires paid data subscription)."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        return []

    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        return []

    try:
        client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        end = datetime.now(UTC)
        start = end - timedelta(minutes=lookback_minutes)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        response = client.get_stock_bars(request)
    except Exception:
        return []

    bars: list[dict[str, Any]] = []
    bars_by_symbol = response.data if hasattr(response, "data") else response
    for sym in symbols:
        sym_bars = bars_by_symbol.get(sym, [])
        for bar in sym_bars:
            bars.append({
                "symbol": sym,
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "price": float(bar.close),
            })

    bars.sort(key=lambda b: b["timestamp"] if b["timestamp"] else datetime.min.replace(tzinfo=UTC))
    return bars


def _preload_yfinance_bars(
    symbols: list[str], lookback_minutes: int,
) -> list[dict[str, Any]]:
    """Fetch historical bars from yfinance (free, 1-min granularity).

    yfinance provides up to 7 days of 1-min data. Bars may be delayed up to
    15 minutes during market hours.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    period_map = {
        60: "1d",
        120: "5d",
        240: "5d",
        480: "5d",
    }
    period = period_map.get(lookback_minutes, "5d")

    bars: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval="1m")
            if df.empty:
                continue
            for ts, row in df.iterrows():
                bars.append({
                    "symbol": sym,
                    "timestamp": ts.to_pydatetime().replace(tzinfo=UTC),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "price": float(row["Close"]),
                })
        except Exception:
            continue

    bars.sort(key=lambda b: b["timestamp"] if b["timestamp"] else datetime.min.replace(tzinfo=UTC))
    return bars


# ---------------------------------------------------------------------------
# Feed + broker factory
# ---------------------------------------------------------------------------


def create_feed_and_broker(
    data_source: str,
    broker_mode: str,
    symbols: list[str],
    risk_cfg: RiskConfig,
    logger: logging.Logger,
    live_mode: bool = False,
    resample_minutes: int = 1,
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
    elif data_source == "alpaca_ws":
        from quantcore.feeds.alpaca_feed import LiveAlpacaFeed
        feed = LiveAlpacaFeed(symbols=symbols)
        feed.connect()
        logger.info("Using Alpaca live WebSocket feed — symbols: %s", symbols)
    elif data_source == "ibkr":
        from quantcore.feeds.ibkr_feed import IBKRFeed
        feed = IBKRFeed(symbols=symbols)
        logger.info("Using IBKR data feed — symbols: %s", symbols)
    elif data_source == "intraday":
        feed = IntradayDataFeed(symbols=symbols, resample_minutes=resample_minutes)
        logger.info("Using intraday data feed (Alpaca IEX) — symbols: %s", symbols)
    elif data_source == "yahoo":
        logger.error("Yahoo data feed not yet wired into paper_trade.py")
        raise ValueError("Yahoo data feed not available")
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # ---- Broker ----
    if broker_mode == "alpaca":
        if live_mode:
            from quantcore.brokers.alpaca_broker import AlpacaBroker
            from quantcore.risk.drawdown import DrawdownTracker

            broker = AlpacaBroker(mode="live")
            broker.drawdown_tracker = DrawdownTracker(config=risk_cfg.drawdown)
            logger.warning("Using Alpaca LIVE broker — real money orders")
        else:
            from quantcore.live.paper import PaperBroker, PaperConfig
            from quantcore.risk.drawdown import DrawdownTracker

            paper_cfg = PaperConfig(initial_capital=100_000.0)  # capital set by caller
            broker = PaperBroker(config=paper_cfg)
            broker.drawdown_tracker = DrawdownTracker(config=risk_cfg.drawdown)
            logger.info("Using Alpaca paper broker")
    elif broker_mode == "ibkr":
        from quantcore.brokers.ibkr_broker import IBKRBroker
        from quantcore.risk.drawdown import DrawdownTracker

        broker = IBKRBroker(timeout=30.0)
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

    # ---- Smart execution router (env-var activated) ----
    if os.environ.get("QUANTCORE_SMART_EXECUTION") == "1":
        from quantcore.execution.smart_router import SmartExecutionRouter, SmartRouterConfig
        cost_report = os.environ.get(
            "QUANTCORE_COST_REPORT",
            str(Path(__file__).resolve().parent.parent / "reports" / "execution_costs.csv"),
        )
        router_cfg = SmartRouterConfig(cost_report_path=cost_report)
        broker = SmartExecutionRouter(inner_broker=broker, config=router_cfg)
        logger.info("SmartExecutionRouter active — cost report: %s", cost_report)

    # Validate combination
    if broker_mode == "ibkr" and data_source in ("alpaca", "alpaca_ws"):
        logger.info("Hybrid mode: Alpaca%s + IBKR execution", " WebSocket" if data_source == "alpaca_ws" else " REST")

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
    framework: str | None = None,
    symbols: list[str] | None = None,
    framework_params: str | None = None,
    resample_minutes: int = 1,
    live_mode: bool = False,
    scale_mode: str = "full",
    session_id: str = "",
    factory_config: str | None = None,
) -> int:
    """Run the paper trading loop for the specified duration. Returns exit code.

    Parameters
    ----------
    framework:
        If set, loads a framework strategy from the registry (e.g.
        ``equity_mean_reversion``) and wraps it in ``LiveStrategyAdapter``.
        Overrides the built-in ``strategy_name``.
    factory_config:
        If set, loads a ConfigurableStrategy from a YAML config file in
        ``quantcore/strategy_factory/configs/``. Overrides both ``framework``
        and ``strategy_name``.
    symbols:
        List of symbols to trade. Defaults to ``["SPY"]`` for live data
        or ``["STK0000"]`` for mock.
    framework_params:
        JSON dict of strategy params to override defaults (e.g.
        ``{"rsi_window": 10, "risk_budget_pct": 0.25}``).
    """
    logger, log_path = setup_logging()
    csv_path, csv_writer, csv_file = setup_metrics_csv()
    session_store = SessionStore(session_id=session_id)
    risk_cfg = risk_config or load_risk_config()

    # ---- Preflight checks ----
    data_feed_ok = data_source in ("mock", "alpaca", "alpaca_ws", "intraday")
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

    # ---- Resolve symbols ----
    if symbols is None:
        symbols = ["SPY"] if data_source != "mock" else ["STK0000"]

    logger.info("=" * 60)
    if live_mode:
        logger.warning("*** LIVE TRADING MODE — REAL MONEY ***")
    logger.info("Paper Trading Run Starting")
    logger.info(f"  Duration: {duration_seconds}s ({duration_seconds/3600:.1f}h)")
    if framework:
        logger.info(f"  Strategy: {framework} (framework) on {symbols}")
    else:
        logger.info(f"  Strategy: {strategy_name}")
    logger.info(f"  Data source: {data_source}")
    logger.info(f"  Broker: {broker_mode}")
    logger.info(f"  Initial capital: ${initial_capital:,.0f}")
    logger.info(f"  Scale mode: {scale_mode}")
    logger.info(f"  Resume: {'yes' if resume_state else 'no'}")
    logger.info("  Risk config: config/risk.yaml")
    logger.info(f"  Log file: {log_path}")
    logger.info(f"  Metrics CSV: {csv_path}")
    logger.info("=" * 60)

    # Initialize data feed and broker via factory
    try:
        feed, broker = create_feed_and_broker(
            data_source=data_source,
            broker_mode=broker_mode,
            symbols=symbols,
            risk_cfg=risk_cfg,
            logger=logger,
            live_mode=live_mode,
            resample_minutes=resample_minutes,
        )
    except ValueError as e:
        logger.error(str(e))
        return 2
    except Exception as e:
        logger.error("Failed to initialize feed/broker: %s", e)
        return 2

    # Auto-detect capital from broker for live modes
    if broker_mode == "ibkr":
        acct_dict = _get_account_dict(broker)
        detected_pv = acct_dict.get("portfolio_value", 0.0)
        if detected_pv > 0:
            initial_capital = detected_pv
            logger.info(f"  Auto-detected capital from broker: ${initial_capital:,.0f}")
    elif hasattr(broker, "config"):
        broker.config.initial_capital = initial_capital

    # Initialize strategy
    is_factory = bool(factory_config)
    is_framework = bool(framework) and not is_factory
    if is_factory:
        from quantcore.backtest.adapter import LiveStrategyAdapter
        from quantcore.strategy_factory.configurable_strategy import (
            ConfigurableStrategy,
            load_factory_config_by_name,
        )

        try:
            fc = load_factory_config_by_name(factory_config)
        except FileNotFoundError as e:
            logger.error("Factory config not found: %s", e)
            return 2

        strat = ConfigurableStrategy(fc)
        if not symbols:
            symbols = ["SPY"]
        strategy = LiveStrategyAdapter(strat, symbols, resample_minutes=resample_minutes)
        logger.info(
            "Factory strategy initialized: %s (signal=%s) on %s",
            fc.name, fc.signal.formula, symbols,
        )

        # Preload historical bars for warmup
        if data_source in ("alpaca", "ibkr"):
            logger.info("Preloading historical bars for strategy warmup...")
            historical = _preload_historical_bars(symbols, lookback_minutes=120)
            if historical:
                strategy.warmup(historical)
                logger.info(
                    "Warmed up with %d historical bars across %d symbols",
                    len(historical), len(symbols),
                )
            else:
                logger.warning(
                    "No historical bars loaded — strategy starts cold "
                    "(may need ~21 bars to generate first signal)"
                )
    elif is_framework:
        if framework not in _FRAMEWORK_STRATEGIES:
            logger.error(
                "Unknown framework strategy: %s. Available: %s",
                framework, sorted(_FRAMEWORK_STRATEGIES),
            )
            return 2
        from quantcore.backtest.adapter import LiveStrategyAdapter
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.registry import create as create_fw

        stype = StrategyType.EQUITY
        if framework.startswith("macro_"):
            stype = StrategyType.MACRO
        elif framework.startswith("options_"):
            stype = StrategyType.OPTIONS

        import json

        fw_params: dict[str, Any] = {}
        if framework_params:
            try:
                fw_params = json.loads(framework_params)
            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse --fw-params '%s', using defaults",
                    framework_params,
                )

        config = StrategyConfig(
            name=framework,
            strategy_type=stype,
            max_position_notional=initial_capital,
            risk_budget_pct=float(
                fw_params.pop("risk_budget_pct", 0.25)
            ),
            params=fw_params,
        )
        fw = create_fw(framework, config)
        strategy = LiveStrategyAdapter(fw, symbols, resample_minutes=resample_minutes)
        logger.info("Framework strategy initialized: %s on %s", framework, symbols)

        # Preload historical bars for warmup
        if data_source in ("alpaca", "ibkr"):
            logger.info("Preloading historical bars for strategy warmup...")
            historical = _preload_historical_bars(symbols, lookback_minutes=120)
            if historical:
                strategy.warmup(historical)
                logger.info(
                    "Warmed up with %d historical bars across %d symbols",
                    len(historical), len(symbols),
                )
            else:
                logger.warning(
                    "No historical bars loaded — strategy starts cold "
                    "(may need ~21 bars to generate first signal)"
                )
    else:
        strat_cls = _STRATEGIES.get(strategy_name)
        if strat_cls is None:
            logger.error(
                "Unknown strategy: %s. Available: %s",
                strategy_name, list(_STRATEGIES),
            )
            return 2
        strategy = strat_cls(symbol=symbols[0])
        logger.info("Strategy initialized: %s", strategy_name)

    # Wire up IBKR fill tracking via event callback
    ibkr_pending_fills: list = []
    if broker_mode == "ibkr" and hasattr(broker, "stream_fills"):
        broker.stream_fills(lambda f: ibkr_pending_fills.append(f))
        logger.info("IBKR fill callback registered")

    # Initialize OMS and risk with YAML-configured limits
    from quantcore.execution.oms import OrderManager
    from quantcore.risk.limits import CircuitBreaker

    oms = OrderManager()
    circuit_breaker = CircuitBreaker(limits=risk_cfg.limits)
    logger.info("OMS + risk engine + circuit breaker initialized (limits from risk.yaml)")

    # Snapshot pre-existing positions so shutdown only closes what we opened
    _initial_positions: set[str] = {
        sym for sym, pos in _get_broker_positions(broker).items()
        if abs(float(pos.quantity)) > 1e-10
    }

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
            if data_source in ("alpaca", "alpaca_ws") and not market_open:
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

            # Generate signals from the same events used for price updates
            signals = strategy.generate_signals(events)

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

                # Position sizing: use strategy-computed notional if provided,
                # otherwise fall back to fixed percentage of equity.
                notional = sig.get("notional_size")
                price = float(sig["price"])
                equity = float(_get_account_dict(broker)["portfolio_value"])

                if notional and notional > 0:
                    qty = max(1, int(notional / price))
                else:
                    trade_pct = risk_cfg.paper_execution.trade_size_pct_equity
                    min_qty = risk_cfg.paper_execution.min_trade_quantity
                    qty = max(min_qty, int(equity * trade_pct / price))

                # Scale mode override
                if scale_mode == "single_share":
                    qty = 1
                elif scale_mode == "fractional":
                    qty = max(1, qty // 10)

                order = Order(
                    id=f"paper_{int(time.time() * 1e6)}_{symbol}",
                    symbol=symbol, side=side, type=OrderType.MARKET,
                    quantity=Decimal(str(qty)), strategy_id=strategy_name,
                    limit_price=Decimal(str(round(price, 4))),
                )

                # Pre-trade risk check
                from decimal import Decimal as D

                acct_dict = _get_account_dict(broker)
                ps = PortfolioSnapshot(
                    cash=D(str(max(acct_dict["cash"], 0.0))),
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

                # Extract stop-loss / take-profit from signal for bracket orders
                sl = sig.get("stop_loss")
                tp = sig.get("take_profit")
                sl_dec = Decimal(str(sl)) if sl else None
                tp_dec = Decimal(str(tp)) if tp else None

                try:
                    _submit_order(broker, order, broker_mode,
                                  stop_loss=sl_dec, take_profit=tp_dec)
                    orders_submitted += 1
                    consecutive_rejections = 0
                except Exception as e:
                    logger.warning("Order submission failed: %s", e)
                    consecutive_rejections += 1
                    if consecutive_rejections >= 5:
                        logger.error(
                            "5 consecutive order failures — activating circuit breaker"
                        )
                        kill_switch_active = True
                        break

            # Process fills
            fills = _process_fills(broker, broker_mode, ibkr_pending_fills)
            orders_filled += len(fills)
            for fill in fills:
                try:
                    strategy.on_fill(fill)
                except Exception:
                    pass

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

    # Disconnect WebSocket feed if active
    if data_source == "alpaca_ws" and hasattr(feed, 'disconnect'):
        try:
            feed.disconnect()
        except Exception:
            pass

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

    # Close only positions opened during this run (skip pre-existing)
    broker_positions = _get_broker_positions(broker)
    from decimal import Decimal as D
    for sym, pos in broker_positions.items():
        qty = float(pos.quantity)
        if abs(qty) < 1e-10:
            continue
        if sym in _initial_positions:
            continue  # Don't touch positions that were already there
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
            "  python scripts/paper_trade.py --duration 5d --factory-config carry\n"
            "  python scripts/paper_trade.py --duration 5d --factory-config time_series_momentum\n"
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
        choices=["mock", "alpaca", "alpaca_ws", "ibkr", "yahoo", "intraday"],
        help="Data feed source (default: mock). 'alpaca' = Alpaca REST, 'alpaca_ws' = Alpaca WebSocket. 'intraday' = Alpaca IEX intraday bars.",
    )
    parser.add_argument(
        "--broker", default="mock",
        choices=["mock", "alpaca", "ibkr"],
        help="Broker for execution (default: mock). "
        "'ibkr' = Interactive Brokers, 'alpaca' = Alpaca.",
    )
    parser.add_argument(
        "--capital", type=float, default=135_000.0,
        help="Initial capital in dollars (default: 135,000)",
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
    parser.add_argument(
        "--framework", default=None,
        help=(
            "Framework strategy to use instead of built-in. "
            "E.g., equity_mean_reversion, equity_momentum, equity_trend_following. "
            "Available: "
            + ", ".join(sorted(_FRAMEWORK_STRATEGIES))
        ),
    )
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols to trade (default: SPY). E.g., SPY,QQQ,AAPL",
    )
    parser.add_argument(
        "--fw-params", default=None,
        help=(
            "JSON dict of framework strategy params to override defaults. "
            'E.g., \'{"risk_budget_pct": 0.25, "rsi_window": 10}\''
        ),
    )
    parser.add_argument(
        "--resample", type=int, default=None,
        help=(
            "Resample 1-min bars to higher timeframe before passing to strategy. "
            "E.g., 5 for 5-min bars, 15 for 15-min bars."
        ),
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Switch to LIVE trading with real money. Use --broker to select: "
        "'alpaca' (requires ALPACA_API_KEY/ALPACA_API_SECRET) or "
        "'ibkr' (requires IBKR_HOST/IBKR_PORT/IBKR_CLIENT_ID).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive 'Type YES' confirmation (for non-interactive launches).",
    )
    parser.add_argument(
        "--scale", default="full", choices=["single_share", "fractional", "full"],
        help="Position scaling mode: single_share (1 share per order), "
        "fractional (10%% of normal size), full (100%% strategy signals). Default: full.",
    )
    parser.add_argument(
        "--session-id", default="",
        help="Unique session identifier for state file isolation. "
        "Each strategy should use a different ID to avoid file conflicts.",
    )
    parser.add_argument(
        "--factory-config", default=None,
        help=(
            "Strategy factory YAML config name (without .yaml extension). "
            "Loads a ConfigurableStrategy from quantcore/strategy_factory/configs/. "
            "E.g., 'time_series_momentum', 'carry', 'pairs_trading'. "
            "Overrides --framework and --strategy."
        ),
    )

    args = parser.parse_args()

    # ---- Live trading safety gate ----
    if args.live:
        _broker_display = {
            "alpaca": "Alpaca LIVE (ALPACA_API_KEY / ALPACA_API_SECRET)",
            "ibkr": "Interactive Brokers LIVE (IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID)",
        }
        _broker_name = _broker_display.get(args.broker, args.broker.upper())
        print("=" * 60)
        print("  *** WARNING: LIVE TRADING WITH REAL MONEY ***")
        print("=" * 60)
        print(f"  Broker:   {_broker_name}")
        print(f"  Scale:    {args.scale}")
        print(f"  Capital:  ${args.capital:,.0f}")
        print(f"  Duration: {args.duration}")
        print(f"  Strategy: {args.factory_config or args.framework or args.strategy}")
        print()
        if args.broker == "ibkr":
            port = int(os.environ.get("IBKR_PORT", "7497"))
            _mode = {7497: "TWS Paper", 7496: "TWS Live", 4002: "Gateway Paper", 4001: "Gateway Live"}.get(port, f"port {port}")
            print(f"  IBKR connection: {_mode} (port {port})")
            if port in (7497, 4002):
                print("  WARNING: IBKR port is a PAPER trading port. Set IBKR_PORT=7496 or 4001 for live.")
        else:
            print("  Real orders will be submitted to Alpaca's live trading API.")
        print("  You may lose money. Confirm you accept this risk.")
        print()

        if args.yes:
            print("  --yes flag set. Proceeding with live trading.")
        elif not sys.stdin.isatty():
            print("  Non-interactive mode detected (no tty). Proceeding with live trading.")
        else:
            resp = input("  Type 'YES' to proceed with live trading: ").strip()
            if resp != "YES":
                print("Aborted. No live orders sent.")
                return 0

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

    # Parse symbols list
    symbols_list: list[str] | None = None
    if args.symbols:
        symbols_list = [s.strip().upper() for s in args.symbols.split(",")]

    return run_paper_trade(
        duration_seconds=duration_sec,
        strategy_name=args.strategy,
        data_source=args.data_source,
        broker_mode=args.broker,
        initial_capital=args.capital,
        resume=args.resume,
        risk_config=risk_cfg,
        framework=args.framework,
        symbols=symbols_list,
        framework_params=args.fw_params,
        resample_minutes=args.resample or 1,
        live_mode=args.live,
        scale_mode=args.scale,
        session_id=args.session_id,
        factory_config=args.factory_config,
    )


if __name__ == "__main__":
    sys.exit(main())
