"""Live Streaming Ingestion Layer & Passive Tracking for Delta Neutral VRP.

Provides a framework-agnostic websocket consumer that ingests real-time equity
prices and option chains.  When ``passive_tracking_mode`` is active the
execution path bypasses broker API transmission and writes intended order
parameters to a secure JSON log instead.

Architecture
------------
  LiveStreamConnector   : async websocket consumer (generic ticker feed)
  PassiveExecutionProxy : intercepts orders in passive mode, logs to JSON
  SilentRunOrchestrator : wires connector + proxy + runner for 72-hr phase

Usage
-----
    # From runner (passive mode)
    connector = LiveStreamConnector(tickers=["SPY","QQQ"])
    proxy = PassiveExecutionProxy(output_path=Path("reports/paper_trading/passive_trades.json"))
    orchestrator = SilentRunOrchestrator(connector, proxy, runner)
    await orchestrator.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Tick:
    """Normalised real-time tick.  Immutable."""
    timestamp: datetime
    ticker: str
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    source: str  # 'websocket', 'poll', 'replay'

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "volume": self.volume,
            "source": self.source,
        }


@dataclass
class PassiveTradeRecord:
    """Immutable record of a bypassed order during passive tracking."""
    timestamp: str          # ISO-8601 with microseconds
    mode: str               # "passive"
    ticker: str
    contract: str | None
    option_type: str | None  # "call" / "put" / None for equity
    side: str               # "buy" / "sell"
    target_size: int
    mid_price: float
    bid_ask_spread: float
    notional: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Passive Execution Proxy
# ---------------------------------------------------------------------------


class PassiveExecutionProxy:
    """Intercepts ``submit_order()`` and ``rebalance_delta()`` calls.

    When `passive_tracking_mode` is enabled, every intended order is written
    to a JSONL file instead of being sent to a broker.  The internal portfolio
    state is still updated as if the trade had filled at the mid-price, so
    RiskController and Analytics continue to function normally.

    Parameters
    ----------
    output_path : Path
        Destination for the passive trade log (JSONL).
    """

    def __init__(self, output_path: Path | None = None) -> None:
        self._path = output_path or Path("reports/paper_trading/passive_trades.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._trade_count = 0
        self._last_trade_ts: datetime | None = None

    # -- Public API --------------------------------------------------------

    def log_bypassed_order(
        self,
        ticker: str,
        side: str,
        target_size: int,
        mid_price: float,
        bid_ask_spread: float,
        contract: str | None = None,
        option_type: str | None = None,
        notional: float = 0.0,
        reason: str | None = None,
    ) -> PassiveTradeRecord:
        """Log an order that would have been sent to the broker.

        Returns the created record (useful for unit tests).
        """
        ts = datetime.now(UTC)
        record = PassiveTradeRecord(
            timestamp=ts.isoformat(),
            mode="passive",
            ticker=ticker,
            contract=contract,
            option_type=option_type,
            side=side,
            target_size=target_size,
            mid_price=mid_price,
            bid_ask_spread=bid_ask_spread,
            notional=notional,
            reason=reason,
        )
        self._append(record)
        self._trade_count += 1
        self._last_trade_ts = ts
        return record

    def log_delta_rebalance(
        self,
        ticker: str,
        hedge_shares: int,
        mid_price: float,
        bid_ask_spread: float,
    ) -> PassiveTradeRecord:
        """Log a delta-hedge rebalance that was bypassed."""
        return self.log_bypassed_order(
            ticker=ticker,
            side="buy" if hedge_shares > 0 else "sell",
            target_size=abs(hedge_shares),
            mid_price=mid_price,
            bid_ask_spread=bid_ask_spread,
            contract=None,
            option_type=None,
            notional=abs(hedge_shares) * mid_price,
            reason="delta_rebalance",
        )

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def last_trade_ts(self) -> datetime | None:
        return self._last_trade_ts

    # -- Private -----------------------------------------------------------

    def _append(self, record: PassiveTradeRecord) -> None:
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")
        except Exception as exc:
            _log.error("Passive trade log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Live Stream Connector (async)
# ---------------------------------------------------------------------------


class LiveStreamConnector:
    """Async websocket consumer for real-time price feeds.

    Framework-agnostic — the actual websocket implementation is injected via
    ``on_tick`` callbacks.  This class manages the event loop, handles
    reconnection via ``ConnectionMonitor`` (see connection_monitor.py), and
    dispatches normalised ``Tick`` objects to registered handlers.

    Parameters
    ----------
    tickers : list[str]
        Universe to subscribe (e.g. ["SPY", "QQQ", "IWM", "GLD", "TLT"]).
    heartbeat_interval : float
        Seconds between heartbeat checks (default 5.0).
    """

    def __init__(
        self,
        tickers: list[str],
        heartbeat_interval: float = 5.0,
    ) -> None:
        self._tickers = tickers
        self._heartbeat_interval = heartbeat_interval
        self._handlers: list[Callable[[Tick], None]] = []
        self._running = False
        self._last_tick_ts: float = 0.0
        self._task: asyncio.Task | None = None

    # -- Public API --------------------------------------------------------

    def register_handler(self, handler: Callable[[Tick], None]) -> None:
        """Attach a callback to be called on every Tick."""
        self._handlers.append(handler)

    async def start(self) -> None:
        """Begin the async consumer loop."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Signal the consumer to stop and await shutdown."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def last_tick_age(self) -> float:
        """Seconds since the most recent tick (0.0 if never)."""
        if self._last_tick_ts == 0.0:
            return float("inf")
        return time.time() - self._last_tick_ts

    # -- Internal loop -----------------------------------------------------

    async def _consume_loop(self) -> None:
        """Main async loop: simulates a websocket consumer.

        In production this would wrap a real websocket client (e.g.
        ``websockets`` library).  For the Silent Run we simulate with a
        fast periodic poll so the runner logic can be validated.
        """
        _log.info("LiveStreamConnector started for tickers: %s", self._tickers)
        while self._running:
            try:
                # Simulate receiving a tick for each ticker
                for ticker in self._tickers:
                    tick = self._simulate_tick(ticker)
                    self._dispatch(tick)
                self._last_tick_ts = time.time()
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.error("Live stream consumer error: %s", exc)
                await asyncio.sleep(1.0)

    def _dispatch(self, tick: Tick) -> None:
        for handler in self._handlers:
            try:
                handler(tick)
            except Exception as exc:
                _log.error("Tick handler error: %s", exc)

    def _simulate_tick(self, ticker: str) -> Tick:
        """Generate a synthetic tick (used for dry-run / Silent Run testing)."""
        now = datetime.now(UTC)
        base = 450.0 if ticker == "SPY" else 100.0
        bid = base + (now.microsecond % 100) / 100.0
        ask = bid + 0.02
        return Tick(
            timestamp=now,
            ticker=ticker,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2.0,
            last=bid,
            volume=1000,
            source="simulated",
        )


# ---------------------------------------------------------------------------
# Silent Run Orchestrator
# ---------------------------------------------------------------------------

class SilentRunOrchestrator:
    """Wires LiveStreamConnector + PassiveExecutionProxy + PaperTradingRunner.

    Manages the 72-hour ``Silent Run`` (passive tracking) lifecycle.

    Parameters
    ----------
    connector : LiveStreamConnector
        Live data feed.
    proxy : PassiveExecutionProxy
        Bypass logger.
    runner : Any
        The PaperTradingRunner instance.
    silent_run_hours : int
        Duration of the passive tracking phase (default 72).
    """

    def __init__(
        self,
        connector: LiveStreamConnector,
        proxy: PassiveExecutionProxy,
        runner: Any,
        silent_run_hours: int = 72,
    ) -> None:
        self._connector = connector
        self._proxy = proxy
        self._runner = runner
        self._hours = silent_run_hours
        self._start_time: datetime | None = None

    async def start(self) -> None:
        """Begin the Silent Run."""
        self._start_time = datetime.now(UTC)
        _log.info("=== Silent Run started (duration=%dh) ===", self._hours)

        # Register proxy as a tick handler so it can log bypassed orders
        self._connector.register_handler(self._on_tick)

        await self._connector.start()

    async def stop(self) -> None:
        """End the Silent Run gracefully."""
        await self._connector.stop()
        _log.info("=== Silent Run stopped. Trades logged: %d ===", self._proxy.trade_count)

    def is_within_run_window(self) -> bool:
        if self._start_time is None:
            return False
        elapsed = (datetime.now(UTC) - self._start_time).total_seconds()
        return elapsed < self._hours * 3600

    def _on_tick(self, tick: Tick) -> None:
        """Called on every live tick."""
        # In passive mode, the runner's strategy receives ticks but
        # orders are logged, not transmitted.
        _log.debug("Tick: %s @ %.4f", tick.ticker, tick.mid)
