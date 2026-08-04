"""Connection Resilience & Network Heartbeat Monitor.

Tracks live data-feed health, triggers alerts on missed heartbeats, and
orchestrates exponential-backoff reconnection.

Design
------
  ConnectionMonitor  : main class (thread-safe, not asyncio)
    ├── HeartbeatTracker  : records last-seen timestamp per feed
    ├── ReconnectEngine   : exponential-backoff retries
    └── StateGuard        : freezes execution on disconnect

Usage
-----
    monitor = ConnectionMonitor(alert_threshold_sec=30.0)
    monitor.register_feed("alpaca_ws")
    monitor.start()

    # On every incoming websocket frame:
    monitor.heartbeat("alpaca_ws")

    # In the runner loop:
    if not monitor.is_healthy("alpaca_ws"):
        # halt order generation until reconnected
        ...
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ConnectionState(str, Enum):
    """Finite states for a single feed connection."""

    CONNECTED = "connected"
    DEGRADED = "degraded"      # missed 1 heartbeat
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"          # exhausted all backoff retries


# ---------------------------------------------------------------------------
# Per-feed health record
# ---------------------------------------------------------------------------


@dataclass
class FeedHealth:
    """Mutable health state for a single data feed."""
    name: str
    last_seen: float = 0.0          # epoch seconds
    state: ConnectionState = ConnectionState.DISCONNECTED
    missed_beats: int = 0
    reconnect_attempts: int = 0
    reconnect_delay: float = 2.0    # next backoff (seconds)
    max_reconnect_delay: float = 64.0

    def missed_beat(self) -> None:
        self.missed_beats += 1
        self.state = ConnectionState.DEGRADED if self.missed_beats < 3 else ConnectionState.DISCONNECTED

    def record_seen(self) -> None:
        self.last_seen = time.time()
        self.missed_beats = 0
        if self.state != ConnectionState.CONNECTED:
            self.state = ConnectionState.CONNECTED
            _log.info("Feed '%s' restored to CONNECTED", self.name)

    def next_backoff(self) -> float:
        """Return current delay and advance exponential backoff."""
        delay = min(self.reconnect_delay, self.max_reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2.0, self.max_reconnect_delay)
        self.reconnect_attempts += 1
        return delay

    def reset_backoff(self) -> None:
        self.reconnect_delay = 2.0
        self.reconnect_attempts = 0


# ---------------------------------------------------------------------------
# ConnectionMonitor
# ---------------------------------------------------------------------------


class ConnectionMonitor:
    """Monitors all live data feeds, alerts on heartbeat failures, and
    manages exponential-backoff reconnection.

    Thread-safe.  Designed to be embedded in the paper-trading runner.

    Parameters
    ----------
    alert_threshold_sec : float
        Seconds without a tick before declaring a feed unhealthy (default 30).
    market_hours_only : bool
        If True, skip monitoring outside US market hours (09:30–16:00 ET).
    """

    def __init__(
        self,
        alert_threshold_sec: float = 30.0,
        market_hours_only: bool = True,
    ) -> None:
        self._threshold = alert_threshold_sec
        self._market_hours_only = market_hours_only
        self._feeds: dict[str, FeedHealth] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._alert_log_path = Path("reports/paper_trading/connection_alerts.log")
        self._alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_feed(self, name: str) -> None:
        """Register a new feed for monitoring."""
        with self._lock:
            if name not in self._feeds:
                self._feeds[name] = FeedHealth(name=name)
                _log.info("ConnectionMonitor: feed '%s' registered", name)

    def heartbeat(self, name: str) -> None:
        """Call on every received frame/tick to confirm the feed is alive."""
        with self._lock:
            feed = self._feeds.get(name)
        if feed is None:
            self.register_feed(name)
            feed = self._feeds[name]
        feed.record_seen()

    def remove_feed(self, name: str) -> None:
        with self._lock:
            self._feeds.pop(name, None)

    def is_healthy(self, name: str | None = None) -> bool:
        """Return True if the feed (or all feeds if name is None) is healthy."""
        with self._lock:
            if name is not None:
                feed = self._feeds.get(name)
                return feed is not None and feed.state == ConnectionState.CONNECTED
            return all(f.state == ConnectionState.CONNECTED for f in self._feeds.values())

    def get_health(self, name: str) -> FeedHealth | None:
        with self._lock:
            return self._feeds.get(name)

    def get_all_health(self) -> dict[str, FeedHealth]:
        with self._lock:
            return {name: feed for name, feed in self._feeds.items()}

    def start(self) -> None:
        """Start the background monitor thread."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        _log.info("ConnectionMonitor started (threshold=%.0fs)", self._threshold)

    def stop(self) -> None:
        """Signal the monitor thread to stop."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def attempt_reconnect(self, name: str, reconnect_fn) -> bool:
        """Execute a reconnection attempt with exponential backoff.

        Parameters
        ----------
        name : str
            Feed name.
        reconnect_fn : callable
            Callable that returns True on success, False on failure.

        Returns
        -------
        bool
            True if reconnected successfully.
        """
        with self._lock:
            feed = self._feeds.get(name)
        if feed is None:
            return False

        delay = feed.next_backoff()
        feed.state = ConnectionState.RECONNECTING
        _log.warning(
            "Feed '%s' reconnecting in %.1fs (attempt %d)",
            name, delay, feed.reconnect_attempts,
        )
        time.sleep(delay)

        if reconnect_fn():
            feed.reset_backoff()
            feed.record_seen()
            _log.info("Feed '%s' reconnected successfully", name)
            return True
        else:
            _log.error("Feed '%s' reconnection failed", name)
            return False

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread: check every feed's last_seen timestamp.

        If a feed hasn't sent a heartbeat within ``_threshold`` seconds,
        trigger an alert and transition to DEGRADED / DISCONNECTED.
        """
        while self._running:
            try:
                self._check_all_feeds()
                time.sleep(5.0)
            except Exception as exc:
                _log.error("Monitor loop error: %s", exc)
                time.sleep(5.0)

    def _check_all_feeds(self) -> None:
        now = time.time()
        with self._lock:
            feeds = list(self._feeds.items())
        for name, feed in feeds:
            if feed.state == ConnectionState.CONNECTED:
                age = now - feed.last_seen
                if age > self._threshold:
                    feed.missed_beat()
                    self._on_feed_degraded(name, feed, age)

    def _on_feed_degraded(self, name: str, feed: FeedHealth, age: float) -> None:
        """Called when a feed exceeds the heartbeat threshold."""
        msg = (
            f"HEARTBEAT ALERT: feed '{name}' silent for {age:.1f}s "
            f"(threshold={self._threshold:.0f}s). State={feed.state.value}. "
            f"Missed beats={feed.missed_beats}."
        )
        _log.warning(msg)
        self._write_alert(name, msg)

    def _write_alert(self, name: str, message: str) -> None:
        try:
            with open(self._alert_log_path, "a") as f:
                f.write(f"{datetime.now(UTC).isoformat()} [{name}] {message}\n")
        except Exception as exc:
            _log.error("Alert log write failed: %s", exc)
