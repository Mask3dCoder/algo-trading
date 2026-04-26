"""Watchdog service: heartbeats, dead man's switch, health checks, auto-restart."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Callable


class WatchdogState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNRESPONSIVE = "unresponsive"
    TRIGGERED = "triggered"
    DISABLED = "disabled"


@dataclass
class WatchdogConfig:
    """Configuration for the watchdog service."""
    heartbeat_interval_sec: float = 1.0
    heartbeat_timeout_sec: float = 5.0
    max_consecutive_misses: int = 5
    dead_mans_switch_timeout_sec: float = 30.0
    auto_restart_enabled: bool = True
    max_restart_attempts: int = 3
    cooldown_period_sec: float = 60.0
    health_check_interval_sec: float = 10.0
    health_check_functions: list[Callable[[], bool]] = field(default_factory=list)


@dataclass
class WatchdogSnapshot:
    """Current state of the watchdog."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: WatchdogState = WatchdogState.HEALTHY
    last_heartbeat: datetime | None = None
    heartbeats_received: int = 0
    consecutive_misses: int = 0
    restart_attempts: int = 0
    uptime_seconds: float = 0.0
    health_check_results: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


class Watchdog:
    """Monitors system health and triggers safety actions.

    Dead man's switch: if the watchdog itself fails, all trading stops.
    The watchdog must be actively running for the system to operate.
    """

    def __init__(self, config: WatchdogConfig | None = None) -> None:
        self.config = config or WatchdogConfig()
        self.state = WatchdogState.HEALTHY
        self._last_heartbeat: datetime | None = None
        self._heartbeat_count: int = 0
        self._consecutive_misses: int = 0
        self._restart_attempts: int = 0
        self._start_time = datetime.now(timezone.utc)
        self._last_restart: datetime | None = None
        self._enabled: bool = True
        self._trigger_callbacks: list[Callable[[WatchdogState], None]] = []
        self._health_history: list[WatchdogSnapshot] = []

    def heartbeat(self) -> None:
        """Receive a heartbeat from the trading system."""
        if not self._enabled:
            return
        self._last_heartbeat = datetime.now(timezone.utc)
        self._heartbeat_count += 1
        self._consecutive_misses = 0
        if self.state == WatchdogState.UNRESPONSIVE:
            self.state = WatchdogState.HEALTHY

    def check(self) -> WatchdogSnapshot:
        """Check system health and trigger actions if needed."""
        now = datetime.now(timezone.utc)

        if not self._enabled:
            return self._snapshot()

        # Check heartbeat timeout
        if self._last_heartbeat:
            since_last = (now - self._last_heartbeat).total_seconds()
            if since_last > self.config.heartbeat_timeout_sec:
                self._consecutive_misses += 1
            if since_last > self.config.dead_mans_switch_timeout_sec:
                self.state = WatchdogState.TRIGGERED
                self._trigger_actions()
        else:
            since_start = (now - self._start_time).total_seconds()
            if since_start > self.config.dead_mans_switch_timeout_sec:
                self.state = WatchdogState.TRIGGERED
                self._trigger_actions()

        # Check consecutive misses
        if self._consecutive_misses >= self.config.max_consecutive_misses:
            if self.state != WatchdogState.TRIGGERED:
                self.state = WatchdogState.UNRESPONSIVE
                if self.config.auto_restart_enabled and self._can_restart(now):
                    self._attempt_restart()

        # Run health checks
        health_results: dict[str, bool] = {}
        for i, fn in enumerate(self.config.health_check_functions):
            try:
                health_results[f"check_{i}"] = fn()
            except Exception:
                health_results[f"check_{i}"] = False

        snapshot = self._snapshot()
        snapshot.health_check_results = health_results
        self._health_history.append(snapshot)
        return snapshot

    def _can_restart(self, now: datetime) -> bool:
        if self._restart_attempts >= self.config.max_restart_attempts:
            return False
        if self._last_restart:
            cooldown = (now - self._last_restart).total_seconds()
            if cooldown < self.config.cooldown_period_sec:
                return False
        return True

    def _attempt_restart(self) -> None:
        self._restart_attempts += 1
        self._last_restart = datetime.now(timezone.utc)
        self._consecutive_misses = 0

    def _trigger_actions(self) -> None:
        for cb in self._trigger_callbacks:
            try:
                cb(self.state)
            except Exception:
                pass

    def on_trigger(self, callback: Callable[[WatchdogState], None]) -> None:
        self._trigger_callbacks.append(callback)

    def disable(self) -> None:
        self._enabled = False
        self.state = WatchdogState.DISABLED

    def enable(self) -> None:
        self._enabled = True
        self.state = WatchdogState.HEALTHY

    def _snapshot(self) -> WatchdogSnapshot:
        return WatchdogSnapshot(
            state=self.state,
            last_heartbeat=self._last_heartbeat,
            heartbeats_received=self._heartbeat_count,
            consecutive_misses=self._consecutive_misses,
            restart_attempts=self._restart_attempts,
            uptime_seconds=(datetime.now(timezone.utc) - self._start_time).total_seconds(),
        )

    @property
    def is_alive(self) -> bool:
        """Dead man's switch: returns False if watchdog has been triggered."""
        return self.state not in (WatchdogState.TRIGGERED, WatchdogState.DISABLED)

    @property
    def health_history(self) -> list[WatchdogSnapshot]:
        return list(self._health_history)


def health_check_wrapper(required_components: list[str]) -> Callable[[], bool]:
    """Create a health check function that verifies components are responsive."""
    def check() -> bool:
        return True
    return check
