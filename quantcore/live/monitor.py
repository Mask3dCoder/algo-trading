"""Real-time monitoring: health checks, latency, position drift, alerting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from quantcore.risk.drawdown import DrawdownTracker


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class MonitorConfig:
    """Configuration for the live monitor."""
    # Latency
    max_order_latency_ms: float = 500.0
    max_data_staleness_sec: float = 5.0

    # Risk
    max_position_drift_pct: float = 0.05  # 5% drift from target
    max_drawdown_warning_pct: float = 0.05
    max_drawdown_critical_pct: float = 0.10

    # Heartbeat
    heartbeat_timeout_sec: float = 10.0
    max_consecutive_misses: int = 3

    # Alerts
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


@dataclass
class HealthSnapshot:
    """Current health status of the live trading system."""
    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Latency
    avg_order_latency_ms: float = 0.0
    max_order_latency_ms: float = 0.0
    data_staleness_sec: float = 0.0
    # Risk
    current_drawdown_pct: float = 0.0
    position_drift_pct: float = 0.0
    # Heartbeat
    last_heartbeat_age_sec: float = 0.0
    consecutive_misses: int = 0
    # Summary
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class LiveMonitor:
    """Real-time health and risk monitor for the live trading system.

    Tracks latency, data freshness, position drift, drawdown, and
    heartbeat health. Triggers callbacks on status changes.
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        on_alert: Callable[[HealthStatus, str], None] | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self.on_alert = on_alert
        self._last_heartbeat: datetime | None = None
        self._consecutive_misses: int = 0
        self._order_latencies: list[float] = []
        self._last_data_time: datetime | None = None
        self._status_history: list[HealthSnapshot] = []

    def record_heartbeat(self) -> None:
        """Record a heartbeat from the live runner."""
        self._last_heartbeat = datetime.utcnow()
        self._consecutive_misses = 0

    def record_order_latency(self, latency_ms: float) -> None:
        """Record order round-trip latency."""
        self._order_latencies.append(latency_ms)
        self._order_latencies = self._order_latencies[-1000:]

    def record_data_update(self) -> None:
        """Record a market data update timestamp."""
        self._last_data_time = datetime.utcnow()

    def check(self) -> HealthSnapshot:
        """Run a full health check and return current status."""
        snapshot = HealthSnapshot()
        now = datetime.utcnow()

        # Heartbeat check
        if self._last_heartbeat is not None:
            snapshot.last_heartbeat_age_sec = (now - self._last_heartbeat).total_seconds()
            if snapshot.last_heartbeat_age_sec > self.config.heartbeat_timeout_sec:
                self._consecutive_misses += 1
            snapshot.consecutive_misses = self._consecutive_misses

            if self._consecutive_misses >= self.config.max_consecutive_misses:
                snapshot.errors.append(
                    f"No heartbeat for {snapshot.last_heartbeat_age_sec:.0f}s "
                    f"({self._consecutive_misses} consecutive misses)"
                )

        # Latency check
        if self._order_latencies:
            snapshot.avg_order_latency_ms = float(np.mean(self._order_latencies[-100:]))
            snapshot.max_order_latency_ms = float(np.max(self._order_latencies[-100:]))
            if snapshot.avg_order_latency_ms > self.config.max_order_latency_ms:
                snapshot.warnings.append(
                    f"Order latency {snapshot.avg_order_latency_ms:.0f}ms > "
                    f"{self.config.max_order_latency_ms:.0f}ms"
                )

        # Data staleness
        if self._last_data_time is not None:
            snapshot.data_staleness_sec = (now - self._last_data_time).total_seconds()
            if snapshot.data_staleness_sec > self.config.max_data_staleness_sec:
                snapshot.errors.append(
                    f"Data stale: {snapshot.data_staleness_sec:.1f}s since last update"
                )

        # Determine status
        if snapshot.errors:
            snapshot.status = HealthStatus.UNHEALTHY
        elif snapshot.warnings:
            snapshot.status = HealthStatus.DEGRADED
        else:
            snapshot.status = HealthStatus.HEALTHY

        self._status_history.append(snapshot)
        self._status_history = self._status_history[-100:]

        # Alert on status change
        if len(self._status_history) >= 2:
            prev = self._status_history[-2]
            if snapshot.status != prev.status and self.on_alert:
                if snapshot.status == HealthStatus.UNHEALTHY and self.config.alert_on_unhealthy:
                    self.on_alert(snapshot.status, "; ".join(snapshot.errors))
                elif snapshot.status == HealthStatus.DEGRADED and self.config.alert_on_degraded:
                    self.on_alert(snapshot.status, "; ".join(snapshot.warnings))

        return snapshot

    @property
    def is_healthy(self) -> bool:
        if not self._status_history:
            return True
        return self._status_history[-1].status == HealthStatus.HEALTHY

    @property
    def last_status(self) -> HealthSnapshot | None:
        return self._status_history[-1] if self._status_history else None


def health_check(
    monitor: LiveMonitor,
    drawdown_tracker: DrawdownTracker | None = None,
    positions: dict[str, Any] | None = None,
    target_weights: dict[str, float] | None = None,
) -> HealthSnapshot:
    """Run a comprehensive health check including risk metrics.

    Args:
        monitor: The LiveMonitor instance.
        drawdown_tracker: Optional drawdown tracker for DD checks.
        positions: Current positions for drift check.
        target_weights: Target weights for drift calculation.

    Returns:
        HealthSnapshot with full status.
    """
    snapshot = monitor.check()

    # Drawdown check
    if drawdown_tracker is not None:
        dd = abs(drawdown_tracker.current_drawdown)
        cfg = monitor.config
        snapshot.current_drawdown_pct = dd * 100
        if dd > cfg.max_drawdown_critical_pct:
            snapshot.errors.append(f"Critical drawdown: {dd:.1%}")
        elif dd > cfg.max_drawdown_warning_pct:
            snapshot.warnings.append(f"Elevated drawdown: {dd:.1%}")

    # Position drift check
    if positions and target_weights:
        total_value = sum(float(p.get("market_value", 0)) for p in positions.values())
        if total_value > 0:
            current_weights = {
                sym: float(p.get("market_value", 0)) / total_value
                for sym, p in positions.items()
            }
            max_drift = 0.0
            for sym, tw in target_weights.items():
                cw = current_weights.get(sym, 0)
                drift = abs(cw - tw)
                max_drift = max(max_drift, drift)
            snapshot.position_drift_pct = max_drift * 100
            if max_drift > monitor.config.max_position_drift_pct:
                snapshot.warnings.append(f"Position drift: {max_drift:.1%}")

    # Re-evaluate status
    if snapshot.errors:
        snapshot.status = HealthStatus.UNHEALTHY
    elif snapshot.warnings:
        snapshot.status = HealthStatus.DEGRADED

    return snapshot
