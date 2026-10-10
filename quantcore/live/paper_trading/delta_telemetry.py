"""Delta Telemetry — Real-Time Net Delta Monitoring & Logging Middleware.

Wraps the active execution loop to capture the net aggregate Delta of all
open option legs and underlying hedge positions, writing to the Module 3
logging schema (daily_log.csv) at configurable intervals.

Design
------
  1. Captures net delta from PaperExecutionEnvironment + PositionSizer
  2. Logs a structured telemetry snapshot at every interval ticks
  3. Writes to the daily_log CSV schema (net_delta column) plus a dedicated
     delta_telemetry.csv for audit-grade granularity
  4. Emits alerts if delta drift exceeds configurable thresholds

Usage
-----
    from quantcore.live.paper_trading.delta_telemetry import DeltaTelemetry

    telemetry = DeltaTelemetry(config=telemetry_config)
    telemetry.record_snapshot(
        dt=date.today(),
        positions=current_positions,
        hedge_shares=current_hedge_shares,
        equity=current_equity,
        vix_close=current_vix,
    )
"""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaTelemetryConfig:
    """Configuration for delta telemetry monitoring."""

    # -- Output paths -----------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("reports/paper_trading"))

    # -- Logging interval -------------------------------------------------------
    # Number of seconds between telemetry snapshots during live market hours.
    # Set to 0 to log on every call (tick-level), 300 for every 5 min, etc.
    log_interval_seconds: int = 300  # 5 minutes

    # -- Delta drift alert thresholds -------------------------------------------
    # Absolute delta drift that triggers alerts
    drift_warning_threshold: float = 5.0   # |net_delta| > 5 → WARNING
    drift_critical_threshold: float = 15.0  # |net_delta| > 15 → CRITICAL

    # -- Rolling window for drift statistics ------------------------------------
    drift_lookback: int = 20  # last 20 snapshots for statistics

    # -- Shares per contract (must match PositionSizer) --------------------------
    shares_per_contract: int = 100

    def __post_init__(self) -> None:
        # Ensure output_dir is a Path (frozen dataclass workaround)
        object.__setattr__(self, "output_dir", Path(self.output_dir))


# ---------------------------------------------------------------------------
# Telemetry snapshot
# ---------------------------------------------------------------------------

@dataclass
class DeltaSnapshot:
    """Immutable telemetry snapshot for a single point in time."""

    timestamp: datetime
    date: date
    gross_option_delta: float
    hedge_delta: float
    net_delta: float
    target_delta: float
    delta_drift: float
    num_positions: int
    hedge_shares: float
    equity: float
    vix_close: float | None = None
    alert_level: str = "ok"  # "ok" | "warning" | "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "date": self.date.isoformat(),
            "gross_option_delta": round(self.gross_option_delta, 4),
            "hedge_delta": round(self.hedge_delta, 4),
            "net_delta": round(self.net_delta, 4),
            "target_delta": round(self.target_delta, 4),
            "delta_drift": round(self.delta_drift, 4),
            "num_positions": self.num_positions,
            "hedge_shares": round(self.hedge_shares, 2),
            "equity": round(self.equity, 2),
            "vix_close": self.vix_close,
            "alert_level": self.alert_level,
        }


# ---------------------------------------------------------------------------
# Delta Telemetry
# ---------------------------------------------------------------------------

class DeltaTelemetry:
    """Real-time net delta monitor and logger.

    Captures the net aggregate Delta of all open option legs and underlying
    hedge positions, writing structured telemetry to CSV at configurable
    intervals for audit-grade delta drift tracking.

    Parameters
    ----------
    config : DeltaTelemetryConfig
        Configuration including output paths and thresholds.
    """

    def __init__(self, config: DeltaTelemetryConfig | None = None) -> None:
        self._cfg = config or DeltaTelemetryConfig()
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Rolling history for statistics
        self._history: deque[DeltaSnapshot] = deque(maxlen=self._cfg.drift_lookback)
        self._last_log_time: float = 0.0  # epoch seconds of last CSV write

        # File paths
        self._telemetry_csv_path = self._cfg.output_dir / "delta_telemetry.csv"
        self._alert_path = self._cfg.output_dir / "alerts.log"
        self._init_csv()

        _log.info(
            "DeltaTelemetry initialised: interval=%ds, "
            "drift_warning=%.1f, drift_critical=%.1f",
            self._cfg.log_interval_seconds,
            self._cfg.drift_warning_threshold,
            self._cfg.drift_critical_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        dt: date,
        positions: list[dict[str, Any]],
        hedge_shares: float,
        equity: float,
        vix_close: float | None = None,
        target_delta: float = 0.0,
        force_log: bool = False,
    ) -> DeltaSnapshot:
        """Compute and record a delta telemetry snapshot.

        Call this from the execution loop at configurable intervals.

        Parameters
        ----------
        dt : date
            Trading date.
        positions : list[dict]
            Open positions. Each must have keys:
              - call_delta (float): per-share delta for the short call
              - put_delta (float): per-share delta for the short put
              - contracts (int): number of contracts
        hedge_shares : float
            Current hedge position in shares (positive = long).
        equity : float
            Current portfolio equity.
        vix_close : float | None
            VIX close for macro context.
        target_delta : float
            Target net portfolio delta (default 0.0).
        force_log : bool
            Force a CSV write regardless of interval timer.

        Returns
        -------
        DeltaSnapshot
            The computed telemetry snapshot.
        """
        # ── Compute gross option delta ──
        gross_option_delta = 0.0
        for pos in positions:
            call_delta = pos.get("call_delta", 0.0)
            put_delta = pos.get("put_delta", 0.0)
            contracts = pos.get("contracts", 0)
            # Short call delta = -delta; short put delta = -put_delta
            short_call_delta = -call_delta * self._cfg.shares_per_contract * contracts
            short_put_delta = -put_delta * self._cfg.shares_per_contract * contracts
            gross_option_delta += short_call_delta + short_put_delta

        # ── Compute net delta ──
        hedge_delta = hedge_shares
        net_delta = gross_option_delta + hedge_delta
        delta_drift = net_delta - target_delta

        # ── Determine alert level ──
        abs_drift = abs(delta_drift)
        if abs_drift > self._cfg.drift_critical_threshold:
            alert_level = "critical"
        elif abs_drift > self._cfg.drift_warning_threshold:
            alert_level = "warning"
        else:
            alert_level = "ok"

        # ── Build snapshot ──
        snapshot = DeltaSnapshot(
            timestamp=datetime.utcnow(),
            date=dt,
            gross_option_delta=gross_option_delta,
            hedge_delta=hedge_delta,
            net_delta=net_delta,
            target_delta=target_delta,
            delta_drift=delta_drift,
            num_positions=len(positions),
            hedge_shares=hedge_shares,
            equity=equity,
            vix_close=vix_close,
            alert_level=alert_level,
        )

        # ── Append to history ──
        self._history.append(snapshot)

        # ── Log to CSV (interval-gated) ──
        now = time.time()
        elapsed = now - self._last_log_time
        if force_log or elapsed >= self._cfg.log_interval_seconds or self._last_log_time == 0.0:
            self._write_telemetry(snapshot)
            self._last_log_time = now

        # ── Emit alerts ──
        if alert_level == "critical":
            _log.critical(
                "DELTA DRIFT CRITICAL: net_delta=%.2f drift=%.2f | "
                "positions=%d hedge=%.0f equity=$%.2f",
                net_delta, delta_drift, len(positions),
                hedge_shares, equity,
            )
            self._write_alert(snapshot)
        elif alert_level == "warning":
            _log.warning(
                "DELTA DRIFT WARNING: net_delta=%.2f drift=%.2f | "
                "positions=%d hedge=%.0f",
                net_delta, delta_drift, len(positions), hedge_shares,
            )
            self._write_alert(snapshot)

        return snapshot

    def get_current_net_delta(self) -> float | None:
        """Return the most recent net delta from history."""
        if not self._history:
            return None
        return self._history[-1].net_delta

    def get_drift_statistics(self) -> dict[str, Any]:
        """Compute rolling drift statistics over the lookback window."""
        if not self._history:
            return {"status": "no_data"}

        drifts = [s.delta_drift for s in self._history]
        arr = np.array(drifts)

        return {
            "status": "ok",
            "count": len(drifts),
            "mean_drift": round(float(np.mean(arr)), 4),
            "std_drift": round(float(np.std(arr, ddof=1)) if len(drifts) > 1 else 0.0, 4),
            "max_abs_drift": round(float(np.max(np.abs(arr))), 4),
            "latest_net_delta": round(self._history[-1].net_delta, 4),
            "latest_drift": round(self._history[-1].delta_drift, 4),
            "warning_threshold": self._cfg.drift_warning_threshold,
            "critical_threshold": self._cfg.drift_critical_threshold,
            "breaches_warning": int(np.sum(np.abs(arr) > self._cfg.drift_warning_threshold)),
            "breaches_critical": int(np.sum(np.abs(arr) > self._cfg.drift_critical_threshold)),
        }

    def get_daily_log_compatible_row(
        self,
        snapshot: DeltaSnapshot | None = None,
    ) -> dict[str, Any]:
        """Return a dict compatible with the Module 3 daily_log.csv schema.

        The analytics dashboard expects a 'net_delta' column in the daily log.
        This method produces the delta-related fields in the exact format
        expected by PaperTradingAnalytics.record_daily_result().
        """
        snap = snapshot or (self._history[-1] if self._history else None)
        if snap is None:
            return {"net_delta": 0.0}

        return {
            "net_delta": round(snap.net_delta, 2),
            "vix_close": snap.vix_close,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Create the telemetry CSV with headers if it doesn't exist."""
        if self._telemetry_csv_path.exists():
            return

        headers = [
            "timestamp", "date", "gross_option_delta", "hedge_delta",
            "net_delta", "target_delta", "delta_drift",
            "num_positions", "hedge_shares", "equity",
            "vix_close", "alert_level",
        ]
        try:
            with open(self._telemetry_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
        except Exception as exc:
            _log.error("Telemetry CSV init failed: %s", exc)

    def _write_telemetry(self, snapshot: DeltaSnapshot) -> None:
        """Append a snapshot to the telemetry CSV."""
        row = snapshot.to_dict()
        # Flatten timestamp for CSV
        row["timestamp"] = snapshot.timestamp.isoformat()
        row["date"] = snapshot.date.isoformat()
        try:
            with open(self._telemetry_csv_path, "a", newline="") as f:
                headers = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
        except Exception as exc:
            _log.error("Telemetry CSV write failed: %s", exc)

    def _write_alert(self, snapshot: DeltaSnapshot) -> None:
        """Append an alert to the alerts log file."""
        try:
            with open(self._alert_path, "a") as f:
                f.write(
                    f"{snapshot.date.isoformat()} [{snapshot.alert_level.upper()}] "
                    f"DELTA_DRIFT: net_delta={snapshot.net_delta:.2f} "
                    f"drift={snapshot.delta_drift:.2f} "
                    f"positions={snapshot.num_positions} "
                    f"hedge={snapshot.hedge_shares:.0f}\n"
                )
        except Exception as exc:
            _log.error("Delta alert write failed: %s", exc)


# ---------------------------------------------------------------------------
# Middleware wrapper for the execution loop
# ---------------------------------------------------------------------------

class DeltaTelemetryMiddleware:
    """Middleware that wraps a PaperTradingRunner's execution loop.

    Hooks into the daily run cycle to capture delta telemetry at each
    step. Designed to be called from PaperTradingRunner.run_once().

    Usage in runner.py::

        telemetry_middleware = DeltaTelemetryMiddleware(telemetry_config)

        # Inside run_once(), after computing positions:
        snapshot = telemetry_middleware.on_positions_updated(
            dt=as_of,
            positions=current_positions,
            hedge_shares=delta_record["hedge_shares"],
            equity=self._current_equity,
            vix_close=vix,
        )

        # Pass net_delta to analytics:
        net_delta = snapshot.net_delta

    Parameters
    ----------
    config : DeltaTelemetryConfig | None
        Configuration. Defaults to standard thresholds.
    """

    def __init__(self, config: DeltaTelemetryConfig | None = None) -> None:
        self._telemetry = DeltaTelemetry(config)

    def on_positions_updated(
        self,
        dt: date,
        positions: list[dict[str, Any]],
        hedge_shares: float,
        equity: float,
        vix_close: float | None = None,
        target_delta: float = 0.0,
        force_log: bool = False,
    ) -> DeltaSnapshot:
        """Called when positions are updated in the execution loop.

        Replace direct calls to PaperExecutionEnvironment.record_delta_hedge()
        with this middleware to get automatic telemetry on every position update.
        """
        snapshot = self._telemetry.record_snapshot(
            dt=dt,
            positions=positions,
            hedge_shares=hedge_shares,
            equity=equity,
            vix_close=vix_close,
            target_delta=target_delta,
            force_log=force_log,
        )
        return snapshot

    @property
    def telemetry(self) -> DeltaTelemetry:
        """Access the underlying DeltaTelemetry for statistics."""
        return self._telemetry

    def get_drift_stats(self) -> dict[str, Any]:
        """Convenience proxy for DeltaTelemetry.get_drift_statistics()."""
        return self._telemetry.get_drift_statistics()
