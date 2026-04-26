"""Shared health monitoring state for the /health endpoint.

This module acts as a lightweight in-process singleton that tracks runtime
metrics updated by the trading engine, feed handlers, and OMS. The API server
reads from it for health checks without requiring a database round-trip.

Thread safety: each attribute is a simple scalar protected by the GIL.
For multi-process deployments, replace with a Redis-backed implementation.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

# --- Module-level state ---

_start_time: float = time.time()
_start_datetime: datetime = datetime.now(UTC)

_last_data_timestamp: datetime | None = None
_last_order_timestamp: datetime | None = None
_kill_switch_active: bool = False
_open_positions_count: int = 0


# --- State readers (used by /health endpoint) ---

def get_health_state() -> dict[str, Any]:
    """Return the full health state as a JSON-serializable dict."""
    now = datetime.now(UTC)
    uptime = time.time() - _start_time

    # Memory: prefer psutil, fall back to resource module
    memory_rss_mb: float | None = None
    try:
        import psutil
        memory_rss_mb = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 1)
    except ImportError:
        try:
            import resource
            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
            if os.name == "nt" or (hasattr(os, "uname") and os.uname().sysname == "Darwin"):
                memory_rss_mb = round(raw / (1024.0 * 1024.0), 1)
            else:
                memory_rss_mb = round(raw / 1024.0, 1)
        except (ImportError, AttributeError):
            pass

    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 3),
        "last_data_timestamp": _last_data_timestamp.isoformat() if _last_data_timestamp else None,
        "open_positions_count": _open_positions_count,
        "last_order_timestamp": _last_order_timestamp.isoformat() if _last_order_timestamp else None,
        "kill_switch_active": _kill_switch_active,
        "memory_rss_mb": memory_rss_mb,
        "timestamp": now.isoformat(),
    }


# --- State writers (called by trading engine / feed / OMS) ---

def record_data_point() -> None:
    """Record that a market data point was received right now."""
    global _last_data_timestamp
    _last_data_timestamp = datetime.now(UTC)


def record_order_submitted() -> None:
    """Record that an order was submitted right now."""
    global _last_order_timestamp
    _last_order_timestamp = datetime.now(UTC)


def set_kill_switch_active(active: bool) -> None:
    """Set the kill switch state."""
    global _kill_switch_active
    _kill_switch_active = active


def set_open_positions_count(count: int) -> None:
    """Update the count of currently open positions."""
    global _open_positions_count
    _open_positions_count = count
