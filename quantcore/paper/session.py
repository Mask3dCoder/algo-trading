"""Paper trading session persistence and restart recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_STATE_DIR = Path("data/paper_state")
STATE_FILE = "session_state.json"


@dataclass
class SessionSnapshot:
    """Serializable snapshot of paper trading session state."""
    timestamp: str = ""
    total_pnl: float = 0.0
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    duration_elapsed_sec: float = 0.0
    max_drawdown: float = 0.0
    last_signal_timestamp: str = ""
    kill_switch_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_pnl": self.total_pnl,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions": self.positions,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "duration_elapsed_sec": self.duration_elapsed_sec,
            "max_drawdown": self.max_drawdown,
            "last_signal_timestamp": self.last_signal_timestamp,
            "kill_switch_active": self.kill_switch_active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionSnapshot:
        return cls(
            timestamp=d.get("timestamp", ""),
            total_pnl=float(d.get("total_pnl", 0)),
            portfolio_value=float(d.get("portfolio_value", 0)),
            cash=float(d.get("cash", 0)),
            positions=d.get("positions", {}),
            orders_submitted=int(d.get("orders_submitted", 0)),
            orders_filled=int(d.get("orders_filled", 0)),
            orders_rejected=int(d.get("orders_rejected", 0)),
            duration_elapsed_sec=float(d.get("duration_elapsed_sec", 0)),
            max_drawdown=float(d.get("max_drawdown", 0)),
            last_signal_timestamp=d.get("last_signal_timestamp", ""),
            kill_switch_active=bool(d.get("kill_switch_active", False)),
        )


class SessionStore:
    """Persist paper trading session state for restart recovery."""

    def __init__(self, state_dir: str | Path = DEFAULT_STATE_DIR, session_id: str = "") -> None:
        base = Path(state_dir)
        if session_id:
            base = base / session_id
        self._dir = base
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / STATE_FILE

    def save(self, snapshot: SessionSnapshot) -> None:
        """Write current session state to disk atomically."""
        snapshot.timestamp = datetime.now(UTC).isoformat()
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(snapshot.to_dict(), indent=2))
        tmp_path.replace(self._path)

    def load(self) -> SessionSnapshot | None:
        """Load the last saved session state. Returns None if no saved state."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text())
            return SessionSnapshot.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def clear(self) -> None:
        """Remove saved state (call after successful run completion)."""
        if self._path.exists():
            self._path.unlink()

    def has_saved_state(self) -> bool:
        return self._path.exists()

    @property
    def state_path(self) -> Path:
        return self._path
