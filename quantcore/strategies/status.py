"""Strategy phase status helpers for runtime gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_STATUS_PATH = Path("strategy_status.yaml")
LIVE_STATUSES = {"LIVE"}
PAPER_STATUSES = {"PAPER"}
RUNTIME_ALLOWED_STATUSES = LIVE_STATUSES | PAPER_STATUSES


def load_strategy_status(path: str | Path = DEFAULT_STATUS_PATH) -> dict[str, Any]:
    """Load strategy_status.yaml. Returns an empty dict when absent."""
    status_path = Path(path)
    if not status_path.exists():
        return {}
    data = yaml.safe_load(status_path.read_text(encoding="utf-8")) or {}
    strategies = data.get("strategies", {})
    return strategies if isinstance(strategies, dict) else {}


def get_strategy_phase(strategy_name: str, path: str | Path = DEFAULT_STATUS_PATH) -> str | None:
    """Return the phase for a strategy, or None when the status file is absent."""
    status = load_strategy_status(path)
    if not status:
        return None
    record = status.get(strategy_name, {})
    if not isinstance(record, dict):
        return None
    phase = record.get("phase")
    return str(phase) if phase is not None else None


def is_runtime_allowed(
    strategy_name: str,
    path: str | Path = DEFAULT_STATUS_PATH,
    allowed: set[str] | None = None,
) -> bool:
    """Return whether a strategy may be loaded by a runtime config.

    If strategy_status.yaml is absent, the function is permissive for backward
    compatibility. Once the file exists, only LIVE/PAPER strategies may be loaded.
    """
    phase = get_strategy_phase(strategy_name, path)
    if phase is None:
        return True
    allowed_statuses = allowed or RUNTIME_ALLOWED_STATUSES
    return phase in allowed_statuses


def validate_runtime_config(
    strategy_names: list[str],
    path: str | Path = DEFAULT_STATUS_PATH,
    allowed: set[str] | None = None,
) -> list[str]:
    """Return strategies that should be skipped by runtime gating."""
    skipped: list[str] = []
    for name in strategy_names:
        if not is_runtime_allowed(name, path, allowed):
            skipped.append(name)
    return skipped
