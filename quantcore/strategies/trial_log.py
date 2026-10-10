"""Prospective trial logger for strategy lifecycle management.

Provides CSV-based tracking of prospective and historical strategy trials,
including rejection rationale, rebuild criteria, and status transitions.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quantcore.strategies.base import StrategyStatus


@dataclass
class TrialRecord:
    """A single strategy trial record."""

    trial_id: int
    strategy_name: str
    version: str
    status: str
    date: str
    reason: str
    proposed_rebuild_criteria: str
    notes: str = ""


# Default path for the prospective trials CSV
DEFAULT_TRIALS_PATH = Path(__file__).resolve().parents[2] / "prospective_trials.csv"

# CSV header
_TRIAL_HEADERS = [
    "trial_id",
    "strategy_name",
    "version",
    "status",
    "date",
    "reason",
    "proposed_rebuild_criteria",
    "notes",
]


def _csv_path(path: str | os.PathLike | None = None) -> Path:
    if path is None:
        return Path(DEFAULT_TRIALS_PATH)
    return Path(path)


def init_trials_file(path: str | os.PathLike | None = None, *, force: bool = False) -> Path:
    """Initialize the trials CSV with headers if it doesn't exist.

    Args:
        path: Optional override for the trials CSV path.
        force: If True, overwrite even if the file already exists.

    Returns:
        Path to the initialized file.
    """
    p = _csv_path(path)
    if not p.exists() or force:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_TRIAL_HEADERS)
    return p


def log_trial(
    trial_id: int,
    strategy_name: str,
    version: str,
    status: StrategyStatus | str,
    reason: str,
    proposed_rebuild_criteria: str,
    *,
    notes: str = "",
    date: str | None = None,
    path: str | os.PathLike | None = None,
) -> TrialRecord:
    """Log a new trial record to the prospective trials CSV.

    Args:
        trial_id: Unique trial identifier.
        strategy_name: Strategy name (e.g. 'macro_cross_sectional_momentum').
        version: Version string (e.g. 'v2').
        status: Strategy status (e.g. StrategyStatus.REJECTED).
        reason: Rationale for the current status.
        proposed_rebuild_criteria: Criteria for revisiting the strategy.
        notes: Optional additional notes.
        date: Optional date string (defaults to today's date).
        path: Optional override for the trials CSV path.

    Returns:
        The logged TrialRecord.
    """
    p = _csv_path(path)
    init_trials_file(p)

    status_str = status.value if isinstance(status, StrategyStatus) else str(status)
    date_str = date or datetime.now().strftime("%Y-%m-%d")

    record = TrialRecord(
        trial_id=trial_id,
        strategy_name=strategy_name,
        version=version,
        status=status_str,
        date=date_str,
        reason=reason,
        proposed_rebuild_criteria=proposed_rebuild_criteria,
        notes=notes,
    )

    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            record.trial_id,
            record.strategy_name,
            record.version,
            record.status,
            record.date,
            record.reason,
            record.proposed_rebuild_criteria,
            record.notes,
        ])

    return record


def load_trials(
    path: str | os.PathLike | None = None,
) -> list[TrialRecord]:
    """Load all trial records from the CSV.

    Args:
        path: Optional override for the trials CSV path.

    Returns:
        List of TrialRecord objects.
    """
    p = _csv_path(path)
    if not p.exists():
        return []

    records: list[TrialRecord] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(TrialRecord(**{k: row[k] for k in _TRIAL_HEADERS}))
    return records


def next_trial_id(path: str | os.PathLike | None = None) -> int:
    """Return the next available trial_id (max + 1).

    Args:
        path: Optional override for the trials CSV path.

    Returns:
        Next available trial_id.
    """
    records = load_trials(path)
    if not records:
        return 1
    return max(r.trial_id for r in records if isinstance(r.trial_id, int)) + 1
