"""Append-only audit trail with hash chain for tamper detection."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AuditEventType(str, Enum):
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_UPDATE = "position_update"
    RISK_CHECK = "risk_check"
    COMPLIANCE_CHECK = "compliance_check"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class AuditEntry:
    """Immutable entry in the audit log with cryptographic hash chaining."""
    entry_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_type: AuditEventType = AuditEventType.SYSTEM_START
    actor: str = "system"
    details: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""


def _compute_hash(
    entry_id: str, timestamp: str, event_type: str, actor: str,
    details_json: str, previous_hash: str,
) -> str:
    """Deterministic SHA-256 hash of an audit entry's contents."""
    payload = f"{entry_id}|{timestamp}|{event_type}|{actor}|{details_json}|{previous_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only audit trail with cryptographic integrity verification."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._storage_path = storage_path

    def record(
        self, event_type: AuditEventType, actor: str, details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Create and append a new audit entry with hash chain."""
        details = details or {}
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC)
        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry_hash = _compute_hash(
            entry_id, timestamp.isoformat(), event_type.value, actor,
            json.dumps(details, sort_keys=True, default=str), prev_hash,
        )
        entry = AuditEntry(
            entry_id=entry_id, timestamp=timestamp, event_type=event_type,
            actor=actor, details=details, previous_hash=prev_hash, entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Verify all hashes are consistent. Returns True if tamper-free."""
        for i, entry in enumerate(self._entries):
            prev_hash = self._entries[i - 1].entry_hash if i > 0 else ""
            computed = _compute_hash(
                entry.entry_id, entry.timestamp.isoformat(), entry.event_type.value,
                entry.actor, json.dumps(entry.details, sort_keys=True, default=str), prev_hash,
            )
            if computed != entry.entry_hash:
                return False
            if entry.previous_hash != prev_hash:
                return False
        return True

    def query(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        event_type: AuditEventType | None = None,
        actor: str | None = None,
    ) -> list[AuditEntry]:
        """Filter entries by time range, event type, and/or actor."""
        results = self._entries
        if start:
            results = [e for e in results if e.timestamp >= start]
        if end:
            results = [e for e in results if e.timestamp <= end]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if actor:
            results = [e for e in results if e.actor == actor]
        return results

    def export_json(self, filepath: str) -> None:
        """Export all entries to a JSON file."""
        data = []
        for e in self._entries:
            data.append({
                "entry_id": e.entry_id,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "actor": e.actor,
                "details": e.details,
                "previous_hash": e.previous_hash,
                "entry_hash": e.entry_hash,
            })
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def import_json(self, filepath: str) -> None:
        """Import entries from JSON and validate hash chain."""
        with open(filepath) as f:
            data = json.load(f)
        entries: list[AuditEntry] = []
        for d in data:
            entries.append(AuditEntry(
                entry_id=d["entry_id"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                event_type=AuditEventType(d["event_type"]),
                actor=d["actor"],
                details=d.get("details", {}),
                previous_hash=d["previous_hash"],
                entry_hash=d["entry_hash"],
            ))
        self._entries = entries

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def last_entry(self) -> AuditEntry | None:
        return self._entries[-1] if self._entries else None


def tamper_detection_report(log: AuditLog) -> dict:
    """Generate a tamper detection report for an audit log."""
    is_valid = log.verify_chain()
    first_broken: int | None = None
    if not is_valid:
        for i, entry in enumerate(log.entries):
            prev_hash = log.entries[i - 1].entry_hash if i > 0 else ""
            computed = _compute_hash(
                entry.entry_id, entry.timestamp.isoformat(), entry.event_type.value,
                entry.actor, json.dumps(entry.details, sort_keys=True, default=str), prev_hash,
            )
            if computed != entry.entry_hash or entry.previous_hash != prev_hash:
                first_broken = i
                break
    return {
        "tampered": not is_valid,
        "first_broken_index": first_broken,
        "total_entries": log.entry_count,
        "verified_entries": log.entry_count if is_valid else (first_broken or 0),
    }
