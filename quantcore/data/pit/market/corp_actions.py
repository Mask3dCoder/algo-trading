"""Normalise Polygon corporate actions into PIT rows and adjustment factors.

Two representations are kept in sync:

* **Store rows** (``CORP_ACTION_COLUMNS``) — the durable, as-of-aware records.
* **``AdjustmentEvent``** — the existing lightweight model from
  ``quantcore.data.corporate_actions``, reused by the backtest engine.

The PIT gate: a backtest at ``T`` applies an action only if
``effective_date <= T`` **and** (``announced_at`` is null or ``announced_at <= T``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from quantcore.data.corporate_actions import ActionType, AdjustmentEvent, get_adjustment_factors
from quantcore.data.pit.market.schema import CORP_ACTION_COLUMNS


def _utc_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def normalize_splits(symbol: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Polygon ``/v3/reference/splits`` rows to PIT store rows."""
    out: list[dict[str, Any]] = []
    for r in rows:
        split_from = r.get("split_from")
        split_to = r.get("split_to")
        ratio = None
        try:
            ratio = float(split_to) / float(split_from) if split_from and split_to else None
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
        out.append({
            "action_type": ActionType.SPLIT.value,
            "symbol": symbol,
            "effective_date": r.get("execution_date"),
            "announced_at": _utc_ts(r.get("declared_date")),
            "value": ratio,
            "ticker_old": None,
            "ticker_new": None,
            "description": f"{split_from}-for-{split_to}" if ratio else "",
            "source": "polygon",
        })
    return out


def normalize_dividends(symbol: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Polygon ``/v3/reference/dividends`` rows to PIT store rows."""
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "action_type": ActionType.DIVIDEND.value,
            "symbol": symbol,
            "effective_date": r.get("ex_dividend_date"),
            "announced_at": _utc_ts(r.get("declaration_date")),
            "value": r.get("cash_amount"),
            "ticker_old": None,
            "ticker_new": None,
            "description": f"{r.get('dividend_type') or 'cash'} dividend",
            "source": "polygon",
        })
    return out


def delisting_row(symbol: str, delisted_utc: str | None) -> dict[str, Any] | None:
    """Build a delisting action row from a ticker reference ``delisted_utc``."""
    ts = _utc_ts(delisted_utc)
    if ts is None:
        return None
    return {
        "action_type": ActionType.DELISTING.value,
        "symbol": symbol,
        "effective_date": ts.date().isoformat(),
        "announced_at": ts,
        "value": None,
        "ticker_old": symbol,
        "ticker_new": None,
        "description": "delisted",
        "source": "polygon",
    }


def frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Coerce a list of action dicts into the canonical store frame."""
    if not rows:
        return pd.DataFrame(columns=list(CORP_ACTION_COLUMNS))
    df = pd.DataFrame(rows)
    for col in CORP_ACTION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date
    df["announced_at"] = pd.to_datetime(df["announced_at"], utc=True, errors="coerce")
    return df[list(CORP_ACTION_COLUMNS)]


def to_adjustment_events(df: pd.DataFrame) -> list[AdjustmentEvent]:
    """Convert PIT action rows to the backtest engine's ``AdjustmentEvent``."""
    events: list[AdjustmentEvent] = []
    if df.empty:
        return events
    for _, row in df.iterrows():
        try:
            action_type = ActionType(str(row["action_type"]))
        except ValueError:
            continue
        effective = pd.Timestamp(row["effective_date"]).date() if pd.notna(row["effective_date"]) else date.today()
        value = float(row["value"]) if pd.notna(row["value"]) else 0.0
        events.append(AdjustmentEvent(
            symbol=str(row["symbol"]),
            date=effective,
            action_type=action_type,
            value=value,
            description=str(row.get("description") or ""),
        ))
    return events


def build_adjustment_factors(
    symbol: str,
    actions: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    prices: pd.DataFrame | None = None,
) -> pd.Series:
    """Cumulative adjustment factor series over ``date_range`` for one symbol."""
    events = [e for e in to_adjustment_events(actions) if e.symbol == symbol]
    return get_adjustment_factors(events, date_range, prices)
