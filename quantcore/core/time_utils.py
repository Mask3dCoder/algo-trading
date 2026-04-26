"""Timezone-aware time utilities for market calendar operations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta

import pytz
from exchange_calendars import get_calendar

from quantcore.core.constants import (
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
)

ET = pytz.timezone("US/Eastern")
UTC = pytz.utc

NYSE = get_calendar("XNYS")

MARKET_OPEN = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
MARKET_CLOSE = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)


def to_utc(dt: datetime) -> datetime:
    """Convert any timezone-aware datetime to UTC."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(UTC)


def to_eastern(dt: datetime) -> datetime:
    """Convert any timezone-aware datetime to US/Eastern."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(ET)


def is_market_open(dt: datetime) -> bool:
    """Check if the given datetime falls within NYSE regular trading hours."""
    dt_et = to_eastern(dt) if dt.tzinfo else ET.localize(dt)
    if not is_trading_day(dt_et.date()):
        return False
    return MARKET_OPEN <= dt_et.time() < MARKET_CLOSE


def is_trading_day(d: date) -> bool:
    """Check if a date is a valid NYSE trading day."""
    return d.weekday() < 5 and NYSE.is_session(d.isoformat())


def next_market_open(dt: datetime) -> datetime:
    """Find the next market open after the given datetime."""
    dt_et = to_eastern(dt) if dt.tzinfo else ET.localize(dt)
    current_date = dt_et.date()
    while True:
        if is_trading_day(current_date):
            market_open = ET.localize(
                datetime.combine(current_date, MARKET_OPEN)
            )
            if market_open > dt_et:
                return to_utc(market_open)
        current_date += timedelta(days=1)


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (exclusive of end)."""
    sessions = NYSE.sessions_in_range(start.isoformat(), end.isoformat())
    # sessions_in_range is inclusive; subtract 1 if end is a session
    count = len(sessions)
    if is_trading_day(end):
        count -= 1
    return max(count, 0)


def trading_days(start: date, end: date) -> Sequence[date]:
    """Return all trading days between start and end (inclusive)."""
    sessions = NYSE.sessions_in_range(start.isoformat(), end.isoformat())
    return [date.fromisoformat(s) for s in sessions]
