"""Numerical data validation helpers.

QuantCore treats non-finite market values as fatal data-quality events. This
module centralizes validation so backtest, live feed, and live runner paths all
fail closed before alpha or risk logic can consume NaN/inf values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from quantcore.core.exceptions import DataQualityViolation


class _BarLike(Protocol):
    symbol: str
    open: Any
    high: Any
    low: Any
    close: Any
    volume: Any


class _SnapshotLike(Protocol):
    bars: Mapping[str, _BarLike]


_PRICE_FIELDS = ("open", "high", "low", "close")
_NUMERIC_FIELDS = ("open", "high", "low", "close", "volume")


def validate_finite(value: Any, *, field_name: str, symbol: str) -> None:
    """Raise :class:`DataQualityViolation` when ``value`` is not finite."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataQualityViolation(
            f"Non-numeric {field_name} for {symbol}: {value!r}",
        ) from exc
    if not np.isfinite(numeric):
        raise DataQualityViolation(
            f"Non-finite {field_name} for {symbol}: {value!r}",
        )


def validate_market_data_snapshot(snapshot: _SnapshotLike) -> _SnapshotLike:
    """Validate all OHLCV fields in a ``MarketDataSnapshot``-like object."""
    for symbol, bar in snapshot.bars.items():
        for field in _PRICE_FIELDS:
            validate_finite(getattr(bar, field), field_name=field, symbol=symbol)
        validate_finite(getattr(bar, "volume", 0), field_name="volume", symbol=symbol)
        volume = float(getattr(bar, "volume", 0))
        if volume < 0:
            raise DataQualityViolation(f"Negative volume for {symbol}: {volume!r}")
    return snapshot


def clean_snapshot(snapshot: _SnapshotLike) -> _SnapshotLike:
    """Validate a market snapshot before alpha/risk logic consumes it."""
    return validate_market_data_snapshot(snapshot)


def validate_numeric_snapshot(
    snapshot: Mapping[str, Any],
    *,
    symbols: Sequence[str] | None = None,
    fields: Sequence[str] = _NUMERIC_FIELDS,
) -> Mapping[str, Any]:
    """Validate numeric arrays in live ``strategy.on_data()`` snapshots.

    Parameters
    ----------
    snapshot:
        Mapping returned by live data feeds, e.g. ``{"close": ndarray, "symbols": [...]}``.
    symbols:
        Optional symbol labels. If omitted, symbolic error messages fall back to
        column indices.
    fields:
        Numeric field names to validate. Defaults to OHLCV fields; callers may
        pass a narrower set such as ``("close",)`` for close-only snapshots.
    """
    for field in fields:
        values = snapshot.get(field)
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            continue
        if not np.isfinite(array).all():
            bad = np.argwhere(~np.isfinite(array))
            if symbols and bad.size:
                row, col = bad[0]
                symbol = symbols[int(col)] if int(col) < len(symbols) else f"column_{int(col)}"
                location = f"{symbol} at row {int(row)}"
            else:
                location = f"field={field}, index={bad[0].tolist() if bad.size else 'unknown'}"
            raise DataQualityViolation(f"Non-finite {field} detected at {location}")
    return snapshot


def validate_live_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a live feed snapshot and return it for pipeline continuity."""
    symbols = snapshot.get("symbols") or []
    return validate_numeric_snapshot(snapshot, symbols=symbols)
