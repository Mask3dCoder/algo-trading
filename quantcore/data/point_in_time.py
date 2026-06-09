"""Point-in-time feature storage and bias-detection helpers.

The rebuild requires every feature record to carry both:

* ``as_of`` — what calendar/bar date the observation describes.
* ``valid_from`` — when that observation could first have been known.

The unified schema used by this module is:

``source, symbol, as_of, valid_from, feature_name, value``

``PointInTimeStore`` and ``get_features_as_of()`` are intentionally simple and
dependency-light so they can be used by tests, ingestion code, and the backtest
engine before any strategy code is written.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "source",
    "symbol",
    "as_of",
    "valid_from",
    "feature_name",
    "value",
)

_DEFAULT_SOURCE = "memory"
_DEFAULT_STORE: PointInTimeStore | None = None


@dataclass(frozen=True)
class FeatureRecord:
    """One point-in-time feature observation."""

    source: str
    symbol: str
    as_of: Any
    valid_from: Any
    feature_name: str
    value: Any


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    """Return a timezone-aware UTC pandas timestamp."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FEATURE_COLUMNS))


def _normalize_record(record: FeatureRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, FeatureRecord):
        data = {
            "source": record.source,
            "symbol": record.symbol,
            "as_of": record.as_of,
            "valid_from": record.valid_from,
            "feature_name": record.feature_name,
            "value": record.value,
        }
    else:
        data = dict(record)

    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Feature record missing required columns: {missing}")

    normalized = {
        "source": str(data["source"] or _DEFAULT_SOURCE),
        "symbol": str(data["symbol"]).upper(),
        "as_of": _coerce_timestamp(data["as_of"]),
        "valid_from": _coerce_timestamp(data["valid_from"]),
        "feature_name": str(data["feature_name"]),
        "value": data["value"],
    }
    if not normalized["symbol"]:
        raise ValueError("Feature record symbol cannot be empty")
    if not normalized["feature_name"]:
        raise ValueError("Feature record feature_name cannot be empty")
    return normalized


def feature_frame_from_records(
    records: Iterable[FeatureRecord | dict[str, Any]] | pd.DataFrame | None,
) -> pd.DataFrame:
    """Normalize records into the canonical feature-frame schema."""

    if records is None:
        return _empty_feature_frame()

    if isinstance(records, pd.DataFrame):
        if records.empty:
            return _empty_feature_frame()
        df = records.copy()
    else:
        df = pd.DataFrame([_normalize_record(record) for record in records])

    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Feature frame missing required columns: {missing}")

    df = df[list(FEATURE_COLUMNS)].copy()
    df["source"] = df["source"].fillna(_DEFAULT_SOURCE).astype(str)
    df["symbol"] = df["symbol"].fillna("").astype(str).str.upper()
    df["feature_name"] = df["feature_name"].fillna("").astype(str)

    if (df["symbol"] == "").any():
        raise ValueError("Feature frame contains empty symbol values")
    if (df["feature_name"] == "").any():
        raise ValueError("Feature frame contains empty feature_name values")

    df["as_of"] = df["as_of"].map(_coerce_timestamp)
    df["valid_from"] = df["valid_from"].map(_coerce_timestamp)
    return _sort_feature_frame(df)


def _sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_df = df.copy()
    sort_df["_as_of_sort"] = sort_df["as_of"].map(_coerce_timestamp)
    sort_df["_valid_from_sort"] = sort_df["valid_from"].map(_coerce_timestamp)
    sort_df = sort_df.sort_values(
        ["symbol", "feature_name", "_as_of_sort", "_valid_from_sort", "source"],
        kind="mergesort",
    )
    return sort_df.drop(columns=["_as_of_sort", "_valid_from_sort"]).reset_index(drop=True)


class PointInTimeStore:
    """In-memory canonical feature store with point-in-time query semantics."""

    def __init__(self, records: Iterable[FeatureRecord | dict[str, Any]] | pd.DataFrame | None = None) -> None:
        self._df = feature_frame_from_records(records)

    @property
    def data(self) -> pd.DataFrame:
        """Return a sorted copy of all stored records."""

        return self._df.copy()

    def append(self, records: Iterable[FeatureRecord | dict[str, Any]] | pd.DataFrame) -> int:
        """Append records and return the number of rows added."""

        new_df = feature_frame_from_records(records)
        if new_df.empty:
            return 0
        self._df = _sort_feature_frame(pd.concat([self._df, new_df], ignore_index=True))
        return len(new_df)

    def upsert(self, records: Iterable[FeatureRecord | dict[str, Any]] | pd.DataFrame) -> int:
        """Replace existing records with the same natural key and append the rest."""

        new_df = feature_frame_from_records(records)
        if new_df.empty:
            return 0

        key_cols = ["source", "symbol", "as_of", "valid_from", "feature_name"]
        old_keys = self._df[key_cols].drop_duplicates() if not self._df.empty else _empty_feature_frame()
        new_keys = new_df[key_cols].drop_duplicates()
        if old_keys.empty:
            existing = self._df.copy()
        else:
            existing = self._df.merge(
                new_keys.assign(_duplicate=True),
                on=key_cols,
                how="left",
                indicator=True,
            )
            existing = existing.loc[existing["_merge"] == "left_only"].drop(columns=["_merge", "_duplicate"])

        self._df = _sort_feature_frame(pd.concat([existing, new_df], ignore_index=True))
        return len(new_df)

    def query(
        self,
        symbol: str,
        bar_timestamp: Any,
        *,
        feature_names: Iterable[str] | None = None,
        source: str | None = None,
        as_of: Any | None = None,
        latest_only: bool = True,
    ) -> pd.DataFrame:
        """Return feature values known at ``bar_timestamp``.

        The only hard gate is ``valid_from <= bar_timestamp``. If ``latest_only``
        is true, the latest known value per ``feature_name`` is returned.
        """

        if self._df.empty:
            return _empty_feature_frame()

        bar_ts = _coerce_timestamp(bar_timestamp)
        df = self._df.copy()
        mask = df["valid_from"].map(_coerce_timestamp) <= bar_ts
        mask &= df["symbol"].eq(symbol.upper())

        if source is not None:
            mask &= df["source"].eq(source)
        if feature_names is not None:
            mask &= df["feature_name"].isin(list(feature_names))
        if as_of is not None:
            mask &= df["as_of"].eq(_coerce_timestamp(as_of))

        out = df.loc[mask, list(FEATURE_COLUMNS)].copy()
        if out.empty:
            return out

        if latest_only:
            out["_valid_from_sort"] = out["valid_from"].map(_coerce_timestamp)
            out = (
                out.sort_values(["feature_name", "_valid_from_sort"], kind="mergesort")
                .drop_duplicates("feature_name", keep="last")
                .drop(columns=["_valid_from_sort"])
            )

        return out.sort_values("feature_name", kind="mergesort").reset_index(drop=True)

    def get_features_as_of(
        self,
        symbol: str,
        bar_timestamp: Any,
        *,
        feature_names: Iterable[str] | None = None,
        source: str | None = None,
        as_of: Any | None = None,
        latest_only: bool = True,
    ) -> pd.DataFrame:
        """Alias for ``query`` matching the required API name."""

        return self.query(
            symbol,
            bar_timestamp,
            feature_names=feature_names,
            source=source,
            as_of=as_of,
            latest_only=latest_only,
        )


def get_default_store() -> PointInTimeStore:
    """Return the module-level default store used by ``get_features_as_of``."""

    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = PointInTimeStore()
    return _DEFAULT_STORE


def clear_default_store() -> None:
    """Clear the module-level default store. Intended for tests."""

    global _DEFAULT_STORE
    _DEFAULT_STORE = None


def get_features_as_of(
    symbol: str,
    bar_timestamp: Any,
    *,
    store: PointInTimeStore | None = None,
    feature_names: Iterable[str] | None = None,
    source: str | None = None,
    as_of: Any | None = None,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Query the canonical feature store as of ``bar_timestamp``.

    This is the required point-in-time API: records with ``valid_from`` after the
    bar timestamp are never returned.
    """

    active_store = store or get_default_store()
    return active_store.get_features_as_of(
        symbol,
        bar_timestamp,
        feature_names=feature_names,
        source=source,
        as_of=as_of,
        latest_only=latest_only,
    )


def find_future_leakage(records: Iterable[FeatureRecord | dict[str, Any]] | pd.DataFrame, bar_timestamp: Any) -> pd.DataFrame:
    """Return records that would be leaked if queried at ``bar_timestamp``."""

    df = feature_frame_from_records(records)
    if df.empty:
        return df
    bar_ts = _coerce_timestamp(bar_timestamp)
    return df.loc[df["valid_from"].map(_coerce_timestamp) > bar_ts].reset_index(drop=True)


def assert_no_future_leakage(records: Iterable[FeatureRecord | dict[str, Any]] | pd.DataFrame, bar_timestamp: Any) -> None:
    """Raise ``AssertionError`` if any record is valid after ``bar_timestamp``."""

    leaked = find_future_leakage(records, bar_timestamp)
    assert leaked.empty, f"Future leakage detected: {leaked.to_dict(orient='records')}"


def detect_survivorship_bias(universe_asof: Iterable[str], universe_current: Iterable[str]) -> list[str]:
    """Return current-universe symbols absent from the historical as-of universe."""

    return sorted(set(str(s).upper() for s in universe_current) - set(str(s).upper() for s in universe_asof))


def _records_frame(records: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame(list(records))


def detect_corporate_action_lookahead(
    prices: pd.DataFrame,
    actions: Iterable[dict[str, Any]] | pd.DataFrame,
    as_of: Any,
) -> pd.DataFrame:
    """Find price rows where a corporate action was applied before it was known.

    ``actions`` must include ``date`` and ``valid_from``. A violation is raised
    when a price row is on/after the action effective date but before the action
    ``valid_from`` timestamp.
    """

    as_of_ts = _coerce_timestamp(as_of)
    price_df = prices.copy()
    if "timestamp" not in price_df.columns:
        raise ValueError("prices must contain a 'timestamp' column")
    price_df["timestamp"] = price_df["timestamp"].map(_coerce_timestamp)
    price_df = price_df.loc[price_df["timestamp"] <= as_of_ts]

    action_df = _records_frame(actions)
    if action_df.empty or price_df.empty:
        return pd.DataFrame()
    if "date" not in action_df.columns or "valid_from" not in action_df.columns:
        raise ValueError("actions must contain 'date' and 'valid_from' columns")

    action_df["date"] = action_df["date"].map(_coerce_timestamp)
    action_df["valid_from"] = action_df["valid_from"].map(_coerce_timestamp)

    violations: list[pd.DataFrame] = []
    for _, price in price_df.iterrows():
        future_actions = action_df.loc[
            (action_df["date"] <= price["timestamp"])
            & (action_df["valid_from"] > price["timestamp"])
        ]
        if not future_actions.empty:
            row = price.to_frame().T
            row["_future_action_count"] = len(future_actions)
            violations.append(row)

    if not violations:
        return pd.DataFrame()
    return pd.concat(violations, ignore_index=True)


def _event_lookahead(
    events: Iterable[dict[str, Any]] | pd.DataFrame,
    bar_timestamp: Any,
    required_columns: tuple[str, str],
) -> pd.DataFrame:
    as_of_col, valid_from_col = required_columns
    df = _records_frame(events)
    if df.empty:
        return df
    missing = [col for col in (as_of_col, valid_from_col) if col not in df.columns]
    if missing:
        raise ValueError(f"events must contain columns: {missing}")

    bar_ts = _coerce_timestamp(bar_timestamp)
    df = df.copy()
    df[as_of_col] = df[as_of_col].map(_coerce_timestamp)
    df[valid_from_col] = df[valid_from_col].map(_coerce_timestamp)
    return df.loc[(df[as_of_col] <= bar_ts) & (df[valid_from_col] > bar_ts)].reset_index(drop=True)


def detect_earnings_lookahead(earnings: Iterable[dict[str, Any]] | pd.DataFrame, bar_timestamp: Any) -> pd.DataFrame:
    """Find earnings records whose announcement was not known at the bar."""

    return _event_lookahead(earnings, bar_timestamp, ("as_of", "valid_from"))


def detect_macro_release_lag(macro: Iterable[dict[str, Any]] | pd.DataFrame, bar_timestamp: Any) -> pd.DataFrame:
    """Find macro observations released after the bar timestamp."""

    return _event_lookahead(macro, bar_timestamp, ("as_of", "valid_from"))
