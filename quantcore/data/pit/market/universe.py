"""Survivorship-free liquid-equity universe selection (Polygon-backed).

The core survivorship fix: we keep **delisted** tickers (``active=false``), so a
backtest at ``T`` never assumes a symbol still trades that had already delisted.
A delisted ticker is simply carried in the universe table with a non-null
``delisted_utc``; the as-of reader drops it for dates after delisting.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from quantcore.data.pit.market.client import PolygonPitClient
from quantcore.data.pit.market.schema import UNIVERSE_COLUMNS
from quantcore.data.pit.market.store import ParquetPitStore

_DEFAULT_MIN_MARKET_CAP = 2_000_000_000  # $2B — a liquid large/mid-cap floor


def _row_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalise raw ticker-reference rows into the canonical universe schema."""
    normalized = []
    for r in rows:
        delisted = r.get("delisted_utc")
        normalized.append({
            "symbol": r.get("ticker"),
            "name": r.get("name"),
            "market_cap": r.get("market_cap"),
            "shares_outstanding": r.get("shares_outstanding"),
            "type": r.get("type"),
            "delisted_utc": delisted,
            "last_updated": r.get("updated") or datetime.now(UTC).isoformat(),
        })
    if not normalized:
        return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))
    df = pd.DataFrame(normalized)
    for col in ("market_cap", "shares_outstanding"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delisted_utc"] = pd.to_datetime(df["delisted_utc"], utc=True, errors="coerce")
    df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
    return df[list(UNIVERSE_COLUMNS)]


def select_universe(
    client: PolygonPitClient,
    *,
    min_market_cap: float = _DEFAULT_MIN_MARKET_CAP,
    top_n: int | None = 500,
    max_delisted: int | None = None,
) -> pd.DataFrame:
    """Return liquid active names (top ``top_n`` by market cap) plus the delisted tail.

    ``min_market_cap`` and ``top_n`` rank **active** names by *current* market
    cap. Delisted tickers are appended afterwards **without** the market-cap
    floor: their current ``market_cap`` is typically null/0, and applying the
    floor there would silently drop every company that died (or was acquired)
    during the backtest window — rebuilding the survivorship bias this layer
    exists to remove.

    ``max_delisted`` optionally bounds the historical tail (the full inactive
    reference can be tens of thousands of names; ingesting bars for all of them
    is expensive). Keep it ``None`` for correctness, or set a ceiling for the
    first pilot.

    .. note:: Present-day market-cap ranking is a *proxy* for point-in-time
       membership. A fully PIT universe needs historical constituent
       membership (e.g. S&P 500 constituents history) or PIT market-cap
       ranking computed from the ingested bars (``shares_outstanding_t *
       price_t``); until then a name that was large in the window but is dead
       today only survives if it still ranks here. See the audit's Step 3.
    """
    active = client.all_tickers(active=True)
    active_df = _row_to_frame(active)

    # The delisted tail is survivorship-critical — retain it unconditionally.
    delisted = client.all_tickers(active=False)
    delisted_df = _row_to_frame(delisted)

    if min_market_cap is not None:
        active_df = active_df[active_df["market_cap"].fillna(0.0) >= min_market_cap]
    active_df = active_df.sort_values("market_cap", ascending=False)
    if top_n is not None:
        active_df = active_df.head(top_n)

    if max_delisted is not None:
        delisted_df = delisted_df.sort_values(
            "market_cap", ascending=False, na_position="last"
        ).head(max_delisted)

    df = pd.concat([active_df, delisted_df], ignore_index=True).drop_duplicates(
        subset=["symbol"], keep="first"
    )
    return df.reset_index(drop=True)


def save_universe(
    store: ParquetPitStore,
    universe_df: pd.DataFrame,
    asof: date | None = None,
) -> None:
    """Persist a universe snapshot keyed by its as-of date (survivorship-free)."""
    asof = asof or date.today()
    if universe_df.empty:
        raise ValueError("Refusing to save an empty universe snapshot")
    store.write("universe", shard=asof.isoformat(), df=universe_df, period="snapshot")


def current_symbols(store: ParquetPitStore, asof: date | None = None) -> list[str]:
    """Return the symbols that were trading on ``asof``.

    Uses the most recent universe snapshot (survivorship is handled by
    ``delisted_utc``, so a fresh snapshot is always valid for any historical
    as-of). Returns ``[]`` if no snapshot exists yet — run
    ``polygon_ingest universe`` first.
    """
    snapshots = store.list_shards("universe")
    if not snapshots:
        return []
    chosen = snapshots[-1]  # most recent snapshot
    df = store.read("universe", shards=[chosen])
    if df.empty:
        return []
    asof_ts = (
        pd.Timestamp(asof).tz_localize("UTC")
        if asof is not None
        else pd.Timestamp.now("UTC")
    )
    delisted = pd.to_datetime(df["delisted_utc"], utc=True, errors="coerce")
    return sorted(df.loc[delisted.isna() | (delisted > asof_ts), "symbol"].dropna().tolist())
