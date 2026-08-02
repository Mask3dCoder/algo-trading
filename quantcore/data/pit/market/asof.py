"""As-of readers — return exactly what the market knew at instant ``T``.

The point-in-time invariant for market data:

    bar  ``ts <= T``            (a daily bar is known at its close)
    quote ``ts <= T``           (an NBBO print is known when printed)
    action effective_date <= T  AND (announced_at is null OR announced_at <= T)
    symbol not yet delisted at T

Nothing that became known *after* ``T`` is ever returned, so these readers feed
lookahead-free backtests directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from quantcore.data.pit.market import universe as universe_mod
from quantcore.data.pit.market.store import ParquetPitStore


def _asof_ts(asof: str | date | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(asof)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class PitReader:
    """Query the PIT market-data store as of an instant."""

    def __init__(self, store: ParquetPitStore) -> None:
        self.store = store

    def equity_bars_asof(
        self,
        symbols: Iterable[str],
        asof: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return daily bars with ``ts <= asof`` for the given symbols."""
        target = _asof_ts(asof)
        df = self.store.read("equity_bars", shards=list(symbols))
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        out = df.loc[df["ts"] <= target].sort_values(["symbol", "ts"])
        # Collapse any accidental duplicate (symbol, ts) rows, keeping the last.
        return out.drop_duplicates(subset=["symbol", "ts"], keep="last").reset_index(drop=True)

    def corp_actions_asof(
        self,
        symbols: Iterable[str],
        asof: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return actions effective and announced on/before ``asof``."""
        target = _asof_ts(asof)
        df = self.store.read("corp_actions", shards=["all"])
        if df.empty:
            return df
        if symbols is not None:
            df = df.loc[df["symbol"].isin(set(symbols))]
        # effective_date is a calendar date (naive after the parquet round-trip);
        # compare it to asof at UTC midnight to keep tz-awareness consistent.
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
        df["announced_at"] = pd.to_datetime(df["announced_at"], utc=True, errors="coerce")
        known = df["effective_date"] <= target.tz_localize(None)
        announced = df["announced_at"].isna() | (df["announced_at"] <= target)
        out = df.loc[known & announced].sort_values(["symbol", "effective_date"])
        return out.reset_index(drop=True)

    def options_quotes_asof(
        self,
        contracts: Iterable[str],
        asof: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return NBBO quote prints with ``ts <= asof``."""
        target = _asof_ts(asof)
        df = self.store.read("options_quotes", shards=list(contracts))
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        out = df.loc[df["ts"] <= target].sort_values(["contract_symbol", "ts"])
        return out.drop_duplicates(subset=["contract_symbol", "ts"], keep="last").reset_index(drop=True)

    def options_contracts_asof(
        self,
        underlyings: Iterable[str],
        asof: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return option contract references for underlyings, created on/before asof."""
        target = _asof_ts(asof)
        df = self.store.read("options_contracts", shards=list(underlyings))
        if df.empty:
            return df
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        out = df.loc[df["created_at"].isna() | (df["created_at"] <= target)]
        return out.sort_values(["underlying_symbol", "contract_symbol"]).reset_index(drop=True)

    def universe_symbols_asof(self, asof: str | date | pd.Timestamp) -> list[str]:
        """Return the symbols trading as of ``asof`` (survivorship-free)."""
        target = _asof_ts(asof)
        return universe_mod.current_symbols(self.store, asof=target.date())

    def latest_close(
        self,
        symbol: str,
        asof: str | date | pd.Timestamp,
    ) -> float | None:
        """Return the most recent close ``<= asof`` for a symbol, or None."""
        df = self.equity_bars_asof([symbol], asof)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])
