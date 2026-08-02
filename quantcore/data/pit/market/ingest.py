"""Incremental, watermark-aware ingestion from Polygon into the PIT store.

Design
------
* **Idempotent** — re-running a sync never duplicates rows (upsert by natural
  key) and never re-fetches what a watermark already covers.
* **Atomic** — each dataset/shard write and every watermark commit is a
  temp-file + ``os.replace`` so a crash leaves a consistent store.
* **Serial + rate-limited** — the Polygon Starter tier allows ~5 calls/min;
  ``backfill_bars`` resumes from the watermark if interrupted.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import pandas as pd

from quantcore.data.pit.market import universe as universe_mod
from quantcore.data.pit.market.client import PolygonPitClient
from quantcore.data.pit.market.corp_actions import (
    delisting_row,
    frame_from_rows,
    normalize_dividends,
    normalize_splits,
)
from quantcore.data.pit.market.schema import (
    EQUITY_BAR_COLUMNS,
    OPTION_CONTRACT_COLUMNS,
    OPTION_QUOTE_COLUMNS,
    period_shard,
)
from quantcore.data.pit.market.store import ParquetPitStore


def _iso_date(value: Any) -> str:
    d = pd.Timestamp(value)
    return d.strftime("%Y-%m-%d")


def _aware(value: Any) -> pd.Timestamp:
    """Coerce a timestamp to timezone-aware UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _normalize_bars(symbol: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(EQUITY_BAR_COLUMNS))
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["t"], unit="ms", utc=True)
    out = pd.DataFrame({
        "symbol": symbol,
        "ts": ts,
        "open": pd.to_numeric(df["o"], errors="coerce"),
        "high": pd.to_numeric(df["h"], errors="coerce"),
        "low": pd.to_numeric(df["l"], errors="coerce"),
        "close": pd.to_numeric(df["c"], errors="coerce"),
        "volume": pd.to_numeric(df["v"], errors="coerce"),
        "vwap": pd.to_numeric(df["vw"], errors="coerce"),
        "transactions": pd.to_numeric(df["n"], errors="coerce"),
    })
    return out.dropna(subset=["close"]).reset_index(drop=True)


def _normalize_contracts(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(OPTION_CONTRACT_COLUMNS))
    df = pd.DataFrame(rows)
    right = df["contract_type"].map(lambda v: str(v).upper()[0] if v else None)
    out = pd.DataFrame({
        "contract_symbol": df["ticker"],
        "underlying_symbol": df["underlying_ticker"],
        "expiration_date": pd.to_datetime(df["expiration_date"], errors="coerce").dt.date,
        "strike_price": pd.to_numeric(df["strike_price"], errors="coerce"),
        "right": right,
        "exercise_style": df.get("exercise_style"),
        "multiplier": pd.to_numeric(df["multiplier"], errors="coerce"),
        "cfi": df.get("cfi"),
        "primary_exchange": df.get("primary_exchange"),
        "created_at": pd.to_datetime(df["created_at"], utc=True, errors="coerce"),
        "updated_at": pd.to_datetime(df["updated_at"], utc=True, errors="coerce"),
    })
    return out.dropna(subset=["contract_symbol"]).reset_index(drop=True)


def _normalize_quotes(underlying: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(OPTION_QUOTE_COLUMNS))
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["t"], unit="ms", utc=True, errors="coerce")
    bid = pd.to_numeric(df["bid_price"], errors="coerce")
    ask = pd.to_numeric(df["ask_price"], errors="coerce")
    out = pd.DataFrame({
        "contract_symbol": df["sym"],
        "underlying_symbol": underlying,
        "ts": ts,
        "bid_price": bid,
        "bid_size": pd.to_numeric(df["bid_size"], errors="coerce"),
        "ask_price": ask,
        "ask_size": pd.to_numeric(df["ask_size"], errors="coerce"),
        "midpoint": (bid + ask) / 2.0,
        "spread": (ask - bid),
        "iv": None,
        "delta": None,
    })
    valid = out["bid_price"].notna() & out["ask_price"].notna() & out["ts"].notna()
    return out.loc[valid].reset_index(drop=True)


class PolygonPitIngestor:
    """Orchestrates Polygon fetches into the PIT parquet store."""

    def __init__(self, client: PolygonPitClient, store: ParquetPitStore) -> None:
        self.client = client
        self.store = store

    # -- universe -------------------------------------------------------------
    def ingest_universe(
        self,
        *,
        min_market_cap: float = 2_000_000_000,
        top_n: int | None = 500,
        asof: date | None = None,
    ) -> pd.DataFrame:
        asof = asof or date.today()
        df = universe_mod.select_universe(
            self.client, min_market_cap=min_market_cap, top_n=top_n
        )
        universe_mod.save_universe(self.store, df, asof=asof)
        return df

    # -- equity bars ----------------------------------------------------------
    def ingest_bars(
        self,
        symbols: Iterable[str],
        start: str | date,
        end: str | date,
        *,
        verbose: bool = True,
    ) -> dict[str, int]:
        """Sync daily bars for ``symbols`` over ``[start, end]`` (incremental)."""
        symbols = list(symbols)
        total = len(symbols)
        end_dt = _aware(end)
        report: dict[str, int] = {}
        for i, symbol in enumerate(symbols, start=1):
            watermark = self.store.get_watermark("equity_bars", symbol)
            if watermark is not None and watermark >= end_dt:
                report[symbol] = 0  # already current
                continue
            fetch_start = _aware(start)
            if watermark is not None:
                fetch_start = watermark.normalize()
            rows = self.client.eod_bars(symbol, _iso_date(fetch_start), _iso_date(end))
            df = _normalize_bars(symbol, rows)
            added = 0
            if not df.empty:
                added = self.store.upsert_shard(
                    "equity_bars", symbol, df, dedupe_cols=["symbol", "ts"]
                )
            self.store.set_watermark("equity_bars", symbol, end_dt)
            report[symbol] = len(df)
            if verbose and (i % 50 == 0 or i == 1 or i == total):
                print(f"  bars {i:>4}/{total} {symbol}: {added} rows")
        return report

    def backfill_bars(
        self,
        symbols: Iterable[str],
        start: str | date,
        end: str | date,
    ) -> dict[str, int]:
        """Full-history sweep with watermark resume (safe to interrupt)."""
        return self.ingest_bars(list(symbols), start, end)

    # -- corporate actions ----------------------------------------------------
    def ingest_corp_actions(
        self,
        symbols: Iterable[str],
        *,
        verbose: bool = True,
    ) -> int:
        rows: list[dict[str, Any]] = []
        symbols = list(symbols)
        for i, symbol in enumerate(symbols, start=1):
            for split in self.client.splits(symbol):
                rows.extend(normalize_splits(symbol, [split]))
            for div in self.client.dividends(symbol):
                rows.extend(normalize_dividends(symbol, [div]))
            info = self.client.ticker(symbol)
            delisted = info.get("delisted_utc")
            if delisted:
                row = delisting_row(symbol, delisted)
                if row:
                    rows.append(row)
            if verbose and (i % 100 == 0 or i == len(symbols)):
                print(f"  actions {i}/{len(symbols)} {symbol}")
        df = frame_from_rows(rows)
        if df.empty:
            return 0
        # Corp actions are global, stored under a single 'all' shard. upsert_shard
        # dedupes against whatever is already on disk, so re-runs are idempotent.
        return self.store.upsert_shard(
            "corp_actions",
            "all",
            df,
            dedupe_cols=["action_type", "symbol", "effective_date"],
        )

    # -- options --------------------------------------------------------------
    def ingest_options_contracts(
        self,
        underlyings: Iterable[str],
        as_of: str | date | None = None,
    ) -> int:
        asof_str = _iso_date(as_of) if as_of else None
        total = 0
        for underlying in underlyings:
            rows = self.client.options_contracts(underlying, as_of=asof_str)
            df = _normalize_contracts(rows)
            if df.empty:
                continue
            total += self.store.upsert_shard(
                "options_contracts",
                underlying,
                df,
                dedupe_cols=["contract_symbol"],
            )
        return total

    def ingest_options_quotes(
        self,
        contracts: Iterable[str],
        start: str | date,
        end: str | date,
        *,
        underlying_map: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> int:
        """Sync NBBO quotes for ``contracts`` over a range (Advanced tier).

        Quotes are written per contract-month partition (fresh files), so
        re-runs over the same month are idempotent rewrites, not duplicates.
        """
        end_dt = _aware(end)
        total = 0
        contracts = list(contracts)
        for i, contract in enumerate(contracts, start=1):
            watermark = self.store.get_watermark("options_quotes", contract)
            if watermark is not None and watermark >= end_dt:
                continue
            fetch_start = pd.Timestamp(start)
            if watermark is not None:
                fetch_start = watermark.normalize()
            rows = self.client.options_quotes(contract, _iso_date(fetch_start), _iso_date(end))
            underlying = (underlying_map or {}).get(contract, "")
            df = _normalize_quotes(underlying, rows)
            if not df.empty:
                df = df.sort_values("ts")
                for period, frame in df.groupby(df["ts"].map(lambda t: period_shard(t, "options_quotes"))):
                    self.store.write("options_quotes", contract, frame, str(period))
                    total += len(frame)
            self.store.set_watermark("options_quotes", contract, end_dt)
            if verbose and (i % 10 == 0 or i == 1 or i == len(contracts)):
                print(f"  quotes {i:>4}/{len(contracts)} {contract}: {len(df)} rows")
        return total

    def close(self) -> None:
        self.client.close()
