"""Alpaca ingestion into the PIT parquet store.

Design constraints that shaped this module:

* **Raw and adjusted series are stored separately.** The engine trades on raw
  prices; mixing a vendor-adjusted series into ``equity_bars`` would delete
  splits from the execution price path. Keeping both also makes the implied
  adjustment factor computable, which is what stands in for the corporate-
  actions ledger Alpaca does not offer.
* **Only unambiguous actions are derived.** A split shows up as a >50%
  step in the adjustment factor and is recoverable. A dividend shows up as a
  sub-percent drift that cannot be attributed to a specific ex-date from this
  data with any confidence, so no dividend rows are fabricated. The
  delisting date IS recoverable and is the single most valuable row in the
  table for a survivorship-free panel: it is what tells a reader at time T
  that a name is dead.
* **The dead tail is first-class.** Terminal bars are never filtered, even at
  zero volume. That bar is the evidence the name stopped trading.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from quantcore.data.pit.market.ingest import _aware, _iso_date, _normalize_bars
from quantcore.data.pit.market.schema import ADJUSTMENT_FACTOR_COLUMNS
from quantcore.data.pit.market.store import ParquetPitStore
from quantcore.data.sources.alpaca import AlpacaDataClient

#: A factor step larger than this is treated as a split. Dividend-driven drift
#: on a liquid large-cap is well under 1%; a 2:1 split is a 100% step.
SPLIT_FACTOR_JUMP = 0.5

#: A name whose last bar is this many days old is treated as delisted rather
#: than merely stale (holidays, halted tickers, data lag all end much sooner).
DELISTED_GRACE_DAYS = 30

ADJUSTMENT_SOURCE = "alpaca_derived"

#: Shard for derived corporate actions, kept separate from any actions ingested
#: from a real corporate-actions feed so a rebuild can replace its own rows.
ACTION_SHARD = "alpaca_derived"


class AlpacaPitIngestor:
    """Orchestrates Alpaca fetches into the PIT parquet store."""

    def __init__(self, client: AlpacaDataClient, store: ParquetPitStore) -> None:
        self.client = client
        self.store = store

    # -- bars -----------------------------------------------------------------
    def ingest_bars(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        *,
        adjustment: str = "raw",
        dataset: str = "equity_bars",
        verbose: bool = True,
    ) -> dict[str, int]:
        """Sync daily bars into ``dataset`` (incremental, watermark-resumable).

        A symbol whose fetch returns nothing is recorded as 0 and NOT given a
        watermark advance on data, so a later retry can still pick it up.
        """
        symbols = list(symbols)
        total = len(symbols)
        end_dt = _aware(end)
        report: dict[str, int] = {}

        for i, symbol in enumerate(symbols, start=1):
            watermark = self.store.get_watermark(dataset, symbol)
            if watermark is not None and watermark >= end_dt:
                report[symbol] = 0
                continue

            fetch_start = _aware(start)
            if watermark is not None:
                fetch_start = watermark.normalize()

            rows = self.client.daily_bars(
                symbol, _iso_date(fetch_start), _iso_date(end), adjustment=adjustment
            )
            df = _normalize_bars(symbol, rows)
            added = 0
            if not df.empty:
                added = self.store.upsert_shard(dataset, symbol, df, dedupe_cols=["symbol", "ts"])
                self.store.set_watermark(dataset, symbol, end_dt)
            report[symbol] = len(df)
            if verbose and (i % 50 == 0 or i == 1 or i == total):
                print(f"  {dataset} {i:>4}/{total} {symbol}: {added} rows")

        return report

    # -- derived layer --------------------------------------------------------
    def derive_adjustment_factors(self, symbols: Iterable[str], *, verbose: bool = True) -> int:
        """Store factor = adjusted_close / raw_close per trading day.

        Returns the number of rows written. Symbols missing either series are
        skipped rather than filled with 1.0 — a fabricated factor of 1.0 would
        assert "no corporate action", which is exactly the kind of assumption
        that silently corrupts a backtest.
        """
        symbols = list(symbols)
        total = len(symbols)
        written = 0

        for i, symbol in enumerate(symbols, start=1):
            raw = self.store.read("equity_bars", [symbol])
            adj = self.store.read("equity_bars_total_return", [symbol])
            if raw is None or adj is None or raw.empty or adj.empty:
                continue

            merged = raw[["ts", "close"]].merge(
                adj[["ts", "close"]], on="ts", how="inner", suffixes=("_raw", "_adj")
            )
            merged = merged[(merged["close_raw"] > 0) & (merged["close_adj"] > 0)]
            if merged.empty:
                continue

            factors = pd.DataFrame(
                {
                    "symbol": symbol,
                    "ts": merged["ts"],
                    "factor": merged["close_adj"] / merged["close_raw"],
                    "source": ADJUSTMENT_SOURCE,
                }
            )
            written += self.store.upsert_shard(
                "adjustment_factors", symbol, factors[list(ADJUSTMENT_FACTOR_COLUMNS)],
                dedupe_cols=["symbol", "ts"],
            )
            if verbose and (i % 100 == 0 or i == total):
                print(f"  factors {i:>4}/{total} {symbol}: {len(factors)} rows")

        return written

    def derive_corp_actions(
        self,
        symbols: Iterable[str],
        *,
        coverage_end: Any = None,
        verbose: bool = True,
    ) -> int:
        """Write the corporate-action rows this source can actually support.

        Only two kinds qualify:

        * ``split`` — a factor step beyond SPLIT_FACTOR_JUMP, with the implied
          ratio recorded as ``value``. Derived, and flagged as such.
        * ``delisting`` — the terminal bar of a name that stopped trading,
          inferred from the last available bar rather than from a vendor field.

        ``coverage_end`` is the last date the panel actually covers, NOT today.
        Using today would mark every still-trading name as delisted the moment
        the ingest window ends in the past - a false positive on every symbol
        in a backfilled historical panel. A name counts as delisted only if its
        final bar predates coverage by more than DELISTED_GRACE_DAYS.

        Dividend rows are deliberately NOT written: the adjustment series moves
        by a fraction of a percent on an ex-date, which is not separable from
        vendor methodology noise using this data. Emitting them would be a
        fabricated ledger wearing a provenance tag.
        """
        symbols = list(symbols)
        total = len(symbols)
        rows: list[dict[str, Any]] = []
        cutoff = None
        if coverage_end is not None:
            cutoff = _aware(coverage_end) - pd.Timedelta(days=DELISTED_GRACE_DAYS)

        for i, symbol in enumerate(symbols, start=1):
            factors = self.store.read("adjustment_factors", [symbol])
            if factors is not None and not factors.empty:
                factors = factors.sort_values("ts")
                prev = factors["factor"].shift(1)
                jumps = factors[(factors["factor"] / prev - 1.0).abs() > SPLIT_FACTOR_JUMP]
                for _, r in jumps.iterrows():
                    ratio = float(r["factor"] / prev.loc[r.name])
                    rows.append(
                        {
                            "action_type": "split",
                            "symbol": symbol,
                            "effective_date": pd.Timestamp(r["ts"]).date(),
                            "announced_at": pd.NaT,
                            "value": ratio,
                            "ticker_old": symbol,
                            "ticker_new": symbol,
                            "description": (
                                f"derived from adjustment-factor step "
                                f"(factor x{ratio:.3f}); confirm against a primary source"
                            ),
                            "source": ADJUSTMENT_SOURCE,
                        }
                    )

            bars = self.store.read("equity_bars", [symbol])
            if bars is not None and not bars.empty and cutoff is not None:
                last_ts = pd.Timestamp(bars["ts"].max())
                if last_ts < cutoff:
                    rows.append(
                        {
                            "action_type": "delisting",
                            "symbol": symbol,
                            "effective_date": last_ts.date(),
                            "announced_at": pd.NaT,
                            "value": float(bars.sort_values("ts")["close"].iloc[-1]),
                            "ticker_old": symbol,
                            "ticker_new": None,
                            "description": (
                                f"no bars after {last_ts.date()}; inferred from terminal "
                                "bar, not from a vendor delisting field"
                            ),
                            "source": ADJUSTMENT_SOURCE,
                        }
                    )

            if verbose and (i % 100 == 0 or i == total):
                print(f"  actions {i:>4}/{total} {symbol}: {len(rows)} rows so far")

        if not rows:
            return 0
        df = pd.DataFrame(rows)
        # Written to a source-named shard, not 'all': a derived rebuild must be
        # able to replace its own rows wholesale without touching actions that
        # came from a real corporate-actions feed.
        return self.store.upsert_shard(
            "corp_actions",
            ACTION_SHARD,
            df,
            dedupe_cols=["action_type", "symbol", "effective_date"],
        )

    # -- one-shot panel -------------------------------------------------------
    def ingest_panel(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        *,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Fetch raw + total-return bars, then derive factors and actions.

        Order matters: the derived layer reads the stored raw and adjusted
        series rather than the in-flight payloads, so a resumed run derives
        from what is actually on disk. ``end`` doubles as the coverage
        boundary for delisting inference.
        """
        symbols = list(symbols)
        summary: dict[str, Any] = {"symbols": len(symbols)}

        raw = self.ingest_bars(symbols, start, end, adjustment="raw", dataset="equity_bars", verbose=verbose)
        adj = self.ingest_bars(
            symbols, start, end, adjustment="all", dataset="equity_bars_total_return", verbose=verbose
        )
        # Distinguish "already current, nothing fetched" from "no data exists".
        # Reporting these together makes a resumed run look like data loss.
        summary["raw_fetched"] = sum(1 for v in raw.values() if v)
        summary["raw_current"] = sum(1 for v in raw.values() if not v)
        summary["adj_fetched"] = sum(1 for v in adj.values() if v)
        summary["adj_current"] = sum(1 for v in adj.values() if not v)
        summary["factor_rows"] = self.derive_adjustment_factors(symbols, verbose=verbose)
        summary["action_rows"] = self.derive_corp_actions(symbols, coverage_end=end, verbose=verbose)
        return summary
