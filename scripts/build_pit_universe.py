#!/usr/bin/env python
"""Phase 6, Step 1: Build a point-in-time S&P 500 price panel.

For every trading day T, the universe = S&P 500 constituents as of T, INCLUDING
delisted names (they trade until they drop, then vanish). Implemented by:
  1. Fetch current + historical constituents from Wikipedia (browser User-Agent).
  2. Fetch daily OHLCV from Alpaca per ticker (this alpaca version rejects lists).
  3. For each ticker, KEEP only bars within its index-membership window [added, removed).
  4. Concat -> long/tidy panel the BacktestEngine consumes.

Caveats (honest labeling):
  - Wikipedia's removed-company table is best-effort, not institutional grade.
  - Alpaca price history for delisted names is as-available.
  - This REDUCES survivorship bias (no winner-picking; dead names included) but
    does not fully eliminate it (constituent add/remove dates are approximate).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except Exception:
    pass

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_membership():
    import requests
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(WIKI, headers=h, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(r.text)

    cur = tables[0][["Symbol", "Date added"]].copy()
    cur["added"] = pd.to_datetime(cur["Date added"], errors="coerce")
    members = {}
    for _, row in cur.iterrows():
        s = str(row["Symbol"]).strip().replace(".", "-")
        members[s] = (row["added"], pd.NaT)

    removed = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("emoved" in c for c in cols) and any("ffective" in c for c in cols):
            removed = t
            break
    if removed is not None:
        rem = removed.copy()
        rem.columns = ["_".join(str(c) for c in col).strip("_") for col in rem.columns]
        ticker_col = [c for c in rem.columns if c.endswith("Ticker") and "Removed" in c]
        date_col = [c for c in rem.columns if "ffective" in c]
        add_col = [c for c in rem.columns if c.endswith("Ticker") and "Added" in c]
        if ticker_col and date_col:
            for _, row in rem.iterrows():
                s = str(row[ticker_col[0]]).strip().replace(".", "-")
                eff = pd.to_datetime(row[date_col[0]], errors="coerce")
                added = pd.to_datetime(row[add_col[0]], errors="coerce") if add_col else pd.NaT
                if s in members and pd.isna(members[s][1]):
                    members[s] = (added if pd.notna(added) else members[s][0], eff)
                else:
                    members[s] = (added, eff)
    return members


def fetch_alpaca_prices(symbols, start, end):
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"],
        os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        api_version="v2")
    out = {}
    for sym in symbols:
        qsym = sym.replace("-", ".")  # Alpaca uses dots (BRK.B)
        for attempt in range(3):
            try:
                bars = api.get_bars(qsym, tradeapi.TimeFrame.Day, start, end, adjustment="all")
                df = bars.df
                if df is None or len(df) == 0:
                    break
                df = df.reset_index()
                df["symbol"] = sym
                out[sym] = df
                break
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        time.sleep(0.05)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--output", default="data/sp500_pit_universe_2018_2024.parquet")
    ap.add_argument("--limit", type=int, default=0, help="Test mode: only first N tickers")
    ap.add_argument("--csv", action="store_true", help="also write a CSV sidecar")
    args = ap.parse_args()

    members = fetch_membership()
    print(f"[membership] {len(members)} tickers from Wikipedia")
    syms = list(members.keys())
    # Skip tickers delisted entirely before the range (yield zero in-range bars).
    syms = [s for s in syms
            if not (pd.notna(members[s][1]) and pd.Timestamp(members[s][1]) <= pd.Timestamp(args.start))]
    print(f"[membership] {len(syms)} tickers overlap the range (after pre-delisting filter)")
    if args.limit:
        syms = syms[:args.limit]

    panels = fetch_alpaca_prices(syms, args.start, args.end)
    print(f"[prices] fetched {len(panels)}/{len(syms)}")

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    rows = []
    for sym, df in panels.items():
        added, removed = members.get(sym, (start_ts, pd.NaT))
        if pd.isna(added):
            added = start_ts
        if pd.isna(removed):
            removed = end_ts
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None).dt.normalize()
        d = d[(d["timestamp"] >= added) & (d["timestamp"] < removed) &
              (d["timestamp"] >= start_ts) & (d["timestamp"] <= end_ts)]
        if len(d) == 0:
            continue
        for c in ["open", "high", "low", "close", "volume"]:
            d[c] = d[c].ffill(limit=5)
        d = d.dropna(subset=["close"])
        rows.append(d[["symbol", "timestamp", "open", "high", "low", "close", "volume"]])

    long = (pd.concat(rows, ignore_index=True)
              .sort_values(["timestamp", "symbol"]).reset_index(drop=True))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    # CSV is the primary consumption format (engine reads CSV; avoids pyarrow dep).
    csv_path = out.with_suffix(".csv")
    long.to_csv(csv_path, index=False)
    saved.append(str(csv_path))
    try:
        long.to_parquet(out, index=False)
        saved.append(str(out))
    except Exception as e:
        print(f"[warn] parquet skipped: {type(e).__name__}: {e}")
    if args.csv and not str(out).endswith(".csv"):
        long.to_csv(str(out.with_suffix("")) + "_csv.csv", index=False)
    print(f"[done] tickers={long['symbol'].nunique()} rows={len(long)} "
          f"coverage {long['timestamp'].min().date()}..{long['timestamp'].max().date()}")
    print(f"       saved: {saved}")


if __name__ == "__main__":
    raise SystemExit(main())
