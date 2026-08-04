#!/usr/bin/env python
"""Phase 5, Step A: fetch a REAL, survivorship-aware OHLCV panel.

Correction to the blueprint prose
--------------------------------
The blueprint's Step A2 says "pivot to a wide DataFrame (columns = tickers)".
That is NOT what quantcore.backtest.engine.BacktestEngine consumes: it reads
TIDY/long bars with columns [symbol, timestamp, open, high, low, close, volume]
and groups by timestamp. So this script writes the LONG format as the primary
output (the path the sweep reads) and also writes a *_wide.csv sidecar for
human inspection. Do not "pivot" the primary artifact or the sweep will fail.

Survivorship note (honest labeling)
-----------------------------------
The blueprint calls this "zero survivorship bias". That is inaccurate: a static
list of KNOWN mega-cap survivors is survivorship-SELECTED (it biases toward
winners and cannot show a strategy bleeding on a name that later died). What it
DOES avoid is the "backfill illusion" (yfinance dropping dead tickers so the
panel looks artificially clean). The --drop_dead_before guard below is an
anti-delisting check, not an anti-survivorship cure. We label the output
accordingly and treat any edge found as upper-bound-biased until validated on a
point-in-time constituent list.

Sources
-------
  equities/etfs : Alpaca (IBKR gateway is offline in this env; Alpaca API is up)
  crypto        : ccxt/Binance (--source ccxt)

Usage
-----
  .venv/Scripts/python.exe scripts/fetch_real_panel.py \
      --start 2018-01-01 --end 2024-12-31 \
      --output data/real_panel_2018_2024.csv
  .venv/Scripts/python.exe scripts/fetch_real_panel.py --source ccxt \
      --symbols BTC/USDT,ETH/USDT,SOL/USDT --start 2020-01-01 \
      --output data/crypto_panel.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except Exception:
    pass

# Static, known-survivor universe (biases toward winners by construction).
EQUITY_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "BRK-B", "JPM", "UNH", "JNJ", "VTI",
]


# ---------------------------------------------------------------------------
# Alpaca (equities / ETFs)
# ---------------------------------------------------------------------------
def fetch_alpaca(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import alpaca_trade_api as tradeapi

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not (key and secret):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in .env")

    api = tradeapi.REST(key, secret, base, api_version="v2")
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            # Alpaca uses a dot in the symbol (BRK.B), not a hyphen.
            qsym = sym.replace("-", ".")
            bars = api.get_bars(qsym, tradeapi.TimeFrame.Day, start, end, adjustment="all")
            df = bars.df
            if df is None or len(df) == 0:
                print(f"  [alpaca] {sym}: empty")
                continue
            df = df.reset_index()
            df["symbol"] = sym
            out[sym] = df
            print(f"  [alpaca] {sym}: {len(df)} bars "
                  f"{df['timestamp'].min().date()}..{df['timestamp'].max().date()}")
        except Exception as exc:
            print(f"  [alpaca] {sym}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
    if not out:
        raise RuntimeError("Alpaca returned no data for any symbol (check API key / plan limits)")
    return out


# ---------------------------------------------------------------------------
# ccxt / Binance (crypto)
# ---------------------------------------------------------------------------
def fetch_ccxt(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            rows = ex.fetch_ohlcv(sym, "1d", since=since_ms, limit=3000)
            if not rows:
                print(f"  [ccxt] {sym}: empty")
                continue
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = sym
            out[sym] = df
            print(f"  [ccxt] {sym}: {len(df)} bars "
                  f"{df['timestamp'].min().date()}..{df['timestamp'].max().date()}")
        except Exception as exc:
            print(f"  [ccxt] {sym}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
    if not out:
        raise RuntimeError("ccxt/Binance returned no data for any symbol")
    return out


# ---------------------------------------------------------------------------
# Assemble tidy panel
# ---------------------------------------------------------------------------
def assemble(panels: dict[str, pd.DataFrame], drop_dead_before: str):
    cols = ["open", "high", "low", "close", "volume"]
    keep = []
    dropped = []
    for sym, df in panels.items():
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None).dt.normalize()
        if d["timestamp"].max() < pd.Timestamp(drop_dead_before):
            dropped.append((sym, str(d["timestamp"].max().date())))
            continue
        keep.append(d[["symbol", "timestamp"] + cols])

    if not keep:
        raise RuntimeError(f"All symbols delisted before {drop_dead_before}")

    grid = (pd.concat([k[["symbol", "timestamp"]] for k in keep])
              .drop_duplicates()
              .sort_values(["timestamp", "symbol"])
              .reset_index(drop=True))
    assembled = grid.copy()
    for c in cols:
        w = pd.concat([k[["symbol", "timestamp", c]] for k in keep]).pivot(
            index="timestamp", columns="symbol", values=c)
        w = w.ffill(limit=5).bfill(limit=2)
        melted = w.reset_index().melt(id_vars="timestamp", var_name="symbol", value_name=c)
        assembled = assembled.merge(melted, on=["timestamp", "symbol"], how="left")
    assembled = assembled.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    survivors = [s for s in panels if s not in [d[0] for d in dropped]]
    if dropped:
        print(f"[assemble] dropped delisted: {dropped}")
    return assembled, survivors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["alpaca", "ccxt"], default="alpaca")
    ap.add_argument("--symbols", default=None,
                    help="Comma-separated. Equity default uses built-in survivor list.")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--drop_dead_before", default="2024-06-30",
                    help="Drop symbols whose last bar is before this date (anti-delist guard).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.source == "alpaca":
        symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else EQUITY_UNIVERSE
        print(f"[fetch] Alpaca equities: {len(symbols)} symbols {args.start}..{args.end}")
        panels = fetch_alpaca(symbols, args.start, args.end)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"[fetch] ccxt/Binance: {symbols} {args.start}..{args.end}")
        panels = fetch_ccxt(symbols, args.start, args.end)

    long, survivors = assemble(panels, args.drop_dead_before)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False)

    wide = long.pivot(index="timestamp", columns="symbol", values="close")
    wide.to_csv(Path(str(out.with_suffix("")) + "_wide.csv"))

    min_date = long["timestamp"].min().date()
    max_date = long["timestamp"].max().date()
    print(f"\n[done] survivors={len(survivors)}/{len(panels)}  rows={len(long)}")
    print(f"       coverage: {min_date} .. {max_date}")
    print(f"       long  -> {out}")
    if min_date > pd.Timestamp(args.start).date():
        print(f"  WARNING: data starts {min_date}, not {args.start} "
              f"(possible API plan window limit). Label accordingly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
