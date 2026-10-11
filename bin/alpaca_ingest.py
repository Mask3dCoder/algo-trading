#!/usr/bin/env python3
"""CLI for the Alpaca PIT market-data ingestion pipeline.

Alpaca is the primary source for the decisive panel (amended 2026-10-11 in
``docs/cross_sectional_factor_protocol.md``); Polygon/Massive is not
affordable. Output lands in ``$QUANTCORE_PIT_MARKET_DIR`` or
``<repo>/data/pit/market`` (gitignored).

Examples
--------
    python bin/alpaca_ingest.py status
    python bin/alpaca_ingest.py panel --symbols-file data/sp500_pit_universe_2018_2024.csv \
        --start 2018-01-02 --end 2024-12-31
    python bin/alpaca_ingest.py bars --symbols AAPL,ATVI --start 2018-01-02 --end 2024-12-31
    python bin/alpaca_ingest.py panel --from-store --start 2018-01-02 --end 2024-12-31

Requires a real ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` in ``.env``.

Delisted names are a supported input, not an error: the dead tail is the point
of a survivorship-free panel and Alpaca retains that history.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from quantcore.data.pit.market import default_market_data_dir  # noqa: E402
from quantcore.data.pit.market.alpaca_ingest import AlpacaPitIngestor  # noqa: E402
from quantcore.data.pit.market.store import ParquetPitStore  # noqa: E402
from quantcore.data.sources.alpaca import (  # noqa: E402
    DATA_BASE_URL,
    AlpacaClientError,
    AlpacaDataClient,
)

_PLACEHOLDER_MARKERS = ("your", "here", "placeholder", "changeme")

#: Alpaca writes share classes with a dot (BRK.B); some universe files carry a
#: dash (BRK-B). The data API 400s on the dash form, so normalise before use.
_TICKER_FIXES = {"BRK-B": "BRK.B", "BF-B": "BF.B"}


def _resolve_credentials() -> tuple[str, str]:
    key = (os.environ.get("ALPACA_API_KEY") or "").strip()
    secret = (os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET") or "").strip()

    if not key or not secret:
        raise AlpacaClientError(
            "ALPACA_API_KEY / ALPACA_API_SECRET are not set in .env. Sign up at "
            "https://app.alpaca.markets (free IEX tier) and add both keys."
        )
    for name, value in (("ALPACA_API_KEY", key), ("ALPACA_API_SECRET", secret)):
        if any(m in value.lower() for m in _PLACEHOLDER_MARKERS) or len(value) < 16:
            raise AlpacaClientError(
                f"{name} in .env looks like a placeholder ({value[:4]}...)."
            )
    return key, secret


def _build_client() -> AlpacaDataClient:
    key, secret = _resolve_credentials()
    # ALPACA_BASE_URL points at the *paper trading* host in this repo's .env,
    # which is a different service from the market-data API.
    base = os.environ.get("ALPACA_DATA_REST_URL") or DATA_BASE_URL
    return AlpacaDataClient(key, secret, base_url=base)


def _normalise_symbol(raw: str) -> str:
    sym = raw.strip().upper()
    return _TICKER_FIXES.get(sym, sym)


def _symbols_from_file(path: Path) -> list[str]:
    """Read a symbol list from a CSV (first column or a 'symbol' column)."""
    df = pd.read_csv(path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return sorted({_normalise_symbol(s) for s in df[col].dropna().unique()})


def _resolve_symbols(args: argparse.Namespace, store: ParquetPitStore) -> list[str]:
    if getattr(args, "symbols", None):
        return [_normalise_symbol(s) for s in str(args.symbols).split(",") if s.strip()]
    if getattr(args, "symbols_file", None):
        return _symbols_from_file(Path(args.symbols_file))
    if getattr(args, "from_store", False):
        shards = store.list_shards("equity_bars")
        return sorted(shards)
    raise SystemExit("Provide --symbols, --symbols-file, or --from-store.")


def _build_store() -> ParquetPitStore:
    return ParquetPitStore(default_market_data_dir())


def cmd_status(_: argparse.Namespace, store: ParquetPitStore) -> int:
    report = store.status()
    if not report:
        print("Store is empty. Start with `alpaca_ingest.py panel`.")
        return 0
    for dataset, info in sorted(report.items()):
        print(f"  {dataset:26s} shards={info['shards']:>5}  bytes={info['bytes']:>12,}")
    return 0


def cmd_bars(args: argparse.Namespace, store: ParquetPitStore) -> int:
    symbols = _resolve_symbols(args, store)
    with _build_client() as client:
        ingestor = AlpacaPitIngestor(client, store)
        report = ingestor.ingest_bars(symbols, args.start, args.end, adjustment=args.adjustment)
    synced = sum(1 for v in report.values() if v)
    print(f"Bars ({args.adjustment}) synced for {synced}/{len(symbols)} symbols -> {store.root}")
    return 0 if synced else 1


def cmd_panel(args: argparse.Namespace, store: ParquetPitStore) -> int:
    symbols = _resolve_symbols(args, store)
    print(f"Ingesting panel for {len(symbols)} symbols: {args.start} -> {args.end}")
    with _build_client() as client:
        ingestor = AlpacaPitIngestor(client, store)
        summary = ingestor.ingest_panel(symbols, args.start, args.end)
    print(f"raw series   : {summary['raw_fetched']} fetched, {summary['raw_current']} already current")
    print(f"adjusted     : {summary['adj_fetched']} fetched, {summary['adj_current']} already current")
    print(f"factor rows  : {summary['factor_rows']:,}")
    print(f"action rows  : {summary['action_rows']:,}")
    return 0


def cmd_derive(args: argparse.Namespace, store: ParquetPitStore) -> int:
    """Re-derive factors/actions from what is already on disk (no network)."""
    symbols = _resolve_symbols(args, store)
    ingestor = AlpacaPitIngestor(_build_client(), store)
    factors = ingestor.derive_adjustment_factors(symbols)
    actions = ingestor.derive_corp_actions(symbols, coverage_end=args.end)
    print(f"Derived {factors:,} factor rows and {actions:,} corporate-action rows.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Alpaca market data into the PIT store")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", default="2018-01-02")
    common.add_argument("--end", default="2024-12-31")
    common.add_argument("--symbols", help="Comma-separated list, e.g. AAPL,ATVI")
    common.add_argument("--symbols-file", help="CSV with a symbol column")
    common.add_argument("--from-store", action="store_true", help="Use existing equity_bars shards")

    s = sub.add_parser("status", help="Show store contents")
    s.set_defaults(func=cmd_status)

    b = sub.add_parser("bars", parents=[common], help="Fetch daily bars only")
    b.add_argument("--adjustment", default="raw", choices=["raw", "split", "all"])
    b.set_defaults(func=cmd_bars)

    pa = sub.add_parser("panel", parents=[common], help="Raw + adjusted bars, then derive factors/actions")
    pa.set_defaults(func=cmd_panel)

    d = sub.add_parser("derive", parents=[common], help="Re-derive factors/actions from disk (offline)")
    d.set_defaults(func=cmd_derive)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    store = _build_store()
    try:
        return int(args.func(args, store))
    except AlpacaClientError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
