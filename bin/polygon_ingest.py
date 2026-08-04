#!/usr/bin/env python3
"""CLI for the PIT market-data ingestion pipeline.

Examples
--------
    python bin/polygon_ingest.py status
    python bin/polygon_ingest.py universe --top-n 500
    python bin/polygon_ingest.py bars --from-universe --start 2019-01-01 --end 2026-08-01
    python bin/polygon_ingest.py corp-actions --from-universe
    python bin/polygon_ingest.py backfill --start 2019-01-01 --end 2026-08-01 --top-n 500

Requires a **real** ``POLYGON_API_KEY`` in ``.env`` (the current value is a
placeholder). Output lands in ``$QUANTCORE_PIT_MARKET_DIR`` or
``<repo>/data/pit/market`` (gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from quantcore.data.pit.market import default_market_data_dir  # noqa: E402
from quantcore.data.pit.market import universe as universe_mod  # noqa: E402
from quantcore.data.pit.market.asof import PitReader  # noqa: E402
from quantcore.data.pit.market.client import PolygonClientError, PolygonPitClient  # noqa: E402
from quantcore.data.pit.market.ingest import PolygonPitIngestor  # noqa: E402
from quantcore.data.pit.market.store import ParquetPitStore  # noqa: E402

_PLACEHOLDER_MARKERS = ("your", "here", "placeholder", "changeme")


def _resolve_key() -> str:
    key = os.environ.get("MASSIVE_API_KEY", "") or os.environ.get("POLYGON_API_KEY", "")
    key = (key or "").strip()
    lowered = key.lower()
    if not key:
        raise PolygonClientError(
            "MASSIVE_API_KEY / POLYGON_API_KEY is not set in .env. Sign up at "
            "https://massive.com and add the real key to .env "
            "(either MASSIVE_API_KEY or POLYGON_API_KEY)."
        )
    if any(m in lowered for m in _PLACEHOLDER_MARKERS) or len(key) < 20:
        raise PolygonClientError(
            "MASSIVE_API_KEY / POLYGON_API_KEY in .env looks like a placeholder "
            f"({key[:4]}…{key[-4:]}). Replace it with a real Massive key before ingesting."
        )
    return key


def _build_store() -> ParquetPitStore:
    return ParquetPitStore(default_market_data_dir())


def _resolve_symbols(args: argparse.Namespace, store: ParquetPitStore) -> list[str]:
    if getattr(args, "symbols", None):
        return [s.strip().upper() for s in args.symbols if s.strip()]
    if getattr(args, "from_universe", False):
        symbols = universe_mod.current_symbols(store, asof=date.today())
        if not symbols:
            raise SystemExit("No universe snapshot found. Run `polygon_ingest.py universe` first.")
        return symbols
    raise SystemExit("Provide --symbols AAPL,MSFT or --from-universe.")


# --------------------------------------------------------------------------- commands
def cmd_check(_: argparse.Namespace, store: ParquetPitStore) -> int:
    """Verify connectivity with the real key against the current Massive hosts."""
    import httpx

    from quantcore.data.pit.market.client import BASE_URL
    from quantcore.data.pit.market.client import FLAT_FILE_ENDPOINT, FLAT_FILE_BUCKET, DAY_AGG_PREFIX

    key = _resolve_key()
    print(f"REST base URL : {BASE_URL}")
    print(f"Flat endpoint : {FLAT_FILE_ENDPOINT}  bucket={FLAT_FILE_BUCKET}  prefix={DAY_AGG_PREFIX}")

    ok = True
    try:
        with httpx.Client(base_url=BASE_URL, timeout=30.0,
                          headers={"Authorization": f"Bearer {key}"}) as client:
            resp = client.get("/v3/reference/tickers/SPY")
            print(f"REST /v3/reference/tickers/SPY -> HTTP {resp.status_code}")
            if resp.status_code == 200:
                r = resp.json().get("results") or {}
                print(f"  auth OK — ticker {r.get('ticker')} ({r.get('name')})")
            elif resp.status_code == 401:
                print("  ERROR: HTTP 401 - key rejected. Check MASSIVE_API_KEY / POLYGON_API_KEY.")
                ok = False
            else:
                print(f"  WARN: unexpected HTTP {resp.status_code}")
                ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: REST connectivity failed: {exc}")
        ok = False

    # Flat-file readiness: endpoint reachability + S3 cred presence (no download).
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(FLAT_FILE_ENDPOINT + "/")
            print(f"Flat endpoint reachable -> HTTP {r.status_code} (expected 403 = auth-gated, host live)")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: flat endpoint unreachable: {exc}")
    s3_id = os.environ.get("MASSIVE_S3_ACCESS_ID") or os.environ.get("POLYGON_S3_Access_ID")
    s3_secret = os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY") or os.environ.get("POLYGON_Secret_Access_Key")
    print(f"Flat-file S3 creds: {'present' if (s3_id and s3_secret) else 'MISSING (flat-file download will not work)'}")

    return 0 if ok else 2


def cmd_status(_: argparse.Namespace, store: ParquetPitStore) -> int:
    report = store.status()
    if not report:
        print("Store is empty. Start with `polygon_ingest.py universe`.")
        return 0
    for dataset, info in sorted(report.items()):
        print(f"  {dataset:20s} shards={info['shards']:>5}  bytes={info['bytes']:>12,}")
    return 0


def cmd_universe(args: argparse.Namespace, store: ParquetPitStore) -> int:
    key = _resolve_key()
    with PolygonPitClient(key, rate_limit_per_minute=args.rate_limit) as client:
        ingestor = PolygonPitIngestor(client, store)
        df = ingestor.ingest_universe(
            min_market_cap=args.min_market_cap,
            top_n=args.top_n,
            asof=date.today(),
        )
    print(f"Saved {len(df)}-symbol universe snapshot (survivorship-free).")
    return 0


def cmd_bars(args: argparse.Namespace, store: ParquetPitStore) -> int:
    key = _resolve_key()
    symbols = _resolve_symbols(args, store)
    with PolygonPitClient(key, rate_limit_per_minute=args.rate_limit) as client:
        ingestor = PolygonPitIngestor(client, store)
        report = ingestor.ingest_bars(symbols, args.start, args.end)
    synced = sum(1 for v in report.values() if v)
    print(f"Bars synced for {synced}/{len(symbols)} symbols -> {store.root}")
    return 0


def cmd_corp_actions(args: argparse.Namespace, store: ParquetPitStore) -> int:
    key = _resolve_key()
    symbols = _resolve_symbols(args, store)
    with PolygonPitClient(key, rate_limit_per_minute=args.rate_limit) as client:
        ingestor = PolygonPitIngestor(client, store)
        total = ingestor.ingest_corp_actions(symbols)
    print(f"Corporate actions stored: {total} rows")
    return 0


def cmd_backfill(args: argparse.Namespace, store: ParquetPitStore) -> int:
    """The main 500-ticker sweep: universe -> bars -> corporate actions."""
    key = _resolve_key()
    with PolygonPitClient(key, rate_limit_per_minute=args.rate_limit) as client:
        ingestor = PolygonPitIngestor(client, store)
        print("1/3 universe")
        ingestor.ingest_universe(min_market_cap=args.min_market_cap, top_n=args.top_n)
        symbols = universe_mod.current_symbols(store, asof=date.today())
        print(f"2/3 bars  ({len(symbols)} symbols, {args.start}..{args.end})")
        ingestor.ingest_bars(symbols, args.start, args.end)
        print("3/3 corporate actions")
        ingestor.ingest_corp_actions(symbols)
    print(f"Backfill complete -> {store.root}")
    return 0


def cmd_options_contracts(args: argparse.Namespace, store: ParquetPitStore) -> int:
    key = _resolve_key()
    underlyings = [u.strip().upper() for u in args.underlyings if u.strip()]
    with PolygonPitClient(key, rate_limit_per_minute=args.rate_limit) as client:
        ingestor = PolygonPitIngestor(client, store)
        total = ingestor.ingest_options_contracts(underlyings, as_of=args.as_of)
    print(f"Options contracts stored: {total} rows")
    return 0


def cmd_options_quotes(args: argparse.Namespace, store: ParquetPitStore) -> int:
    key = _resolve_key()
    contracts = [c.strip().upper() for c in args.contracts if c.strip()]
    with PolygonPitClient(key, rate_limit_per_minute=args.rate_limit) as client:
        ingestor = PolygonPitIngestor(client, store)
        total = ingestor.ingest_options_quotes(contracts, args.start, args.end)
    print(f"Options NBBO quotes stored: {total} rows (Advanced tier data)")
    return 0


def cmd_pit(args: argparse.Namespace, store: ParquetPitStore) -> int:
    """Print the as-of view for a symbol — the PIT correctness smoke test."""
    reader = PitReader(store)
    asof = args.as_of
    bars = reader.equity_bars_asof([args.symbol], asof)
    print(f"Bars for {args.symbol} known at {asof}: {len(bars)}")
    print(bars.tail(5).to_string(index=False))
    actions = reader.corp_actions_asof([args.symbol], asof)
    if not actions.empty:
        print(f"Corporate actions known at {asof}:")
        print(actions[["action_type", "symbol", "effective_date", "value"]].to_string(index=False))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polygon_ingest", description=__doc__)
    parser.add_argument("--rate-limit", type=int, default=5, help="Polygon calls/minute (Starter=5)")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Verify REST + flat-file connectivity with the real key").set_defaults(func=cmd_check)

    sub.add_parser("status", help="Show what's in the store").set_defaults(func=cmd_status)

    u = sub.add_parser("universe", help="Fetch survivorship-free liquid universe")
    u.add_argument("--min-market-cap", type=float, default=2_000_000_000)
    u.add_argument("--top-n", type=int, default=500)
    u.set_defaults(func=cmd_universe)

    b = sub.add_parser("bars", help="Sync daily bars (incremental)")
    b.add_argument("--symbols", help="Comma-separated tickers")
    b.add_argument("--from-universe", action="store_true")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.set_defaults(func=cmd_bars)

    c = sub.add_parser("corp-actions", help="Sync splits/dividends/delistings")
    c.add_argument("--symbols")
    c.add_argument("--from-universe", action="store_true")
    c.set_defaults(func=cmd_corp_actions)

    backfill = sub.add_parser("backfill", help="Universe + bars + corp-actions sweep")
    backfill.add_argument("--start", required=True)
    backfill.add_argument("--end", required=True)
    backfill.add_argument("--min-market-cap", type=float, default=2_000_000_000)
    backfill.add_argument("--top-n", type=int, default=500)
    backfill.set_defaults(func=cmd_backfill)

    oc = sub.add_parser("options-contracts", help="Sync option contract references")
    oc.add_argument("--underlyings", required=True)
    oc.add_argument("--as-of")
    oc.set_defaults(func=cmd_options_contracts)

    oq = sub.add_parser("options-quotes", help="Sync options NBBO quotes (Advanced tier)")
    oq.add_argument("--contracts", required=True)
    oq.add_argument("--start", required=True)
    oq.add_argument("--end", required=True)
    oq.set_defaults(func=cmd_options_quotes)

    pit = sub.add_parser("pit", help="Print the as-of PIT view for one symbol")
    pit.add_argument("--symbol", required=True)
    pit.add_argument("--as-of", default=date.today().isoformat())
    pit.set_defaults(func=cmd_pit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = _build_store()
    try:
        return args.func(args, store)
    except PolygonClientError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
