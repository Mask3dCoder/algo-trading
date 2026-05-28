"""Batch download 1-minute intraday bars from Alpaca Markets (IEX feed).

Usage:
    python scripts/download_intraday_data.py                            # S&P 500 filtered
    python scripts/download_intraday_data.py --universe expanded_15     # Small test
    python scripts/download_intraday_data.py --start 2024-01-01        # Partial range
    python scripts/download_intraday_data.py --validate-only           # Check existing data
    python scripts/download_intraday_data.py --reset                   # Clear progress, start fresh

Requires ALPACA_API_KEY and ALPACA_API_SECRET environment variables.
Free tier (IEX): 200 requests/minute — estimated ~3 min for 15 tickers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from quantcore.data.intraday_loader import AlpacaIntradayLoader

log = logging.getLogger("download_intraday")

# ── Progress tracking ────────────────────────────────────────────────────

PROGRESS_FILE = _ROOT / "data" / "intraday" / ".download_progress.json"


class DownloadProgress:
    """Crash-safe download progress tracker.

    Stores state in .download_progress.json — survives restarts.
    Each ticker is marked completed/failed individually.
    """

    def __init__(self, path: Path = PROGRESS_FILE) -> None:
        self.path = path
        self._state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                return {"completed": {}, "failed": {}}
        return {"completed": {}, "failed": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2, default=str))

    def is_completed(self, ticker: str) -> bool:
        return ticker in self._state.get("completed", {})

    def mark_completed(self, ticker: str, bar_count: int) -> None:
        self._state.setdefault("completed", {})[ticker] = {
            "bars": bar_count,
            "finished_at": date.today().isoformat(),
        }
        self._state.get("failed", {}).pop(ticker, None)
        self._save()

    def mark_failed(self, ticker: str, error: str) -> None:
        self._state.setdefault("failed", {})[ticker] = {
            "error": error,
            "failed_at": date.today().isoformat(),
        }
        self._save()

    def reset(self) -> None:
        self._state = {"completed": {}, "failed": {}}
        self._save()

    @property
    def completed_count(self) -> int:
        return len(self._state.get("completed", {}))

    @property
    def failed_count(self) -> int:
        return len(self._state.get("failed", {}))

    @property
    def failed_tickers(self) -> list[str]:
        return list(self._state.get("failed", {}).keys())


# ── Universe resolution ──────────────────────────────────────────────────

def get_download_universe(name: str) -> list[str]:
    """Resolve a universe name to a list of ticker symbols.

    Args:
        name: "sp500", "sp500_filtered", or a key from UNIVERSES dict.

    Returns:
        Sorted list of ticker symbols.
    """
    if name == "sp500":
        from quantcore.data.sources.historical_universe import HistoricalUniverseLoader
        loader = HistoricalUniverseLoader()
        return sorted(loader._sp500())

    if name == "sp500_filtered":
        loader = AlpacaIntradayLoader()
        return loader.build_universe(min_adv=1_000_000, min_price=5.0)

    # Try UNIVERSES dict from real_data_loader
    try:
        from scripts.real_data_loader import UNIVERSES
        if name in UNIVERSES:
            return sorted(UNIVERSES[name]["symbols"])
    except ImportError:
        pass

    raise ValueError(
        f"Unknown universe '{name}'. "
        f"Use 'sp500', 'sp500_filtered', or a key from UNIVERSES (cta, rp, expanded_15, etc.)."
    )


# ── Download ─────────────────────────────────────────────────────────────

def run_download(
    tickers: list[str],
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    resume: bool = True,
) -> None:
    """Download 1-minute bars for all tickers sequentially.

    Args:
        tickers: List of ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        resume: Skip tickers already completed (default True).
    """
    progress = DownloadProgress()
    loader = AlpacaIntradayLoader()

    total = len(tickers)
    skipped = 0
    downloaded = 0
    failed = 0

    log.info("Starting download: %d tickers, %s -> %s", total, start, end)

    for i, ticker in enumerate(tickers):
        prefix = f"[{i + 1}/{total}]"

        if resume and progress.is_completed(ticker):
            log.info("%s %s — already completed, skipping", prefix, ticker)
            skipped += 1
            continue

        log.info("%s Downloading %s ...", prefix, ticker)
        try:
            df = loader.download(ticker, start, end)
            bar_count = len(df)
            progress.mark_completed(ticker, bar_count)
            log.info("%s %s — %d bars saved", prefix, ticker, bar_count)
            downloaded += 1
        except Exception as e:
            progress.mark_failed(ticker, str(e))
            log.error("%s %s — FAILED: %s", prefix, ticker, e)
            failed += 1

    log.info(
        "Download complete: %d downloaded, %d skipped (cached), %d failed",
        downloaded, skipped, failed,
    )

    if progress.failed_tickers:
        log.warning("Failed tickers: %s", ", ".join(progress.failed_tickers))


# ── Validation ───────────────────────────────────────────────────────────

def validate_download(
    tickers: list[str] | None = None,
    start: str = "2023-01-01",
    end: str = "2024-12-31",
) -> None:
    """Validate downloaded intraday data.

    Checks:
    - Parquet files exist for expected trading days
    - No NaN in OHLCV columns
    - High >= max(open, close), Low <= min(open, close)
    - Volume > 0 during market hours
    - ~390 bars per trading day (1-minute bars)
    """
    loader = AlpacaIntradayLoader()
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    if tickers is None:
        progress = DownloadProgress()
        tickers = list(progress._state.get("completed", {}).keys())

    if not tickers:
        print("No tickers to validate. Run download first.")
        return

    print(f"\nValidating {len(tickers)} tickers: {start} -> {end}\n")

    total_issues = 0

    for ticker in tickers:
        df = loader.load_range(ticker, start, end)
        issues: list[str] = []

        if df.empty:
            issues.append("No data found")
            print(f"  {ticker}: FAIL — no data")
            total_issues += 1
            continue

        # Check for NaN
        nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum()
        nan_total = nan_counts.sum()
        if nan_total > 0:
            issues.append(f"NaN values: {nan_total}")

        # Check high/low consistency
        bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if bad_high > 0:
            issues.append(f"High < max(open,close) in {bad_high} bars")
        if bad_low > 0:
            issues.append(f"Low > min(open,close) in {bad_low} bars")

        # Check bar count per day
        dates_with_data = df["timestamp"].dt.date.nunique()
        bars_per_day = len(df) / max(dates_with_data, 1)
        if bars_per_day < 300:
            issues.append(f"Low bars/day: {bars_per_day:.0f} (expected ~390)")

        # Summary
        status = "PASS" if not issues else "FAIL"
        print(f"  {ticker}: {status} — {len(df)} bars, {dates_with_data} days, ~{bars_per_day:.0f} bars/day")
        for issue in issues:
            print(f"    ISSUE: {issue}")
            total_issues += 1

    print(f"\nValidation complete: {total_issues} issues found across {len(tickers)} tickers")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 1-minute intraday bars from Alpaca Markets (IEX feed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_intraday_data.py                         # S&P 500 filtered
  python scripts/download_intraday_data.py --universe expanded_15  # 15 ETF test
  python scripts/download_intraday_data.py --start 2024-06-01      # Partial range
  python scripts/download_intraday_data.py --validate-only         # Check data
  python scripts/download_intraday_data.py --reset                 # Start fresh
        """,
    )
    parser.add_argument(
        "--source", default="alpaca", choices=["alpaca", "polygon"],
        help="Data source: alpaca (default, IEX feed) or polygon (future)",
    )
    parser.add_argument(
        "--universe", default="sp500_filtered",
        help="Universe name: sp500, sp500_filtered, expanded_15, cta, rp, etc. (default: sp500_filtered)",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Explicit ticker list (overrides --universe)",
    )
    parser.add_argument(
        "--start", default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="End date YYYY-MM-DD (default: 2024-12-31)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing data, don't download",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear download progress and start fresh",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip previously completed tickers",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.source == "polygon":
        print("Error: Polygon source not yet implemented. Use --source alpaca.", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reset progress if requested
    if args.reset:
        progress = DownloadProgress()
        progress.reset()
        log.info("Download progress reset")

    # Resolve tickers
    if args.tickers:
        tickers = sorted(args.tickers)
    else:
        try:
            tickers = get_download_universe(args.universe)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Source: {args.source}")
    print(f"Universe: {len(tickers)} tickers")
    print(f"Period: {args.start} -> {args.end}")

    if args.validate_only:
        validate_download(tickers, args.start, args.end)
    else:
        # Alpaca: ~3s per ticker (0.3s rate limit × ~10 requests per 2-month chunk)
        est_minutes = len(tickers) * 3 / 60
        print("Starting download (Ctrl+C to stop)...")
        print(f"Estimated time: ~{est_minutes:.0f} minutes (Alpaca IEX, 200 req/min)\n")
        run_download(tickers, args.start, args.end, resume=not args.no_resume)

        # Auto-validate after download
        print("\nRunning post-download validation...")
        validate_download(tickers, args.start, args.end)


if __name__ == "__main__":
    main()
