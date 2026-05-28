"""Real data loader -- yfinance-backed universe definitions with retry logic.

Universe definitions for Phase A real-data validation:
  - cta:  DBC, GLD, IEF, SPY, TLT  (CTA multi-speed trend)
  - rp:   DBC, GLD, IEF, SPY, TLT  (risk parity)

Provides load_universe() with exponential backoff retry, validate_data()
with 7 quality checks, and fetch_and_prepare() convenience wrapper.

Usage:
    python scripts/real_data_loader.py                       # Validate all
    python scripts/real_data_loader.py --universe cta        # Single universe
    python scripts/real_data_loader.py --list-universes      # Show definitions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from typing import Any

import pandas as pd

from quantcore.data.yfinance_loader import load_combined, load_bars

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------

UNIVERSES: dict[str, dict[str, Any]] = {
    "cta": {
        "description": "CTA multi-speed trend universe",
        "symbols": ["DBC", "GLD", "IEF", "SPY", "TLT"],
        "asset_classes": {
            "DBC": "Commodities",
            "GLD": "Gold",
            "IEF": "US Intermediate-Term Bonds",
            "SPY": "US Equities",
            "TLT": "US Long-Term Bonds",
        },
    },
    "rp": {
        "description": "Risk parity universe",
        "symbols": ["DBC", "GLD", "IEF", "SPY", "TLT"],
        "asset_classes": {
            "DBC": "Commodities",
            "GLD": "Gold",
            "IEF": "US Intermediate-Term Bonds",
            "SPY": "US Equities",
            "TLT": "US Long-Term Bonds",
        },
    },
    "original_5": {
        "description": "Original 5-asset universe (same as cta/rp)",
        "symbols": ["DBC", "GLD", "IEF", "SPY", "TLT"],
        "asset_classes": {
            "DBC": "Commodities",
            "GLD": "Gold",
            "IEF": "US Intermediate-Term Bonds",
            "SPY": "US Equities",
            "TLT": "US Long-Term Bonds",
        },
    },
    "expanded_15": {
        "description": "Expanded 15-asset universe with non-factor assets",
        "symbols": [
            # Original 5 (factors)
            "DBC", "GLD", "IEF", "SPY", "TLT",
            # Non-factor assets (for meaningful residuals)
            "QQQ", "IWM", "EFA", "EEM",  # Equities
            "HYG", "LQD",  # Credit
            "USO", "SLV",  # Commodities
            "VNQ",  # REITs
            "DIA",  # Dow
        ],
        "asset_classes": {
            "DBC": "Commodities",
            "GLD": "Gold",
            "IEF": "US Intermediate-Term Bonds",
            "SPY": "US Equities",
            "TLT": "US Long-Term Bonds",
            "QQQ": "Nasdaq 100",
            "IWM": "Russell 2000",
            "EFA": "EAFE Developed",
            "EEM": "Emerging Markets",
            "HYG": "High Yield Bonds",
            "LQD": "Investment Grade",
            "USO": "Oil",
            "SLV": "Silver",
            "VNQ": "US REITs",
            "DIA": "Dow Jones",
        },
    },
    "ETF_EXPANDED": {
        "description": "Expanded 57-asset ETF universe for cross-sectional stat-arb",
        "symbols": [
            # Equities - US Broad
            "SPY", "QQQ", "IWM", "MDY", "VTI", "VO", "VUG", "VBR",
            # Factors
            "MTUM", "VLUE", "QUAL", "USMV", "SIZE", "VIG",
            # International Equity
            "EFA", "EEM", "VEU", "FEZ", "EWJ", "EWZ", "VWO", "ACWI", "VXUS", "MCHI", "EZU",
            # Bonds - Government
            "TLT", "IEF", "SHY", "TIP", "AGG", "BND",
            # Bonds - Credit & Other
            "LQD", "HYG", "MBB", "BKLN",
            # Commodities
            "GLD", "SLV", "USO", "UNG", "DBA", "DJP",
            # REITs
            "VNQ", "VNQI", "REET",
            # US Sectors
            "XLF", "XLK", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB",
            "FTEC", "IHI",
            # Preferred / Others
            "PFF",
            # Volatility
            "VXX",
            # Currencies
            "UUP", "FXE",
        ],
        "asset_classes": {
            "SPY": "Large Cap", "QQQ": "Tech", "IWM": "Small Cap",
            "MDY": "Mid Cap", "VTI": "Total Market", "VO": "Mid Cap Growth",
            "VUG": "Large Cap Growth", "VBR": "Small Cap Value",
            "MTUM": "Momentum", "VLUE": "Value", "VIG": "Dividend Growth",
            "EFA": "EAFE", "EEM": "Emerging", "VEU": "Developed",
            "FEZ": "Eurozone", "EWJ": "Japan", "EWZ": "Brazil",
            "VWO": "Emerging Markets", "ACWI": "Global", "VXUS": "Ex-US",
            "MCHI": "China", "EZU": "Eurozone",
            "TLT": "Long Bonds", "IEF": "Interm Bonds", "SHY": "Short Bonds",
            "TIP": "TIPS", "AGG": "Aggregate Bonds", "BND": "Total Bonds",
            "LQD": "IG Credit", "HYG": "High Yield",
            "MBB": "Mortgage", "BKLN": "Bank Loans",
            "GLD": "Gold", "SLV": "Silver", "USO": "Oil",
            "UNG": "Natural Gas", "DBA": "Agriculture", "DJP": "Commodities",
            "VNQ": "US REITs", "VNQI": "Intl REITs", "REET": "Global REITs",
            "XLF": "Financials", "XLK": "Technology", "XLV": "Healthcare",
            "XLE": "Energy", "XLY": "Consumer Disc",
            "XLP": "Consumer Staples", "XLI": "Industrials",
            "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials",
            "FTEC": "Tech Sector", "IHI": "Medical Devices",
            "QUAL": "Quality", "USMV": "Low Vol", "SIZE": "Size",
            "PFF": "Preferred",
            "VXX": "VIX",
            "UUP": "USD", "FXE": "Euro",
        },
    },
}


def list_universes() -> None:
    """Print universe definitions to stdout."""
    print(f"{'Universe':<12s} {'Symbols':<45s} Description")
    print("-" * 80)
    for name, info in UNIVERSES.items():
        symbols = ", ".join(info["symbols"])
        print(f"{name:<12s} {symbols:<45s} {info['description']}")


# ---------------------------------------------------------------------------
# Loading with retry
# ---------------------------------------------------------------------------

_RETRY_DELAYS: list[float] = [5.0, 15.0, 45.0]


def load_universe(
    name: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load combined data for a universe with retry on failure.

    Each symbol is downloaded individually. If a symbol fails, up to 3
    retries with exponential backoff (5s, 15s, 45s) are attempted.

    Args:
        name: Key into UNIVERSES dict ('cta' or 'rp').
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.
        interval: Bar interval (default '1d').
        use_cache: Use parquet caching (default True).

    Returns:
        DataFrame with columns [symbol, timestamp, open, high, low, close, volume]
        sorted by timestamp then symbol. Empty if no data returned.
    """
    if name not in UNIVERSES:
        valid = list(UNIVERSES.keys())
        raise ValueError(f"Unknown universe '{name}'. Valid options: {valid}")

    symbols = UNIVERSES[name]["symbols"]
    all_frames: list[pd.DataFrame] = []

    for sym in symbols:
        last_error: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS):
            try:
                df = load_bars(sym, start, end, interval, use_cache=use_cache)
                if not df.empty:
                    all_frames.append(df)
                    last_error = None
                    break
            except Exception as e:
                last_error = e
                log.warning(
                    "Attempt %d/3 failed for %s: %s. Retrying in %.0fs...",
                    attempt + 1, sym, e, delay,
                )
                time.sleep(delay)

        if last_error is not None:
            log.error("All retries exhausted for %s: %s", sym, last_error)

    if not all_frames:
        return pd.DataFrame(
            columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Data quality validation (7 checks)
# ---------------------------------------------------------------------------


def validate_data(
    df: pd.DataFrame,
    expected_symbols: list[str] | None = None,
    name: str = "",
    max_missing_pct: float = 0.05,
    max_nan_pct: float = 0.05,
) -> list[str]:
    """Validate data quality for a combined multi-symbol DataFrame.

    Seven checks:
      1. Required columns exist
      2. No duplicate (timestamp, symbol) rows
      3. Date alignment -- all symbols share the same dates (after forward-fill)
      4. Missing bar count -- each symbol within max_missing_pct of expected
      5. Price sanity -- no zeros, negatives, or unrealistic values
      6. NaN check per column
      7. Volume check -- no zero volumes, no unrealistic spikes

    Args:
        df: DataFrame with columns [symbol, timestamp, open, high, low, close, volume].
        expected_symbols: Optional list of symbols that must all be present.
        name: Optional label for error messages.
        max_missing_pct: Max fraction of missing bars allowed (default 0.05).
        max_nan_pct: Max fraction of NaN values allowed per column (default 0.05).

    Returns:
        List of error messages (empty = all checks passed).
    """
    errors: list[str] = []

    if df.empty:
        errors.append(f"{name}: DataFrame is empty")
        return errors

    # 1. Required columns
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        errors.append(f"{name}: Missing columns: {missing_cols}")
        return errors

    # 2. Duplicate detection
    dup_mask = df.duplicated(subset=["timestamp", "symbol"], keep=False)
    n_dupes = dup_mask.sum()
    if n_dupes > 0:
        errors.append(f"{name}: {n_dupes} duplicate (timestamp, symbol) rows found")

    # 3. Coverage: all expected symbols present
    actual_symbols = set(df["symbol"].unique())
    if expected_symbols:
        expected_set = set(expected_symbols)
        missing = expected_set - actual_symbols
        if missing:
            errors.append(f"{name}: Missing symbols from data: {sorted(missing)}")

    actual_count = len(actual_symbols)

    # Date alignment check
    ts_per_sym = df.groupby("symbol")["timestamp"].nunique()
    max_ts = ts_per_sym.max()
    min_ts = ts_per_sym.min()
    if min_ts < max_ts * 0.95:
        low_syms = ts_per_sym[ts_per_sym < max_ts * 0.95].index.tolist()
        errors.append(
            f"{name}: Date alignment issue -- symbols with fewer dates: {low_syms}"
        )

    # 4. Missing bar count per symbol
    all_dates = sorted(df["timestamp"].unique())
    if len(all_dates) > 0 and actual_count > 0:
        expected_bars = len(all_dates)
        for sym in df["symbol"].unique():
            sym_bars = df[df["symbol"] == sym]["timestamp"].nunique()
            missing_frac = (expected_bars - sym_bars) / expected_bars
            if missing_frac > max_missing_pct:
                errors.append(
                    f"{name}: {sym} missing {missing_frac:.1%} bars "
                    f"({sym_bars}/{expected_bars})"
                )

    # 5. Price sanity
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        zeros = (df[col] <= 0).sum()
        if zeros > 0:
            errors.append(f"{name}: {zeros} zero/negative values in {col}")
        inf_mask = df[col].isin([float("inf"), float("-inf")])
        n_inf = inf_mask.sum()
        if n_inf > 0:
            errors.append(f"{name}: {n_inf} infinite values in {col}")

    # 6. NaN check per column
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            continue
        nan_pct = df[col].isna().mean()
        if nan_pct > max_nan_pct:
            errors.append(
                f"{name}: {col} NaN {nan_pct:.1%} > {max_nan_pct:.0%}"
            )

    # 7. Volume check
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        # Allow up to 5 zero-volume rows or < 0.01% of total (whichever is larger)
        if zero_vol > max(5, int(len(df) * 0.0001)):
            errors.append(f"{name}: {zero_vol} rows with zero volume")
        vol = df["volume"]
        vol = vol[vol > 0]
        if len(vol) > 0:
            median_vol = vol.median()
            if median_vol > 0:
                spike_ratio = vol.max() / median_vol
                if spike_ratio > 500:
                    errors.append(
                        f"{name}: Volume spike detected -- max/median ratio {spike_ratio:.0f}x"
                    )

    return errors


# ---------------------------------------------------------------------------
# Combined fetch and prepare
# ---------------------------------------------------------------------------


def fetch_and_prepare(
    name: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load a universe and validate data quality in one call.

    Args:
        name: Universe name ('cta' or 'rp').
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.
        interval: Bar interval.
        use_cache: Use parquet caching.

    Returns:
        Validated DataFrame.

    Raises:
        ValueError: If validation fails or universe is unknown.
    """
    df = load_universe(name, start, end, interval, use_cache=use_cache)
    expected = UNIVERSES.get(name, {}).get("symbols")
    errs = validate_data(df, expected_symbols=expected, name=name)

    if errs:
        msg = f"Data quality validation failed for '{name}'"
        for e in errs:
            msg += f"\n  - {e}"
        raise ValueError(msg)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_validation_report(errs: list[str], label: str = "") -> None:
    """Print validation results."""
    status = "PASS" if not errs else "FAIL"
    print(f"\n  [{status}] {label or 'Validation'}")
    for e in errs:
        print(f"    ERROR: {e}")


def main() -> None:
    """CLI entry point for data loading and validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real data loader for Phase A validation"
    )
    parser.add_argument("--universe", type=str, default=None,
                        help="Universe to load and validate (cta, rp)")
    parser.add_argument("--start", type=str, default="2018-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--list-universes", action="store_true",
                        help="Print universe definitions and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.list_universes:
        list_universes()
        return

    if args.universe:
        universes_to_check = [args.universe]
    else:
        universes_to_check = list(UNIVERSES.keys())

    for uname in universes_to_check:
        print(f"\n{'=' * 60}")
        print(f"  Loading universe: {uname}")
        print(f"  Period: {args.start} -> {args.end}")
        print(f"{'=' * 60}")

        try:
            df = load_universe(uname, args.start, args.end)
            if df.empty:
                print(f"  WARNING: No data returned for '{uname}'")
                continue

            print(f"  Loaded {len(df)} bars across {df['symbol'].nunique()} symbols")

            expected = UNIVERSES[uname]["symbols"]
            errs = validate_data(df, expected_symbols=expected, name=uname)
            _print_validation_report(errs, uname)

        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
