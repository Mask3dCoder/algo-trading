"""Real data loader -- yfinance-backed universe definitions and data quality validation.

Provides asset universe definitions for Phase A real-data backtesting,
data loading with parquet caching, and quality validation (NaN checks,
gap detection, coverage ratios).

Usage:
    python scripts/real_data_loader.py              # Validate all universes
    python scripts/real_data_loader.py --universe equity  # Single universe
    python scripts/real_data_loader.py --list-universes   # Show definitions
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from typing import Any

import pandas as pd

from quantcore.data.yfinance_loader import load_combined, load_multiple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------

UNIVERSES: dict[str, dict[str, Any]] = {
    "equity": {
        "description": "Risk parity universe: equities, bonds, gold, commodities",
        "symbols": ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"],
        "asset_classes": {
            "SPY": "US Equities",
            "EFA": "International Equities",
            "TLT": "US Long-Term Bonds",
            "IEF": "US Intermediate-Term Bonds",
            "GLD": "Gold",
            "DBC": "Commodities",
        },
    },
    "cta": {
        "description": "CTA universe: equities, bonds, gold, commodities, FX",
        "symbols": ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC", "FXE"],
        "asset_classes": {
            "SPY": "US Equities",
            "EFA": "International Equities",
            "TLT": "US Long-Term Bonds",
            "IEF": "US Intermediate-Term Bonds",
            "GLD": "Gold",
            "DBC": "Commodities",
            "FXE": "US Dollar Index (EUR)",
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
# Custom exception
# ---------------------------------------------------------------------------


class DataQualityError(ValueError):
    """Raised when data quality checks fail."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_universe(
    name: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load combined data for a universe definition.

    Args:
        name: Key into UNIVERSES dict ('equity' or 'cta').
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
    df = load_combined(symbols, start, end, interval, use_cache=use_cache)

    if df.empty:
        log.warning("No data returned for universe '%s' from %s to %s", name, start, end)

    return df


# ---------------------------------------------------------------------------
# Data quality validation
# ---------------------------------------------------------------------------


def validate_data(
    df: pd.DataFrame,
    name: str = "",
    min_coverage: float = 0.90,
    max_nan_pct: float = 0.05,
    max_gap_days: int = 10,
) -> dict[str, Any]:
    """Validate data quality for a combined multi-symbol DataFrame.

    Checks:
    - Non-empty DataFrame
    - Required columns present
    - NaN percentage per symbol
    - Data coverage ratio (actual / expected business days)
    - Gap detection (consecutive missing business days per symbol)

    Args:
        df: DataFrame with [symbol, timestamp, open, high, low, close, volume].
        name: Optional label for error messages.
        min_coverage: Minimum fraction of expected bars that must exist (default 0.90).
        max_nan_pct: Maximum fraction of NaN values allowed per column (default 0.05).
        max_gap_days: Maximum allowed consecutive missing business days (default 10).

    Returns:
        Dict with validation results:
            'passed': bool
            'n_symbols': int
            'total_bars': int
            'date_range': (str, str) | None
            'symbol_results': dict[str, dict]
            'errors': list[str]

    Raises:
        DataQualityError: If critical quality checks fail.
    """
    result: dict[str, Any] = {
        "passed": False,
        "n_symbols": 0,
        "total_bars": 0,
        "date_range": None,
        "symbol_results": {},
        "errors": [],
    }

    # Check non-empty
    if df.empty:
        result["errors"].append(f"{name}: DataFrame is empty")
        return result

    # Required columns
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        result["errors"].append(f"{name}: Missing columns: {missing_cols}")
        return result

    result["total_bars"] = len(df)
    result["n_symbols"] = df["symbol"].nunique()

    # Date range
    ts = pd.to_datetime(df["timestamp"])
    result["date_range"] = (str(ts.min().date()), str(ts.max().date()))

    total_errors: list[str] = []
    symbol_results: dict[str, dict] = {}

    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym].copy()
        sub = sub.sort_values("timestamp")
        sym_errors: list[str] = []

        # NaN check per column
        for col in ["open", "high", "low", "close", "volume"]:
            nan_pct = sub[col].isna().mean()
            if nan_pct > max_nan_pct:
                msg = f"{sym}: {col} NaN {nan_pct:.1%} > {max_nan_pct:.0%}"
                sym_errors.append(msg)

        # Coverage ratio
        sub_ts = pd.to_datetime(sub["timestamp"])
        expected = pd.bdate_range(sub_ts.min(), sub_ts.max())
        actual = sub_ts.nunique()
        coverage = actual / len(expected) if len(expected) > 0 else 1.0
        if coverage < min_coverage:
            msg = f"{sym}: coverage {coverage:.1%} < {min_coverage:.0%} ({actual}/{len(expected)} days)"
            sym_errors.append(msg)

        # Gap detection
        sorted_dates = sorted(sub_ts.unique())
        if len(sorted_dates) > 1:
            business_day_diffs = []
            for i in range(1, len(sorted_dates)):
                diff = pd.bdate_range(sorted_dates[i - 1], sorted_dates[i])
                gap = len(diff) - 1
                if gap > 0:
                    business_day_diffs.append(gap)
            max_gap = max(business_day_diffs) if business_day_diffs else 0
            if max_gap > max_gap_days:
                msg = f"{sym}: max gap {max_gap}d > {max_gap_days}d"
                sym_errors.append(msg)
        else:
            max_gap = 0

        symbol_results[sym] = {
            "bars": len(sub),
            "coverage": coverage,
            "max_gap_days": max_gap,
            "errors": sym_errors,
        }
        total_errors.extend(sym_errors)

    result["symbol_results"] = symbol_results
    result["errors"] = total_errors
    result["passed"] = len(total_errors) == 0

    return result


def load_and_validate(
    name: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
    **validate_kwargs: Any,
) -> pd.DataFrame:
    """Load a universe and validate data quality in one call.

    Args:
        name: Universe name ('equity' or 'cta').
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.
        interval: Bar interval.
        use_cache: Use parquet caching.
        **validate_kwargs: Passed through to validate_data().

    Returns:
        Validated DataFrame.

    Raises:
        DataQualityError: If validation fails.
    """
    df = load_universe(name, start, end, interval, use_cache=use_cache)
    vresult = validate_data(df, name=name, **validate_kwargs)

    if not vresult["passed"]:
        msg = f"Data quality validation failed for '{name}'"
        for err in vresult["errors"]:
            msg += f"\n  - {err}"
        raise DataQualityError(msg)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_validation_report(vresult: dict[str, Any], label: str = "") -> None:
    """Print validation results in a readable format."""
    status = "PASS" if vresult["passed"] else "FAIL"
    print(f"\n  [{status}] {label or 'Validation'}")
    if vresult["date_range"]:
        print(f"    Range: {vresult['date_range'][0]} -> {vresult['date_range'][1]}")
    print(f"    Symbols: {vresult['n_symbols']}")
    print(f"    Total bars: {vresult['total_bars']}")

    for sym, sr in vresult["symbol_results"].items():
        errs = sr["errors"]
        sym_status = "PASS" if not errs else "FAIL"
        print(f"    [{sym_status}] {sym}: {sr['bars']} bars, "
              f"coverage {sr['coverage']:.1%}, max gap {sr['max_gap_days']}d")
        for e in errs:
            print(f"           ERROR: {e}")

    if vresult["errors"]:
        print(f"    Total errors: {len(vresult['errors'])}")


def main() -> None:
    """CLI entry point for data loading and validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Real data loader for Phase A backtesting")
    parser.add_argument("--universe", type=str, default=None,
                        help="Universe to load and validate (equity, cta)")
    parser.add_argument("--start", type=str, default="2008-01-01",
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

            vresult = validate_data(df, name=uname)
            _print_validation_report(vresult, uname)

        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
