"""Synchronous yfinance data loader with parquet caching.

Provides load_bars() and load_multiple() for backtest data preparation.
Data is cached to ~/.quantcore/cache/ for fast re-retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".quantcore" / "cache"


def _cache_path(symbol: str, start: str, end: str, interval: str) -> Path:
    """Build a deterministic cache file path."""
    safe = symbol.replace("/", "_").replace(":", "_")
    key = f"{safe}_{start}_{end}_{interval}"
    return CACHE_DIR / f"{key}.parquet"


def _normalize_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance column names to internal format."""
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in ("date", "datetime", "datetimes_etc"):
            rename_map[col] = "timestamp"
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        for col in ("date", "datetime"):
            if col in df.columns:
                df = df.rename(columns={col: "timestamp"})
                break

    # yfinance returns 'adj close' or 'adj_close' — drop it, we don't need it
    for adj_col in ("adj_close", "adj close"):
        if adj_col in df.columns:
            df = df.drop(columns=[adj_col])

    df["symbol"] = symbol

    keep_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    available = [c for c in keep_cols if c in df.columns]
    return df[available]


def load_bars(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data for a single symbol via yfinance.

    Args:
        symbol: Ticker symbol (e.g. "SPY").
        start: Start date string "YYYY-MM-DD".
        end: End date string "YYYY-MM-DD".
        interval: Bar interval ("1d", "1h", "15m", etc.).
        use_cache: If True, check and use cached parquet files.

    Returns:
        DataFrame with columns [symbol, timestamp, open, high, low, close, volume].
    """
    cache_file = _cache_path(symbol, start, end, interval)

    if use_cache and cache_file.exists():
        log.info("Loading %s from cache: %s", symbol, cache_file)
        df = pd.read_parquet(cache_file)
        if not df.empty:
            return df

    import yfinance as yf

    log.info("Downloading %s from %s to %s (%s)", symbol, start, end, interval)
    ticker = yf.Ticker(symbol)
    raw = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)

    if raw.empty:
        log.warning("No data returned for %s (%s → %s)", symbol, start, end)
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

    df = _normalize_columns(raw, symbol)

    # Cache for next time
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file, index=False)
    log.info("Cached %d bars to %s", len(df), cache_file)

    return df


def load_multiple(
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple symbols.

    Returns a dict mapping symbol → DataFrame. Failed symbols are logged
    and omitted from the result.
    """
    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_bars(sym, start, end, interval, use_cache=use_cache)
            if not df.empty:
                results[sym] = df
        except Exception:
            log.exception("Failed to load data for %s", sym)
    return results


def load_combined(
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download and combine OHLCV data for multiple symbols into one DataFrame.

    Result is sorted by timestamp then symbol — ready for BacktestEngine.run().
    """
    all_frames = []
    for sym in symbols:
        try:
            df = load_bars(sym, start, end, interval, use_cache=use_cache)
            if not df.empty:
                all_frames.append(df)
        except Exception:
            log.exception("Failed to load data for %s", sym)

    if not all_frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return combined
