"""Compute non-price-based carry signals for each asset type.

Sources:
  Equities: real TTM dividend yield from yfinance Ticker.dividends
  Bonds:    real term premium from ^TNX (10Y) - ^IRX (3M) via yfinance
  Commodities: 21d - 252d return (curve slope proxy, explicitly permitted)

All series are z-scored (rolling 252-bar window) for cross-asset comparability.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".quantcore" / "cache"

ASSET_CARRY_TYPE: dict[str, str] = {
    "SPY": "equity",
    "EFA": "equity",
    "TLT": "bond",
    "IEF": "bond",
    "GLD": "commodity",
    "DBC": "commodity",
}


# ---------------------------------------------------------------------------
# Dividend data
# ---------------------------------------------------------------------------

def _load_dividends(
    symbol: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.Series:
    """Load dividend history for a symbol, cached, with 400d buffer for TTM."""
    data_start = pd.Timestamp(start) - pd.Timedelta(days=400)
    cache_path = CACHE_DIR / f"raw_dividends_{symbol}_{data_start.strftime('%Y%m%d')}_{end}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path).iloc[:, 0]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ticker = yf.Ticker(symbol)
    div = ticker.dividends
    if div.index.tz is not None:
        div.index = div.index.tz_localize(None)
    div = div.loc[data_start:end]
    div.to_frame().to_parquet(cache_path)
    return div


def _load_close_prices(
    symbol: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.Series:
    """Load adjusted close prices (uses the same caching as yfinance_loader)."""
    cache_path = CACHE_DIR / f"carry_close_{symbol}_{start}_{end}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path).iloc[:, 0]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, auto_adjust=True)
    close = hist["Close"]
    close.to_frame().to_parquet(cache_path)
    return close


# ---------------------------------------------------------------------------
# Per-asset raw carry computation
# ---------------------------------------------------------------------------

def _equity_carry_raw(
    symbol: str,
    close_series: pd.Series,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.Series:
    """TTM dividend yield = trailing_12m_dividends / price.

    Uses calendar-year rolling window (252 trading days ≈ 12 months).
    Returns Series aligned with close_series dates.
    """
    cache_path = CACHE_DIR / f"div_yield_{symbol}_{start}_{end}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path).iloc[:, 0]

    dividends = _load_dividends(symbol, start, end, use_cache)

    # Create daily dividend series aligned with close dates
    div_daily = pd.Series(0.0, index=close_series.index)
    for dt, val in dividends.items():
        if dt in div_daily.index:
            div_daily.loc[dt] = val

    # TTM sum (252 trading days)
    ttm_div = div_daily.rolling(252, min_periods=1).sum()
    div_yield = ttm_div / close_series

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    div_yield.to_frame().to_parquet(cache_path)
    return div_yield


def _bond_term_premium(
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.Series:
    """Term premium = ^TNX (10Y yield) - ^IRX (3M yield).

    Returns daily Series indexed by date. Forward-filled to match
    business-day calendar.
    """
    cache_path = CACHE_DIR / f"term_premium_{start}_{end}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path).iloc[:, 0]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        tnx = yf.Ticker("^TNX")
        irx = yf.Ticker("^IRX")

        # Load with 30d buffer for alignment
        buf_start = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        tnx_h = tnx.history(start=buf_start, end=end)
        irx_h = irx.history(start=buf_start, end=end)

        if tnx_h.empty or irx_h.empty:
            log.warning("No yield data returned for ^TNX / ^IRX — using zeros")
            dates = pd.bdate_range(start=start, end=end)
            result = pd.Series(0.0, index=dates)
            result.to_frame().to_parquet(cache_path)
            return result

        tnx_close = tnx_h["Close"]
        irx_close = irx_h["Close"]

        # Common dates, term premium in percentage points
        common = tnx_close.index.intersection(irx_close.index)
        tp = tnx_close.loc[common] - irx_close.loc[common]

        # Strip timezone for consistent date handling
        if tp.index.tz is not None:
            tp.index = tp.index.tz_localize(None)

        tp = tp.loc[start:end]

        result = tp
    except Exception as e:
        log.warning("Failed to load yield data (%s) — using zeros", e)
        dates = pd.bdate_range(start=start, end=end)
        result = pd.Series(0.0, index=dates)

    result.to_frame().to_parquet(cache_path)
    return result


def _commodity_carry_raw(close_series: pd.Series) -> pd.Series:
    """Roll yield proxy = 21d return - 252d return.

    Positive when near-term momentum exceeds long-term (backwardation proxy).
    """
    ret_21 = close_series / close_series.shift(21) - 1.0
    ret_252 = close_series / close_series.shift(252) - 1.0
    return (ret_21 - ret_252).fillna(0.0)


# ---------------------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------------------

def _rolling_zscore(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling z-score with no lookahead.

    First min_periods bars are filled with 0.0 (neutral carry).
    """
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_carry_matrix(
    df: pd.DataFrame,
    use_cache: bool = True,
) -> np.ndarray:
    """Compute z-scored carry for all assets in the universe.

    Args:
        df: DataFrame from load_universe() with columns
            [symbol, timestamp, open, high, low, close, volume].
        use_cache: Cache raw data (dividends, yields) and computed series.

    Returns:
        (n_timestamps, n_assets) float64 array of z-scored carry values.
        Columns are in sorted(symbols) order — matches _df_to_data() output.
    """
    symbols = sorted(df["symbol"].unique())
    pivot = df.pivot_table(index="timestamp", columns="symbol", values="close")

    # Strip timezone from all indices for consistent date handling
    if pivot.index.tz is not None:
        pivot.index = pivot.index.tz_localize(None)

    timestamps = pivot.index
    n_periods = len(timestamps)
    n_assets = len(symbols)

    start = str(timestamps[0].date())
    end = str(timestamps[-1].date())

    # Pre-load term premium once (shared across all bonds)
    bond_carry = None

    carry_matrix = np.zeros((n_periods, n_assets))

    for j, sym in enumerate(symbols):
        close_series = pivot[sym].dropna()
        asset_type = ASSET_CARRY_TYPE.get(sym, "equity")

        if asset_type == "equity":
            raw = _equity_carry_raw(sym, close_series, start, end, use_cache)
        elif asset_type == "bond":
            if bond_carry is None:
                bond_carry = _bond_term_premium(start, end, use_cache)
            raw = bond_carry.reindex(close_series.index).ffill().fillna(0.0)
        elif asset_type == "commodity":
            raw = _commodity_carry_raw(close_series)
        else:
            raw = pd.Series(0.0, index=close_series.index)

        # Reindex to full timestamp grid (NaN filled to 0 before z-score)
        raw = raw.reindex(timestamps).fillna(0.0)

        # Z-score (rolling, no lookahead)
        carry_matrix[:, j] = _rolling_zscore(raw).values

    return carry_matrix
