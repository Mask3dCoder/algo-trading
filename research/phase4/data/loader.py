"""Phase 4 — Institutional Dataset Builder.

Builds a clean, validated, panel dataset for cross-sectional factor research.
Wraps the existing yfinance_loader with institutional-grade preprocessing:
  - Adjusted OHLCV with corporate actions applied
  - Verified forward returns (no look-ahead bias)
  - Benchmark alignment
  - VIX regime data from yfinance
  - Parquet caching for reproducibility
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the project root
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantcore.data.yfinance_loader import load_bars, load_multiple

log = logging.getLogger(__name__)

# ── Universe ──────────────────────────────────────────────────────────────────
# S&P 100 constituents (current as of ~2024).
# Expanded from 25 → 100 for statistical power in cross-sectional IC.
PHASE4_UNIVERSE: list[str] = [
    # Technology
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE",
    "CRM", "ACN", "IBM", "INTC", "TXN", "QCOM", "NOW", "INTU", "AMAT",
    # Communication Services
    "GOOGL", "META", "NFLX", "TMUS", "CMCSA", "DIS", "CHTR",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "COST", "WMT",
    "DG", "DLTR", "ROST", "BBY", "ORLY",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "USB", "PNC",
    "TFC", "COF", "SCHW", "CB", "MMC", "AON", "SPGI", "MCO",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT", "MRK", "LLY", "TMO", "ABBV", "BMY", "AMGN",
    "GILD", "ISRG", "MDT", "SYK", "ZTS", "REGN", "VRTX",
    # Industrials
    "UPS", "CAT", "RTX", "HON", "GE", "DE", "BA", "LMT", "MMM", "EMR",
    "ETN", "FDX", "CSX", "NSC", "WM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    # Utilities / Real Estate / Staples
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SPG", "AMT", "PLD", "EQIX",
    # Materials
    "LIN", "APD", "ECL", "SHW", "FCX",
]

# Deduplicate while preserving order
PHASE4_UNIVERSE = list(dict.fromkeys(PHASE4_UNIVERSE))

BENCHMARK = "SPY"
VIX_TICKER = "^VIX"

TRAIN_START = "2013-01-01"
TRAIN_END   = "2020-12-31"
OOS_START   = "2021-01-01"
OOS_END     = "2024-12-31"
FULL_START  = TRAIN_START
FULL_END    = OOS_END


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_dated_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex from timestamp column, tz-naive."""
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def _build_price_panel(
    symbols: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download and stack adjusted close prices into a (date × symbol) pivot.

    Downloads in batches of 20 to avoid yfinance rate-limiting, then joins
    into a single panel. Each symbol's data is cached independently.
    """
    BATCH_SIZE = 20
    frames: list[pd.DataFrame] = []

    log.info("Downloading %d symbols in batches of %d ...", len(symbols), BATCH_SIZE)
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        try:
            raw = load_multiple(batch, start, end, use_cache=use_cache)
        except Exception as e:
            log.warning("Batch downloading failed for %s: %s — retrying individually", batch[:3], e)
            raw = {}
            for sym in batch:
                try:
                    df = load_bars(sym, start, end, use_cache=use_cache)
                    if df is not None and not df.empty:
                        raw[sym] = df
                except Exception:
                    log.warning("Failed to download %s", sym)

        for sym, df in raw.items():
            if df is None or df.empty:
                continue
            df = _to_dated_index(df)
            if "close" not in df.columns:
                continue
            df_sub = df[["close", "volume"]].rename(
                columns={"close": sym, "volume": f"{sym}_vol"}
            )
            frames.append(df_sub)

    if not frames:
        raise RuntimeError("No data downloaded. Check network / yfinance installation.")

    panel = frames[0]
    for f in frames[1:]:
        panel = panel.join(f, how="outer")

    return panel.sort_index()


# ── Compounding helpers ────────────────────────────────────────────────────────

def _compound_returns(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Compound h-day forward returns with no look-ahead.

    For h=1:  r[t] = returns[t+1]                        (single period)
    For h>1:  r[t] = sum(returns[t+1 .. t+h])           (h-period log return)

    Using log-returns throughout: log(∏(1+R_i)) = Σlog(1+R_i),
    so summing log returns is exact for the log of the compound return.

    The index t in the output corresponds to the signal date — we observe
    at close of day t and the return is realized from t+1 .. t+h (no look-ahead).
    """
    if h == 1:
        return returns.shift(-1)
    else:
        # Rolling sum of h forward returns, shifted back by h so the index
        # in the output corresponds to the signal date t
        return returns.rolling(h).sum().shift(-h)


# ── Public API ────────────────────────────────────────────────────────────────

def build_research_dataset(
    symbols: list[str] | None = None,
    start: str = FULL_START,
    end: str = FULL_END,
    fwd_horizons: tuple[int, ...] = (1, 5, 21),
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the complete Phase 4 research dataset.

    Returns a dict with keys:
      'prices'    — (date × symbol) adjusted close prices
      'volumes'   — (date × symbol) raw volumes
      'returns'   — (date × symbol) daily log returns
      'fwd'       — dict[int, pd.DataFrame] forward return DataFrames per horizon
                    fwd[h][t] = compound return from t+1 to t+h
      'benchmark' — (date,) SPY daily log returns
      'vix'       — (date,) VIX closing level
      'symbols'   — list of available symbol names
      'train_mask' — boolean Series, True for in-sample dates (before OOS_START)
    """
    if symbols is None:
        symbols = PHASE4_UNIVERSE

    log.info("Building research dataset: %d symbols, %s → %s", len(symbols), start, end)
    panel = _build_price_panel(symbols, start, end, use_cache=use_cache)

    # Separate price and volume columns
    sym_cols = [s for s in symbols if s in panel.columns]
    vol_cols = [f"{s}_vol" for s in symbols if f"{s}_vol" in panel.columns]

    prices = panel[sym_cols].astype(float)
    volumes = panel[vol_cols].rename(columns={f"{s}_vol": s for s in symbols})

    # Log returns: log(P_t / P_{t-1})
    # Using log-returns means the rolling sum is exactly the log-compound return.
    returns = np.log(prices / prices.shift(1))

    # ── Forward returns (multi-horizon) ──────────────────────────────────────
    fwd: dict[int, pd.DataFrame] = {}
    for h in fwd_horizons:
        fwd[h] = _compound_returns(returns, h)
        log.info(
            "  fwd[%d]: %d valid entries, range [%s, %s]",
            h, fwd[h].notna().sum().sum(), fwd[h].index[0].date(), fwd[h].index[-1].date(),
        )

    # ── Benchmark ────────────────────────────────────────────────────────────
    log.info("Downloading SPY benchmark ...")
    spy_df = load_bars(BENCHMARK, start, end, use_cache=use_cache)
    spy_df = _to_dated_index(spy_df)
    spy_ret = np.log(spy_df["close"] / spy_df["close"].shift(1)).rename("spy_return")

    # ── VIX ──────────────────────────────────────────────────────────────────
    log.info("Downloading VIX ...")
    try:
        vix_df = load_bars(VIX_TICKER, start, end, use_cache=use_cache)
        vix_df = _to_dated_index(vix_df)
        vix = vix_df["close"].rename("vix")
    except Exception:
        log.warning("VIX download failed. Regime analysis will use SPY vol only.")
        vix = pd.Series(dtype=float, name="vix")

    log.info(
        "Dataset built: %d trading days, %d symbols, horizons=%s",
        len(prices), len(sym_cols), fwd_horizons,
    )

    return {
        "prices":       prices,
        "volumes":      volumes,
        "returns":      returns,
        "fwd":          fwd,
        "benchmark":    spy_ret,
        "vix":          vix,
        "symbols":      sym_cols,
        "train_mask":   prices.index < pd.Timestamp(OOS_START),
    }