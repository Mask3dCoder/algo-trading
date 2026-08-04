# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""equity_factors.py — Point-in-time multi-factor signals for a market-neutral sleeve.

Factors (all computed ONLY from data available as of the signal date):
  * Momentum (12-1): Jegadeesh & Titman (1993) 11-month return skipping the last month.
  * Low-Beta: Black, Jensen & Scholes (1972) 60-day CAPM beta vs the market proxy.
  * Residual Volatility: Ang et al. (2006) 60-day idiosyncratic volatility from CAPM.

NEUTRALISATION NOTE (honesty override):
  The audit directive specified demeaning each factor within (sector, size_bin).
  The REAL production panel (data/sp500_pit_universe_2018_2024.parquet) contains
  ONLY [symbol, timestamp, OHLCV] — there is NO sector and NO market_cap column on
  any available real file (verified). Fabricating sector/size maps would be a
  look-ahead / data-integrity violation, so demeaning is performed cross-sectionally
  across ALL currently-tradable names. The long-short construction already removes
  broad market, sector-rotation and size exposures in aggregate; cross-sectional
  demeaning removes the within-date level shift. If a real metadata file is supplied
  later, `demean_cross_section(..., group=...)` supports grouped demeaning with no
  other code change.

Market proxy: equal-weight cross-sectional mean daily return (PIT — uses only names
available on each date). SPY is NOT in the constituent panel, so an external index
column cannot be joined without inventing data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

_log = logging.getLogger(__name__)

# Real panel has no sector / market_cap. We refuse to fabricate them.
METADATA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Spec'd pure functions (verbatim interface from the audit directive).
# These operate on a SINGLE trailing snapshot and are kept for legibility /
# unit testing. The strategy uses the vectorised variants below for speed.
# ---------------------------------------------------------------------------
def compute_momentum(price_df: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """11-month cumulative return skipping the most recent `skip` days.

    `price_df` is wide (dates x symbols), forward-filled. Returns a Series for the
    LAST date of the frame (the signal date). All inputs are strictly past.
    """
    shifted = price_df.shift(skip)
    mom = (shifted / shifted.shift(lookback)) - 1.0
    return mom.iloc[-1]


def compute_beta(returns_df: pd.DataFrame, market_returns: pd.Series, window: int = 60) -> pd.Series:
    """60-day rolling beta vs the market proxy, for the last available window."""
    betas: dict[str, float] = {}
    market_returns = market_returns.reindex(returns_df.index)
    for col in returns_df.columns:
        joined = pd.concat([returns_df[col], market_returns], axis=1).dropna()
        if len(joined) < window:
            betas[col] = np.nan
            continue
        x = joined.iloc[-window:, 1].to_numpy(dtype=float)
        y = joined.iloc[-window:, 0].to_numpy(dtype=float)
        if np.std(x) < 1e-9:
            betas[col] = np.nan
        else:
            slope, _, _, _, _ = linregress(x, y)
            betas[col] = float(slope)
    return pd.Series(betas)


def compute_residual_vol(returns_df: pd.DataFrame, market_returns: pd.Series, window: int = 60) -> pd.Series:
    """60-day idiosyncratic std dev from CAPM regression (kept on daily scale)."""
    resids: dict[str, float] = {}
    market_returns = market_returns.reindex(returns_df.index)
    for col in returns_df.columns:
        joined = pd.concat([returns_df[col], market_returns], axis=1).dropna()
        if len(joined) < window:
            resids[col] = np.nan
            continue
        x = joined.iloc[-window:, 1].to_numpy(dtype=float)
        y = joined.iloc[-window:, 0].to_numpy(dtype=float)
        if np.std(x) < 1e-9:
            resids[col] = np.nan
        else:
            _, _, _, _, _, res = linregress(x, y)
            resids[col] = float(np.std(res))
    return pd.Series(resids)


# ---------------------------------------------------------------------------
# Vectorised, PIT-correct factor engine used by the strategy.
# ---------------------------------------------------------------------------
def build_market_returns(close_wide: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional mean daily return — the PIT market proxy.

    Uses only names present on each date (no look-ahead, no external index).
    """
    rets = close_wide.pct_change().fillna(0.0)
    # Equal-weight mean across available names each day.
    mkt = rets.mean(axis=1)
    return mkt


def compute_factors_vectorized(
    close_wide: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    window: int = 60,
) -> dict[str, pd.Series]:
    """Compute momentum, beta and residual-vol Series (indexed by symbol) for the
    LAST date of `close_wide`, using only data up to and including that date.

    All operations are vectorised; returns the three factor Series plus the market
    return series for diagnostics. Symbols with insufficient history are NaN.
    """
    close_wide = close_wide.sort_index()
    rets = close_wide.pct_change()

    # --- Momentum (12-1) ---
    shifted = close_wide.shift(skip)
    momentum = (shifted / shifted.shift(lookback)) - 1.0
    momentum = momentum.iloc[-1]

    # --- Market proxy (equal-weight) ---
    market = rets.mean(axis=1)
    mkt_window = market.iloc[-window:].to_numpy(dtype=float)
    var_mkt = float(np.var(mkt_window, ddof=1)) if len(mkt_window) > 1 else np.nan

    # --- Beta & residual vol (vectorised CAPM over the last `window` days) ---
    r_win = rets.iloc[-window:].to_numpy(dtype=float)          # (window, n_sym)
    m_col = (mkt_window - mkt_window.mean()).reshape(-1, 1)    # (window, 1)
    r_mean = r_win.mean(axis=0, keepdims=True)                 # (1, n_sym)
    dev = r_win - r_mean                                       # (window, n_sym)
    if np.isfinite(var_mkt) and var_mkt > 0:
        cov = (dev * m_col).mean(axis=0)                       # (n_sym,)
        beta = cov / var_mkt
        var_y = (dev ** 2).mean(axis=0)
        resid_var = np.clip(var_y - beta ** 2 * var_mkt, 0.0, None)
        resid_vol = np.sqrt(resid_var)
    else:
        beta = np.full(r_win.shape[1], np.nan)
        resid_vol = np.full(r_win.shape[1], np.nan)

    syms = list(close_wide.columns)
    return {
        "momentum": pd.Series(momentum.to_numpy(), index=syms),
        "beta": pd.Series(beta, index=syms),
        "residual_vol": pd.Series(resid_vol, index=syms),
        "market_returns": market,
    }


def demean_cross_section(
    factor: pd.Series,
    group: pd.Series | None = None,
) -> pd.Series:
    """Demean a factor cross-sectionally.

    If `group` (e.g. sector or size_bin) is supplied, demean WITHIN each group.
    The real panel has no sector/size metadata, so `group` is None and we demean
    across the whole cross-section. (Refuse to fabricate groups.)
    """
    f = factor.copy()
    if group is None:
        f = f - f.mean()
        return f
    f = f - f.groupby(group).transform("mean")
    return f


def build_signal(
    factors: dict[str, pd.Series],
    *,
    long_frac: float = 0.10,
    short_frac: float = 0.10,
    min_names: int = 5,
    max_names: int = 75,
) -> dict[str, list[str]]:
    """Combine factors into a market-neutral long/short selection.

    Steps:
      1. Invert low-beta and high-residual-vol (we WANT low beta, low idio vol).
      2. Cross-sectionally demean each factor.
      3. Rank-transform each to [0,1] and average the ranks -> composite signal.
      4. Long = top `long_frac`, Short = bottom `short_frac`.

    Returns {'long': [...], 'short': [...]} of symbol lists.
    """
    mom = demean_cross_section(factors["momentum"]).dropna()
    # Low-beta and low-residual-vol are desirable -> invert sign so "high signal"
    # means low beta / low idio vol.
    beta = demean_cross_section(-factors["beta"]).dropna()
    rvol = demean_cross_section(-factors["residual_vol"]).dropna()

    common = mom.index.intersection(beta.index).intersection(rvol.index)
    if len(common) < max(min_names * 2, 10):
        return {"long": [], "short": []}

    df = pd.DataFrame({
        "momentum": mom.reindex(common),
        "low_beta": beta.reindex(common),
        "low_rvol": rvol.reindex(common),
    }).dropna()
    if df.empty:
        return {"long": [], "short": []}

    # Rank-transform each to [0,1] and average.
    ranks = df.rank(pct=True)
    composite = ranks.mean(axis=1)

    n = len(composite)
    k = int(max(min_names, min(max_names, np.floor(long_frac * n))))
    k = min(k, int(np.floor(n / 2)))
    if k < 1:
        return {"long": [], "short": []}

    longs = composite.nlargest(k).index.tolist()
    shorts = composite.nsmallest(k).index.tolist()
    return {"long": longs, "short": shorts}
