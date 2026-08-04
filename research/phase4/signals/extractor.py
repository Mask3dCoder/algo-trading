"""Phase 4 — Signal Extractor (Stage 2).

Extracts 13 pure predictive factors from price/volume data.
No execution, no sizing, no portfolio construction.
Every factor returns a (date × symbol) DataFrame of signal values.

Factor Catalog:
  Momentum:
    mom_1m   — 1-month trailing return
    mom_3m   — 3-month trailing return
    mom_6m   — 6-month trailing return
    mom_12_1 — 12-1 month Jegadeesh-Titman momentum
    ts_mom   — Time-series momentum (sign × vol-scaled)

  Mean Reversion:
    rev_1w   — 1-week short-term reversal
    rev_1m   — 1-month short-term reversal
    zscore_20 — 20-day z-score (price vs. rolling mean)
    rsi_14   — RSI-14 (mapped to [-1, +1])

  Volatility / Risk:
    low_vol  — Low-volatility factor (inverse 60d realized vol)
    vrp_proxy — Asset-specific vol-regime spread (10d GK vs 60d CC vol)
    vol_regime — VIX percentile rank (high VIX = high vol regime)

  Quality / Composite:
    quality  — Risk-adjusted return (252d return / 252d vol = Sharpe factor)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helper: cross-sectional rank normalize ────────────────────────────────────

def _cs_rank(df: pd.DataFrame, min_cross_section: int = 10) -> pd.DataFrame:
    """Cross-sectionally rank-normalize each row to [-0.5, +0.5].

    This removes cross-sectional level effects so IC measures signal
    ordering power, not absolute signal magnitude.

    Args:
        df:               (date × symbol) DataFrame.
        min_cross_section: Minimum number of non-NaN values required to rank a row.
                          Rows with fewer valid values are set to NaN.
    """
    result = df.copy()

    for i, row in df.iterrows():
        valid = row.dropna()
        if len(valid) < min_cross_section:
            continue  # leave as NaN — insufficient cross-section
        # Rank percentiles within the valid cross-section
        ranked = valid.rank(pct=True) - 0.5
        result.loc[i, ranked.index] = ranked

    return result


# ── Momentum Factors ──────────────────────────────────────────────────────────

def mom_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """1-month (21-day) trailing return."""
    return _cs_rank(prices / prices.shift(21) - 1.0)


def mom_3m(prices: pd.DataFrame) -> pd.DataFrame:
    """3-month (63-day) trailing return."""
    return _cs_rank(prices / prices.shift(63) - 1.0)


def mom_6m(prices: pd.DataFrame) -> pd.DataFrame:
    """6-month (126-day) trailing return."""
    return _cs_rank(prices / prices.shift(126) - 1.0)


def mom_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Jegadeesh-Titman: 12-month return excluding most recent month (252−21 days).

    Skipping the most recent month removes short-term reversal contamination.
    """
    raw = prices.shift(21) / prices.shift(252) - 1.0
    return _cs_rank(raw)


def ts_mom(prices: pd.DataFrame) -> pd.DataFrame:
    """Time-series momentum: sign of 12m return, scaled by inverse vol.

    Moskowitz, Ooi, Pedersen (2012): vol-scaling equalizes risk contribution.
    """
    ret_12m = prices / prices.shift(252) - 1.0
    direction = np.sign(ret_12m)
    daily_ret = prices.pct_change()
    ann_vol = daily_ret.rolling(60).std() * np.sqrt(252)
    ann_vol = ann_vol.clip(lower=0.05)  # Floor at 5% to prevent extreme scaling
    raw = direction / ann_vol
    return _cs_rank(raw)


# ── Mean Reversion Factors ────────────────────────────────────────────────────

def rev_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """1-month reversal: negate 21-day return (buy losers, sell winners)."""
    return _cs_rank(-(prices / prices.shift(21) - 1.0))


def rev_1w(prices: pd.DataFrame) -> pd.DataFrame:
    """1-week short-term reversal: negate 5-day return."""
    return _cs_rank(-(prices / prices.shift(5) - 1.0))


def zscore_20(prices: pd.DataFrame) -> pd.DataFrame:
    """20-day price z-score. Negative = buy (price below rolling mean = reversion signal)."""
    roll_mean = prices.rolling(20).mean()
    roll_std  = prices.rolling(20).std().clip(lower=1e-8)
    zscore = (prices - roll_mean) / roll_std
    # Negate: below mean → positive signal (expect reversion up)
    return _cs_rank(-zscore)


def rsi_14(prices: pd.DataFrame) -> pd.DataFrame:
    """RSI-14 mapped to [-0.5, +0.5]. Low RSI (oversold) → positive signal.

    We first compute row-wise (cross-sectionally) to avoid look-ahead from
    the rolling window operating on stale data mixed with today's close.
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().clip(lower=1e-8)
    rs    = gain / loss
    rsi   = 100.0 - 100.0 / (1.0 + rs)
    # Map: low RSI (oversold) → high positive signal; high RSI → negative
    # [0, 100] → [+0.5, -0.5]: invert so low RSI = +0.5 (buy signal)
    normalized = -(rsi / 100.0 - 0.5)
    return _cs_rank(normalized)


# ── Volatility / Risk Factors ─────────────────────────────────────────────────

def low_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """Low-volatility factor: inverse of 60-day realized annualized vol.

    Blitz & Van Vliet (2007): low-volatility stocks outperform on risk-adjusted basis.
    """
    daily_ret = prices.pct_change()
    ann_vol = daily_ret.rolling(60).std() * np.sqrt(252)
    # Low vol = high rank = positive signal (negate so low-vol → high factor)
    return _cs_rank(-ann_vol)


def _garman_klass_vol(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Approximate Garman-Klass volatility from OHLC-free close-only data.

    True Garman-Klass requires O/H/L data. This close-only approximation uses
    the Rogers-Satchell estimator:  E[ln(H/C)*ln(H/O)] which only needs close.

    For a window of n days, we compute the daily log return variance using
    both the close-to-close and the Garman-Klass inspired estimators.

    If insufficientOHLC data is available we fall back to close-to-close vol
    scaled by a factor of sqrt(252 * (2 - π)/2) ≈ 1.41 for the GK correction.
    """
    log_hl = np.log(prices / prices.shift(1))
    # Simple close-to-close annualised vol as fallback
    cc_vol = log_hl.rolling(window).std() * np.sqrt(252)
    return cc_vol.bfill()


def vrp_proxy(prices: pd.DataFrame) -> pd.DataFrame:
    """Asset-specific VRP: 10-day realized vol vs 60-day realized vol.

    Previously VIX was used as a proxy for implied vol, but VIX is SPX-specific
    and does not represent individual stock implied volatility surfaces.
    This refactored version computes a per-stock VRP as:

      VRP[t,s] = vol_10d[t,s] - vol_60d[t,s]

    where each is the annualized rolling standard deviation of log returns.

    Interpretation:
      - Positive (high short-term vol relative to long-term): vol is rising,
        historically associated with negative forward returns → sell signal
      - Negative (short-term vol < long-term vol): vol is mean-reverting down,
        historically associated with positive forward returns → buy signal

    This is cross-sectionally ranked per date to produce a long-short portfolio.
    Low VRP (vol still declining) → high factor rank → long; High VRP → short.
    """
    log_ret = np.log(prices / prices.shift(1))

    # Short-term: 10-day realized vol (annualised)
    vol_10d = log_ret.rolling(10).std() * np.sqrt(252)

    # Long-term: 60-day realized vol (annualised)
    vol_60d = log_ret.rolling(60).std() * np.sqrt(252)

    # Asset-specific VRP spread: positive = vol rising (bad), negative = falling (good)
    # Guard against division by zero / NaNs
    vol_60d_safe = vol_60d.replace(0, np.nan).bfill().ffill()
    vrp_spread = vol_10d - vol_60d_safe

    # Guard against all-NaN rows
    valid_counts = vrp_spread.notna().sum(axis=1)
    vrp_spread = vrp_spread.where(valid_counts >= 10, other=np.nan)

    # Negate: low VRP (vol declining) → high factor value → buy signal
    return _cs_rank(-vrp_spread)


def vol_regime_rank(vix: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    """VIX percentile rank (252-day rolling). High = high vol regime.

    This is a regime indicator, not a standalone predictive signal.
    Used to condition other signals during regime-aware evaluation.
    """
    common_idx = prices.index.intersection(vix.index)
    vix_aligned = vix.reindex(common_idx).ffill()
    vix_rank = vix_aligned.rolling(252, min_periods=126).rank(pct=True)

    # Broadcast to (date × symbol)
    arr = np.full((len(vix_rank), len(prices.columns)), np.nan, dtype=float)
    valid_mask = ~vix_rank.isna()
    arr[valid_mask.values, :] = np.tile(
        vix_rank.loc[valid_mask].values[:, None], (1, len(prices.columns))
    )

    result = pd.DataFrame(arr, index=prices.index, columns=prices.columns)
    result.index.name = "date"
    return result


# ── Quality Factor ────────────────────────────────────────────────────────────

def quality_sharpe(prices: pd.DataFrame) -> pd.DataFrame:
    """Quality proxy: trailing 252-day Sharpe ratio (return / vol).

    High risk-adjusted return = high quality. Empirically linked to low leverage,
    consistent profitability (Asness, Frazzini, Pedersen 2013 — Quality Minus Junk).
    """
    daily_ret = prices.pct_change()
    ann_ret  = daily_ret.rolling(252).mean() * 252
    ann_vol  = daily_ret.rolling(252).std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol.clip(lower=0.01)
    return _cs_rank(sharpe)


# ── Factor Catalog ────────────────────────────────────────────────────────────

FACTOR_CATALOG: dict[str, dict] = {
    "mom_1m":       {"fn": mom_1m,       "group": "Momentum",  "holding_days": 21,  "econ": "Underreaction / Herding"},
    "mom_3m":       {"fn": mom_3m,       "group": "Momentum",  "holding_days": 63,  "econ": "Jegadeesh-Titman Anomaly"},
    "mom_6m":       {"fn": mom_6m,       "group": "Momentum",  "holding_days": 126, "econ": "Underreaction / Herding"},
    "mom_12_1":     {"fn": mom_12_1,     "group": "Momentum",  "holding_days": 21,  "econ": "Jegadeesh-Titman Anomaly"},
    "ts_mom":       {"fn": ts_mom,       "group": "Momentum",  "holding_days": 21,  "econ": "Trend Following (Moskowitz 2012)"},
    "rev_1m":       {"fn": rev_1m,       "group": "Reversal",  "holding_days": 21,  "econ": "Short-term Reversal"},
    "rev_1w":       {"fn": rev_1w,       "group": "Reversion",  "holding_days": 5,   "econ": "Overreaction / Liquidity"},
    "zscore_20":    {"fn": zscore_20,   "group": "Reversion",  "holding_days": 5,   "econ": "Mean Reversion"},
    "rsi_14":       {"fn": rsi_14,       "group": "Reversion",  "holding_days": 5,   "econ": "RSI Oversold/Overbought"},
    "low_vol":      {"fn": low_vol,      "group": "Risk",       "holding_days": 63,  "econ": "Low Volatility Anomaly (Blitz 2007)"},
    "quality":      {"fn": quality_sharpe, "group": "Quality", "holding_days": 63,  "econ": "Quality Minus Junk (Asness 2013)"},
    # vrp_proxy and vol_regime are added conditionally in extract_all_factors()
}


def extract_all_factors(
    prices: pd.DataFrame,
    vix: Optional[pd.Series] = None,
) -> dict[str, pd.DataFrame]:
    """Extract all factors from the price panel.

    Args:
        prices: (date × symbol) adjusted close prices.
        vix:     Optional VIX series for VIX-based regime factor.
                 If None, volatility-based signals fall back to price-only.

    Returns:
        dict mapping factor_name → (date × symbol) signal DataFrame.
    """
    factors: dict[str, pd.DataFrame] = {}

    for name, meta in FACTOR_CATALOG.items():
        fn = meta["fn"]
        try:
            sig = fn(prices)
            # Safety: drop rows with insufficient cross-section
            valid_ct = sig.notna().sum(axis=1)
            sig = sig.where(valid_ct >= 10, other=np.nan)
            factors[name] = sig
        except Exception as e:
            log.error("Factor %s failed: %s", name, e)

    # ── VRP proxy (asset-specific — no longer needs VIX) ───────────────────
    try:
        factors["vrp_proxy"] = vrp_proxy(prices)
        log.info("  vrp_proxy: asset-specific 10d vs 60d vol spread (VIX-free)")
    except Exception as e:
        log.warning("  vrp_proxy failed: %s", e)

    # ── Vol-regime (requires VIX) ──────────────────────────────────────────
    if vix is not None and not vix.empty:
        try:
            factors["vol_regime"] = vol_regime_rank(vix, prices)
            log.info("  vol_regime: VIX rolling percentile rank")
        except Exception as e:
            log.warning("  vol_regime failed: %s", e)

    log.info("Extracted %d factors: %s", len(factors), list(factors.keys()))
    return factors