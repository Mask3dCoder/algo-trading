"""Synthetic Universe Builder for Zero-Data Historical Simulation.

Public API: download_spy_vix_history(), build_synthetic_chain(),
            build_chain_history(), make_snapshot()
"""
from __future__ import annotations
import logging
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

_log = logging.getLogger(__name__)
DEFAULT_CACHE = Path("data/synthetic_universe/spy_vix_2020_2025.csv")


def download_spy_vix_history(start=date(2020, 1, 1), end=date(2025, 12, 31), cache_path=None, force_redownload=False):
    cache = Path(cache_path) if cache_path else DEFAULT_CACHE
    if not force_redownload and cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.loc[str(start):str(end)]
    _log.info("Downloading SPY/VIX via yfinance ...")
    import yfinance as yf
    spy = yf.download("SPY", start=str(start), end=str(end), progress=False)
    vix = yf.download("^VIX", start=str(start), end=str(end), progress=False)
    df = pd.DataFrame(
        {"SPY_close": spy["Close"].squeeze(), "SPY_volume": spy["Volume"].squeeze(),
         "VIX_close": vix["Close"].squeeze()},
        index=pd.to_datetime(spy.index),
    ).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def build_synthetic_chain(spot, as_of, vix, strikes_pct_otm=(-0.10, 0.10), strike_step=1.0, dte=45,
                          rf=0.045, smile=True, spread_coef=0.05):
    lo = np.floor(spot * (1 + strikes_pct_otm[0]) / strike_step) * strike_step
    hi = np.ceil(spot * (1 + strikes_pct_otm[1]) / strike_step) * strike_step + 0.5
    strikes = np.arange(lo, hi, strike_step)
    base_iv = max(vix / 100.0, 0.01)
    m = strikes / spot - 1.0
    iv_smile = np.where(m < 0, 0.08 * np.abs(m), 0.03 * m) - 0.02 if smile else np.zeros_like(strikes)
    ivs = np.clip(base_iv + iv_smile, 0.01, 2.0)
    T, r = dte / 365.0, rf
    d1 = (np.log(spot / strikes) + (r + 0.5 * ivs ** 2) * T) / (ivs * np.sqrt(T))
    d2 = d1 - ivs * np.sqrt(T)
    c_prc = spot * norm.cdf(d1) - strikes * np.exp(-r * T) * norm.cdf(d2)
    p_prc = strikes * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    c_delta, p_delta = norm.cdf(d1), norm.cdf(d1) - 1.0
    spread = 0.01 + spread_coef * max((vix - 15.0) / 100.0, 0.0)
    call = pd.DataFrame({
        "strike": strikes, "right": "C", "bid": c_prc * (1 - spread / 2), "ask": c_prc * (1 + spread / 2),
        "mid": c_prc, "delta": c_delta, "iv_used": ivs, "dte": dte,
    })
    put = pd.DataFrame({
        "strike": strikes, "right": "P", "bid": p_prc * (1 - spread / 2), "ask": p_prc * (1 + spread / 2),
        "mid": p_prc, "delta": p_delta, "iv_used": ivs, "dte": dte,
    })
    return pd.concat([call, put], ignore_index=True)


def build_chain_history(history, dte=45):
    return {
        dt.date() if hasattr(dt, "date") else dt:
            build_synthetic_chain(float(row["SPY_close"]), dt,
                                   float(row["VIX_close"]) if not pd.isna(row["VIX_close"]) else 15.0, dte=dte)
        for dt, row in history.iterrows()
        if not pd.isna(row["SPY_close"])
    }


def make_snapshot(history, as_of, lookback=252):
    w = history.loc[:as_of].tail(lookback)
    c = w["SPY_close"].to_numpy().reshape(-1, 1)
    iv = w["SPY_close"].pct_change().rolling(21, min_periods=5).std().mul(np.sqrt(252)).fillna(0.20).to_numpy().reshape(-1, 1)
    return {"close": c, "implied_vol": iv, "symbols": ["SPY"]}
