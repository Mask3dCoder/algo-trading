#!/usr/bin/env python
"""Scratch validation: is the real-data Sharpe an EDGE or NVDA-beta + survivorship?"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.sweep_vol_target_momentum import run_backtest  # type: ignore

TEST_START, TEST_END = "2022-01-01", "2024-12-31"
TOP1 = {"momentum_lookback": 180, "vol_target_annual": 0.25,
        "vol_estimation_window": 60, "rebalance_frequency": 5}

df = pd.read_csv(REPO / "data" / "real_panel_2018_2024.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
test = df[(df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)].copy()
test = test.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

def bh_return(sub):
    """Equal-weight buy & hold total return over the test window."""
    last = sub.groupby("symbol")["close"].last()
    first = sub.groupby("symbol")["close"].first()
    rets = (last / first - 1.0)
    return float(rets.mean()), dict(rets.round(3))

print("=== Buy & Hold benchmarks (equal-weight, 2022-2024) ===")
ew, per = bh_return(test)
print(f"  full universe  : +{ew*100:.1f}%   (NVDA={per.get('NVDA')})")
ewx, _ = bh_return(test[test.symbol != "NVDA"])
print(f"  ex-NVDA        : +{ewx*100:.1f}%")

print("\n=== Strategy (Top-1 params) ===")
m_full = run_backtest(test, TOP1)
print(f"  full universe  : Sharpe={m_full['oos_sharpe']:.2f}  ret={m_full['oos_total_return']*100:.1f}%  "
      f"MaxDD={m_full['oos_max_dd']*100:.1f}%  win={m_full['win_rate']*100:.1f}%  trades={m_full['trade_count']}")
m_nv = run_backtest(test[test.symbol != "NVDA"], TOP1)
print(f"  ex-NVDA        : Sharpe={m_nv['oos_sharpe']:.2f}  ret={m_nv['oos_total_return']*100:.1f}%  "
      f"MaxDD={m_nv['oos_max_dd']*100:.1f}%  win={m_nv['win_rate']*100:.1f}%  trades={m_nv['trade_count']}")

print(f"\n=== Verdict ===")
print(f"  Edge share of return (ex-NVDA / full): {m_nv['oos_total_return']/max(m_full['oos_total_return'],1e-9)*100:.0f}%")
print(f"  NVDA contribution to portfolio return: {(m_full['oos_total_return']-m_nv['oos_total_return'])*100:.1f} pts")
