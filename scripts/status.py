"""QuantCore live status — reads reports/live_equity.csv (no IBKR login needed)."""
import sys
from pathlib import Path
import pandas as pd

eq = Path("reports/live_equity.csv")
if not eq.exists():
    print("No reports/live_equity.csv yet. Is the runner running?")
    sys.exit(0)

df = pd.read_csv(eq)
if df.empty:
    print("live_equity.csv is empty.")
    sys.exit(0)

last = df.iloc[-1]
first_eq = float(df.iloc[0]["total_equity"])
cur_eq = float(last["total_equity"])
pnl = cur_eq - first_eq
peak = float(df["total_equity"].max())
dd = (cur_eq - peak) / peak * 100 if peak else 0.0

print("QuantCore Paper Status")
print("=" * 40)
print(f"As of        : {last['date']}")
print(f"Equity       : ${cur_eq:,.2f}")
print(f"Cash         : ${float(last['cash']):,.2f}")
print(f"Positions val: ${float(last['positions_value']):,.2f}")
print(f"Cum P&L      : ${pnl:,.2f}  ({pnl/first_eq*100:+.2f}%)")
print(f"Drawdown     : {dd:.2f}% (peak ${peak:,.2f})")
print(f"Equity rows  : {len(df)} (trading days logged)")
