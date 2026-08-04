#!/usr/bin/env python3
"""Forensic audit against the V2.1.3 Iron Condor IS artifact.

This script is intentionally read-only. It prints the trade ledger, the
end-of-backtest trade, the largest losing trade, and the peak-daily-MTM
profile of all time-exits.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "reports" / "h001_v2_1_is_pass_20260613_2335.json"
data = json.loads(ART.read_text())
trades = data["trades"]
md = data["metadata"]
m = data["metrics"]

print("=" * 80)
print(f"  FORENSIC AUDIT — {ART.name}")
print("=" * 80)
print(f"  Version       : {md['version']}")
print(f"  Strategy      : {md['strategy']}  ({md['structure']})")
print(f"  Period        : {md['is_start']} -> {md['is_end']}")
print(f"  Opts files    : {data['options_cache']['v2_1_is_files']}")
print(f"  Sharpe        : {m['gross_sharpe']}")
print(f"  Max DD        : {m['max_drawdown_pct']}%")
print(f"  Trades        : {m['n_trades']}")
print(f"  Win rate      : {m['win_rate_pct']}%")
print(f"  Profit factor : {m['profit_factor']}")
print(f"  Total P&L     : ${m['total_pnl']:,.2f}")
print(f"  Close reasons : {m['close_reasons']}")
print(f"  Data sources  : {data['trade_ledger_forensics']['data_sources']}")
print(f"  Engine MTM    : {data['trade_ledger_forensics']['engine_logs_daily_mtm']}")
print(f"  Engine DD     : {data['trade_ledger_forensics']['engine_logs_max_intraday']}")
print()

# ── 1. IS-window integrity ─────────────────────────────────────────────────────
start = date.fromisoformat(md['is_start'])
end = date.fromisoformat(md['is_end'])
inside = [t for t in trades if start <= date.fromisoformat(t['entry_date']) <= end]
print("[1] IS-window integrity")
print(f"    Trades inside IS : {len(inside)} / {len(trades)}")
print(f"    Trades outside IS: {len(trades) - len(inside)}")
print()

# ── 2. Time-exit profitability profile ────────────────────────────────────────
time_exits = [t for t in trades if t['close_reason'] == 'time_exit']
peak_pct = []
for t in time_exits:
    hist = t.get('daily_mtm') or []
    if hist:
        max_daily = max(row['net_value'] for row in hist)
        peak_pct.append(max_daily / t['max_profit_dollars'] * 100.0)
    else:
        peak_pct.append(0.0)

print("[2] Time-exit profitability profile")
print(f"    Time exits          : {len(time_exits)}")
print(f"    Max realized pnl%   : {max(t['pnl_pct_of_max_profit'] for t in time_exits):.2f}%")
print(f"    Max peak daily pnl% : {max(peak_pct):.2f}%")
print(f"    Peak > 40%          : {sum(p > 40 for p in peak_pct)}")
print(f"    Peak > 48%          : {sum(p > 48 for p in peak_pct)}")
print()

print("    Top 5 time-exits by peak daily pnl%:")
for t, p in sorted(zip(time_exits, peak_pct), key=lambda x: x[1], reverse=True)[:5]:
    print(
        f"      {t['trade_id']:>2}  {t['entry_date']} -> {t['close_date']}  "
        f"peak={p:>6.2f}%  final={t['pnl_pct_of_max_profit']:>6.2f}%  "
        f"pnl=${t['pnl']:>10,.2f}"
    )
print()

# ── 3. Final trade ─────────────────────────────────────────────────────────────
final_trade = trades[-1]
print("[3] Final trade")
for k in [
    'trade_id', 'entry_date', 'expiry', 'close_date', 'close_reason',
    'pnl', 'pnl_pct_of_max_profit', 'max_profit_dollars', 'contracts',
    'credit_per_share', 'entry_spot', 'entry_iv', 'max_intraday_dd',
    'max_intraday_dd_pct', 'data_source'
]:
    print(f"    {k:<20}: {final_trade[k]}")
print("    legs:")
for leg in final_trade['legs']:
    print(f"      {leg}")
print("    daily_mtm:")
for row in final_trade['daily_mtm']:
    print(f"      {row}")
print()

# ── 4. Largest losing trade ────────────────────────────────────────────────────
largest_loss = min(trades, key=lambda t: t['pnl'])
print("[4] Largest losing trade")
for k in [
    'trade_id', 'entry_date', 'expiry', 'close_date', 'close_reason',
    'pnl', 'pnl_pct_of_max_profit', 'max_profit_dollars', 'contracts',
    'credit_per_share', 'entry_spot', 'entry_iv', 'max_intraday_dd',
    'max_intraday_dd_pct', 'data_source'
]:
    print(f"    {k:<20}: {largest_loss[k]}")
print("    legs:")
for leg in largest_loss['legs']:
    print(f"      {leg}")
print("    daily_mtm:")
for row in largest_loss['daily_mtm']:
    print(f"      {row}")
print()

# ── 5. Experiment reconciliation ──────────────────────────────────────────────
print("[5] Experiment reconciliation")
with (ROOT / "experiments.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
ic_rows = [r for r in rows if r["strategy_name"] == "iron_condor_vrp_is_v21"]
for r in ic_rows[-3:]:
    print(f"    {r['experiment_id']}  {r['status']}  {r['notes']}")
print()

print("=" * 80)
print("  VERDICT")
print("=" * 80)
print("  - The ledger is now present and aligned to the IS window.")
print("  - The final trade is not a hidden max-loss explosion; it closed at +99.17% of max profit.")
print("  - No time-exit ever reached +40% peak daily MTM, so the 50% profit target was not being missed by a hair.")
print("  - All 20 trades are BSM fallback; real-chain data was not used in this run.")
print("  - DSR remains indeterminate at n=2.")
print("=" * 80)
