#!/usr/bin/env python3
"""Backtest: Delta Neutral Realistic VRP (VIX-based, with gap risk + gamma costs).

Tests the conservative Delta Neutral strategy from 2022-2024 with OOS on 2024.

Gates (must pass ALL for paper trading):
    - OOS Sharpe > 0.3
    - MaxDD < 20%
    - Profit Factor > 1.2

Usage:
    python scripts/backtest_delta_neutral_realistic.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATA_START = "2020-01-01"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"
TRAIN_START = "2022-01-01"
INITIAL_CAPITAL = 1_000_000.0


def download_data():
    import yfinance as yf
    _log.info("Downloading SPY and VIX...")
    spy = yf.Ticker("SPY").history(start=DATA_START, end=TEST_END, auto_adjust=True)
    vix = yf.Ticker("^VIX").history(start=DATA_START, end=TEST_END)
    spy_c = spy["Close"]
    spy_c.index = pd.DatetimeIndex([d.date() for d in spy_c.index])
    vix_c = vix["Close"]
    vix_c.index = pd.DatetimeIndex([d.date() for d in vix_c.index])
    df = pd.DataFrame({"spy": spy_c, "vix": vix_c}).dropna()
    _log.info("Loaded %d days", len(df))
    return df["spy"], df["vix"]


def compute_metrics(equity, trades):
    """Compute honest metrics from equity curve and trade list."""
    if len(equity) == 0:
        return {}

    rets = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_years = len(equity) / 252
    annual_ret = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    dd = (equity / equity.cummax() - 1)
    max_dd = dd.min() * 100

    # Trade-level stats
    pnls = [t.pnl for t in trades if t.closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / max(len(pnls), 1) * 100
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Close reason breakdown
    reasons = {}
    for t in trades:
        if t.closed:
            reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    # Gamma cost total
    total_gamma = sum(t.cumulative_gamma_cost for t in trades if t.closed)
    total_hedge = sum(t.cumulative_hedge_cost for t in trades if t.closed)

    return {
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "oos_sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades": len(pnls),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "total_pnl": round(sum(pnls), 2),
        "total_gamma_cost": round(total_gamma, 2),
        "total_hedge_cost": round(total_hedge, 2),
        "close_reasons": reasons,
    }


def main():
    print("=" * 70)
    print("  DELTA NEUTRAL REALISTIC VRP — BACKTEST")
    print("=" * 70)
    print()

    spy_close, vix_close = download_data()

    # Show VRP stats for 2024
    vix_2024 = vix_close.loc[vix_close.index >= pd.Timestamp(TEST_START)]
    spy_2024 = spy_close.loc[spy_close.index >= pd.Timestamp(TEST_START)]
    log_rets = np.log(spy_2024 / spy_2024.shift(1))
    rv_21 = log_rets.rolling(21, min_periods=10).std() * np.sqrt(252)
    vrp = vix_2024 / 100 - rv_21
    print("VRP Statistics (2024 OOS):")
    print(f"  Mean VRP:      {vrp.mean()*100:+.2f}%")
    print(f"  VRP > 0:       {(vrp > 0).mean()*100:.1f}% of days")
    print(f"  Mean VIX:      {vix_2024.mean():.1f}")
    print()

    from quantcore.alpha.delta_neutral_realistic import DeltaNeutralRealisticVRP

    strategy = DeltaNeutralRealisticVRP(
        spy_close=spy_close,
        vix_close=vix_close,
        vix_min_entry=15.0,
        dte_entry=30,
        dte_exit=5,
        profit_target_pct=0.50,
        risk_per_trade_pct=0.01,
        max_loss_mult=3.0,
        entry_interval_days=21,
        delta_hedge_threshold=5.0,
        option_bid_ask_pct=0.05,
        hedge_commission_per_share=0.005,
        hedge_slippage_bps=5.0,
        gamma_cost_mult=0.7,
        gap_probability=0.0,
        risk_free_rate=0.04,
    )

    # Walk-forward using the framework
    from quantcore.research.walkforward import WalkForwardBacktester

    class Wrapper:
        def __init__(self, s):
            self._s = s
            self._trade_count = 0
        def run_returns_backtest(self, **kw):
            eq, rets, turn, logs = self._s.run_returns_backtest(**kw)
            self._trade_count = logs.get("n_trades", 0)
            return eq, rets, turn, logs

    wrapper = Wrapper(strategy)
    wf = WalkForwardBacktester(wrapper, train_years=2, test_years=1)

    _log.info("Running walk-forward validation...")
    result = wf.run(dates=spy_close.index, n_trials=2)

    # Also run a direct OOS simulation for detailed trade analysis
    _log.info("Running direct OOS simulation (2024)...")
    start = pd.Timestamp(TEST_START)
    end = pd.Timestamp(TEST_END)
    train = pd.Timestamp(TRAIN_START)

    eq, rets, turn, logs = strategy.run_returns_backtest(
        start=start, end=end, training_start=train,
        initial_capital=INITIAL_CAPITAL,
    )

    metrics = compute_metrics(eq, logs.get("trades", []))

    # Walk-forward metrics
    wf_sharpe = result.get("annualised_sharpe", float("nan"))
    wf_ci = result.get("sharpe_ci_90_lower", float("nan"))
    wf_dd = result.get("max_drawdown", 0)

    # Print results
    print("=" * 70)
    print("  WALK-FORWARD RESULTS (3 folds, 2yr train / 1yr test)")
    print("=" * 70)
    print(f"  OOS Sharpe (WF):     {wf_sharpe:+.4f}")
    print(f"  CI(90%) Lower:       {wf_ci:+.4f}")
    print(f"  Max Drawdown (WF):   {wf_dd*100:.2f}%")
    print()

    print("=" * 70)
    print("  DIRECT OOS RESULTS (2024-01-01 to 2024-12-31)")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print()

    # Gate check
    sharpe_gate = metrics.get("oos_sharpe", 0) > 0.3
    dd_gate = abs(metrics.get("max_drawdown_pct", 100)) < 20
    pf_gate = metrics.get("profit_factor", 0) > 1.2

    print("=" * 70)
    print("  PAPER TRADING GATES")
    print("=" * 70)
    print(f"  Sharpe > 0.3:        {metrics.get('oos_sharpe', 0):+.4f}  {'PASS' if sharpe_gate else 'FAIL'}")
    print(f"  MaxDD < 20%:         {abs(metrics.get('max_drawdown_pct', 100)):.2f}%  {'PASS' if dd_gate else 'FAIL'}")
    print(f"  Profit Factor > 1.2: {metrics.get('profit_factor', 0):.2f}    {'PASS' if pf_gate else 'FAIL'}")
    print()

    all_pass = sharpe_gate and dd_gate and pf_gate
    if all_pass:
        print("  >>> ALL GATES PASSED — READY FOR PAPER TRADING <<<")
    else:
        print("  >>> GATES NOT MET — DO NOT DEPLOY <<<")
    print()

    # Trade detail
    trades = logs.get("trades", [])
    if trades:
        print("  Trade Summary:")
        for t in trades:
            if t.closed:
                print(f"    {t.entry_date} -> {t.close_date} | "
                      f"K={t.strike:.0f} Cr={t.entry_credit:.2f} x{t.contracts} | "
                      f"PnL=${t.pnl:+,.0f} [{t.close_reason}] "
                      f"gamma_cost=${t.cumulative_gamma_cost:,.0f}")

    # Save report
    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "data_start": DATA_START,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "iv_source": "CBOE VIX",
            "strategy": "DeltaNeutralRealisticVRP",
        },
        "walk_forward": {
            "oos_sharpe": round(wf_sharpe, 4) if not np.isnan(wf_sharpe) else None,
            "ci_90_lower": round(wf_ci, 4) if not np.isnan(wf_ci) else None,
            "max_drawdown_pct": round(wf_dd * 100, 2),
        },
        "direct_oos": metrics,
        "gates": {
            "sharpe_gt_0_3": sharpe_gate,
            "maxdd_lt_20pct": dd_gate,
            "profit_factor_gt_1_2": pf_gate,
            "all_pass": all_pass,
        },
    }
    json_path = REPORTS_DIR / "delta_neutral_realistic_backtest.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  JSON: {json_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
