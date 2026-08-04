#!/usr/bin/env python3
"""HYPOTHESIS_001 — In-Sample (IS) Calibration Pass: 2016-2021.

This is the first un-cheated run of the Delta Neutral VRP strategy.
Expected outcome: DataQualityViolation cascades on the IS period because
no real options contracts are cached from 2016-2021. This is the
correct, honest result — not a bug.

Usage:
    python research/h001_opra_flow/run_insample.py [--strategy delta_neutral]

Outputs:
    experiments.csv        — one row per hypothesis run
    reports/h001_is_pass_<date>.json
"""

from __future__ import annotations

import logging
import os
import sys
import traceback as _tb
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from quantcore.alpha.delta_neutral import DeltaNeutralVRP
from quantcore.core.exceptions import DataQualityViolation
from quantcore.data.options_loader import OptionsDataLoader
from quantcore.research.experiment_tracker import Experiment, log_experiment
from quantcore.research.splitter.data_splitter import DataSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

REPORTS_DIR = _ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

IS_START = "2016-01-01"
IS_END   = "2021-12-31"

INITIAL_CAPITAL = 1_000_000.0
UNDERLYING      = "SPY"
N_TRIALS_DSR    = 1  # first trial — no multiplicity correction yet


# ── Data ────────────────────────────────────────────────────────────────

def load_spy(start: str, end: str) -> pd.Series:
    import yfinance as yf
    _log.info("Downloading SPY OHLCV (%s -> %s)...", start, end)
    t = yf.Ticker(UNDERLYING)
    hist = t.history(start=start, end=end, auto_adjust=True)
    close = hist["Close"]
    close.index = pd.DatetimeIndex([d.date() for d in close.index])
    _log.info("  %d trading days loaded", len(close))
    return close


def check_options_cache() -> tuple[int, list[str]]:
    """Return count of cached parquet files and list of available dates."""
    cache_dir = _ROOT / "data" / "options" / UNDERLYING
    if not cache_dir.exists():
        return 0, []
    files = sorted(cache_dir.glob("*.parquet"))
    dates = [f.stem for f in files]
    return len(files), dates


# ── Strategy helpers ────────────────────────────────────────────────────

def _make_strategy(
    close: pd.Series,
    options_loader: OptionsDataLoader | None = None,
    structure: str = "straddle",
) -> DeltaNeutralVRP:
    return DeltaNeutralVRP(
        close=close,
        options_loader=options_loader,
        underlying=UNDERLYING,
        structure=structure,
        dte_entry_min=30,
        dte_entry_max=45,
        dte_exit=14,
        profit_target_pct=0.50,
        risk_per_trade_pct=0.02,
        max_loss_mult=3.0,
        entry_interval_days=30,
        ivr_entry=0.0,  # unconditional entry for max trade count
        delta_hedge_threshold=5.0,
        risk_free_rate=0.04,
        commission_per_contract=0.65,
        commission_per_share=0.005,
    )


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(eq: pd.Series, trades: list) -> dict:
    if len(eq) == 0:
        return {}
    rets = eq.pct_change().dropna()
    total_ret  = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years   = len(eq) / 252
    annual_ret = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
    sharpe     = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    dd         = (eq / eq.cummax() - 1)
    max_dd     = dd.min() * 100

    pnls = [t.pnl for t in trades if t.closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / max(len(pnls), 1) * 100
    gross_profit = sum(wins) if wins else 0.0
    gross_loss   = abs(sum(losses)) if losses else 1.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    close_reasons: dict[str, int] = {}
    for t in trades:
        if t.closed:
            close_reasons[t.close_reason] = close_reasons.get(t.close_reason, 0) + 1

    return {
        "total_return_pct":  round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "gross_sharpe":      round(sharpe, 4),
        "max_drawdown_pct":  round(max_dd, 2),
        "n_trades":          len(pnls),
        "win_rate_pct":      round(win_rate, 1),
        "profit_factor":     round(pf, 2),
        "total_pnl":         round(sum(pnls), 2),
        "close_reasons":     close_reasons,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main() -> int:
    structure = "straddle"
    if len(sys.argv) > 1 and sys.argv[1] == "--strategy":
        structure = sys.argv[2] if len(sys.argv) > 2 else "straddle"

    print("=" * 70)
    print("  HYPOTHESIS_001 - In-Sample Calibration Pass")
    print("  IS Period: %s -> %s" % (IS_START, IS_END))
    print("  Structure: %s" % structure.upper())
    print("=" * 70)

    # ── 1. DataSplitter (IS phase — most restrictive) ──────────────
    splitter = DataSplitter(current_phase="is")
    splitter_report = splitter.report()
    _log.info("\nPhase status:\n  " + "\n  ".join(splitter_report["phases"]))

    # ── 2. Load spy (use splitter to slice — confirm isolation) ──────
    warmup_start = "2015-06-01"  # 300 days before IS_START for IVR
    spy = load_spy(warmup_start, "2022-01-03")
    spy_is = splitter.get(spy, "is")  # enforce IS-only slice
    _log.info("IS spy window: %s -> %s (%d days)", spy_is.index[0], spy_is.index[-1], len(spy_is))

    # ── 3. Check options cache ───────────────────────────────────────
    n_cached, cached_dates = check_options_cache()
    _log.info("Options cache: %d files (%s ... %s)",
              n_cached, cached_dates[0] if cached_dates else "-",
              cached_dates[-1] if cached_dates else "-")

    # ── 4. Build options loader (from cache) ───────────────────────
    options_loader: OptionsDataLoader | None = None
    if n_cached > 0:
        options_loader = OptionsDataLoader()  # cached mode
        _log.info("Using cached options loader")
    else:
        _log.warning("No cached options data - synthetic mode will produce fake results")

    # ── 5. Run backtest and catch every violation ───────────────────
    strategy = _make_strategy(spy, options_loader, structure=structure)
    start_ts = pd.Timestamp(IS_START)
    end_ts   = pd.Timestamp(IS_END)

    violations: list[dict] = []
    metrics: dict = {}
    trades: list = []
    equity_curve: pd.Series = pd.Series(dtype=float)

    _log.info("")
    _log.info("─" * 50)
    _log.info("Running IS backtest (2016-2021)...")
    _log.info("─" * 50)

    try:
        eq, rets, turn, logs = strategy.run_returns_backtest(
            start=start_ts,
            end=end_ts,
            training_start=start_ts,
            initial_capital=INITIAL_CAPITAL,
        )
        equity_curve = eq
        trades = logs.get("trades", [])
        metrics = compute_metrics(eq, trades)

    except DataQualityViolation as e:
        violations.append({
            "type": "DataQualityViolation",
            "message": str(e),
        })
        _log.error("DataQualityViolation caught: %s", e)

    except Exception as e:
        violations.append({
            "type": type(e).__name__,
            "message": str(e),
            "traceback": _tb.format_exc(),
        })
        _log.error("Unexpected error: %s\n%s", e, _tb.format_exc())

    # ── 6. Compute DSR (n=1 trial — first pass) ───────────────────
    if len(equity_curve) > 0:
        rets_arr = equity_curve.pct_change().dropna().values
        ret_yr   = rets_arr.mean() * 252
        vol_yr   = rets_arr.std()  * np.sqrt(252)
        sharpe   = ret_yr / vol_yr if vol_yr > 0 else 0.0
        # DSR with n=1: stat ~= sharpe, p-value ~= 0.5 (inconclusive without multiple trials)
        dsr_stat  = sharpe
        dsr_pval  = 1.0  # indeterminate with a single trial
    else:
        ret_yr = vol_yr = sharpe = dsr_stat = 0.0
        dsr_pval = 1.0

    # ── 7. Print results ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  IS CALIBRATION RESULTS (2016-2021)")
    print("=" * 70)
    print(f"  Gross Sharpe (annualized):  {sharpe:+.4f}")
    print(f"  Total Return:               {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Annual Return:              {metrics.get('annual_return_pct', 0):+.2f}%")
    print(f"  Max Drawdown:               {metrics.get('max_drawdown_pct', 0):+.2f}%")
    print(f"  Number of Trades:           {metrics.get('n_trades', 0)}")
    print(f"  Win Rate:                   {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"  Profit Factor:              {metrics.get('profit_factor', 0):.2f}")
    print(f"  Total P&L:                  ${metrics.get('total_pnl', 0):+,.2f}")
    print()
    print(f"  DSR statistic:              {dsr_stat:+.4f}")
    print(f"  DSR p-value (n=1):         {dsr_pval:.4f}  [indeterminate - need more trials]")
    print(f"  DataQualityViolations:      {len(violations)}")
    if violations:
        for v in violations:
            print(f"    [{v['type']}] {v['message'][:120]}")
    print()
    print(f"  Options data cache files:   {n_cached}")
    print(f"  Cache period:              {cached_dates[0] if cached_dates else '-'} -> {cached_dates[-1] if cached_dates else '-'}")
    print(f"  IS period in cache:         {any(IS_START <= d[:10] <= IS_END for d in cached_dates)}")
    print()

    # ── 8. Save JSON report ──────────────────────────────────────────
    import json
    json_path = REPORTS_DIR / f"h001_is_pass_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "phase": "is",
            "strategy": "DeltaNeutralVRP",
            "structure": structure,
            "is_start": IS_START,
            "is_end": IS_END,
            "n_trials": N_TRIALS_DSR,
        },
        "data_availability": {
            "options_cache_files": n_cached,
            "cache_start": cached_dates[0] if cached_dates else None,
            "cache_end": cached_dates[-1] if cached_dates else None,
            "is_period_in_cache": any(IS_START <= d[:10] <= IS_END for d in cached_dates),
        },
        "metrics": {
            "gross_sharpe": round(sharpe, 4),
            "total_return_pct": metrics.get("total_return_pct", 0),
            "annual_return_pct": metrics.get("annual_return_pct", 0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
            "n_trades": metrics.get("n_trades", 0),
            "win_rate_pct": metrics.get("win_rate_pct", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "total_pnl": metrics.get("total_pnl", 0),
        },
        "dsr": {
            "stat": round(dsr_stat, 4),
            "pvalue": round(dsr_pval, 4),
            "n_trials": N_TRIALS_DSR,
        },
        "data_quality": {
            "n_violations": len(violations),
            "violations": violations,
        },
    }
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  JSON: {json_path}")

    # ── 9. Log to experiments.csv ───────────────────────────────────
    experiment = Experiment(
        strategy_name="delta_neutral_vrp_is",
        parameters={
            "phase": "is",
            "structure": structure,
            "is_start": IS_START,
            "is_end": IS_END,
        },
        start_date=IS_START,
        end_date=IS_END,
        oos_sharpe=round(sharpe, 4),
        deflated_sharpe_pvalue=round(dsr_pval, 4),
        ci_90_lower=float("nan"),
        status="data_gap" if violations else ("no_trades" if metrics.get("n_trades", 0) == 0 else "is_complete"),
        notes=f"n_violations={len(violations)} n_trades={metrics.get('n_trades', 0)}",
    )
    exp_id = log_experiment(experiment, csv_path=_ROOT / "experiments.csv")
    print(f"  Experiment logged: {exp_id}")
    print()
    print("=" * 70)

    # ── 10. Interpretation ─────────────────────────────────────────
    cache_is_overlap = [d for d in cached_dates if IS_START <= d <= IS_END]
    if not cache_is_overlap:
        print("  INTERPRETATION:")
        print("  No options contracts cached for IS period (2016-2021).")
        print("  The cache only starts on %s." % (cached_dates[0] if cached_dates else "?"))
        print("  Historical options data (pre-2021) requires a data vendor API key.")
        print("  Options: Polygon.io, CBOE, or interactive Brokers API.")
    elif violations:
        print("  INTERPRETATION:")
        print("  Real options data exists for IS period but some contracts")
        print("  are missing. This is the expected honest result - the")
        print("  DataQualityViolation is working correctly.")
    else:
        print("  IS CALIBRATION COMPLETE - check experiments.csv")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())