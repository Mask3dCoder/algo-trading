#!/usr/bin/env python3
"""HYPOTHESIS_001 V2 — In-Sample Calibration Pass: 2021-06-01 -> 2022-12-31.

RUNNING THIS SCRIPT WITH DATA BEFORE 2023 IS PROHIBITED.
This file references HYPOTHESIS_001 V2 only.

Usage:
    python research/h001_opra_flow/run_insample_v2.py

Outputs:
    experiments.csv        — one row per hypothesis run
    reports/h001_v2_is_pass_<date>.json
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

# V2 AMENDED DATES (HYPOTHESIS_001 V2 — do not reference V1)
IS_START = "2021-06-01"
IS_END   = "2022-12-31"
VAL_START = "2023-01-01"  # strictly off-limits until IS complete
VAULT_START = "2024-01-01"  # strictly off-limits until Val complete

INITIAL_CAPITAL = 1_000_000.0
UNDERLYING      = "SPY"
N_TRIALS_DSR    = 1  # first trial — needs at least 10+ trials for DSR p-value


def load_spy(start: str, end: str) -> pd.Series:
    import yfinance as yf
    _log.info("Downloading SPY OHLCV (%s -> %s)...", start, end)
    t = yf.Ticker(UNDERLYING)
    hist = t.history(start=start, end="2023-01-10", auto_adjust=True)
    close = hist["Close"]
    close.index = pd.DatetimeIndex([d.date() for d in close.index])
    _log.info("  %d trading days loaded", len(close))
    return close


def check_options_cache(start: str, end: str) -> tuple[int, list[str]]:
    """Return count of cached parquet files within [start, end]."""
    cache_dir = _ROOT / "data" / "options" / UNDERLYING
    if not cache_dir.exists():
        return 0, []
    files = sorted(cache_dir.glob("*.parquet"))
    dates = [f.stem for f in files if start <= f.stem[:10] <= end]
    return len(dates), dates


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
        ivr_entry=0.0,  # unconditional — maximize trade count
        delta_hedge_threshold=5.0,
        risk_free_rate=0.04,
        commission_per_contract=0.65,
        commission_per_share=0.005,
    )


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
    wins   = [p for p in pnls if p > 0]
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


def main() -> int:
    print("=" * 70)
    print("  HYPOTHESIS_001 V2 — In-Sample Calibration Pass")
    print("  IS Period: %s -> %s" % (IS_START, IS_END))
    print("  Reference: HYPOTHESIS_001 V2 (governance patch)")
    print("=" * 70)

    # ── 1. DataSplitter locked to V2 boundaries ──────────────────────
    # Override SPLIT_RANGES temporarily to enforce V2 windows
    from quantcore.research.splitter import data_splitter as ds
    original_ranges = dict(ds.SPLIT_RANGES)
    ds.SPLIT_RANGES["is"]    = (IS_START,    IS_END)
    ds.SPLIT_RANGES["val"]   = (VAL_START,    "2023-12-31")
    ds.SPLIT_RANGES["vault"] = (VAULT_START,  "2024-12-31")
    for phase, mask in ds._MASK_CACHE.items():
        phase_key = {"is": "is", "val": "val", "vault": "vault"}.get(phase, phase)
        start, end = ds.SPLIT_RANGES[phase]
        mask.start = start
        mask.end   = end

    splitter = DataSplitter(current_phase="is")
    _log.info("\nPhase status (V2):")
    for line in splitter.report()["phases"]:
        _log.info("  %s", line)

    # ── 2. Load SPY ─────────────────────────────────────────────────
    warmup_start = "2021-01-01"  # enough for 150-day IVR warmup
    spy = load_spy(warmup_start, "2023-01-10")
    spy_is = splitter.get(spy, "is")
    _log.info("IS spy window: %s -> %s (%d days)",
              spy_is.index[0], spy_is.index[-1], len(spy_is))

    # ── 3. Check options cache ───────────────────────────────────────
    n_cached, cached = check_options_cache("2021-06-01", "2022-12-31")
    _log.info("IS options cache: %d files (%s ... %s)",
              n_cached, cached[0] if cached else "-", cached[-1] if cached else "-")

    # ── 4. Build loader ─────────────────────────────────────────────
    options_loader: OptionsDataLoader | None = None
    if n_cached >= 10:
        options_loader = OptionsDataLoader()
        _log.info("Using cached options loader")
    else:
        _log.error("Insufficient options cache for V2 IS period: %d files", n_cached)
        print("  ERROR: Options cache insufficient for V2 IS period.")
        print("  Run backfill first: python research/h001_opra_flow/backfill_is_data.py")
        return 1

    # ── 5. Run backtest ─────────────────────────────────────────────
    strategy = _make_strategy(spy, options_loader, structure="straddle")
    start_ts = pd.Timestamp(IS_START)
    end_ts   = pd.Timestamp(IS_END)

    violations: list[dict] = []
    metrics: dict = {}
    trades: list = []
    equity_curve: pd.Series = pd.Series(dtype=float)

    _log.info("")
    _log.info("Running IS backtest (V2: %s -> %s)...", IS_START, IS_END)

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
        violations.append({"type": "DataQualityViolation", "message": str(e)})
        _log.error("DataQualityViolation: %s", e)

    except Exception as e:
        violations.append({"type": type(e).__name__, "message": str(e),
                           "traceback": _tb.format_exc()})
        _log.error("Unexpected error: %s", e)

    # ── 6. DSR (n=1 — indeterminate) ───────────────────────────────
    if len(equity_curve) > 0:
        rets_arr = equity_curve.pct_change().dropna().values
        ret_yr  = rets_arr.mean() * 252
        vol_yr  = rets_arr.std() * np.sqrt(252)
        sharpe  = ret_yr / vol_yr if vol_yr > 0 else 0.0
        dsr_stat = sharpe
        dsr_pval = 1.0  # n=1 — indeterminate until more trials
    else:
        sharpe = dsr_stat = 0.0
        dsr_pval = 1.0

    # ── 7. Print results ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("  IS CALIBRATION RESULTS (V2: 2021-06-01 -> 2022-12-31)")
    print("=" * 70)
    print("  Gross Sharpe (annualized):  %+0.4f" % sharpe)
    print("  Total Return:               %+0.2f%%" % metrics.get("total_return_pct", 0))
    print("  Annual Return:              %+0.2f%%" % metrics.get("annual_return_pct", 0))
    print("  Max Drawdown:               %+0.2f%%" % metrics.get("max_drawdown_pct", 0))
    print("  Number of Trades:           %d" % metrics.get("n_trades", 0))
    print("  Win Rate:                   %0.1f%%" % metrics.get("win_rate_pct", 0))
    print("  Profit Factor:              %0.2f" % metrics.get("profit_factor", 0))
    print("  Total P&L:                 $%+0.2f" % metrics.get("total_pnl", 0))
    print()
    print("  DSR statistic (n=1):        %+0.4f  [indeterminate — need more trials]" % dsr_stat)
    print("  DSR p-value (n=1):          1.0000  [need 10+ trials for valid p-value]")
    print("  DataQualityViolations:      %d" % len(violations))
    if violations:
        for v in violations:
            print("    [%s] %s" % (v["type"], v["message"][:120]))
    print()
    print("  V2 IS options cache:         %d files" % n_cached)
    if cached:
        print("  V2 IS cache period:         %s -> %s" % (cached[0], cached[-1]))
    print()
    print("=" * 70)

    # ── 8. Save report ──────────────────────────────────────────────
    import json
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    json_path = REPORTS_DIR / ("h001_v2_is_pass_" + ts + ".json")
    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "version": "V2",
            "reference": "HYPOTHESIS_001_V2.md",
            "phase": "is",
            "strategy": "DeltaNeutralVRP",
            "is_start": IS_START,
            "is_end": IS_END,
        },
        "options_cache": {
            "v2_is_files": n_cached,
            "cache_start": cached[0] if cached else None,
            "cache_end": cached[-1] if cached else None,
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
        "dsr": {"stat": round(dsr_stat, 4), "pvalue": 1.0, "n_trials": 1},
        "data_quality": {"n_violations": len(violations), "violations": violations},
    }
    json_path.write_text(json.dumps(report, indent=2))
    print("  JSON: %s" % json_path)

    # ── 9. Log to experiments.csv ───────────────────────────────────
    experiment = Experiment(
        strategy_name="delta_neutral_vrp_is_v2",
        parameters={
            "version": "V2", "phase": "is",
            "is_start": IS_START, "is_end": IS_END,
            "reference": "HYPOTHESIS_001_V2.md",
        },
        start_date=IS_START,
        end_date=IS_END,
        oos_sharpe=round(sharpe, 4),
        deflated_sharpe_pvalue=round(dsr_pval, 4),
        ci_90_lower=float("nan"),
        status="is_v2_pending_trials",
        notes=("n_violations=%d n_trades=%d dsr_pvalue=1.0(n=1,need 10+trials)"
               % (len(violations), metrics.get("n_trades", 0))),
    )
    exp_id = log_experiment(experiment, csv_path=_ROOT / "experiments.csv")
    print("  Experiment logged: %s" % exp_id)

    # ── 10. Restore SPLIT_RANGES ────────────────────────────────────
    ds.SPLIT_RANGES.clear()
    ds.SPLIT_RANGES.update(original_ranges)
    for phase, mask in ds._MASK_CACHE.items():
        start, end = original_ranges.get(phase, original_ranges.get(phase, (IS_START, IS_END)))
        mask.start = start
        mask.end   = end

    print()
    print("  NEXT STEPS:")
    if violations:
        print("  - DataQualityViolation raised. Fix data gaps before proceeding.")
    elif metrics.get("n_trades", 0) == 0:
        print("  - No trades executed. Check entry logic.")
    else:
        print("  - IS results computed. DSR p-value indeterminate until 10+ trials.")
        print("  - Continue IS calibration or advance to Val phase.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())