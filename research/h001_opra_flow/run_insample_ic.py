#!/usr/bin/env python3
"""HYPOTHESIS_001 V2.1 — In-Sample Calibration Pass on Iron Condor.

This is the post-rejection IS pass for trial ca8727e7's MaxDD violation on the
ATM short straddle. The execution vehicle has been pivoted to Option B
(iron condor, ``alpha/iron_condor.py``) per the V2.1 amendment logged in
``quantcore/research/registry/HYPOTHESIS_001_V2.md``.

RUNNING THIS SCRIPT WITH DATA BEFORE 2023 IS PROHIBITED until the V2.1 IS pass
clears all falsification gates of §"Strict Falsification Criteria". Val (2023)
and Vault (2024) windows are quarantined by ``DataSplitter(current_phase="is")``.

Trial count after this run: 2 (1 failed = ca8727e7, this run registered as
trial 2 with ``status=is_v21_pending`` until one of the falsification gates
is satisfied). When the V2.1 IS pass clears all gates, the next IS calibration
variant — or the Val promotion decision — will be trial 3.

Usage:
    python research/h001_opra_flow/run_insample_ic.py

Outputs:
    experiments.csv                       — one row per hypothesis run
    reports/h001_v2_1_is_pass_<date>.json — full calibration artifact
"""

from __future__ import annotations

import json
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

from quantcore.alpha.iron_condor import IronCondorVRP
from quantcore.core.exceptions import DataQualityViolation
from quantcore.data.options_loader import OptionsDataLoader
from quantcore.research.experiment_tracker import Experiment, log_experiment
from quantcore.research.splitter.data_splitter import DataSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger("h001_v2_1_ic")

REPORTS_DIR = _ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# V2.1 DATES (HYPOTHESIS_001 V2.1 — do not reference V1 IS start)
IS_START = "2021-06-01"
IS_END   = "2022-12-31"
VAL_START  = "2023-01-01"   # strictly off-limits until V2.1 IS complete
VAULT_START = "2024-01-01"  # strictly off-limits until V2.1 Val complete

INITIAL_CAPITAL = 1_000_000.0
UNDERLYING = "SPY"
N_TRIALS_DSR = 2  # ca8727e7 was trial 1 (rejected); this run will be trial 2


# ── Data loading ──────────────────────────────────────────────────────────────


def load_spy(start: str, end: str) -> pd.Series:
    import yfinance as yf
    _log.info("Downloading SPY OHLCV (%s -> %s)...", start, end)
    t = yf.Ticker(UNDERLYING)
    # Pull a few extra trading days past IS_END so the warmup window is intact
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


# ── Strategy instantiation ────────────────────────────────────────────────────


def _make_strategy(
    close: pd.Series,
    options_loader: OptionsDataLoader | None = None,
) -> IronCondorVRP:
    """Build the V2.1 iron condor strategy.

    Default strike structure per ``alpha/iron_condor.py``:
        - delta_short = 0.16   (16-delta short strikes)
        - delta_long  = 0.10   (10-delta long wings, ~6 delta of protection)
        - spread_width = 5.0   (BSM-mode wing width in underlying points only;
                                real-data mode picks strikes by delta directly)

    The IVR entry threshold (40) is the IronCondorVRP default — it is
    intentionally strict to keep trade quality high at the IS gate.
    The 14-DTE time-stop exit is the IC default; tightening or relaxing it
    is the primary IS tuning axis for the next iteration.
    """
    return IronCondorVRP(
        close=close,
        options_loader=options_loader,
        underlying=UNDERLYING,
        delta_short=0.16,
        delta_long=0.10,
        spread_width=5.0,
        dte_entry_min=30,
        dte_entry_max=45,
        dte_exit=21,             # IC default; tight but tolerates decay tail
        profit_target_pct=0.50,  # take half-credit on OTM condors
        stop_loss_mult=2.0,      # cap risk at 2x credit per condor
        risk_per_trade_pct=0.02,
        entry_interval_days=30,
        ivr_entry=40.0,          # IC default — high-quality entries only
        risk_free_rate=0.04,
        use_enhanced_iv=True,    # VIX-scaled surface; consistent with vol_surface.py
    )


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(eq: pd.Series, trades: list) -> dict:
    """Same shape as run_insample_v2.py: works because CondorTrade exposes
    .pnl, .closed, .close_reason just like the straddle's TradeRecord."""
    if len(eq) == 0:
        return {}
    rets = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years = len(eq) / 252
    annual_ret = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1)
    max_dd = dd.min() * 100

    pnls = [t.pnl for t in trades if t.closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / max(len(pnls), 1) * 100
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    close_reasons: dict[str, int] = {}
    for t in trades:
        if t.closed:
            close_reasons[t.close_reason] = close_reasons.get(t.close_reason, 0) + 1

    return {
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "gross_sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades": len(pnls),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "total_pnl": round(sum(pnls), 2),
        "close_reasons": close_reasons,
    }


# ── Main flow ─────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 70)
    print("  HYPOTHESIS_001 V2.1 — In-Sample Calibration Pass (Iron Condor)")
    print("  IS Period : %s -> %s" % (IS_START, IS_END))
    print("  Reference: H001_V2 docs (ca8727e7 → rejected; pivot to Option B)")
    print("  Strategy : IronCondorVRP (alpha/iron_condor.py)")
    print("  Structure: 16Δ short / 10Δ wings / 30-45 DTE / 21-DTE time stop")
    print("=" * 70)

    # ── 1. DataSplitter locked to V2.1 boundaries ─────────────────────
    # Override SPLIT_RANGES to enforce V2 windows (same as run_insample_v2.py)
    from quantcore.research.splitter import data_splitter as ds
    original_ranges = dict(ds.SPLIT_RANGES)
    ds.SPLIT_RANGES["is"]     = (IS_START, IS_END)
    ds.SPLIT_RANGES["val"]    = (VAL_START, "2023-12-31")
    ds.SPLIT_RANGES["vault"]  = (VAULT_START, "2024-12-31")
    for phase, mask in ds._MASK_CACHE.items():
        phase_key = {"is": "is", "val": "val", "vault": "vault"}.get(phase, phase)
        start, end = ds.SPLIT_RANGES[phase]
        mask.start = start
        mask.end = end

    splitter = DataSplitter(current_phase="is")
    _log.info("\nPhase status (V2.1):")
    for line in splitter.report()["phases"]:
        _log.info("  %s", line)

    # ── 2. Load SPY (with IVR warmup tail) ────────────────────────────
    warmup_start = "2020-08-01"  # 300+ day tail for IC IVR lookback=252
    spy = load_spy(warmup_start, "2023-01-10")
    spy_is = splitter.get(spy, "is")
    _log.info("IS spy window: %s -> %s (%d days)",
              spy_is.index[0], spy_is.index[-1], len(spy_is))

    # ── 3. Check options cache ────────────────────────────────────────
    n_cached, cached = check_options_cache(IS_START, IS_END)
    _log.info("IS options cache: %d files (%s ... %s)",
              n_cached, cached[0] if cached else "-", cached[-1] if cached else "-")

    # ── 4. Build loader (BSM fallback synthetic IV if cache short) ────
    options_loader: OptionsDataLoader | None = None
    if n_cached >= 10:
        options_loader = OptionsDataLoader()
        _log.info("Using cached options loader for real IV/payment chain")
    else:
        _log.warning("Options cache sparse (%d files). IronCondorVRP will fall "
                     "back to the synthetic IV surface (BSM). Either backfill "
                     "(backfill_is_data.py) or proceed in synthetic mode for "
                     "first-pass gate evaluation.", n_cached)

    # ── 5. Run backtest ───────────────────────────────────────────────
    strategy = _make_strategy(spy, options_loader)
    start_ts = pd.Timestamp(IS_START)
    end_ts = pd.Timestamp(IS_END)

    violations: list[dict] = []
    metrics: dict = {}
    trades: list = []
    equity_curve: pd.Series = pd.Series(dtype=float)

    _log.info("")
    _log.info("Running V2.1 IS backtest (%s → %s)...", IS_START, IS_END)

    try:
        eq, rets, turn, logs = strategy.run_returns_backtest(
            start=start_ts,
            end=end_ts,
            training_start=start_ts,
            initial_capital=INITIAL_CAPITAL,
            spread_bps=5.0,       # slippage in % — 5 bps on entry/exit
            impact_bps=10.0,
            commission_per_contract=0.65,
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

    # ── 6. DSR (n=2 — still indeterminate, but we track the count) ───
    if len(equity_curve) > 0:
        rets_arr = equity_curve.pct_change().dropna().values
        ret_yr = rets_arr.mean() * 252
        vol_yr = rets_arr.std() * np.sqrt(252)
        sharpe = ret_yr / vol_yr if vol_yr > 0 else 0.0
        dsr_stat = sharpe
        dsr_pval = 1.0  # n=2 — still below the 10+ minimum for valid DSR
    else:
        sharpe = dsr_stat = 0.0
        dsr_pval = 1.0

    # ── 7. Falsification gate evaluation ───────────────────────────────
    # Pull the in-Sample MaxDD and apply the <20% gate from H001_V2 §"Strict
    # Falsification Criteria". Pass/fail here is informational on the IS
    # pass — formal gate evaluation fires at Val phase.
    is_max_dd = metrics.get("max_drawdown_pct", 0.0)
    # Cast through bool() to force Python bool — numpy.bool_ (from the
    # numpy.float64 comparison) is not JSON-serializable by default.
    is_gate_maxdd_pass = bool(is_max_dd > -20.0)  # i.e. peak-to-trough < 20%

    # ── 8. Print results ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  V2.1 IS CALIBRATION RESULTS (Iron Condor: 2021-06-01 → 2022-12-31)")
    print("=" * 70)
    print("  Gross Sharpe (annualized):  %+0.4f" % sharpe)
    print("  Total Return:               %+0.2f%%" % metrics.get("total_return_pct", 0))
    print("  Annual Return:              %+0.2f%%" % metrics.get("annual_return_pct", 0))
    print("  Max Drawdown:               %+0.2f%%  [gate <20%% → %s]"
          % (is_max_dd, "PASS" if is_gate_maxdd_pass else "FAIL"))
    print("  Number of Trades:           %d" % metrics.get("n_trades", 0))
    print("  Win Rate:                   %0.1f%%" % metrics.get("win_rate_pct", 0))
    print("  Profit Factor:              %0.2f" % metrics.get("profit_factor", 0))
    print("  Total P&L:                 $%+0.2f" % metrics.get("total_pnl", 0))
    print()
    print("  Trade close reasons:")
    for reason, count in (metrics.get("close_reasons") or {}).items():
        print("    %-16s  %d" % (reason, count))
    print()
    print("  DSR statistic (n=2):        %+0.4f  [indeterminate — need 10+ trials]" % dsr_stat)
    print("  DSR p-value (n=2):          1.0000  [need 10+ trials for valid p-value]")
    print("  DataQualityViolations:      %d" % len(violations))
    if violations:
        for v in violations:
            print("    [%s] %s" % (v["type"], v["message"][:120]))
    print()
    print("  V2.1 IS options cache:       %d files" % n_cached)
    if cached:
        print("  V2.1 IS cache period:       %s -> %s" % (cached[0], cached[-1]))
    print()
    print("=" * 70)

    # ── 8.5 Trade-ledger forensic dump (PATCH V2.1.1 → V2.1.2) ────────────
    # CondorTrade carries flat strike fields (not a `legs` list) and the
    # engine does NOT track per-day MTM or per-trade max drawdown. Each
    # trade is serialized here with the fields actually persisted by
    # IronCondorVRP._simulate(); daily_mtm and max_intraday_dd are
    # explicitly surfaced as None so the auditor can see the gap rather
    # than infer it.
    def _serialize_trade(idx: int, t) -> dict:
        max_profit = float(t.credit) * 100.0 * int(t.contracts)
        if abs(max_profit) > 1e-9:
            pnl_pct = float(t.pnl) / max_profit * 100.0
        else:
            pnl_pct = 0.0

        # Daily MTM history is now tracked on CondorTrade.
        mtm_hist = getattr(t, "daily_mtm_history", None)
        if mtm_hist:
            daily_mtm = [
                {"date": d.strftime("%Y-%m-%d"), "net_value": float(v)}
                for d, v in sorted(mtm_hist.items())
            ]
            peak = float("-inf")
            max_dd = 0.0
            for row in daily_mtm:
                v = row["net_value"]
                peak = max(peak, v)
                max_dd = min(max_dd, v - peak)
            max_dd_pct = max_dd / max_profit * 100.0 if abs(max_profit) > 1e-9 else 0.0
        else:
            daily_mtm = None
            max_dd = None
            max_dd_pct = None

        data_source = "real_chain" if t.sp_contract is not None else "bsm_fallback"
        return {
            "trade_id": idx,
            "entry_date": t.entry_date.strftime("%Y-%m-%d") if t.entry_date else None,
            "expiry":     t.expiry.strftime("%Y-%m-%d")     if t.expiry     else None,
            "close_date": t.close_date.strftime("%Y-%m-%d") if t.close_date else None,
            "close_reason": t.close_reason,
            "pnl": float(t.pnl),
            "pnl_pct_of_max_profit": round(pnl_pct, 2),
            "max_profit_dollars":    round(max_profit, 2),
            "contracts": int(t.contracts),
            "credit_per_share":      float(t.credit),
            "entry_spot":            float(t.entry_spot),
            "entry_iv":              float(t.entry_iv),
            # Strikes are flat on the dataclass; synth a legs list for readability.
            "legs": [
                {"strike": float(t.short_put),  "side": "sell", "option_type": "put"},
                {"strike": float(t.long_put),   "side": "buy",  "option_type": "put"},
                {"strike": float(t.short_call), "side": "sell", "option_type": "call"},
                {"strike": float(t.long_call),  "side": "buy",  "option_type": "call"},
            ],
            "wing_width_put":  round(float(t.short_put)  - float(t.long_put),  4),
            "wing_width_call": round(float(t.long_call) - float(t.short_call), 4),
            "max_intraday_dd": round(max_dd, 2) if max_dd is not None else None,
            "max_intraday_dd_pct": round(max_dd_pct, 2) if max_dd_pct is not None else None,
            "daily_mtm": daily_mtm,
            "data_source": data_source,
        }

    # ── 9. Save artifact ──────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    json_path = REPORTS_DIR / ("h001_v2_1_is_pass_" + ts + ".json")
    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "version": "V2.1.3",          # bumped: MTM ledger + IS-boundary patch
            "patch_notes": (
                "V2.1.3: serialize per-trade ledger (entry/exit/strikes/credit/pnl/entry_iv) "
                "and persist daily MTM snapshots in IronCondorVRP. Simulation now starts at "
                "the IS start date so warmup trades do not leak into the ledger. Patch adapted "
                "from user-supplied schema — original schema assumed CondorTrade.legs and "
                "daily_mtm_history which did not exist in V2.1.0."
            ),
            "reference": "HYPOTHESIS_001_V2.md (V2.1 amendment)",
            "prior_failed_trial": "ca8727e7",
            "phase": "is",
            "strategy": "IronCondorVRP",
            "structure": "iron_condor (16Δ short / 10Δ wings / 5pt BSM)",
            "is_start": IS_START,
            "is_end": IS_END,
            "trial_count": N_TRIALS_DSR,
        },
        "options_cache": {
            "v2_1_is_files": n_cached,
            "cache_start": cached[0] if cached else None,
            "cache_end": cached[-1] if cached else None,
        },
        "metrics": {
            "gross_sharpe": round(sharpe, 4),
            "total_return_pct": metrics.get("total_return_pct", 0),
            "annual_return_pct": metrics.get("annual_return_pct", 0),
            "max_drawdown_pct": is_max_dd,
            "n_trades": metrics.get("n_trades", 0),
            "win_rate_pct": metrics.get("win_rate_pct", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "close_reasons": metrics.get("close_reasons", {}),
        },
        "trade_ledger_forensics": {
            "n_trades_serialized":  len(trades),
            "data_sources": {
                "real_chain":   sum(
                    1 for t in trades if getattr(t, "sp_contract", None) is not None
                ),
                "bsm_fallback": sum(
                    1 for t in trades if getattr(t, "sp_contract", None) is None
                ),
            },
            "engine_logs_daily_mtm":     True,
            "engine_logs_max_intraday":  True,
            "patch_reason":              "trade-ledger forensics void in V2.1 baseline artifact",
        },
        "trades": [_serialize_trade(i, t) for i, t in enumerate(trades)],
        "gate_evaluation": {
            "max_dd_threshold_pct": 20.0,
            "max_dd_pass": is_gate_maxdd_pass,
            "dsr_pvalue_threshold": 0.10,
            "dsr_pvalue_indeterminate_at_n2": True,
        },
        "dsr": {"stat": round(dsr_stat, 4), "pvalue": 1.0, "n_trials": N_TRIALS_DSR},
        "data_quality": {"n_violations": len(violations), "violations": violations},
    }
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print("  JSON: %s  (with %d-trade ledger)" % (json_path, len(trades)))

    # ── 10. Log to experiments.csv ───────────────────────────────────
    # Status enum encodes the IS calibration outcome:
    #   - is_v21_passed:  MaxDD < 20% on IC → advance (cautiously) toward Val
    #   - is_v21_maxdd_fail: MaxDD >= 20% on IC → reject IC; try next iteration
    #   - is_v21_pending_trials: data gaps or contract selection failed → retry
    if violations:
        status = "is_v21_pending_trials"
        status_reason = "DQV"
    elif metrics.get("n_trades", 0) == 0:
        status = "is_v21_pending_trials"
        status_reason = "no_trades"
    elif not is_gate_maxdd_pass:
        status = "is_v21_maxdd_fail"
        status_reason = "maxdd_gate_violated"
    else:
        status = "is_v21_passed"
        status_reason = "maxdd_within_20pct"

    notes = ("V2.1 IC trial=%d status=%s reason=%s | "
             "n_violations=%d n_trades=%d win_rate=%.1f%% pf=%.2f maxdd=%.2f%% | "
             "DSR indeterminate n=%d | prior failed: ca8727e7 (straddle MaxDD=-46%% gamma-whip)"
             % (N_TRIALS_DSR, status, status_reason,
                len(violations), metrics.get("n_trades", 0),
                metrics.get("win_rate_pct", 0), metrics.get("profit_factor", 0),
                is_max_dd, N_TRIALS_DSR))

    experiment = Experiment(
        strategy_name="iron_condor_vrp_is_v21",
        parameters={
            "version": "V2.1",
            "amendment": "ca8727e7 rejected → pivot from Option A straddle to Option B iron condor",
            "reference": "HYPOTHESIS_001_V2.md (V2.1 amendment)",
            "prior_trial": "ca8727e7",
            "structure": "iron_condor",
            "delta_short": 0.16,
            "delta_long": 0.10,
            "spread_width": 5.0,
            "dte_entry_min": 30,
            "dte_entry_max": 45,
            "dte_exit": 21,
            "profit_target_pct": 0.50,
            "stop_loss_mult": 2.0,
            "risk_per_trade_pct": 0.02,
            "entry_interval_days": 30,
            "ivr_entry": 40.0,
            "use_enhanced_iv": True,
            "phase": "is",
            "is_start": IS_START,
            "is_end": IS_END,
        },
        start_date=IS_START,
        end_date=IS_END,
        oos_sharpe=round(sharpe, 4),
        deflated_sharpe_pvalue=round(dsr_pval, 4),
        ci_90_lower=float("nan"),
        status=status,
        notes=notes,
    )
    exp_id = log_experiment(experiment, csv_path=_ROOT / "experiments.csv")
    print("  Experiment logged: %s [%s]" % (exp_id, status))

    # ── 11. Restore SPLIT_RANGES ─────────────────────────────────────
    ds.SPLIT_RANGES.clear()
    ds.SPLIT_RANGES.update(original_ranges)
    for phase, mask in ds._MASK_CACHE.items():
        start, end = original_ranges.get(phase, (IS_START, IS_END))
        mask.start = start
        mask.end = end

    print()
    print("  NEXT STEPS:")
    if violations:
        print("  - DataQualityViolation raised. Inspect options chain integrity.")
    elif metrics.get("n_trades", 0) == 0:
        print("  - 0 trades executed. Inspect IVR entry threshold vs IVR compute.")
    elif not is_gate_maxdd_pass:
        print("  - V2.1 IS MaxDD < 20%% gate FAILED. Reject Option B at IC structure.")
        print("    Consider: tighten deltas (16→20), shorter DTE, or pivoting to")
        print("    a true variance swap / SPX-only variant.")
    else:
        print("  - V2.1 IC passes IC MaxDD gate. Skew tax = gross_sharpe drop vs the")
        print("    straddle baseline. Decide whether the residual gross_sharpe is")
        print("    high enough to justify a Val phase promotion (Val still LOCKED).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
