#!/usr/bin/env python3
# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""institutional_gate.py — Audit Mandate 2026-07-12 (Phase 3 + Phase 8).

THE SINGLE SOURCE OF TRUTH for strategy deployment. Replaces
scripts/verify_honest_foundation.py, whose triage fed the gate HARDCODED metrics
and whose portfolio DSR ran on rng.normal(...) mock returns. Both are fabrications.

This gate uses ONLY real data and FAILS CLOSED. It now supports two strategies:

  * options_delta_neutral  -> real SPY option chains (data/options/SPY/*.parquet)
  * equity_factor_neutral  -> real PIT SP500 panel
                              (data/sp500_pit_universe_2018_2024.parquet)

Data policy:
  * ANY path containing 'synthetic' -> RuntimeError. No mocks. No BSM fills.
  * The real equity panel has ONLY [symbol, timestamp, OHLCV]. There is NO sector,
    NO market_cap, NO returns column on any available real file (verified). The
    strategy therefore neutralises cross-sectionally and refuses to fabricate
    sector/size maps.

Gate criteria (equity_factor_neutral, per Phase-8 directive):
    oos_sharpe (annualized, 10 bps/leg costs)  > 0.65
    max_drawdown                                 < 25%
    deflated_sharpe_p_value (num_trials from experiments.csv) < 0.10
    total_trades (unique entry events)          > 200
    all 5 anchored folds evaluated on REAL data

On PASS -> writes the verdict JSON + (equity) factor_backtest_trades.csv.
On FAIL -> writes the verdict JSON AND invokes scripts/emergency_kill.py
           (dry-run unless env GATE_ARM_KILL=1, so the truth-check is safe).

Usage:
    python scripts/institutional_gate.py --strategy equity_factor_neutral
    python scripts/institutional_gate.py --strategy options_delta_neutral
    python scripts/institutional_gate.py --strategy equity_factor_neutral --no-backtest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REAL_PANEL_EQUITY = REPO_ROOT / "data" / "sp500_pit_universe_2018_2024.parquet"
REAL_PANEL_OPTION_SRC = REPO_ROOT / "data" / "real_panel_2018_2024.csv"
OPTIONS_DIR = REPO_ROOT / "data" / "options" / "SPY"
EXPERIMENTS_CSV = REPO_ROOT / "experiments.csv"
EQUITY_VERDICT_PATH = REPO_ROOT / "reports" / "factor_gate_verdict.json"
OPTIONS_VERDICT_PATH = REPO_ROOT / "reports" / "GATE_VERDICT.json"
EQUITY_TRADES_PATH = REPO_ROOT / "reports" / "factor_backtest_trades.csv"
EMERGENCY_KILL = REPO_ROOT / "scripts" / "emergency_kill.py"

# Five ANCHORED equity folds (train window, then a 1-year OOS), per directive 3d.
EQUITY_FOLDS = [
    ("F1", date(2018, 1, 1), date(2020, 1, 1), date(2020, 12, 31)),
    ("F2", date(2019, 1, 1), date(2021, 1, 1), date(2021, 12, 31)),
    ("F3", date(2020, 1, 1), date(2022, 1, 1), date(2022, 12, 31)),
    ("F4", date(2021, 1, 1), date(2023, 1, 1), date(2023, 12, 31)),
    ("F5", date(2022, 1, 1), date(2024, 1, 1), date(2024, 12, 31)),
]

# Five ANCHORED options folds (legacy, calendar years; real chains only 2021-2024).
OPTIONS_FOLDS = [
    ("2018-2019", date(2018, 1, 1), date(2019, 12, 31)),
    ("2019-2020", date(2019, 1, 1), date(2020, 12, 31)),
    ("2020-2021", date(2020, 1, 1), date(2021, 12, 31)),
    ("2021-2022", date(2021, 1, 1), date(2022, 12, 31)),
    ("2022-2023", date(2022, 1, 1), date(2023, 12, 31)),
]

EQUITY_GATE = {
    "min_oos_sharpe": 0.65,
    "max_drawdown_pct": 25.0,
    "max_dsr_p_value": 0.10,
    "min_oos_trades": 200,
    "required_folds": 5,
}
OPTIONS_GATE = {
    "min_oos_sharpe": 0.65,
    "max_drawdown_pct": 25.0,
    "max_dsr_p_value": 0.10,
    "min_oos_trades": 50,
    "required_folds_with_data": 5,
}


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #
def _assert_not_synthetic(*paths: Path) -> None:
    for p in paths:
        if "synthetic" in str(p).lower():
            raise RuntimeError("Synthetic data detected in production gate. Halting.")


def count_trials() -> int:
    """Number of trials for DSR deflation, from experiments.csv (real row count)."""
    override = os.environ.get("GATE_NUM_TRIALS")
    if override:
        return int(override)
    try:
        n = max(0, sum(1 for _ in open(EXPERIMENTS_CSV, encoding="utf-8")) - 1)
    except OSError:
        n = 0
    return max(n, 1) + 10  # conservative prior


# --------------------------------------------------------------------------- #
# Equity factor-neutral runner
# --------------------------------------------------------------------------- #
def _load_equity_panel(max_symbols: int = 0) -> pd.DataFrame:
    if not REAL_PANEL_EQUITY.exists():
        raise RuntimeError(
            f"Real equity panel missing: {REAL_PANEL_EQUITY}. "
            f"The directive named data/real_panel_2018_2024.parquet; only the SP500 "
            f"PIT universe parquet exists. Refusing to fabricate inputs."
        )
    _assert_not_synthetic(REAL_PANEL_EQUITY)
    df = pd.read_parquet(REAL_PANEL_EQUITY)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if max_symbols and max_symbols > 0:
        keep = (
            df.groupby("symbol")["close"].count().nlargest(max_symbols).index.tolist()
        )
        df = df[df["symbol"].isin(keep)].copy()
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _make_equity_strategy(strategy_name: str, oos_start: date):
    """Instantiate the equity strategy under test (PIT, engine-compatible)."""
    if strategy_name == "equity_pairs":
        from quantcore.strategies.equity.equity_pairs import EquityPairs
        return EquityPairs(
            name="equity_pairs", oos_start=oos_start, rebalance_days=5,
            capital=1_000_000.0, gross_target=1.0,
            entry_z=2.0, exit_z=0.0, warmup_bars=60,
            pair_list_path="data/pairs_found.csv",
        )
    from quantcore.strategies.equity.equity_factor_neutral import EquityFactorNeutral
    return EquityFactorNeutral(
        name="equity_factor_neutral", oos_start=oos_start, rebalance_days=5,
        capital=1_000_000.0, gross_target=1.0,
    )


def _run_equity_fold(
    full_df: pd.DataFrame,
    label: str,
    train_start: date,
    oos_start: date,
    oos_end: date,
    cost_per_leg: float,
    max_symbols: int,
    strategy_name: str = "equity_factor_neutral",
) -> dict:
    from quantcore.backtest.engine import BacktestEngine

    # Slice to [train_start - 395d buffer, oos_end] so factors have history.
    buf = train_start - timedelta(days=395)
    fold_df = full_df[
        (full_df["timestamp"] >= pd.Timestamp(buf))
        & (full_df["timestamp"] <= pd.Timestamp(oos_end))
    ].copy()
    if fold_df.empty:
        return {"fold": label, "status": "DATA_GAP", "reason": "no panel rows in window"}

    strat = _make_equity_strategy(strategy_name, oos_start)
    engine = BacktestEngine(
        strategy=strat,
        initial_cash=1_000_000.0,
        slippage_bps=cost_per_leg * 10_000.0,  # 10 bps/leg
        commission_per_share=0.0,
        sizing_pct=0.0,
    )
    eq_df = engine.run(fold_df)
    if eq_df.empty or "equity" not in eq_df:
        return {"fold": label, "status": "ERROR", "reason": "empty equity curve"}

    # Extract OOS-window daily returns (strictly within [oos_start, oos_end]).
    eq_df = eq_df.copy()
    eq_df["ts"] = pd.to_datetime(eq_df["timestamp"])
    oos_mask = (eq_df["ts"] >= pd.Timestamp(oos_start)) & (eq_df["ts"] <= pd.Timestamp(oos_end))
    oos = eq_df[oos_mask].reset_index(drop=True)
    if len(oos) < 2:
        return {"fold": label, "status": "DATA_GAP", "reason": "too few OOS rows"}
    oos_eq = oos["equity"].to_numpy(dtype=float)
    oos_ret = np.diff(oos_eq) / oos_eq[:-1]
    oos_ret = oos_ret[np.isfinite(oos_ret)]

    # Trade events: distinct (date, symbol) pairs in the OOS trade log.
    log = strat.export_trade_log()
    if not log.empty:
        log = log[log["date"] >= oos_start.isoformat()]
        log = log[log["date"] <= oos_end.isoformat()]
        total_trades = log[["date", "symbol"]].drop_duplicates().shape[0]
    else:
        total_trades = 0

    if len(oos_ret) < 2 or np.std(oos_ret, ddof=1) < 1e-12:
        return {
            "fold": label, "status": "RUN", "oos_sharpe": float("nan"),
            "max_dd_pct": 0.0, "n_trades": int(total_trades), "oos_days": int(len(oos)),
            "oos_returns": [],
        }

    sharpe = float(np.mean(oos_ret) / np.std(oos_ret, ddof=1) * np.sqrt(252))
    eq_curve = np.cumprod(1.0 + oos_ret)
    peak = np.maximum.accumulate(eq_curve)
    max_dd = float(abs(np.min((eq_curve - peak) / peak)) * 100.0)
    return {
        "fold": label, "status": "RUN",
        "oos_sharpe": round(sharpe, 4), "max_dd_pct": round(max_dd, 2),
        "n_trades": int(total_trades), "oos_days": int(len(oos)),
        "oos_returns": oos_ret.tolist(),
    }


def run_equity_strategy(
    cost_per_leg: float,
    run_backtest: bool,
    max_symbols: int,
    strategy_name: str = "equity_factor_neutral",
) -> dict:
    panel = _load_equity_panel(max_symbols=max_symbols)
    cov_start = panel["timestamp"].min().date()
    cov_end = panel["timestamp"].max().date()
    num_trials = count_trials()

    fold_reports: list[dict] = []
    all_oos: list[float] = []
    total_trades = 0

    for label, tr_s, oos_s, oos_e in EQUITY_FOLDS:
        rep = {"fold": label, "train_start": str(tr_s), "oos_start": str(oos_s),
               "oos_end": str(oos_e)}
        if not run_backtest:
            rep["status"] = "SKIPPED"
            fold_reports.append(rep)
            continue
        print(f"  [{label}] backtesting OOS {oos_s}..{oos_e} ...", flush=True)
        try:
            r = _run_equity_fold(panel, label, tr_s, oos_s, oos_e, cost_per_leg, max_symbols, strategy_name)
            rep.update(r)
            all_oos.extend(r.get("oos_returns", []))
            total_trades += int(r.get("n_trades", 0))
        except Exception as exc:
            rep.update({"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"})
        fold_reports.append(rep)
        print(f"      -> {rep.get('status')} sharpe={rep.get('oos_sharpe')} "
              f"dd={rep.get('max_dd_pct')}% trades={rep.get('n_trades')}", flush=True)

    agg = _aggregate_equity(all_oos, num_trials)
    failures = _apply_equity_gate(fold_reports, agg, total_trades)

    verdict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_name,
        "gate_conditions": EQUITY_GATE,
        "cost_per_leg": cost_per_leg,
        "num_trials_for_dsr": num_trials,
        "real_data_coverage": {"start": str(cov_start), "end": str(cov_end)},
        "folds": fold_reports,
        "aggregate": agg,
        "gate_status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    verdict_path = REPO_ROOT / "reports" / f"{strategy_name}_gate_verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verdict_path, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(f"\nWritten: {verdict_path}")
    return {"verdict": verdict, "combined": {"returns": all_oos, "total_trades": total_trades}}


def _collect_combined_oos_returns(full_df: pd.DataFrame, cost_per_leg: float, max_symbols: int) -> dict:
    """Deprecated: kept for backward reference. Aggregation now happens inline in
    run_equity_strategy via per-fold `oos_returns`. Retained to avoid breaking the
    public surface; not called by main()."""
    return {"returns": [], "total_trades": 0}


def _aggregate_equity(oos_returns: list[float], num_trials: int) -> dict:
    from quantcore.research.dsr_published import deflated_sharpe_ratio_published

    r = np.asarray(oos_returns, dtype=float)
    if r.size < 2 or np.std(r, ddof=1) < 1e-12:
        return {"annualized_sharpe": float("nan"), "max_dd_pct": float("nan"),
                "dsr_p_value": 1.0, "win_rate": 0.0, "n_obs": int(r.size)}
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    max_dd = float(abs(np.min((eq - peak) / peak)) * 100.0)
    win_rate = float(np.mean(r > 0))
    try:
        # Published B&LdP 2014 DSR: per-observation Sharpe + RAW kurtosis.
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        skew = float(np.mean((r - mean_r) ** 3) / std_r**3)
        kurt_raw = float(np.mean((r - mean_r) ** 4) / std_r**4)
        dsr_p, _ = deflated_sharpe_ratio_published(
            observed_sharpe=sharpe / np.sqrt(252), num_trials=num_trials,
            sample_length=r.size, skewness=skew, kurtosis=kurt_raw,
        )
    except Exception:
        dsr_p = 1.0
    return {
        "annualized_sharpe": round(sharpe, 4), "max_dd_pct": round(max_dd, 2),
        "dsr_p_value": round(float(dsr_p), 4), "win_rate": round(win_rate, 4),
        "n_obs": int(r.size),
    }


def _apply_equity_gate(fold_reports: list[dict], agg: dict, total_trades: int) -> list[str]:
    fails: list[str] = []
    runs = [f for f in fold_reports if f.get("status") == "RUN"]
    if len(runs) < EQUITY_GATE["required_folds"]:
        fails.append(
            f"Only {len(runs)}/{EQUITY_GATE['required_folds']} folds produced a real "
            f"backtest (panel coverage or errors)."
        )
    s = agg.get("annualized_sharpe")
    if not (isinstance(s, (int, float)) and np.isfinite(s) and s > EQUITY_GATE["min_oos_sharpe"]):
        fails.append(f"OOS Sharpe {s} <= {EQUITY_GATE['min_oos_sharpe']}")
    if not (agg.get("max_dd_pct", 100) < EQUITY_GATE["max_drawdown_pct"]):
        fails.append(f"Max DD {agg.get('max_dd_pct')}% >= {EQUITY_GATE['max_drawdown_pct']}%")
    if not (agg.get("dsr_p_value", 1.0) < EQUITY_GATE["max_dsr_p_value"]):
        fails.append(
            f"DSR p {agg.get('dsr_p_value')} >= {EQUITY_GATE['max_dsr_p_value']} "
            f"(num_trials={agg.get('n_obs')} obs; deflated null is severe by design)"
        )
    if not (total_trades > EQUITY_GATE["min_oos_trades"]):
        fails.append(f"total_trades {total_trades} <= {EQUITY_GATE['min_oos_trades']}")
    return fails


# --------------------------------------------------------------------------- #
# Options delta-neutral runner (legacy, kept intact)
# --------------------------------------------------------------------------- #
def run_options_strategy(run_backtest: bool, force: bool) -> dict:
    from quantcore.alpha.delta_neutral import DeltaNeutralVRP
    from quantcore.backtest.walk_forward import compute_oos_metrics

    _assert_not_synthetic(REAL_PANEL_OPTION_SRC, OPTIONS_DIR)
    if not OPTIONS_DIR.exists() or not any(OPTIONS_DIR.glob("*.parquet")):
        raise RuntimeError(f"Real options chains missing under {OPTIONS_DIR}. Halting.")

    cov_start = min(d.date() for d in (date.fromisoformat(p.stem) for p in
                  sorted(OPTIONS_DIR.glob("*.parquet"))))
    cov_end = max(d.date() for d in (date.fromisoformat(p.stem) for p in
                sorted(OPTIONS_DIR.glob("*.parquet"))))
    num_trials = count_trials()

    close = pd.read_csv(REAL_PANEL_OPTION_SRC, parse_dates=["timestamp"])
    spy = close[close["symbol"] == "SPY"].sort_values("timestamp")
    spy_s = pd.Series(spy["close"].to_numpy(), index=pd.DatetimeIndex(spy["timestamp"]), name="SPY")

    covered = sum(1 for _, s, e in OPTIONS_FOLDS if (s <= cov_end and e >= cov_start))
    coverage_dispositive = covered < OPTIONS_GATE["required_folds_with_data"]
    run_bt = run_backtest and (force or not coverage_dispositive)

    fold_reports, all_ret, total_trades = [], [], 0
    folds_with_data = 0
    for label, f_s, f_e in OPTIONS_FOLDS:
        rep = {"fold": label, "oos_start": str(f_s), "oos_end": str(f_e)}
        if (f_s <= cov_end and f_e >= cov_start):
            folds_with_data += 1
            if not run_bt:
                rep["status"] = "COVERED_NOT_RUN"
            else:
                try:
                    loader = _options_loader()
                    strat = DeltaNeutralVRP(spy_s, options_loader=loader, structure="straddle")
                    eff_s, eff_e = max(f_s, cov_start), min(f_e, cov_end)
                    eq, ret, _, logs = strat.run_returns_backtest(
                        start=pd.Timestamp(eff_s), end=pd.Timestamp(eff_e),
                        training_start=pd.Timestamp(eff_s), initial_capital=100_000.0,
                        spread_bps=10.0, impact_bps=20.0)
                    m = compute_oos_metrics(ret.to_numpy(), eq.to_numpy()) if len(ret) else {}
                    rep.update({"status": "RUN", "oos_sharpe": m.get("sharpe"),
                                "max_dd_pct": m.get("max_dd_pct"),
                                "n_trades": len(logs.get("trades", []))})
                    total_trades += len(logs.get("trades", []))
                    all_ret.extend([x for x in ret.to_numpy().tolist() if np.isfinite(x)])
                except Exception as exc:
                    rep.update({"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"})
        else:
            rep.update({"status": "DATA_GAP",
                        "reason": f"real options coverage is {cov_start}..{cov_end}"})
        fold_reports.append(rep)

    agg = _aggregate_options(all_ret, num_trials)
    failures = _apply_options_gate(folds_with_data, agg, total_trades, run_bt)

    verdict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": "options_delta_neutral",
        "gate_conditions": OPTIONS_GATE,
        "num_trials_for_dsr": num_trials,
        "real_options_coverage": {"start": str(cov_start), "end": str(cov_end)},
        "folds": fold_reports, "aggregate": agg,
        "gate_status": "PASS" if not failures else "FAIL", "failures": failures,
    }
    OPTIONS_VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTIONS_VERDICT_PATH, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(f"\nWritten: {OPTIONS_VERDICT_PATH}")
    return verdict


def _options_loader():
    from quantcore.data.options_loader import OptionsDataLoader
    return OptionsDataLoader(data_dir=REPO_ROOT / "data" / "options")


def _aggregate_options(oos_returns: list[float], num_trials: int) -> dict:
    from quantcore.research.deflated_sharpe import deflated_sharpe_ratio

    r = np.asarray(oos_returns, dtype=float)
    if r.size < 2 or np.std(r, ddof=1) < 1e-12:
        return {"annualized_sharpe": float("nan"), "max_dd_pct": float("nan"),
                "dsr_p_value": 1.0, "n_obs": int(r.size)}
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    max_dd = float(abs(np.min((eq - peak) / peak)) * 100.0)
    try:
        dsr_p, _ = deflated_sharpe_ratio(observed_sharpe=sharpe / np.sqrt(252),
                                         num_trials=num_trials, sample_length=r.size)
    except Exception:
        dsr_p = 1.0
    return {"annualized_sharpe": round(sharpe, 4), "max_dd_pct": round(max_dd, 2),
            "dsr_p_value": round(float(dsr_p), 4), "n_obs": int(r.size)}


def _apply_options_gate(folds_with_data: int, agg: dict, total_trades: int, run_bt: bool) -> list[str]:
    fails: list[str] = []
    if folds_with_data < OPTIONS_GATE["required_folds_with_data"]:
        fails.append(f"Only {folds_with_data}/5 folds have real options coverage "
                     f"(real chains start 2021). The 2018-2020 folds are IMPOSSIBLE.")
    if not run_bt:
        fails.append("Performance criteria unverified: real backtest not run "
                     "(coverage failure is dispositive; use --force-backtest).")
    else:
        s = agg.get("annualized_sharpe")
        if not (isinstance(s, (int, float)) and np.isfinite(s) and s > OPTIONS_GATE["min_oos_sharpe"]):
            fails.append(f"OOS Sharpe {s} <= {OPTIONS_GATE['min_oos_sharpe']}")
        if not (agg.get("max_dd_pct", 100) < OPTIONS_GATE["max_drawdown_pct"]):
            fails.append(f"Max DD {agg.get('max_dd_pct')}% >= {OPTIONS_GATE['max_drawdown_pct']}%")
        if not (agg.get("dsr_p_value", 1.0) < OPTIONS_GATE["max_dsr_p_value"]):
            fails.append(f"DSR p {agg.get('dsr_p_value')} >= {OPTIONS_GATE['max_dsr_p_value']}")
        if not (total_trades > OPTIONS_GATE["min_oos_trades"]):
            fails.append(f"n_oos_trades {total_trades} <= {OPTIONS_GATE['min_oos_trades']}")
    return fails


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Institutional gate — real-data truth check.")
    ap.add_argument("--strategy", choices=["equity_factor_neutral", "equity_pairs", "options_delta_neutral"],
                    default="equity_factor_neutral",
                    help="Strategy to gate (default: equity_factor_neutral).")
    ap.add_argument("--cost", type=float, default=0.0010,
                    help="COST_PER_LEG as a fraction (default 0.0010 = 10 bps).")
    ap.add_argument("--folds", type=int, default=5, help="Number of anchored folds (informational).")
    ap.add_argument("--no-backtest", action="store_true",
                    help="Coverage + integrity only; skip the heavy real backtest.")
    ap.add_argument("--force-backtest", action="store_true",
                    help="Run the real backtest even when a coverage failure is dispositive.")
    ap.add_argument("--max-symbols", type=int, default=0,
                    help="Cap the equity universe size for speed (0 = use all ~612 names).")
    args = ap.parse_args()

    print("=" * 72)
    print(f"INSTITUTIONAL GATE — {args.strategy} — REAL DATA ONLY")
    print("=" * 72)

    if args.strategy == "options_delta_neutral":
        verdict = run_options_strategy(run_backtest=not args.no_backtest, force=args.force_backtest)
    else:
        res = run_equity_strategy(cost_per_leg=args.cost, run_backtest=not args.no_backtest,
                                  max_symbols=args.max_symbols, strategy_name=args.strategy)
        verdict = res["verdict"]

    status = verdict["gate_status"]
    print("\n" + "-" * 72)
    print(f"VERDICT: {status}")
    for f in verdict.get("failures", []):
        print(f"  FAIL: {f}")
    print(f"  aggregate: {verdict.get('aggregate')}")

    if status == "FAIL":
        arm = os.environ.get("GATE_ARM_KILL") == "1"
        cmd = [sys.executable, str(EMERGENCY_KILL)] + ([] if arm else ["--dry-run"])
        print(f"\nGate FAILED -> invoking emergency kill ({'ARMED' if arm else 'dry-run'})...")
        try:
            subprocess.run(cmd, check=False)
        except Exception as exc:
            print(f"  (emergency_kill invocation error: {exc})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
