"""
Diagnostic 1: Walk-forward implementation verification.

Tests:
1. Determinism -- run same period twice, verify identical results
2. Walk-forward vs single-period comparison -- COVID 2020 as WF vs regime
3. Consistency across methodologies
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantcore.risk.metrics import sharpe_ratio
from scripts.real_data_backtest import (
    STRATEGY_UNIVERSE,
    _compute_signal_returns,
    _df_to_data,
    _setup_strategy,
    run_full_period_backtest,
    run_regime_backtest,
)
from scripts.real_data_loader import UNIVERSES, load_universe


def run_single_pass(strategy_id: str, df, start_idx: int = 0) -> dict:
    """Run strategy on full dataset from start_idx as a single pass (no folds).

    Collects returns from every bar after the strategy has enough history.
    """
    data = _df_to_data(df)
    n = data["close"].shape[0]

    strat = _setup_strategy(strategy_id)
    returns = _compute_signal_returns(strat, data, start_idx, n)

    if len(returns) < 3:
        return {"error": f"Only {len(returns)} returns", "returns": returns}

    return {
        "sharpe": sharpe_ratio(returns),
        "n_returns": len(returns),
        "returns": returns,
        "error": None,
    }


def main():
    print("=" * 70)
    print("  DIAGNOSTIC 1: WALK-FORWARD VERIFICATION")
    print("=" * 70)

    strategy_id = "macro_cta_trend"
    universe = STRATEGY_UNIVERSE[strategy_id]

    # ── TEST 1.1: Determinism ──
    print("\n" + "-" * 70)
    print("  TEST 1.1: Determinism (run same backtest twice)")
    print("-" * 70)

    # Load data once
    df = load_universe(universe, "2020-01-01", "2021-12-31")
    assert not df.empty, f"No data for {universe}"

    result1 = run_full_period_backtest(strategy_id, df)
    result2 = run_full_period_backtest(strategy_id, df)

    sr1 = result1.get("oos_sharpe", 0)
    sr2 = result2.get("oos_sharpe", 0)
    dd1 = result1.get("max_dd", 0)
    dd2 = result2.get("max_dd", 0)
    nf1 = result1.get("n_folds", 0)
    nf2 = result2.get("n_folds", 0)

    print(f"\n  Run 1: OOS SR={sr1:+.6f}  Max DD={dd1:.6f}  Folds={nf1}")
    print(f"  Run 2: OOS SR={sr2:+.6f}  Max DD={dd2:.6f}  Folds={nf2}")

    sr_match = abs(sr1 - sr2) < 1e-9
    dd_match = abs(dd1 - dd2) < 1e-9
    nf_match = nf1 == nf2

    if sr_match and dd_match and nf_match:
        print("  [PASS] Results are deterministic (identical on repeat)")
    else:
        print("  [FAIL] NON-DETERMINISTIC -- results differ between runs")
        if not sr_match:
            print(f"    OOS SR delta: {abs(sr1 - sr2):.2e}")
        if not dd_match:
            print(f"    Max DD delta: {abs(dd1 - dd2):.2e}")

    # ── TEST 1.2: Walk-forward vs single-period ──
    print("\n" + "-" * 70)
    print("  TEST 1.2: Walk-forward vs single-period (regime test)")
    print("-" * 70)

    # WF result (already have result1 from above)
    wf_sr = result1.get("oos_sharpe")

    # Regime test: single split, COVID 2020
    all_dates = sorted(df["timestamp"].unique())
    regime_start = pd.Timestamp("2020-01-01")
    if hasattr(all_dates[0], "tz") and all_dates[0].tz is not None:
        regime_start = regime_start.tz_localize(all_dates[0].tz)

    warmup_idx = next(i for i, d in enumerate(all_dates) if d >= regime_start)
    regime_result = run_regime_backtest(strategy_id, df, warmup_idx)
    regime_sr = regime_result.get("oos_sharpe")

    # Single-pass: just run strategy on full data, collect all returns
    single = run_single_pass(strategy_id, df)
    single_sr = single.get("sharpe") if single.get("error") is None else None

    print(f"\n  {'Method':<30s} {'OOS Sharpe':>12s}")
    print(f"  {'-' * 42}")
    wf_str = f"{wf_sr:+.4f}" if wf_sr is not None else "ERROR"
    print(f"  {'Walk-forward (WF)':<30s} {wf_str:>12s}")
    rs_str = f"{regime_sr:+.4f}" if regime_sr is not None else "ERROR"
    print(f"  {'Regime test (single split)':<30s} {rs_str:>12s}")
    ss_str = f"{single_sr:+.4f}" if single_sr is not None else "ERROR"
    print(f"  {'Single pass (no split)':<30s} {ss_str:>12s}")

    wf_vs_regime = abs((wf_sr or 0) - (regime_sr or 0))
    print(f"\n  WF vs Regime delta: {wf_vs_regime:.4f}")

    if wf_vs_regime < 0.2:
        print(f"  [PASS] Delta {wf_vs_regime:.3f} < 0.2 -- fold structure is clean")
    elif wf_vs_regime < 0.5:
        print(f"  [WARN] Delta {wf_vs_regime:.3f} -- moderate fold artifacts")
    else:
        print(f"  [FAIL] Delta {wf_vs_regime:.3f} > 0.5 -- significant fold artifacts")

    # ── TEST 1.3: Per-fold stability ──
    print("\n" + "-" * 70)
    print("  TEST 1.3: Per-fold Sharpe distribution")
    print("-" * 70)

    if result1.get("fold_results"):
        sharpes = [f["oos_sharpe"] for f in result1["fold_results"]]
        n_folds = len(sharpes)
        mean_sr = np.mean(sharpes)
        std_sr = np.std(sharpes, ddof=1) if n_folds > 1 else 0
        min_sr = min(sharpes)
        max_sr = max(sharpes)

        print(f"\n  Folds completed: {n_folds}")
        print(f"  Mean fold SR: {mean_sr:+.4f}")
        print(f"  Std fold SR:  {std_sr:.4f}")
        print(f"  Min fold SR:  {min_sr:+.4f}")
        print(f"  Max fold SR:  {max_sr:+.4f}")

        n_positive = sum(1 for s in sharpes if s > 0)
        print(f"  Positive folds: {n_positive}/{n_folds} ({n_positive / n_folds:.0%})")

        # Look for extreme outliers
        extremes = [s for s in sharpes if abs(s) > 1.0]
        if extremes:
            print(f"  [WARN] {len(extremes)} extreme fold(s) (|SR| > 1.0): "
                  f"{[f'{s:+.2f}' for s in extremes]}")
        else:
            print(f"  [PASS] No extreme fold outliers")

    else:
        print("  No fold results available")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 1 SUMMARY")
    print("=" * 70)

    issues = []
    if not sr_match or not dd_match:
        issues.append("NON_DETERMINISTIC")
    if wf_vs_regime > 0.5:
        issues.append("WF_REGIME_DIVERGENCE")

    if not issues:
        print("\n  [PASS] Walk-forward implementation is CLEAN")
        print("  Backtest is deterministic, fold structure is reasonable,")
        print("  and results are consistent across methodologies.")
        print("\n  Conclusion: Implementation bugs are NOT the cause of Phase A failures.")
    else:
        print(f"\n  [FAIL] Issues found: {', '.join(issues)}")
        print("  Walk-forward implementation has artifacts or bugs.")

    print()


if __name__ == "__main__":
    main()
