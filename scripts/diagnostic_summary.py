"""
Diagnostic Summary: Consolidate all 6 diagnostic findings.

Aggregates results from diagnostics 1-6 into a root cause analysis
for Phase A failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json


def main():
    print("=" * 72)
    print("  PHASE A DIAGNOSTIC INVESTIGATION: ROOT CAUSE ANALYSIS")
    print("=" * 72)

    diagnostics = {
        "D1": {
            "name": "Walk-Forward Verification",
            "status": "PASS",
            "verdict": "Implementation is clean. Determinism verified, WF vs regime delta 0.017, no extreme fold outliers.",
            "details": [
                "Determinism: identical results on repeat runs",
                "WF vs regime delta: 0.017 (threshold: 0.20 for PASS)",
                "Per-fold distribution: no extreme outliers",
                "Conclusion: Implementation bugs are NOT the cause of Phase A failures.",
            ],
        },
        "D2": {
            "name": "Transaction Cost Impact",
            "status": "WARN",
            "verdict": "TC impact is significant (delta 0.5-0.9 SR) but NOT the root cause. Even at TC=0, CTA Full SR=-0.147.",
            "details": [
                "CTA Full: SR -0.147 (0bps) -> -0.728 (10bps) -> -2.678 (50bps)",
                "CTA COVID: SR +0.991 (0bps) -> +0.359 (10bps) -> -1.921 (50bps)",
                "RP Full: SR +0.107 (0bps) -> -0.822 (10bps) -> -2.389 (50bps)",
                "Costs degrade performance meaningfully but strategy fails even without them.",
            ],
        },
        "D3": {
            "name": "Signal Generation Comparison",
            "status": "PASS",
            "verdict": "Signal generation works correctly. 300 signals on real vs 263 on synthetic over same period.",
            "details": [
                "Real: 300 signals across 60 bar indices, all 6 symbols hit",
                "Synthetic: 263 signals across 64 bar indices, all 6 symbols hit",
                "Real long bias 70.7% (correct for 2020-2021 bull market recovery)",
                "Data is clean: 0% NaN, 0% zeros, 0% inf in close prices",
                "Data pipeline is NOT distorting signals.",
            ],
        },
        "D4": {
            "name": "Data Quality Verification",
            "status": "PASS",
            "verdict": "Data is fundamentally clean across all 3 datasets (CTA Full, CTA COVID, RP Full).",
            "details": [
                "No absurd returns (>|20%|) across 31680 rows",
                "No excessive gaps (>5 business days)",
                "No stale prices (>3 identical closes)",
                "Minor OHLC inconsistencies: DBC low>close (7x total), TLT high<close (4x total) — rounding only",
                "Volume: zero anomalies",
                "Correlation structure: normal (SPY/EFA +0.87, TLT/IEF +0.91, SPY/TLT -0.18)",
                "Return distributions: all within normal ranges",
            ],
        },
        "D5": {
            "name": "Win Rate & Return Distribution",
            "status": "CRITICAL_INSIGHT",
            "verdict": "CRITICAL: ~80% of OOS bars have ZERO return. On non-zero bars, win rate is 53-63% (decent). Negative Sharpe driven by extreme kurtosis (52.9) — a few catastrophic losses dominate.",
            "details": [
                "CTA Full: 1211/1519 bars = 79.7% zero return",
                "CTA COVID: 413/503 bars = 82.1% zero return",
                "Non-zero win rate: Full=53.2%, COVID=63.3%",
                "Non-zero Sharpe: Full=-0.02, COVID=+0.15",
                "Excess kurtosis: Full=52.9, COVID=24.1",
                "Gain/loss ratio: Full=0.94, COVID=1.48",
                "Mean return: Full=-0.000026 daily, COVID=+0.000113 daily",
                "Win rate calculation verified: matches reported value exactly.",
            ],
        },
        "D6": {
            "name": "Engine Determinism & Sanity",
            "status": "PASS",
            "verdict": "Engine is sound. Fully deterministic, handles edge cases gracefully.",
            "details": [
                "3 identical runs: SR=+0.954574 each (fully deterministic)",
                "Single-symbol universe: works (SR=+0.213, 6 folds, 294 bars)",
                "Minimum bars: ~100 for walk-forward, ~50 for regime",
                "Empty data: graceful error handling",
                "NaN in data: recovers via forward-fill",
            ],
        },
    }

    # ── Print individual results ──
    for d_id, d in diagnostics.items():
        status_symbol = {"PASS": "PASS", "WARN": "WARN", "CRITICAL_INSIGHT": "INFO"}[d["status"]]
        print(f"\n{'=' * 72}")
        print(f"  [{status_symbol}] {d_id}: {d['name']}")
        print(f"{'=' * 72}")
        print(f"  {d['verdict']}")
        for detail in d["details"]:
            print(f"    * {detail}")

    # ── Root cause analysis ──
    print("\n" + "=" * 72)
    print("  ROOT CAUSE ANALYSIS")
    print("=" * 72)

    print("""
  1. PRIMARY FINDING: The strategy signal is structurally weak.

     The CTA trend-following strategy on real data (2018-2024) produces a
     slightly positive edge on non-zero bars (53.2% win rate) but:
     - 80% of bars have zero return (due to 5-day rebalance frequency)
     - The non-zero return stream has extreme kurtosis (52.9)
     - A handful of catastrophic losses wipe out months of small gains

     This is CLASSIC trend-following failure in a non-trending,
     reversal-prone environment. The strategy relies on persistent trends;
     when markets whipsaw, it gets caught on the wrong side with full
     position size.

  2. SECONDARY FINDING: Transaction costs amplify the damage.

     Even at zero TC, the strategy is negative for the full period (-0.147).
     At realistic 10bps: -0.728. Costs are a significant headwind for a
     strategy that rebalances every 5 days across 6 assets, but they are
     NOT the root cause.

  3. REJECTED HYPOTHESES:
     - Implementation bugs:           D1+D6 confirmed clean
     - Data pipeline / NaN:           D3+D4 confirmed clean
     - Win rate calculation bug:      D5 verified correct
     - Synthetic overfitting:         D3 shows real/synthetic signals match

  4. STRATEGY-SPECIFIC OBSERVATIONS:
     - macro_cta_trend: FAILS full period (-0.147 OOS SR), PASSES COVID
       (+0.359 at 10bps TC). Shows the strategy CAN work in strong trends
       but fails in mixed regimes.
     - macro_risk_parity: MARGINAL FAILURE (+0.107 OOS SR at TC=0, vs
       threshold +0.30). Barely positive at zero cost, negative at 10bps.
       Risk parity is more execution-sensitive than trend following.

  5. RECOMMENDATIONS:
     a) Accept real-data performance as the honest baseline. The synthetic
        backtest (+1.79 OOS SR) was overfit to GBM-like regimes.
     b) For Phase B: consider reducing rebalance frequency (lower turnover),
        adding a regime filter (skip low-trend regimes), or implementing
        adaptive position sizing (reduce size during whipsaw).
     c) The COVID regime result (+0.359 at 10bps) validates that the
        strategy has genuine edge in trending markets. The issue is
        capital preservation during non-trending periods.
""")

    # ── Final verdict ──
    print("=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)

    verdict = """
  Phase A -- REAL-DATA VALIDATION: FAIL

  macro_cta_trend:  CATASTROPHIC FAILURE on full period (-0.147 OOS SR)
  macro_risk_parity: MARGINAL FAILURE (+0.107 vs +0.30 threshold)

  Root cause: Strategy signal is structurally weak due to:
  1. Low signal density (80% zero-return bars)
  2. Extreme tail risk (kurtosis 52.9)
  3. Transaction cost drag (amplifies the negative)
  4. Non-trending market regimes (2018-2024)

  The strategy is NOT viable for deployment without modification. Continue
  to Phase B for strategy enhancement, or classify as MONITOR only.
"""
    print(verdict)

    # ── Save to JSON ──
    summary = {
        "phase": "A_DIAGNOSTIC",
        "diagnostics": {
            d_id: {
                "name": d["name"],
                "status": d["status"],
                "verdict": d["verdict"],
            }
            for d_id, d in diagnostics.items()
        },
        "root_cause": {
            "primary": "Strategy signal is structurally weak — 80% zero bars, kurtosis 52.9, negative mean return",
            "secondary": "Transaction costs amplify the damage but are not the root cause",
            "rejected": ["Implementation bugs", "Data pipeline issues", "Win rate calculation errors", "Synthetic overfitting"],
        },
        "verdict": "FAIL",
        "next_steps": "Proceed to Phase B for strategy enhancement or classify as MONITOR only",
    }

    output_path = ROOT / "results" / "diagnostic_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Diagnostic summary saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
