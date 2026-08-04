"""Phase 3 Research Batch: Alpha Discovery Notebook (script form).

Run this script to evaluate all 5 candidate signals against synthetic data.
When real market data is available, replace `_build_synthetic_dataset()` 
with a real data loader adhering to research/datasets/DATASET_SPEC.md.

Usage:
    python research/signal_lab/run_phase3_batch.py
"""
from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from research.signal_lab.engine import (
    DatasetValidator,
    SignalExtractor,
    evaluate_signal,
    run_research_batch,
)


# ---------------------------------------------------------------------------
# Synthetic Dataset Builder (Replace with real data loader)
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Build a minimal synthetic dataset for engine testing.

    IMPORTANT: Real alpha research requires real adjusted OHLCV data.
    This is only for verifying that the evaluation pipeline runs correctly.
    Results on synthetic data have NO research validity.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n_days)

    # Simulate a drifting price with realistic autocorrelation
    daily_ret = rng.normal(0.0004, 0.012, n_days)
    close = 100.0 * np.cumprod(1 + daily_ret)
    volume = rng.lognormal(17, 0.4, n_days)

    # Benchmark (SPY-like)
    bm_ret = daily_ret + rng.normal(0, 0.002, n_days)

    # True forward returns (T+1): use actual next day's return (no look-ahead)
    fwd_1 = np.roll(daily_ret, -1)
    fwd_1[-1] = np.nan
    fwd_5 = np.array([
        np.prod(1 + daily_ret[i+1:i+6]) - 1 if i+5 < n_days else np.nan
        for i in range(n_days)
    ])

    df = pd.DataFrame({
        "close": close,
        "volume": volume,
        "returns": daily_ret,
        "forward_returns_1": fwd_1,
        "forward_returns_5": fwd_5,
        "benchmark_returns": bm_ret,
    }, index=dates)

    return df


# ---------------------------------------------------------------------------
# Main Research Batch
# ---------------------------------------------------------------------------

def run():
    print("\n" + "="*70)
    print("  PHASE 3 ALPHA DISCOVERY — SIGNAL EVALUATION BATCH")
    print("  WARNING: Using SYNTHETIC data. Replace with real market data.")
    print("="*70 + "\n")

    # Step 1: Build / load dataset
    df = _build_synthetic_dataset()
    print(f"Dataset loaded: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}\n")

    # Step 2: Validate dataset
    val_report = DatasetValidator.validate(df)
    val_report.display()

    if not val_report.passed:
        print("\n[STOP] Dataset validation FAILED. Fix issues before continuing.")
        return

    # Step 3: Extract all candidate signals
    print("\n--- Extracting Signals ---")

    signals: dict[str, pd.Series] = {}

    # H001: Cross-Sectional Momentum
    try:
        signals["CS_Momentum"] = SignalExtractor.cross_sectional_momentum(df)
        print("  [OK] CS_Momentum")
    except Exception as e:
        print(f"  [FAIL] CS_Momentum: {e}")

    # H002: Time-Series Momentum
    try:
        signals["TS_Momentum"] = SignalExtractor.time_series_momentum(df)
        print("  [OK] TS_Momentum")
    except Exception as e:
        print(f"  [FAIL] TS_Momentum: {e}")

    # H003: Short-Term Mean Reversion
    try:
        signals["Mean_Reversion"] = SignalExtractor.short_term_mean_reversion(df)
        print("  [OK] Mean_Reversion")
    except Exception as e:
        print(f"  [FAIL] Mean_Reversion: {e}")

    # H004: VRP (requires IV data — will produce NaN with synthetic data)
    try:
        with warnings.catch_warnings(record=True):
            signals["VRP"] = SignalExtractor.volatility_risk_premium(df)
            print("  [WARN] VRP: No IV data — NaN signal. Needs options data.")
    except Exception as e:
        print(f"  [FAIL] VRP: {e}")

    # H005: Earnings Drift (requires earnings data)
    try:
        with warnings.catch_warnings(record=True):
            signals["Earnings_Drift"] = SignalExtractor.earnings_drift(df)
            print("  [WARN] Earnings_Drift: No earnings data — NaN signal.")
    except Exception as e:
        print(f"  [FAIL] Earnings_Drift: {e}")

    # Step 4: Run evaluation batch
    print("\n--- Running Signal Evaluation ---")

    fwd_returns = df["forward_returns_1"]
    bm_returns = df["benchmark_returns"]

    leaderboard = run_research_batch(signals, fwd_returns, bm_returns)

    print("\n=== RESEARCH LEADERBOARD ===")
    print(leaderboard.to_string(index=False))

    # Step 5: Individual signal reports for promotable signals
    print("\n--- Individual Signal Reports ---")
    for name, signal in signals.items():
        report = evaluate_signal(signal, fwd_returns, bm_returns, signal_name=name)
        report.display()

    # Step 6: Gate decision
    promotable = [name for name, s in signals.items()
                  if evaluate_signal(s, fwd_returns, bm_returns, name).recommendation == "PROMOTE"]

    print("\n" + "="*70)
    print(f"  PHASE 3 RESULT: {len(promotable)} signal(s) passed Gate 1.")
    if promotable:
        print(f"  Signals to advance: {', '.join(promotable)}")
        print("  Next step: Proceed to Gate 2 (Purged Cross-Validation).")
    else:
        print("  No signals passed Gate 1.")
        print("  Action: Reject this batch. Document failure reasons. Generate new hypotheses.")
    print("="*70 + "\n")


if __name__ == "__main__":
    run()
