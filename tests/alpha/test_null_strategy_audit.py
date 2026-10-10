"""
Phase 7b — Null-Strategy Sanity Audit
Validates that BacktestEngine + NullStrategy produces Sharpe ≈ 0.0 ± 0.1
and DSR p ≈ 0.5. If Sharpe > 0.2 or DSR p < 0.35, the engine has a
systematic bias (e.g. cost asymmetry, look-ahead, survivorship).

Acceptance:
  PASS if |Sharpe| < 0.2  AND  0.1 < DSR p < 0.9
  FAIL otherwise
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- project imports ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantcore.backtest.engine import BacktestEngine
from quantcore.backtest.null_strategy import NullStrategy
from quantcore.strategies.base import StrategyConfig, StrategyType


# ── synthetic OHLCV generator (same logic as verify_honest_foundation) ──
def make_synthetic(seed: int = 42, days: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days, freq="B")
    n = len(dates)
    symbol = np.array(["AAPL"] * n)
    # random walk: cumulative sum of daily returns
    daily_ret = rng.normal(loc=0.0002, scale=0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low  = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 10_000_000, size=n)

    df = pd.DataFrame({
        "symbol": symbol,
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


# ── Sharpe / DSR helpers ──
def annualised_sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    if len(returns) < 2 or np.std(returns, ddof=1) == 0:
        return 0.0
    excess = returns - rf / periods
    return float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(periods))


def dsr_p_value(returns: np.ndarray, n_sims: int = 100_000) -> float:
    """Two-tailed noise-distribution p-value.

    Generates n_sims standard-normal return series of the same length as
    *returns*, computes their Sharpe ratios, and returns the fraction of
    noise Sharpes whose absolute value exceeds |obs_sharpe|.

    A p-value near 0 means the observed Sharpe is extreme relative to pure
    noise (engine bias); near 1 means it is indistinguishable from noise
    (what we want for a null strategy).
    """
    T = len(returns)
    if T < 2:
        return 1.0
    obs_sharpe = annualised_sharpe(returns)
    abs_obs = abs(obs_sharpe)

    rng_sim = np.random.default_rng(0)
    noise_sharpes = np.empty(n_sims)
    for i in range(n_sims):
        sim = rng_sim.standard_normal(T)
        noise_sharpes[i] = annualised_sharpe(sim)

    # two-tailed: fraction of noise Sharpes with |s| >= |obs|
    p = float(np.mean(np.abs(noise_sharpes) >= abs_obs))
    return p


# ── acceptance gates ──
SHARPE_TOL = 0.20   # |Sharpe| < 0.20
DSR_LOW    = 0.10   # DSR p > 0.10
DSR_HIGH   = 0.90   # DSR p < 0.90


def run_audit(n_days: int = 2000, n_runs: int = 10, seed: int = 42) -> dict:
    results = []
    equity_curves = []  # store equity DataFrames for DSR computation
    for i in range(n_runs):
        df = make_synthetic(seed=seed + i * 7, days=n_days)
        strat = NullStrategy(config=StrategyConfig(name="null", strategy_type=StrategyType.EQUITY, params={"entry_prob": 0.05, "exit_prob": 0.10, "max_risk_pct": 0.02, "long_fraction": 0.50}))
        engine = BacktestEngine(strategy=strat, initial_cash=1_000_000.0, slippage_bps=5.0, commission_per_share=0.005)
        equity_df = engine.run(df)
        report = engine.summary()
        sharpe = report.sharpe
        equity_curves.append(equity_df)
        results.append({"run": i, "sharpe": sharpe, "total_return": report.total_return, "n_days": len(equity_df)})

    sharpes = np.array([r["sharpe"] for r in results])
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1))
    min_sharpe = float(np.min(sharpes))
    max_sharpe = float(np.max(sharpes))

    # compute DSR p on pooled daily returns from equity curves
    pooled_returns = []
    for eq_df in equity_curves:
        equity = eq_df["equity"].values
        if len(equity) > 1:
            daily_ret = np.diff(equity) / equity[:-1]
            pooled_returns.extend(daily_ret.tolist())
    pooled_returns = np.array(pooled_returns)
    dsr_p = dsr_p_value(pooled_returns)

    # acceptance
    pass_sharpe = abs(mean_sharpe) < SHARPE_TOL
    pass_dsr = DSR_LOW < dsr_p < DSR_HIGH
    passed = pass_sharpe and pass_dsr

    return {
        "n_runs": n_runs,
        "mean_sharpe": round(mean_sharpe, 6),
        "std_sharpe": round(std_sharpe, 6),
        "min_sharpe": round(min_sharpe, 6),
        "max_sharpe": round(max_sharpe, 6),
        "dsr_p": round(dsr_p, 4),
        "pass_sharpe": pass_sharpe,
        "pass_dsr": pass_dsr,
        "passed": passed,
        "details": results,
    }


def test_null_strategy_no_systematic_bias():
    """Phase 7b: BacktestEngine + NullStrategy must produce Sharpe ≈ 0.0 ± 0.2."""
    res = run_audit(n_days=2000, n_runs=5, seed=42)
    print(f"\n  Mean Sharpe: {res['mean_sharpe']:.4f} (±{res['std_sharpe']:.4f})")
    print(f"  Range: [{res['min_sharpe']:.4f}, {res['max_sharpe']:.4f}]")
    print(f"  DSR p-value: {res['dsr_p']:.4f}")
    assert res["pass_sharpe"], f"|Sharpe|={abs(res['mean_sharpe']):.4f} > {SHARPE_TOL} — engine has systematic bias"
    assert res["pass_dsr"], f"DSR p={res['dsr_p']:.4f} outside ({DSR_LOW}, {DSR_HIGH}) — unreliable Sharpe"


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 7b — Null Strategy Sanity Audit")
    print("=" * 60)
    res = run_audit(n_days=2000, n_runs=5, seed=42)
    print(f"  Runs:           {res['n_runs']}")
    print(f"  Mean Sharpe:    {res['mean_sharpe']:.4f}  (±{res['std_sharpe']:.4f})")
    print(f"  Range:          [{res['min_sharpe']:.4f}, {res['max_sharpe']:.4f}]")
    print(f"  DSR p-value:    {res['dsr_p']:.4f}")
    print(f"  |Sharpe| < {SHARPE_TOL}: {'PASS' if res['pass_sharpe'] else 'FAIL'}")
    print(f"  DSR p in ({DSR_LOW}, {DSR_HIGH}): {'PASS' if res['pass_dsr'] else 'FAIL'}")
    print("-" * 60)
    if res["passed"]:
        print("  RESULT: PASS — Engine shows no systematic bias")
    else:
        print("  RESULT: FAIL — Engine has systematic bias")
    print("=" * 60)
    sys.exit(0 if res["passed"] else 1)
