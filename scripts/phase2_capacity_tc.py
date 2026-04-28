"""Tests 4-5: capacity sweep + TC sensitivity using audit data (500 bars × 10 symbols).

Fixes the data mismatch: engine now uses the SAME 500-bar data as the walk-forward
analysis, so engine Sharps are directly comparable to signal-based metrics.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.strategy_audit import (
    ALL_PARAMS, _STRATEGY_TYPE_MAP, StrategyConfig, StrategyType,
    make_data,
)
from quantcore.backtest.engine import BacktestEngine
from quantcore.backtest.financing import BorrowCostRegistry, FinancingConfig
from quantcore.backtest.impact_config import MarketImpactConfig
from quantcore.strategies.registry import get as get_strategy_class
from scripts.strategy_adapter import StrategyAdapter


def make_engine_data_from_audit(n_bars=500, n_symbols=10, seed=42):
    """Create engine-compatible OHLCV DataFrame from audit's make_data().

    Uses the same underlying random walk as the audit, but generates OHLCV
    values that satisfy engine constraints: high >= max(open, close), low <= min(open, close).
    Data is sorted (timestamp, symbol) as the engine expects.
    """
    rng = np.random.default_rng(seed)

    # Same correlated random walk as make_data()
    common = rng.normal(0.0003, 0.005, n_bars)
    specific = rng.normal(0.0005, 0.015, (n_bars, n_symbols))
    returns = specific + common[:, np.newaxis] * 0.3

    log_prices = np.zeros((n_bars, n_symbols))
    log_prices[0] = np.log(100.0)
    for i in range(1, n_bars):
        log_prices[i] = log_prices[i - 1] + returns[i]
    close = np.exp(log_prices)

    # Compute open from previous close, then derive high/low to satisfy OHLCV constraints
    open_ = np.zeros_like(close)
    open_[0] = close[0]
    for i in range(1, n_bars):
        open_[i] = close[i - 1] * (1 + rng.normal(0, 0.002))

    # high >= max(open, close), low <= min(open, close)
    hi_noise = np.abs(rng.normal(0, 0.006, (n_bars, n_symbols)))
    lo_noise = np.abs(rng.normal(0, 0.006, (n_bars, n_symbols)))
    high = np.maximum(open_, close) * (1 + hi_noise)
    low = np.minimum(open_, close) * (1 - lo_noise)
    low = np.maximum(low, 0.01)  # prevent negative

    volume = rng.integers(500_000, 5_000_000, (n_bars, n_symbols)).astype(float)

    # Convert to OHLCV DataFrame sorted (timestamp, symbol)
    rows = []
    for i in range(n_bars):
        ts = datetime(2020, 1, 1) + timedelta(days=i)
        for sym_i in range(n_symbols):
            rows.append({
                "symbol": f"SYM{sym_i}",
                "timestamp": ts,
                "open": round(float(open_[i, sym_i]), 2),
                "high": round(float(high[i, sym_i]), 2),
                "low": round(float(low[i, sym_i]), 2),
                "close": round(float(close[i, sym_i]), 2),
                "volume": int(volume[i, sym_i]),
            })
    return pd.DataFrame(rows)


def run_strategy(name, initial_cash, eta=0.0625, engine_data=None):
    """Run engine backtest for a strategy. Returns (engine_sharpe, impact_cost, borrow_cost, margin_cost)."""
    params = ALL_PARAMS.get(name, {})
    stype = _STRATEGY_TYPE_MAP.get(name, StrategyType.EQUITY)
    cfg = StrategyConfig(name=name, strategy_type=stype, params=params)
    cls = get_strategy_class(name)
    n_sym = 6  # Use 6 symbols (matching audit for multi-symbol strategies)
    if engine_data is None:
        engine_data = make_engine_data_from_audit(500, 10, 42)

    # Select only n_sym symbols for the strategy
    syms = [f"SYM{i}" for i in range(n_sym)]
    ed = engine_data[engine_data["symbol"].isin(syms)].copy().reset_index(drop=True)

    adapt = StrategyAdapter(cls, cfg, n_symbols=n_sym)
    engine = BacktestEngine(
        strategy=adapt,
        initial_cash=float(initial_cash),
        sizing_pct=0.30,
        impact_config=MarketImpactConfig(sigma=0.02, adv=500_000, eta=eta),
        financing_config=FinancingConfig(
            borrow_cost_registry=BorrowCostRegistry(default_rate=0.05),
            margin_rate=0.08,
        ),
    )
    engine.run(ed)
    rep = engine.summary()

    # Also compute Sharpe from engine equity curve for verification
    equity = np.array([v for _, v in engine.portfolio.equity_curve])
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    full_sr = np.mean(rets) / np.maximum(np.std(rets, ddof=1), 1e-10) * np.sqrt(252)

    return rep.sharpe, full_sr, rep.total_market_impact_cost, rep.total_borrow_cost, rep.total_margin_interest


# Pre-generate data once (10 symbols, 500 bars)
print("Generating 500-bar x 10-symbol engine data (matching audit)...")
engine_data = make_engine_data_from_audit(500, 10, 42)
print(f"  Data shape: {len(engine_data)} rows")
print(f"  Symbols: {engine_data['symbol'].nunique()}")
print(f"  Date range: {engine_data['timestamp'].min()} to {engine_data['timestamp'].max()}")
print(f"  Close range: ${engine_data['close'].min():.2f} - ${engine_data['close'].max():.2f}")

# Also get walk-forward baseline for comparison
data_audit = make_data()
from scripts.strategy_audit import anchored_walk_forward, analyze_strategy
folds = anchored_walk_forward(500, initial_train=200, test_size=50, embargo_pct=0.02)

for name in ("macro_risk_parity", "macro_cta_trend"):
    label = "2A" if name == "macro_risk_parity" else "2B"

    # Walk-forward baseline
    wf_result = analyze_strategy(name, data_audit, folds, [])
    wf_sr = wf_result.get("oos_sharpe", 0.0)
    print(f"\n{'='*72}")
    print(f"  PHASE {label}: {name}")
    print(f"  Walk-forward OOS Sharpe (baseline): {wf_sr:+.4f}")
    print(f"{'='*72}")

    # ── Test 4: Capacity sweep ────────────────────────────────────────────
    print(f"\n  ── TEST 4: CAPACITY SWEEP ──")
    capital_levels = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]

    header = f"  {'Capital':>12s} {'Eng SR':>8s} {'Imp $':>10s} {'Borrow $':>10s} {'Margin $':>10s} {'vs WF':>8s}"
    print(header)
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    results_cap = []
    for cap in capital_levels:
        raw_sr, full_sr, imp, brw, mgn = run_strategy(name, cap, engine_data=engine_data)
        results_cap.append((cap, full_sr))
        ratio = full_sr / wf_sr if abs(wf_sr) > 0.1 else 0.0
        print(f"  ${cap:>8,} {full_sr:>+8.3f} ${imp:>8,.0f} ${brw:>8,.0f} "
              f"${mgn:>8,.0f} {ratio:>+7.1%}")

    if results_cap:
        peak_sr = max(r[1] for r in results_cap)
        ceiling_found = False
        for cap, sr in results_cap:
            if sr < peak_sr * 0.8:
                print(f"\n  → Capacity ceiling: ${cap:,} "
                      f"(SR {sr:.3f} < 80% of peak {peak_sr:.3f})")
                ceiling_found = True
                break
        if not ceiling_found:
            min_sr = min(r[1] for r in results_cap)
            print(f"\n  → No capacity ceiling up to $5M "
                  f"(min SR {min_sr:.3f} within 80% of peak {peak_sr:.3f})")

    # ── Test 5: TC sensitivity ────────────────────────────────────────────
    print(f"\n  ── TEST 5: TC SENSITIVITY ──")
    tc_mult = [0.5, 1.0, 1.5, 2.0, 3.0]
    base_eta = 0.0625

    print(f"  {'TC mult':>8s} {'Eng SR':>8s} {'Chg':>8s} {'Impact $':>12s} {'vs WF':>8s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")

    tc_results = []
    baseline_sr = None
    for mult in tc_mult:
        raw_sr, full_sr, imp, brw, mgn = run_strategy(name, 1_000_000, eta=base_eta * mult, engine_data=engine_data)
        tc_results.append((mult, full_sr))
        if baseline_sr is None:
            baseline_sr = full_sr
        chg = full_sr - baseline_sr
        ratio = full_sr / wf_sr if abs(wf_sr) > 0.1 else 0.0
        label_tc = "base" if mult == 1.0 else ""
        print(f"  {mult:>7.1f}x {full_sr:>+8.3f} {chg:>+8.3f} ${imp:>10,.0f} {ratio:>+7.1%}")

    # Breakeven analysis
    for mult, sr in tc_results:
        if sr <= 0:
            print(f"\n  → Breakeven TC at {mult:.1f}x base (Sharpe = {sr:.3f})")
            break
    else:
        print(f"\n  → No breakeven up to 3x base TC (all Sharps positive)")

    # Summary: how does engine SR compare to WF baseline?
    print(f"\n  ── SUMMARY ──")
    print(f"  Walk-forward OOS Sharpe: {wf_sr:+.4f}")
    _, best_full_sr, _, _, _ = run_strategy(name, 1_000_000, engine_data=engine_data)
    print(f"  Engine Sharpe ($1M, 1x TC): {best_full_sr:+.4f}")
    print(f"  Ratio (Engine/WF): {best_full_sr/wf_sr:.1%}" if abs(wf_sr) > 0.1 else "  N/A (WF near zero)")

print(f"\n{'='*72}")
print("  ALL ENGINE TESTS COMPLETE (using 500-bar audit data)")
print(f"{'='*72}")
