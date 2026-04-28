"""Re-run Tests 4-5 (capacity + TC sensitivity) for both strategies using correct class-based adapter."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.strategy_audit import (
    ALL_PARAMS, _STRATEGY_TYPE_MAP, StrategyConfig, StrategyType,
)
from quantcore.backtest.engine import BacktestEngine
from quantcore.backtest.financing import BorrowCostRegistry, FinancingConfig
from quantcore.backtest.impact_config import MarketImpactConfig
from quantcore.strategies.registry import get as get_strategy_class
from scripts.strategy_adapter import StrategyAdapter


def make_engine_data(n_bars=252, n_symbols=6):
    rng = np.random.default_rng(142)
    prices = [100.0] * n_symbols
    rows = []
    for i in range(n_bars):
        ts = datetime(2020, 1, 1) + timedelta(days=i)
        for sym_i in range(n_symbols):
            ret = rng.normal(0.0005, 0.015)
            prices[sym_i] = max(prices[sym_i] * (1 + ret), 10.0)
            p = prices[sym_i]
            o = p * (1 + rng.uniform(-0.003, 0.003))
            c = p * (1 + rng.uniform(-0.003, 0.003))
            o, c = max(o, 1.0), max(c, 1.0)
            h = max(o, c) * (1 + abs(rng.uniform(0.001, 0.008)))
            l = min(o, c) * (1 - abs(rng.uniform(0.001, 0.008)))
            rows.append({"symbol": f"SYM{sym_i}", "timestamp": ts,
                         "open": round(o, 2), "high": round(h, 2),
                         "low": round(l, 2), "close": round(c, 2),
                         "volume": 2_000_000})
    return pd.DataFrame(rows)


def run_strategy(name, initial_cash, eta=0.0625):
    """Run engine backtest for a strategy. Returns (sharpe, impact_cost, borrow_cost)."""
    params = ALL_PARAMS.get(name, {})
    stype = _STRATEGY_TYPE_MAP.get(name, StrategyType.EQUITY)
    cfg = StrategyConfig(name=name, strategy_type=stype, params=params)
    cls = get_strategy_class(name)
    n_sym = 6 if name in ("macro_cta_trend", "macro_risk_parity") else 1
    engine_data = make_engine_data(252, n_sym)

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
    engine.run(engine_data)
    rep = engine.summary()
    equity = np.array([v for _, v in engine.portfolio.equity_curve])
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    full_sr = np.mean(rets) / np.maximum(np.std(rets, ddof=1), 1e-10) * np.sqrt(252)
    return rep.sharpe, full_sr, rep.total_market_impact_cost, rep.total_borrow_cost, rep.total_margin_interest


for name in ("macro_risk_parity", "macro_cta_trend"):
    label = "2A" if name == "macro_risk_parity" else "2B"

    # ── Test 4: Capacity sweep ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PHASE {label}: {name} — TEST 4: CAPACITY SWEEP")
    print(f"{'='*72}")

    capital_levels = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    print(f"\n  {'Capital':>12s} {'Raw SR':>8s} {'Full SR':>8s} {'Impact $':>10s} {'Borrow $':>10s} {'Margin $':>10s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    results = []
    for cap in capital_levels:
        raw_sr, full_sr, imp, brw, mgn = run_strategy(name, cap)
        results.append((cap, full_sr))
        print(f"  ${cap:>8,} {raw_sr:>+8.3f} {full_sr:>+8.3f} ${imp:>8,.0f} ${brw:>8,.0f} ${mgn:>8,.0f}")

    if results:
        peak_sr = max(r[1] for r in results)
        ceiling_found = False
        for cap, sr in results:
            if sr < peak_sr * 0.8:
                print(f"\n  → Capacity ceiling: ${cap:,} (SR {sr:.3f} < {peak_sr*0.8:.3f} = 80% of peak {peak_sr:.3f})")
                ceiling_found = True
                break
        if not ceiling_found:
            print(f"\n  → No capacity ceiling up to $5M (min SR {min(r[1] for r in results):.3f} within 80% of peak {peak_sr:.3f})")

    # ── Test 5: TC sensitivity ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PHASE {label}: {name} — TEST 5: TC SENSITIVITY")
    print(f"{'='*72}")

    tc_mult = [0.5, 1.0, 1.5, 2.0, 3.0]
    base_eta = 0.0625
    print(f"\n  {'TC mult':>8s} {'Raw SR':>8s} {'Full SR':>8s} {'Chg':>8s} {'Impact $':>12s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    tc_results = []
    baseline_sr = None
    for mult in tc_mult:
        raw_sr, full_sr, imp, brw, mgn = run_strategy(name, 1_000_000, eta=base_eta * mult)
        tc_results.append((mult, full_sr))
        if baseline_sr is None:
            baseline_sr = full_sr
        chg = full_sr - baseline_sr
        print(f"  {mult:>7.1f}x {raw_sr:>+8.3f} {full_sr:>+8.3f} {chg:>+8.3f} ${imp:>10,.0f}")

    if tc_results:
        for mult, sr in tc_results:
            if sr <= 0:
                print(f"\n  → Breakeven TC at {mult:.1f}x base (Sharpe = {sr:.3f})")
                break
        else:
            print(f"\n  → No breakeven up to 3x base TC")

print(f"\n{'='*72}")
print("  ALL ENGINE TESTS COMPLETE")
print(f"{'='*72}")
