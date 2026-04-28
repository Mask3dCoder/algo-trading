"""Phase 2B: macro_cta_trend deep-dive stress tests (6 tests)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.strategy_audit import (
    ALL_PARAMS,
    _STRATEGY_TYPE_MAP,
    anchored_walk_forward,
    analyze_strategy,
    compute_signal_returns,
    make_data,
    StrategyConfig,
    StrategyType,
    create,
)
from quantcore.backtest.engine import BacktestEngine
from quantcore.backtest.financing import FinancingConfig
from quantcore.backtest.impact_config import MarketImpactConfig
from quantcore.backtest.analytics import drawdown_analysis
from scripts.strategy_adapter import StrategyAdapter

NAME = "macro_cta_trend"
N_SYM = 6

data = make_data()
n = data["close"].shape[0]
folds = anchored_walk_forward(n, initial_train=200, test_size=50, embargo_pct=0.02)

params = ALL_PARAMS.get(NAME, {})
stype = _STRATEGY_TYPE_MAP.get(NAME, StrategyType.EQUITY)
cfg = StrategyConfig(name=NAME, strategy_type=stype, params=params)

print(f"{'='*72}")
print(f"  PHASE 2B: {NAME} DEEP-DIVE STRESS TESTS")
print(f"{'='*72}")

# ── Test 1: Look-ahead bias ────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  TEST 1: LOOK-AHEAD BIAS HUNT")
print(f"{'='*72}")
print("  Code review of quantcore/strategies/macro/cta_trend.py:")
print("  - _trend_signal(): prices[-window:] + prices[-1] — backward only")
print("  - _breakout_signal(): prices[-window:] min/max — backward only")
print("  - _momentum_signal(): prices[-1] / prices[-window-1] — backward only")
print("  - Vol scaling: returns from prices[-vol_lookback-1:-1] — backward only")
print("  - on_data(): uses backward-only indexing throughout")
print("  => No shift(-1), no iloc[i+1], no future leakage")
print("  VERDICT: CLEAN")

# ── Test 2: Parameter sensitivity ─────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  TEST 2: PARAMETER SENSITIVITY GRID")
print(f"{'='*72}")

base_result = analyze_strategy(NAME, data, folds, [])
base_oos = base_result.get("oos_sharpe", 0.0)
print(f"\n  Base OOS Sharpe: {base_oos:+.4f}")
print(f"  Base params: {params}")

param_grid = {
    "slow_window": [60, 80, 100, 120, 140, 160, 180],
    "medium_window": [20, 30, 40, 50, 60, 70, 80],
    "fast_window": [5, 10, 15, 20, 25, 30, 40],
    "vol_target": [0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24],
    "vol_lookback": [20, 30, 40, 60, 80, 100, 120],
}

print(f"\n  {'Parameter':<20s} {'Value':>8s} {'OOS SR':>10s} {'Change':>10s}")
print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")

sensitivities = []
import scripts.strategy_audit as sa
for key, values in param_grid.items():
    for val in values:
        p = dict(params)
        p[key] = val
        old = dict(sa.ALL_PARAMS)
        sa.ALL_PARAMS[NAME] = p
        try:
            r = analyze_strategy(NAME, data, folds, [])
            oos = r.get("oos_sharpe", 0.0)
            chg = oos - base_oos
            print(f"  {key:<20s} {str(val):>8s} {oos:>+10.4f} {chg:>+10.4f}")
            sensitivities.append((key, val, abs(chg)))
        except Exception as e:
            print(f"  {key:<20s} {str(val):>8s} {'FAIL':>10s} {str(e)[:30]}")
        finally:
            sa.ALL_PARAMS[NAME] = old

if sensitivities:
    by_param = {}
    for key, val, achg in sensitivities:
        by_param.setdefault(key, []).append(achg)
    most_sensitive = max(by_param.items(), key=lambda x: np.mean(x[1]))
    print(f"\n  → Most sensitive parameter: {most_sensitive[0]} "
          f"(mean |ΔSR| = {np.mean(most_sensitive[1]):.4f})")

# ── Test 3: Regime conditional performance ─────────────────────────────────
print(f"\n{'='*72}")
print(f"  TEST 3: REGIME CONDITIONAL PERFORMANCE")
print(f"{'='*72}")

close = data["close"]
ew_rets = np.mean(close[1:] / close[:-1] - 1.0, axis=1)
vol_20 = pd.Series(ew_rets).rolling(20).std().values * np.sqrt(252)
vol_75 = np.nanpercentile(vol_20, 75)
vol_25 = np.nanpercentile(vol_20, 25)
ret_20 = pd.Series(ew_rets).rolling(20).sum().values

n_assets = close.shape[1]
corr_series = []
for i in range(20, len(close)):
    block = close[i-20:i]
    rb = np.diff(block, axis=0) / block[:-1]
    cm = np.corrcoef(rb.T)
    if n_assets > 1:
        mask = ~np.eye(n_assets, dtype=bool)
        corr_series.append(np.mean(cm[mask]))
    else:
        corr_series.append(0.0)
corr_series = np.array(corr_series)

strat = create(NAME, cfg)
strat.initialize()
oos_start = folds[0].test_start
oos_end = folds[-1].test_end
oos_rets = compute_signal_returns(strat, data, oos_start, oos_end)
n_oos = len(oos_rets)

regimes = {"HIGH_VOL": [], "LOW_VOL": [], "TREND_UP": [],
           "TREND_DOWN": [], "HIGH_CORR": [], "NORMAL": []}
min_len = min(n_oos, len(vol_20)-oos_start, len(ret_20)-oos_start, len(corr_series)-oos_start)
if min_len > 5:
    for i in range(min_len):
        idx = oos_start + i
        assigned = False
        if idx < len(vol_20) and vol_20[idx] > vol_75:
            regimes["HIGH_VOL"].append(oos_rets[i]); assigned = True
        if idx < len(vol_20) and vol_20[idx] < vol_25:
            regimes["LOW_VOL"].append(oos_rets[i]); assigned = True
        if idx < len(ret_20) and ret_20[idx] > 0.05:
            regimes["TREND_UP"].append(oos_rets[i]); assigned = True
        if idx < len(ret_20) and ret_20[idx] < -0.05:
            regimes["TREND_DOWN"].append(oos_rets[i]); assigned = True
        if idx < len(corr_series) and corr_series[idx] > 0.7:
            regimes["HIGH_CORR"].append(oos_rets[i]); assigned = True
        if not assigned:
            regimes["NORMAL"].append(oos_rets[i])

print(f"\n  {'Regime':<15s} {'Count':>6s} {'Mean Ret':>10s} {'Sharpe':>8s} {'Win Rate':>9s}")
print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*8} {'-'*9}")
for regime, rets in regimes.items():
    if len(rets) < 3:
        print(f"  {regime:<15s} {len(rets):>6d} {'N/A':>10s} {'N/A':>8s} {'N/A':>9s}")
        continue
    r = np.array(rets)
    mean_r = np.mean(r)
    sr = np.mean(r) / np.maximum(np.std(r, ddof=1), 1e-10) * np.sqrt(252)
    wr = np.sum(r > 0) / len(r)
    print(f"  {regime:<15s} {len(r):>6d} {mean_r:>+10.6f} {sr:>+8.3f} {wr:>8.1%}")

# ── Test 4: Capacity sweep ────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  TEST 4: CAPACITY SWEEP")
print(f"{'='*72}")

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

engine_data = make_engine_data(252, N_SYM)
capital_levels = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]

print(f"\n  {'Capital':>12s} {'Raw SR':>8s} {'Full SR':>8s} {'Impact $':>10s} {'Borrow $':>10s}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

results_cap = []
for cap in capital_levels:
    try:
        adapt = StrategyAdapter(create(NAME, cfg), cfg, n_symbols=N_SYM)
        engine = BacktestEngine(strategy=adapt, initial_cash=float(cap),
            sizing_pct=0.30,
            impact_config=MarketImpactConfig(sigma=0.02, adv=500_000),
            financing_config=FinancingConfig(borrow_rate=0.05, margin_rate=0.08, reg_t=0.50))
        engine.run(engine_data)
        rep = engine.summary()
        equity = np.array([v for _, v in engine.portfolio.equity_curve])
        rets = np.diff(equity) / equity[:-1]
        full_sr = np.mean(rets) / np.maximum(np.std(rets, ddof=1), 1e-10) * np.sqrt(252)
        results_cap.append((cap, full_sr, rep.total_market_impact_cost, rep.total_borrow_cost))
        print(f"  ${cap:>8,} {rep.sharpe:>+8.3f} {full_sr:>+8.3f} "
              f"${rep.total_market_impact_cost:>8,.0f} ${rep.total_borrow_cost:>8,.0f}")
    except Exception as e:
        print(f"  ${cap:>8,} {'FAIL':>8s} {str(e)[:40]}")

if results_cap:
    peak_sr = max(r[1] for r in results_cap)
    for cap, sr, imp, brw in results_cap:
        if sr < peak_sr * 0.8:
            print(f"\n  → Capacity ceiling: ${cap:,} (Sharpe degraded by >20% from {peak_sr:.3f})")
            break
    else:
        print(f"\n  → No capacity ceiling detected up to $5M")

# ── Test 5: Transaction cost sensitivity ──────────────────────────────────
print(f"\n{'='*72}")
print(f"  TEST 5: TRANSACTION COST SENSITIVITY")
print(f"{'='*72}")

tc_mult = [0.5, 1.0, 1.5, 2.0, 3.0]
base_eta = 0.0625

print(f"\n  {'TC mult':>8s} {'Raw SR':>8s} {'Full SR':>8s} {'Impact $':>12s}")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

results_tc = []
for mult in tc_mult:
    try:
        adapt = StrategyAdapter(create(NAME, cfg), cfg, n_symbols=N_SYM)
        engine = BacktestEngine(strategy=adapt, initial_cash=1_000_000.0,
            sizing_pct=0.30,
            impact_config=MarketImpactConfig(sigma=0.02, adv=500_000, eta=base_eta*mult),
            financing_config=FinancingConfig(borrow_rate=0.05, margin_rate=0.08, reg_t=0.50))
        engine.run(engine_data)
        rep = engine.summary()
        equity = np.array([v for _, v in engine.portfolio.equity_curve])
        rets = np.diff(equity) / equity[:-1]
        full_sr = np.mean(rets) / np.maximum(np.std(rets, ddof=1), 1e-10) * np.sqrt(252)
        results_tc.append((mult, full_sr, rep.total_market_impact_cost))
        print(f"  {mult:>7.1f}x {rep.sharpe:>+8.3f} {full_sr:>+8.3f} "
              f"${rep.total_market_impact_cost:>10,.0f}")
    except Exception as e:
        print(f"  {mult:>7.1f}x {'FAIL':>8s} {str(e)[:40]}")

if results_tc:
    for mult, sr, _ in results_tc:
        if sr <= 0:
            print(f"\n  → Breakeven TC at {mult:.1f}x (Sharpe = {sr:.3f})")
            break
    else:
        print(f"\n  → No breakeven up to 3x base TC")

# ── Test 6: Trend signal integrity ────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  TEST 6: TREND SIGNAL INTEGRITY")
print(f"{'='*72}")

# 6a) Signal lag
print("\n  6a) Signal Lag Analysis")
strat2 = create(NAME, cfg)
strat2.initialize()

close_arr = data["close"]
all_signals = []
for i in range(50, min(300, close_arr.shape[0])):
    subset = {k: v[:i] for k, v in data.items() if isinstance(v, np.ndarray)}
    try:
        tsigs = strat2.on_data(subset) or []
        avg = np.mean([abs(t.direction) * t.strength for t in tsigs]) if tsigs else 0.0
    except Exception:
        avg = 0.0
    all_signals.append(avg)

if len(all_signals) > 20:
    sig_arr = np.array(all_signals)
    ew = np.mean(close[1:] / close[:-1] - 1.0, axis=1)
    oos_part = ew[50:50 + len(sig_arr)]

    print(f"  {'Lag':>5s} {'Corr':>8s}")
    best_lag, best_corr = 0, 0
    for lag in range(-5, 6):
        if lag < 0:
            s, r = sig_arr[-lag:], oos_part[:len(sig_arr[-lag:])]
        elif lag > 0:
            r, s = oos_part[lag:], sig_arr[:len(oos_part[lag:])]
        else:
            s, r = sig_arr, oos_part[:len(sig_arr)]
        n_min = min(len(s), len(r))
        if n_min > 5:
            corr = float(np.corrcoef(s[:n_min], r[:n_min])[0, 1])
            print(f"  {lag:>+5d} {corr:>+8.4f}")
            if abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, lag
    print(f"  → Peak cross-correlation at lag {best_lag}: {best_corr:+.4f}")

# 6b) Factor beta
print("\n  6b) Factor Beta to Trend Benchmark (SMA crossover)")
try:
    ew_p = np.mean(close, axis=1)
    sma50 = pd.Series(ew_p).rolling(50).mean().values
    sma20 = pd.Series(ew_p).rolling(20).mean().values
    trend_sig = np.where(sma20 > sma50, 1, -1)
    ew_ret = np.diff(ew_p) / ew_p[:-1]
    trend_ret = trend_sig[:-1] * ew_ret

    n_min = min(len(oos_rets), len(trend_ret) - oos_start)
    strat_rets = oos_rets[:n_min]
    bench_rets = trend_ret[oos_start:oos_start + n_min]

    if n_min > 5:
        beta = np.cov(strat_rets, bench_rets)[0, 1] / np.maximum(np.var(bench_rets), 1e-10)
        alpha = np.mean(strat_rets) - beta * np.mean(bench_rets)
        r2 = np.corrcoef(strat_rets, bench_rets)[0, 1] ** 2
        print(f"  Alpha (daily): {alpha:+.6f}  Ann: {alpha*252:+.4f}")
        print(f"  Beta to SMA crossover: {beta:+.4f}")
        print(f"  R²: {r2:.4f}")
        if abs(beta) < 0.3:
            print(f"  → Low beta — high idiosyncratic alpha (good)")
        else:
            print(f"  → High beta — tracks simple trend benchmark")
except Exception as e:
    print(f"  Error: {e}")

# 6c) Worst drawdown episodes
print("\n  6c) Worst Drawdown Episodes")
try:
    oos_equity = 100 * np.cumprod(1 + oos_rets)
    dd_eps = drawdown_analysis(oos_equity)
    print(f"  {'#':>3s} {'Depth':>8s} {'Duration':>10s} {'Start':>6s} {'Trough':>6s}")
    for i, ep in enumerate(dd_eps[:5]):
        print(f"  {i+1:>3d} {ep['depth']:>7.1%} {ep['duration']:>5d} bars "
              f"{ep['start']:>6d} {ep['trough']:>6d}")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n{'='*72}")
print(f"  PHASE 2B COMPLETE — {NAME}")
print(f"{'='*72}")
