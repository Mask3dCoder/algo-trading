#!/usr/bin/env python3
"""validate_survivors_with_costs.py — Cost-aware walk-forward validation for the 4 survivors.

This script re-runs the honest sweep methodology BUT with realistic transaction costs
deducted per signal flip. The original honest_sweep.py computed returns WITHOUT costs,
making all Sharpe ratios unreliable.

Cost model:
  - Equity ETFs (SPY, QQQ, etc.): 2bps one-way slippage, $0.005/share commission
  - Equities (individual stocks): 5bps one-way slippage, $0.005/share commission
  - Macro (futures proxy via ETFs): 2bps one-way slippage, $0.005/share commission
  - Round-trip cost per position flip: ~4-10bps of notional

Methodology:
  - Same walk-forward as honest_sweep.py (504d train / 126d test, anchored, 4+ folds)
  - Track position changes between steps → deduct cost per flip
  - Parameter freeze — use exact same params as original sweep
  - Compare zero-cost vs realistic-cost results side-by-side

Gate:
  - KEEP:    ci_lower > 0 AND dsr_p < 0.1 AND max_dd < 50%
  - MONITOR: ci_lower > 0
  - DISABLE: everything else

Usage:
    python scripts/validate_survivors_with_costs.py
    python scripts/validate_survivors_with_costs.py --output audit/rebuild_2026/survivor_validation.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants — MUST match honest_sweep.py exactly
# ---------------------------------------------------------------------------

TRAIN_SIZE = 504   # 2 years
TEST_SIZE = 126    # 6 months
EMBARGO_PCT = 0.01
N_BOOTSTRAP = 10_000
MIN_FOLDS = 4

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "GLD", "TLT", "DBC",
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META",
    "TSLA", "JPM", "V", "JNJ", "PG", "UNH", "HD", "XOM",
]
MACRO_TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]

DATA_START = "2014-01-01"
DATA_END = "2024-12-31"

# Transaction cost model
# ETFs get 2bps, individual equities get 5bps
ETF_SET = {"SPY", "QQQ", "IWM", "GLD", "TLT", "DBC", "EFA", "EEM", "VNQ", "BND"}
SLIPPAGE_BPS_ETF = 2.0
SLIPPAGE_BPS_EQUITY = 5.0
COMMISSION_PER_SHARE = 0.005  # $0.005/share (Interactive Brokers rate)
AVG_SHARE_PRICE = 150.0  # Approximate average for commission-to-bps conversion

# The 4 survivors with EXACT same params as honest_sweep.py
SURVIVORS: dict[str, dict] = {
    "equity_tsmom": {
        "params": {},
        "tickers": DEFAULT_TICKERS,
        "type": "equity",
    },
    "equity_full_universe_xs_momentum": {
        "params": {"rebalance_freq": 1},
        "tickers": DEFAULT_TICKERS,
        "type": "equity",
    },
    "macro_cta_trend": {
        "params": {},
        "tickers": MACRO_TICKERS,
        "type": "macro",
    },
    "macro_risk_parity": {
        "params": {},
        "tickers": MACRO_TICKERS,
        "type": "macro",
    },
}


# ---------------------------------------------------------------------------
# Data loading — reuse honest_sweep infrastructure
# ---------------------------------------------------------------------------

def load_real_data(tickers: list[str], start: str, end: str) -> dict[str, np.ndarray]:
    """Load real yfinance data. Identical to honest_sweep.py."""
    import yfinance as yf

    print(f"  Loading real data: {len(tickers)} tickers, {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise RuntimeError(f"No data returned from yfinance for {tickers}")

    if len(tickers) == 1:
        close = data["Close"].values.reshape(-1, 1)
        high = data["High"].values.reshape(-1, 1) if "High" in data else close * 1.005
        low = data["Low"].values.reshape(-1, 1) if "Low" in data else close * 0.995
        volume = data["Volume"].values.reshape(-1, 1).astype(float) if "Volume" in data else np.ones_like(close) * 1e6
    else:
        close = data["Close"].values
        high = data["High"].values if "High" in data else close * 1.005
        low = data["Low"].values if "Low" in data else close * 0.995
        volume = data["Volume"].values.astype(float) if "Volume" in data else np.ones_like(close) * 1e6

    valid_mask = ~np.any(np.isnan(close), axis=1)
    close = close[valid_mask]
    high = high[valid_mask]
    low = low[valid_mask]
    volume = volume[valid_mask]

    log_ret = np.diff(np.log(close), axis=0)
    log_ret = np.vstack([np.zeros((1, close.shape[1])), log_ret])
    iv_proxy = np.full_like(close, 0.20)
    lookback = 21
    for i in range(lookback, len(close)):
        rv = np.std(log_ret[i - lookback + 1 : i + 1], axis=0, ddof=1) * np.sqrt(252)
        iv_proxy[i] = np.maximum(rv, 0.05)

    yields_proxy = np.full_like(close, 0.04)

    n_bars = close.shape[0]
    n_assets = close.shape[1]
    print(f"  Loaded {n_bars} bars, {n_assets} assets")

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "implied_vol": iv_proxy,
        "yields": yields_proxy,
        "symbols": list(tickers),
    }


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def get_slippage_bps(symbol: str) -> float:
    """Get one-way slippage in bps for a given symbol."""
    return SLIPPAGE_BPS_ETF if symbol in ETF_SET else SLIPPAGE_BPS_EQUITY


def compute_round_trip_cost_bps(symbol: str) -> float:
    """Compute total round-trip cost in bps for a position flip.

    Round-trip = sell old position + buy new position
    Each leg: slippage + commission_as_bps
    """
    slip_one_way = get_slippage_bps(symbol)
    commission_bps = (COMMISSION_PER_SHARE / AVG_SHARE_PRICE) * 10_000
    # Round-trip: 2 × (slippage + commission)
    return 2.0 * (slip_one_way + commission_bps)


# ---------------------------------------------------------------------------
# Walk-forward + cost-aware return computation
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(returns: np.ndarray, n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """Bootstrap Sharpe ratio with 90% confidence interval."""
    n = len(returns)
    if n < 10:
        return float("nan"), float("nan"), float("nan")

    sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        mean_r = float(np.mean(sample))
        std_r = float(np.std(sample, ddof=1))
        if std_r > 1e-10:
            sharpes.append(mean_r / std_r * np.sqrt(252))

    if len(sharpes) < 100:
        return float("nan"), float("nan"), float("nan")

    sharpes = np.array(sharpes)
    oos_sharpe = float(np.mean(sharpes))
    ci_lower = float(np.percentile(sharpes, 5))
    ci_upper = float(np.percentile(sharpes, 95))
    return oos_sharpe, ci_lower, ci_upper


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown percentage from returns series."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(np.max(dd)) * 100


def compute_dsr_pvalue(sharpe: float, n_trials: int, n_bars: int) -> float:
    """Deflated Sharpe Ratio p-value (simplified)."""
    if math.isnan(sharpe) or n_bars < 10:
        return 1.0
    # Simplified DSR: use Bonferroni-like correction
    # Full DSR uses skew/kurt adjustments; this is conservative
    se = 1.0 / np.sqrt(n_bars / 252) if n_bars > 0 else 1.0
    t_stat = sharpe / se
    from scipy import stats as sp_stats
    p_single = 1 - sp_stats.norm.cdf(t_stat)
    # Bonferroni correction for multiple testing
    p_corrected = min(1.0, p_single * n_trials)
    return p_corrected


def run_survivor_with_costs(
    name: str,
    params: dict,
    data: dict[str, np.ndarray],
    tickers: list[str],
    apply_costs: bool = True,
) -> dict:
    """Run a single survivor through walk-forward with optional cost deduction.

    If apply_costs=True, deducts round-trip costs per position flip.
    If apply_costs=False, computes raw returns (same as original honest_sweep).
    """
    # Import strategy modules to trigger @register decorators
    import importlib
    survivor_modules = [
        "quantcore.strategies.equity.tsmom",
        "quantcore.strategies.equity.full_universe_cross_sectional_momentum",
        "quantcore.strategies.macro.cta_trend",
        "quantcore.strategies.macro.risk_parity",
    ]
    for mod_name in survivor_modules:
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass

    from quantcore.strategies.registry import create
    from quantcore.strategies.base import StrategyConfig, StrategyType
    from quantcore.backtest.walk_forward import anchored_walk_forward, rolling_walk_forward

    # Create fresh strategy instance
    stype = StrategyType.EQUITY
    cfg = StrategyConfig(name=name, strategy_type=stype, params=params)
    strat = create(name, cfg)
    strat.initialize()

    close = data["close"]
    n_bars = close.shape[0]
    n_assets = close.shape[1]
    symbols = data.get("symbols", [f"A{i}" for i in range(n_assets)])

    # Walk-forward folds
    folds = anchored_walk_forward(n_bars, initial_train=TRAIN_SIZE, test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)
    if len(folds) < MIN_FOLDS:
        folds = rolling_walk_forward(n_bars, train_size=TRAIN_SIZE, test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)

    all_oos_returns: list[float] = []
    n_trades_total = 0
    total_oos_bars = 0
    step = 5

    # Track previous positions per asset for cost calculation
    prev_directions: dict[str, int] = {s: 0 for s in symbols}

    for fold in folds:
        # Feed training data first
        train_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                train_data[k] = v[fold.train_start:fold.train_end]
            else:
                train_data[k] = v
        try:
            strat.on_data(train_data)
        except Exception:
            pass

        # Reset positions at fold boundary (fresh start)
        prev_directions = {s: 0 for s in symbols}

        # Pre-compute test-period asset returns
        test_prices = close[fold.test_start:fold.test_end]
        test_asset_rets = np.diff(test_prices, axis=0) / np.maximum(test_prices[:-1], 1e-10)

        n_test = fold.test_end - fold.test_start
        fold_trades = 0

        for t_offset in range(20, n_test - step, step):
            global_end = fold.test_start + t_offset
            bar_data = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    bar_data[k] = v[fold.train_start:global_end]
                else:
                    bar_data[k] = v

            try:
                signals = strat.on_data(bar_data)
            except Exception:
                signals = []

            if not signals:
                continue

            # Build current position direction per asset
            current_directions: dict[str, int] = dict(prev_directions)
            for s in signals:
                if s.symbol in current_directions:
                    current_directions[s.symbol] = s.direction

            # Compute signal-weighted portfolio return
            port_ret = 0.0
            total_weight = 0.0
            cost_this_step = 0.0

            for s in signals:
                if s.direction == 0:
                    continue
                if s.symbol not in symbols:
                    continue

                idx = symbols.index(s.symbol)
                if idx >= n_assets or t_offset - 1 >= len(test_asset_rets):
                    continue

                end_idx = min(t_offset - 1 + step, len(test_asset_rets))
                if end_idx <= t_offset - 1:
                    continue

                # Raw return
                multi_ret = float(np.sum(test_asset_rets[t_offset - 1:end_idx, idx]))
                port_ret += float(s.direction) * float(s.strength) * multi_ret
                total_weight += float(s.strength)

                # Cost: deduct if direction changed from previous step
                if apply_costs and s.symbol in prev_directions:
                    if current_directions[s.symbol] != prev_directions[s.symbol]:
                        # Position flip detected → round-trip cost
                        rt_cost_bps = compute_round_trip_cost_bps(s.symbol)
                        # Cost as fraction of return (approximate: cost / expected_return_period)
                        # For a step-period return, cost is rt_cost_bps / 10000
                        cost_this_step += rt_cost_bps / 10_000.0
                        fold_trades += 1

            if total_weight > 0:
                weighted_ret = port_ret / total_weight
                # Deduct costs from the weighted return
                if apply_costs:
                    weighted_ret -= cost_this_step
                all_oos_returns.append(weighted_ret)
                total_oos_bars += 1

            # Update previous directions for next step
            prev_directions = dict(current_directions)

        n_trades_total += fold_trades

    # Compute statistics
    if len(all_oos_returns) < 10:
        return {
            "strategy": name,
            "oos_sharpe": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "max_dd_pct": float("nan"),
            "n_trades": n_trades_total,
            "dsr_p": 1.0,
            "status": "NO_DATA",
            "n_folds": len(folds),
            "costs_applied": apply_costs,
        }

    oos_array = np.array(all_oos_returns)
    oos_sharpe, ci_lower, ci_upper = bootstrap_sharpe_ci(oos_array)
    max_dd = compute_max_drawdown(oos_array)
    dsr_p = compute_dsr_pvalue(oos_sharpe, n_trials=4, n_bars=total_oos_bars)

    # Triage
    if not math.isnan(ci_lower) and ci_lower > 0 and dsr_p < 0.1 and max_dd < 50:
        status = "KEEP"
    elif not math.isnan(ci_lower) and ci_lower > 0:
        status = "MONITOR"
    else:
        status = "DISABLE"

    return {
        "strategy": name,
        "oos_sharpe": round(oos_sharpe, 4) if math.isfinite(oos_sharpe) else float("nan"),
        "ci_lower": round(ci_lower, 4) if math.isfinite(ci_lower) else float("nan"),
        "ci_upper": round(ci_upper, 4) if math.isfinite(ci_upper) else float("nan"),
        "max_dd_pct": round(max_dd, 2),
        "n_trades": n_trades_total,
        "dsr_p": round(dsr_p, 4),
        "status": status,
        "n_folds": len(folds),
        "costs_applied": apply_costs,
        "mean_oos_return": round(float(np.mean(oos_array)), 6),
        "std_oos_return": round(float(np.std(oos_array, ddof=1)), 6),
        "n_observations": len(all_oos_returns),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Cost-aware survivor validation")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    # Ensure audit directory exists
    audit_dir = REPO_ROOT / "audit" / "rebuild_2026"
    audit_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or str(audit_dir / f"survivor_validation_{datetime.now().strftime('%Y%m%d')}.csv")

    print("=" * 70)
    print("SURVIVOR VALIDATION: Cost-Aware Walk-Forward")
    print("=" * 70)
    print(f"  Train: {TRAIN_SIZE}d | Test: {TEST_SIZE}d | Bootstrap: {N_BOOTSTRAP}")
    print(f"  Cost model: {SLIPPAGE_BPS_ETF}bps ETF / {SLIPPAGE_BPS_EQUITY}bps equity + ${COMMISSION_PER_SHARE}/share")
    print(f"  Survivors: {list(SURVIVORS.keys())}")
    print()

    # Load data (two universes)
    print("[1/3] Loading market data...")
    equity_data = load_real_data(DEFAULT_TICKERS, DATA_START, DATA_END)
    macro_data = load_real_data(MACRO_TICKERS, DATA_START, DATA_END)

    # Run each survivor with ZERO costs and REALISTIC costs
    results = []

    for i, (name, spec) in enumerate(SURVIVORS.items(), 1):
        print(f"\n[{2}/{3}] {name} ({i}/{len(SURVIVORS)})...")

        data = macro_data if spec["type"] == "macro" else equity_data

        # Zero-cost run (baseline — should match original honest_sweep)
        print(f"  Running ZERO costs...")
        result_zero = run_survivor_with_costs(name, spec["params"], data, spec["tickers"], apply_costs=False)
        result_zero["cost_tier"] = "zero"
        print(f"    Sharpe={result_zero['oos_sharpe']:+.2f} CI=[{result_zero['ci_lower']:+.2f}, {result_zero['ci_upper']:+.2f}] "
              f"DD={result_zero['max_dd_pct']:.1f}% trades={result_zero['n_trades']} -> {result_zero['status']}")

        # Realistic-cost run
        print(f"  Running REALISTIC costs ({SLIPPAGE_BPS_ETF}bps ETF / {SLIPPAGE_BPS_EQUITY}bps equity)...")
        result_real = run_survivor_with_costs(name, spec["params"], data, spec["tickers"], apply_costs=True)
        result_real["cost_tier"] = "realistic"
        print(f"    Sharpe={result_real['oos_sharpe']:+.2f} CI=[{result_real['ci_lower']:+.2f}, {result_real['ci_upper']:+.2f}] "
              f"DD={result_real['max_dd_pct']:.1f}% trades={result_real['n_trades']} -> {result_real['status']}")

        # Cost impact
        if math.isfinite(result_zero["oos_sharpe"]) and math.isfinite(result_real["oos_sharpe"]):
            sharpe_delta = result_zero["oos_sharpe"] - result_real["oos_sharpe"]
            print(f"    COST IMPACT: Sharpe dropped by {sharpe_delta:+.2f} ({sharpe_delta/max(result_zero['oos_sharpe'], 0.01)*100:.0f}%)")
            if sharpe_delta < 0.1:
                print(f"    WARNING: Cost impact < 0.1 — cost model may not be deducting properly!")

        results.append(result_zero)
        results.append(result_real)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n[3/3] Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SURVIVOR VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<40} {'Cost Tier':<10} {'Sharpe':>8} {'CI Low':>8} {'Status':<10}")
    print("-" * 80)
    for r in results:
        sharpe = f"{r['oos_sharpe']:+.2f}" if math.isfinite(r['oos_sharpe']) else "N/A"
        ci = f"{r['ci_lower']:+.2f}" if math.isfinite(r['ci_lower']) else "N/A"
        print(f"{r['strategy']:<40} {r['cost_tier']:<10} {sharpe:>8} {ci:>8} {r['status']:<10}")

    # Final verdict
    realistic_results = [r for r in results if r["cost_tier"] == "realistic"]
    any_keep = any(r["status"] == "KEEP" for r in realistic_results)
    any_monitor = any(r["status"] == "MONITOR" for r in realistic_results)
    all_disable = all(r["status"] == "DISABLE" for r in realistic_results)

    print()
    if any_keep:
        print("VERDICT: At least one survivor passes with realistic costs → viable product.")
    elif any_monitor:
        print("VERDICT: Marginal survivors — ci_lower > 0 but DSR not significant. Low-scale only.")
    elif all_disable:
        print("VERDICT: ALL survivors fail with realistic costs. No deployable alpha.")
        print("         STOP. Do not build infrastructure. Report honestly.")
    else:
        print("VERDICT: Mixed results. Review individual strategy results above.")

    # Also save a cost sensitivity audit
    cost_audit_path = str(audit_dir / "cost_sensitivity_audit.csv")
    df.to_csv(cost_audit_path, index=False)
    print(f"\nCost sensitivity audit saved to {cost_audit_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
