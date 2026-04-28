"""
Diagnostic 3: Signal generation comparison (real vs synthetic data).

Tests whether signals generated on real market data differ materially from
signals on clean synthetic data over the same universe and period.

If signals are structurally different (fewer signals, biased directions,
collapsed strength), the issue is in the data pipeline (NaN propagation,
missing symbols, conversion artifacts) rather than strategy logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantcore.strategies.base import TradeSignal
from scripts.real_data_backtest import STRATEGY_UNIVERSE, _df_to_data, _setup_strategy
from scripts.real_data_loader import UNIVERSES, load_universe


def extract_signals(
    strategy_id: str,
    df: pd.DataFrame,
    start_idx: int = 0,
) -> tuple[list[dict], np.ndarray]:
    """Run strategy bar-by-bar and collect emitted signals.

    Returns (signal_records, prices) where signal_records is a list of dicts
    with keys: bar_idx, symbol, direction, strength, composite, and prices
    is the close array from _df_to_data for context.
    """
    data = _df_to_data(df)
    close = data["close"]
    n_bars = close.shape[0]
    symbols = data["symbols"]

    strat = _setup_strategy(strategy_id)
    records: list[dict] = []

    for bar_i in range(start_idx, n_bars - 1):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[: bar_i + 1]
            else:
                bar_data[k] = v

        try:
            signals = strat.on_data(bar_data) or []
        except Exception:
            signals = []

        for s in signals:
            records.append({
                "bar_idx": bar_i,
                "symbol": s.symbol,
                "direction": s.direction,
                "strength": float(s.strength),
                "composite": float(s.metadata.get("composite", 0)) if hasattr(s, "metadata") and s.metadata else 0.0,
                "fast": float(s.metadata.get("fast", 0)) if hasattr(s, "metadata") and s.metadata else 0.0,
                "medium": float(s.metadata.get("medium", 0)) if hasattr(s, "metadata") and s.metadata else 0.0,
                "slow": float(s.metadata.get("slow", 0)) if hasattr(s, "metadata") and s.metadata else 0.0,
                "vol_scale": float(s.metadata.get("vol_scale", 1.0)) if hasattr(s, "metadata") and s.metadata else 1.0,
                "realized_vol": float(s.metadata.get("realized_vol", 0.0)) if hasattr(s, "metadata") and s.metadata else 0.0,
            })

    return records, close


def generate_synthetic_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    annual_vol: float = 0.20,
    annual_mu: float = 0.05,
    n_future: int = 0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data as GBM for given symbols.

    Returns DataFrame with same schema as load_universe().
    """
    dates = pd.bdate_range(start_date, end_date, freq="B")
    n_days = len(dates)
    dt = 1.0 / 252
    daily_sigma = annual_vol * np.sqrt(dt)
    daily_mu = annual_mu * dt

    rng = np.random.default_rng(42)
    rows = []
    for sym in symbols:
        price = 100.0
        for ts in dates:
            ret = daily_mu + daily_sigma * rng.normal()
            price = max(price * np.exp(ret), 10.0)
            daily_ret = rng.uniform(-0.015, 0.015)
            rows.append({
                "symbol": sym,
                "timestamp": ts,
                "open": round(price * (1 + rng.uniform(-0.003, 0.003)), 2),
                "high": round(price * (1 + abs(rng.uniform(0.002, 0.008))), 2),
                "low": round(price * (1 - abs(rng.uniform(0.002, 0.008))), 2),
                "close": round(price, 2),
                "volume": int(rng.integers(1_000_000, 15_000_000)),
            })

    # Add some future data if n_future > 0
    if n_future > 0:
        future_dates = pd.bdate_range(
            dates[-1] + pd.Timedelta(days=1), periods=n_future, freq="B"
        )
        for sym in symbols:
            price = rows[-1]["close"]
            for ts in future_dates:
                ret = daily_mu + daily_sigma * rng.normal()
                price = max(price * np.exp(ret), 10.0)
                rows.append({
                    "symbol": sym,
                    "timestamp": ts,
                    "open": round(price * (1 + rng.uniform(-0.003, 0.003)), 2),
                    "high": round(price * (1 + abs(rng.uniform(0.002, 0.008))), 2),
                    "low": round(price * (1 - abs(rng.uniform(0.002, 0.008))), 2),
                    "close": round(price, 2),
                    "volume": int(rng.integers(1_000_000, 15_000_000)),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


def compute_signal_stats(records: list[dict], label: str) -> dict:
    """Compute summary statistics for a set of signal records."""
    n_signals = len(records)
    if n_signals == 0:
        print(f"  {label}: NO SIGNALS EMITTED")
        return {"n_signals": 0}

    df = pd.DataFrame(records)
    directions = df["direction"]
    strengths = df["strength"]
    composites = df["composite"]
    n_long = int((directions > 0).sum())
    n_short = int((directions < 0).sum())
    n_unique_bars = df["bar_idx"].nunique()
    unique_symbols = df["symbol"].unique()

    print(f"  {label}:")
    print(f"    Total signals:       {n_signals}")
    print(f"    Unique bar indices:  {n_unique_bars}")
    print(f"    Symbols hit:         {sorted(unique_symbols)}")
    print(f"    Long / Short:        {n_long} / {n_short}  ({n_long/n_signals:.1%} long)")
    print(f"    Mean strength:       {strengths.mean():.4f}")
    print(f"    Median strength:     {strengths.median():.4f}")
    print(f"    Strength std:        {strengths.std():.4f}")
    print(f"    Mean composite:      {composites.mean():.4f}")
    print(f"    Median composite:    {composites.median():.4f}")
    print(f"    Composite std:       {composites.std():.4f}")

    # Per-symbol breakdown
    for sym in sorted(unique_symbols):
        sym_df = df[df["symbol"] == sym]
        sym_dir = sym_df["direction"]
        sym_str = sym_df["strength"]
        n_sym_long = int((sym_dir > 0).sum())
        n_sym_short = int((sym_dir < 0).sum())
        print(f"      {sym:6s}: {len(sym_df):4d} signals  "
              f"L={n_sym_long:3d}/S={n_sym_short:3d}  "
              f"mean_strength={sym_str.mean():.4f}")

    # Direction stability: fraction of consecutive signals with same direction
    # (group by symbol, look at direction changes between consecutive bars)
    direction_changes = 0
    for sym in sorted(unique_symbols):
        sym_df = df[df["symbol"] == sym].sort_values("bar_idx")
        dirs = sym_df["direction"].values
        if len(dirs) > 1:
            direction_changes += int((dirs[1:] != dirs[:-1]).sum())

    return {
        "n_signals": n_signals,
        "n_long": n_long,
        "n_short": n_short,
        "mean_strength": float(strengths.mean()),
        "median_strength": float(strengths.median()),
        "strength_std": float(strengths.std()),
        "mean_composite": float(composites.mean()),
        "n_unique_bars": n_unique_bars,
        "direction_changes": direction_changes,
    }


def main():
    print("=" * 70)
    print("  DIAGNOSTIC 3: SIGNAL GENERATION COMPARISON")
    print("  Real market data vs synthetic data")
    print("=" * 70)

    strategy_id = "macro_cta_trend"
    universe = STRATEGY_UNIVERSE[strategy_id]
    symbols = UNIVERSES[universe]["symbols"]
    period = ("2020-01-01", "2021-12-31")  # COVID period

    # ── 1. Load real data ──
    print("\n" + "-" * 70)
    print("  1. Loading REAL market data (CTA universe, 2020-2021)")
    print("-" * 70)
    df_real = load_universe(universe, period[0], period[1])
    print(f"  Loaded {len(df_real)} rows, {df_real['timestamp'].nunique()} bars")
    data_real = _df_to_data(df_real)
    n_bars_real = data_real["close"].shape[0]
    print(f"  Converted to {n_bars_real} bars x {len(data_real['symbols'])} assets")

    # ── 2. Generate synthetic data ──
    print("\n" + "-" * 70)
    print("  2. Generating SYNTHETIC data (GBM, same universe & period)")
    print("-" * 70)
    df_syn = generate_synthetic_data(symbols, period[0], period[1])
    print(f"  Generated {len(df_syn)} rows, {df_syn['timestamp'].nunique()} bars")
    data_syn = _df_to_data(df_syn)
    n_bars_syn = data_syn["close"].shape[0]
    print(f"  Converted to {n_bars_syn} bars x {len(data_syn['symbols'])} assets")

    # ── 3. Extract signals ──
    print("\n" + "-" * 70)
    print("  3. Extracting signals from both datasets")
    print("-" * 70)

    # Warmup: first 200 bars used to build strategy state (same as walk-forward)
    warmup_bars = 200

    real_records, real_prices = extract_signals(strategy_id, df_real, start_idx=warmup_bars)
    print(f"\n  REAL data: extracted {len(real_records)} signals")
    syn_records, syn_prices = extract_signals(strategy_id, df_syn, start_idx=warmup_bars)
    print(f"  SYNTHETIC data: extracted {len(syn_records)} signals")

    # ── 4. Compare signals ──
    print("\n" + "-" * 70)
    print("  4. SIGNAL COMPARISON")
    print("-" * 70)

    print(f"\n  {'=' * 55}")
    print(f"  SIGNAL COUNTS")
    print(f"  {'=' * 55}")
    real_stats = compute_signal_stats(real_records, "  REAL")
    print()
    syn_stats = compute_signal_stats(syn_records, "  SYNTHETIC")

    # ── 5. Structural analysis ──
    print("\n" + "-" * 70)
    print("  5. STRUCTURAL ANALYSIS")
    print("-" * 70)

    issues: list[str] = []

    # 5a. Signal count comparison (expected ~ rebalance_freq=5 => ~100 bars of signals)
    n_real = real_stats.get("n_signals", 0)
    n_syn = syn_stats.get("n_signals", 0)
    period_bars = min(n_bars_real, n_bars_syn) - warmup_bars
    expected_signals = period_bars // 5 * len(symbols)  # approx

    print(f"\n  OOS bars:       {period_bars}")
    print(f"  Expected signals (approx): {expected_signals}")
    print(f"  Real signals:   {n_real}")
    print(f"  Synthetic signals: {n_syn}")

    if n_real < expected_signals * 0.3:
        issues.append(f"REAL signals ({n_real}) << expected ({expected_signals})")
        print(f"  [FAIL] Real signals are far below expected")
    elif n_real < expected_signals * 0.7:
        print(f"  [WARN] Real signals moderately below expected")
    else:
        print(f"  [PASS] Real signal count is reasonable")

    if n_real == 0 and n_syn > 0:
        issues.append("REAL SIGNALS = 0, synthetic has signals")
        print(f"  [FAIL] Strategy emits NO signals on real data")
    elif n_syn == 0:
        issues.append("NO SIGNALS on synthetic data (strategy issue?)")
        print(f"  [FAIL] Strategy emits NO signals on synthetic data either")

    # 5b. Mean strength comparison
    mean_r = real_stats.get("mean_strength", 0)
    mean_s = syn_stats.get("mean_strength", 0)
    print(f"\n  Mean signal strength: real={mean_r:.4f}  synthetic={mean_s:.4f}")
    if n_real > 5 and n_syn > 5:
        delta_str = abs(mean_r - mean_s) / max(abs(mean_s), 1e-10)
        if delta_str > 0.5:
            issues.append(f"Signal strength delta {delta_str:.1f}x between real and synthetic")
            print(f"  [FAIL] Signal strength differs materially ({delta_str:.1f}x)")
        else:
            print(f"  [PASS] Signal strength is comparable")

    # 5c. Long/short bias
    if n_real > 0:
        real_long_pct = real_stats.get("n_long", 0) / n_real
        syn_long_pct = syn_stats.get("n_long", 0) / max(n_syn, 1)
        print(f"\n  Long bias: real={real_long_pct:.1%}  synthetic={syn_long_pct:.1%}")
        long_delta = abs(real_long_pct - syn_long_pct)
        if long_delta > 0.25 and n_real > 10:
            issues.append(f"Long/short bias delta {long_delta:.0%}")
            print(f"  [WARN] Direction bias differs between real and synthetic")
        else:
            print(f"  [PASS] Direction bias is comparable")

    # 5d. Vol scaling comparison
    if n_real > 0:
        real_df = pd.DataFrame(real_records)
        syn_df = pd.DataFrame(syn_records)

        mean_rv_r = real_df["realized_vol"].mean()
        mean_rv_s = syn_df["realized_vol"].mean()
        mean_vs_r = real_df["vol_scale"].mean()
        mean_vs_s = syn_df["vol_scale"].mean()

        print(f"\n  Mean realized vol: real={mean_rv_r:.4f}  synthetic={mean_rv_s:.4f}")
        print(f"  Mean vol scale:   real={mean_vs_r:.4f}  synthetic={mean_vs_s:.4f}")

        if mean_vs_r < 0.3 and mean_vs_s > 0.5:
            issues.append(f"Vol scaling much lower on real data ({mean_vs_r:.2f} vs {mean_vs_s:.2f})")
            print(f"  [FAIL] Vol targeting is severely compressing real-data signals")
        elif mean_vs_r < mean_vs_s * 0.5:
            print(f"  [WARN] Vol scaling is materially lower on real data")

    # 5e. NaN / missing data in the data dict
    print(f"\n  Data quality check:")
    for label, data, df in [("REAL", data_real, df_real), ("SYNTHETIC", data_syn, df_syn)]:
        close_arr = data["close"]
        nan_count = int(np.isnan(close_arr).sum())
        total_elements = close_arr.size
        pct_nan = nan_count / total_elements * 100 if total_elements > 0 else 0
        print(f"    {label:12s}: NaN in close: {nan_count}/{total_elements} ({pct_nan:.2f}%)")

        # Check for zeros in close
        zero_count = int((close_arr == 0).sum())
        pct_zero = zero_count / total_elements * 100 if total_elements > 0 else 0
        print(f"    {'':12s}  Zeros in close: {zero_count}/{total_elements} ({pct_zero:.2f}%)")

        # Check for inf
        inf_count = int(np.isinf(close_arr).sum())
        print(f"    {'':12s}  Inf in close: {inf_count}")

        if nan_count > 0 and pct_nan < 5:
            # Check if NaN is at leading edge (harmless) vs interspersed (problematic)
            non_nan = np.any(~np.isnan(close_arr), axis=1)
            first_valid = int(np.argmax(non_nan)) if non_nan.any() else 0
            print(f"    {'':12s}  First valid bar index: {first_valid}")

    # ── 6. Summary ──
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 3 SUMMARY")
    print("=" * 70)

    if not issues:
        print("\n  [PASS] Signal generation is CONSISTENT between real and synthetic data.")
        print("  The strategy emits similar signals in both environments.")
        print("  Data pipeline artifacts are NOT the cause of Phase A failures.")
    else:
        print(f"\n  [FAIL] {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print("  Data pipeline or conversion may be distorting signals.")

    print()


if __name__ == "__main__":
    main()
