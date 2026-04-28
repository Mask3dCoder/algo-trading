"""
Diagnostic 2: Transaction cost impact analysis.

Tests how performance varies with turnover-based transaction costs:
- TC=0 bps (zero costs)
- TC=5 bps (low)
- TC=10 bps (moderate)
- TC=25 bps (high)
- TC=50 bps (extreme)

If OOS SR improves dramatically at TC=0, strategies are being killed by
trading costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.real_data_backtest import (
    STRATEGY_UNIVERSE,
    run_full_period_backtest,
    run_regime_backtest,
)
from scripts.real_data_loader import UNIVERSES, load_universe


def run_tc_sweep(
    strategy_id: str,
    period_label: str,
    df: pd.DataFrame,
    tc_levels: list[float],
    regime_warmup_idx: int | None = None,
) -> list[dict]:
    """Run backtest at multiple TC levels and return results."""
    results = []
    for tc in tc_levels:
        if regime_warmup_idx is not None:
            r = run_regime_backtest(strategy_id, df, regime_warmup_idx, tc_bps=tc)
        else:
            r = run_full_period_backtest(strategy_id, df, tc_bps=tc)

        results.append({
            "tc_bps": tc,
            "oos_sharpe": r.get("oos_sharpe"),
            "max_dd": r.get("max_dd"),
            "win_rate": r.get("win_rate"),
            "n_bars": r.get("n_oos_bars", 0),
            "error": r.get("error"),
        })
    return results


def print_sweep_table(results: list[dict], label: str):
    """Print TC sensitivity table."""
    print(f"\n  {label}:")
    print(f"  {'TC(bps)':>8s} {'OOS SR':>10s} {'Max DD':>10s} {'Win Rate':>10s} {'Bars':>6s}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 6}")
    for r in results:
        tc = r["tc_bps"]
        sr = f"{r['oos_sharpe']:+.4f}" if r.get("oos_sharpe") is not None else "ERROR"
        dd = f"{r['max_dd']:.2%}" if r.get("max_dd") is not None else "N/A"
        wr = f"{r['win_rate']:.1%}" if r.get("win_rate") is not None else "N/A"
        nb = str(r.get("n_bars", 0))
        print(f"  {tc:>8.0f} {sr:>10s} {dd:>10s} {wr:>10s} {nb:>6s}")


def main():
    print("=" * 70)
    print("  DIAGNOSTIC 2: TRANSACTION COST IMPACT ANALYSIS")
    print("=" * 70)

    tc_levels = [0.0, 5.0, 10.0, 25.0, 50.0]

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════
    # macro_cta_trend / Full period (2018-2024)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("  Loading CTA full period 2018-2024...")
    print("-" * 70)
    df_cta = load_universe("cta", "2018-01-01", "2024-12-31")
    print(f"  Loaded {len(df_cta)} bars")

    r = run_tc_sweep("macro_cta_trend", "CTA Full", df_cta, tc_levels)
    all_results["CTA Full"] = r
    print_sweep_table(r, "CTA Full Period 2018-2024")

    # ═══════════════════════════════════════════════════════════════════
    # macro_cta_trend / COVID 2020
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("  Loading CTA COVID 2020...")
    print("-" * 70)
    df_cta_covid = load_universe("cta", "2019-06-01", "2021-12-31")
    all_dates = sorted(df_cta_covid["timestamp"].unique())
    covid_start = pd.Timestamp("2020-01-01")
    if hasattr(all_dates[0], "tz") and all_dates[0].tz is not None:
        covid_start = covid_start.tz_localize(all_dates[0].tz)
    warmup_idx = next(i for i, d in enumerate(all_dates) if d >= covid_start)

    r = run_tc_sweep("macro_cta_trend", "CTA COVID", df_cta_covid, tc_levels,
                     regime_warmup_idx=warmup_idx)
    all_results["CTA COVID"] = r
    print_sweep_table(r, "CTA COVID 2020")

    # ═══════════════════════════════════════════════════════════════════
    # macro_risk_parity / Full period (2018-2024)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("  Loading RP full period 2018-2024...")
    print("-" * 70)
    df_rp = load_universe("rp", "2018-01-01", "2024-12-31")
    print(f"  Loaded {len(df_rp)} bars")

    r = run_tc_sweep("macro_risk_parity", "RP Full", df_rp, tc_levels)
    all_results["RP Full"] = r
    print_sweep_table(r, "RP Full Period 2018-2024")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 2 SUMMARY")
    print("=" * 70)

    for label, results in all_results.items():
        sr_at_0 = next((r["oos_sharpe"] for r in results if r["tc_bps"] == 0), None)
        sr_at_base = next((r["oos_sharpe"] for r in results if r["tc_bps"] == 10), None)

        if sr_at_0 is not None and sr_at_base is not None:
            delta = sr_at_0 - sr_at_base
            print(f"\n  {label}:")
            print(f"    OOS SR at TC=0:   {sr_at_0:+.4f}")
            print(f"    OOS SR at TC=10:  {sr_at_base:+.4f}")
            print(f"    Delta:            {delta:+.4f}")

            if sr_at_0 > 0.5 and sr_at_base < 0:
                print("    [FAIL] Transaction costs are KILLING the strategy")
                print("    Strategy is profitable at zero cost, negative at 10 bps.")
            elif delta > 0.3:
                print("    [WARN] Transaction costs have SIGNIFICANT impact")
            else:
                print("    [PASS] Transaction costs have reasonable impact")
                print("    Costs are NOT the primary cause of failure.")

    print()


if __name__ == "__main__":
    main()
