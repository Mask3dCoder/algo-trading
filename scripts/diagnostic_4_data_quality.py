"""
Diagnostic 4: Data quality verification.

Deep inspection of real market data quality beyond surface-level checks:
- Absurd returns (>|20%| single-bar)
- Missing dates (gaps > 5 business days)
- Stale prices (identical close for 3+ consecutive bars)
- Cross-sectional correlation collapse / anomalies
- Volume anomalies (spikes, zero-volume days)
- Price level consistency (high >= low, etc.)
- Forward/rolling returns by asset
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.real_data_backtest import FULL_PERIOD, STRATEGY_UNIVERSE
from scripts.real_data_loader import UNIVERSES, load_universe


def check_absurd_returns(df: pd.DataFrame, threshold: float = 0.20) -> list[str]:
    """Flag single-bar returns exceeding threshold (absurd moves)."""
    issues: list[str] = []
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        prices = sym_df["close"].values
        if len(prices) < 2:
            continue
        rets = (prices[1:] - prices[:-1]) / np.maximum(prices[:-1], 1e-10)
        absurd = np.where(np.abs(rets) > threshold)[0]
        if len(absurd) > 0:
            examples = []
            for idx in absurd[:5]:
                ts = sym_df.iloc[idx]["timestamp"]
                examples.append(f"{ts.date()}={rets[idx]:+.2%}")
            issues.append(
                f"{sym}: {len(absurd)} bars >|{threshold:.0%}| return "
                f"({', '.join(examples)}{'...' if len(absurd) > 5 else ''})"
            )
    return issues


def check_missing_dates(df: pd.DataFrame, max_gap_days: int = 5) -> list[str]:
    """Flag gaps between consecutive trading days exceeding max_gap_days."""
    issues: list[str] = []
    all_dates = sorted(df["timestamp"].unique())

    if len(all_dates) < 2:
        return issues

    gaps = []
    for i in range(1, len(all_dates)):
        gap = (all_dates[i] - all_dates[i - 1]).days
        if gap > max_gap_days:
            gaps.append((all_dates[i - 1], all_dates[i], gap))

    if gaps:
        total_gap_days = sum(g[2] for g in gaps)
        examples = [f"{g[0].date()} -> {g[1].date()} ({g[2]}d)" for g in gaps[:5]]
        issues.append(
            f"{len(gaps)} gap(s) >{max_gap_days}d ({total_gap_days} total days "
            f"missing): {', '.join(examples)}{'...' if len(gaps) > 5 else ''}"
        )
    return issues


def check_stale_prices(df: pd.DataFrame, max_consecutive: int = 3) -> list[str]:
    """Flag symbols with identical close prices for max_consecutive+ bars."""
    issues: list[str] = []
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        prices = sym_df["close"].values
        if len(prices) < max_consecutive + 1:
            continue

        # Count consecutive identical values
        diffs = np.diff(prices)
        stale_streaks = []
        current_streak = 0
        for i, d in enumerate(diffs):
            if d == 0:
                current_streak += 1
            else:
                if current_streak >= max_consecutive:
                    stale_streaks.append((i - current_streak, current_streak + 1))
                current_streak = 0
        if current_streak >= max_consecutive:
            stale_streaks.append((len(prices) - 1 - current_streak, current_streak + 1))

        if stale_streaks:
            n_total_stale = sum(s[1] for s in stale_streaks)
            pct_stale = n_total_stale / len(prices) * 100
            examples = [
                f"bar {start}-{start+length-1} ({length}x)"
                for start, length in stale_streaks[:3]
            ]
            issues.append(
                f"{sym}: {n_total_stale}/{len(prices)} stale bars "
                f"({pct_stale:.1f}%): {', '.join(examples)}"
            )
    return issues


def check_price_consistency(df: pd.DataFrame) -> list[str]:
    """Check high >= low, high >= open, high >= close, etc."""
    issues: list[str] = []
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym]

        bad_hl = (sym_df["high"] < sym_df["low"]).sum()
        bad_ho = (sym_df["high"] < sym_df["open"]).sum()
        bad_hc = (sym_df["high"] < sym_df["close"]).sum()
        bad_lo = (sym_df["low"] > sym_df["open"]).sum()
        bad_lc = (sym_df["low"] > sym_df["close"]).sum()
        neg_close = (sym_df["close"] <= 0).sum()
        neg_vol = (sym_df["volume"] <= 0).sum()

        sym_issues = []
        if bad_hl > 0:
            sym_issues.append(f"high<low ({bad_hl}x)")
        if bad_ho > 0:
            sym_issues.append(f"high<open ({bad_ho}x)")
        if bad_hc > 0:
            sym_issues.append(f"high<close ({bad_hc}x)")
        if bad_lo > 0:
            sym_issues.append(f"low>open ({bad_lo}x)")
        if bad_lc > 0:
            sym_issues.append(f"low>close ({bad_lc}x)")
        if neg_close > 0:
            sym_issues.append(f"close<=0 ({neg_close}x)")
        if neg_vol > 0:
            sym_issues.append(f"volume<=0 ({neg_vol}x)")

        if sym_issues:
            issues.append(f"{sym}: {'; '.join(sym_issues)}")
    return issues


def check_volume_anomalies(df: pd.DataFrame, spike_threshold: float = 50.0) -> list[str]:
    """Flag extreme volume spikes or persistently zero volume."""
    issues: list[str] = []
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        volumes = sym_df["volume"].values
        if len(volumes) < 10:
            continue

        # Zero volume
        zero_vol = (volumes == 0).sum()
        if zero_vol > 0:
            pct_zero = zero_vol / len(volumes) * 100
            issues.append(f"{sym}: {zero_vol}/{len(volumes)} zero-volume bars ({pct_zero:.1f}%)")

        # Spikes: ratio of current to rolling median
        rolling_med = pd.Series(volumes).rolling(20, min_periods=5).median().values
        for i in range(5, len(volumes)):
            if rolling_med[i] > 0 and volumes[i] > rolling_med[i] * spike_threshold:
                ratio = volumes[i] / rolling_med[i]
                ts = sym_df.iloc[i]["timestamp"]
                issues.append(
                    f"{sym}: volume spike {ratio:.0f}x on {ts.date()} "
                    f"(vol={volumes[i]:,.0f}, med={rolling_med[i]:,.0f})"
                )
                break  # one example per symbol is enough
    return issues


def check_correlation_structure(df: pd.DataFrame) -> list[str]:
    """Check cross-sectional correlation for collapse or extreme values."""
    issues: list[str] = []
    pivoted = df.pivot_table(index="timestamp", columns="symbol", values="close")

    if pivoted.shape[1] < 2:
        return issues

    rets = pivoted.pct_change().dropna()
    if len(rets) < 20:
        return issues

    # Rolling 20-bar pairwise correlations
    # rolling(20).corr() produces MultiIndex (timestamp, symbol)
    rolling_corr = rets.rolling(20, min_periods=10).corr()

    for i, col in enumerate(rets.columns):
        for j in range(i + 1, len(rets.columns)):
            col2 = rets.columns[j]
            pair_corr = rolling_corr.xs(col, level=1)[col2].dropna()
            if len(pair_corr) < 10:
                continue

            mean_corr = pair_corr.mean()
            min_corr = pair_corr.min()
            max_corr = pair_corr.max()

            if max_corr > 0.99:
                issues.append(
                    f"{col}/{col2}: near-perfect correlation ({max_corr:.3f}) "
                    f"— possible stale data"
                )
            if min_corr < -0.8:
                issues.append(
                    f"{col}/{col2}: extreme negative correlation ({min_corr:.3f})"
                )

    # Overall correlation matrix for the full period
    full_corr = rets.corr()
    print(f"\n  Full-period return correlation matrix:")
    for i, col in enumerate(full_corr.columns):
        row_vals = "  ".join(f"{full_corr.loc[col, c]:+.3f}" for c in full_corr.columns)
        print(f"    {col:6s}: {row_vals}")

    return issues


def check_forward_returns(df: pd.DataFrame) -> list[str]:
    """Check distribution of forward 1-bar returns for each symbol."""
    issues: list[str] = []
    print(f"\n  Per-symbol return statistics:")
    print(f"  {'Symbol':>6s} {'N':>6s} {'Mean':>8s} {'Std':>8s} {'Skew':>8s} "
          f"{'Min':>8s} {'Max':>8s} {'|R|>10%':>8s}")
    print(f"  {'-' * 56}")

    for sym in sorted(df["symbol"].unique()):
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        prices = sym_df["close"].values
        if len(prices) < 3:
            continue
        rets = (prices[1:] - prices[:-1]) / np.maximum(prices[:-1], 1e-10)
        n_extreme = int((np.abs(rets) > 0.10).sum())
        print(f"  {sym:>6s} {len(rets):>6d} {np.mean(rets):+8.4f} {np.std(rets):8.4f} "
              f"{pd.Series(rets).skew():+8.2f} {np.min(rets):+8.4f} {np.max(rets):+8.4f} "
              f"{n_extreme:>8d}")

        if np.max(np.abs(rets)) > 0.20:
            idx = np.argmax(np.abs(rets))
            ts = sym_df.iloc[idx]["timestamp"]
            issues.append(
                f"{sym}: extreme return {rets[idx]:+.2%} on {ts.date()}"
            )

    return issues


def main():
    print("=" * 70)
    print("  DIAGNOSTIC 4: DATA QUALITY VERIFICATION")
    print("=" * 70)

    all_issues: dict[str, list[str]] = {}

    # ── Test datasets ──
    datasets = [
        ("CTA Full (2018-2024)", "cta", FULL_PERIOD["start"], FULL_PERIOD["end"]),
        ("CTA COVID (2019-2021)", "cta", "2019-06-01", "2021-12-31"),
        ("RP Full (2018-2024)", "rp", FULL_PERIOD["start"], FULL_PERIOD["end"]),
    ]

    for label, universe, start, end in datasets:
        print(f"\n{'=' * 70}")
        print(f"  DATASET: {label}")
        print(f"  Universe: {universe}  Period: {start} -> {end}")
        print(f"{'=' * 70}")

        df = load_universe(universe, start, end)
        if df.empty:
            print(f"  ERROR: No data returned")
            continue

        n_rows = len(df)
        n_bars = df["timestamp"].nunique()
        n_symbols = df["symbol"].nunique()
        date_range = f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()}"
        print(f"  Rows: {n_rows}  Bars: {n_bars}  Symbols: {n_symbols}")
        print(f"  Date range: {date_range}")

        dataset_issues: list[str] = []

        # 1. Absurd returns
        absurd = check_absurd_returns(df)
        if absurd:
            dataset_issues.extend(absurd)
            for a in absurd:
                print(f"  [FAIL] Absurd return: {a}")
        else:
            print(f"  [PASS] No absurd returns (>|20%|)")

        # 2. Missing dates
        missing = check_missing_dates(df)
        if missing:
            dataset_issues.extend(missing)
            for m in missing:
                print(f"  [WARN] Missing dates: {m}")
        else:
            print(f"  [PASS] No excessive gaps (>5 business days)")

        # 3. Stale prices
        stale = check_stale_prices(df)
        if stale:
            dataset_issues.extend(stale)
            for s in stale:
                print(f"  [WARN] Stale prices: {s}")
        else:
            print(f"  [PASS] No stale prices (>3 consecutive identical closes)")

        # 4. Price consistency
        consistency = check_price_consistency(df)
        if consistency:
            dataset_issues.extend(consistency)
            for c in consistency:
                print(f"  [FAIL] Price inconsistency: {c}")
        else:
            print(f"  [PASS] OHLCV consistency checks passed")

        # 5. Volume anomalies
        vol = check_volume_anomalies(df)
        if vol:
            dataset_issues.extend(vol)
            for v in vol[:3]:
                print(f"  [WARN] Volume anomaly: {v}")
        else:
            print(f"  [PASS] No volume anomalies")

        # 6. Correlation structure
        print(f"\n  Cross-sectional correlation check:")
        corr = check_correlation_structure(df)
        if corr:
            dataset_issues.extend(corr)
            for c in corr:
                print(f"  [WARN] Correlation: {c}")
        else:
            print(f"  [PASS] Correlation structure normal")

        # 7. Return statistics
        print(f"\n  Return distribution check:")
        ret_issues = check_forward_returns(df)
        if ret_issues:
            dataset_issues.extend(ret_issues)
            for r in ret_issues:
                print(f"  [WARN] Return: {r}")
        else:
            print(f"  [PASS] Return distributions normal")

        all_issues[label] = dataset_issues

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("  DIAGNOSTIC 4 SUMMARY")
    print("=" * 70)

    total_issues = sum(len(v) for v in all_issues.values())
    if total_issues == 0:
        print("\n  [PASS] All data quality checks passed across all datasets.")
        print("  Data quality is NOT the cause of Phase A failures.")
    else:
        print(f"\n  [ISSUES] {total_issues} issue(s) found across {len(all_issues)} datasets:")
        for label, issues in all_issues.items():
            if issues:
                print(f"\n  {label}:")
                for issue in issues:
                    print(f"    - {issue}")

        critical = [i for v in all_issues.values() for i in v
                    if "FAIL" in i or "absurd" in i.lower() or "inconsistency" in i.lower()]
        if critical:
            print(f"\n  [FAIL] {len(critical)} critical data quality issue(s) detected.")
            print("  Data problems MAY contribute to Phase A failures.")
        else:
            print(f"\n  [PASS] No critical data quality issues.")
            print("  Minor warnings only — data quality is NOT the root cause.")

    print()


if __name__ == "__main__":
    main()
