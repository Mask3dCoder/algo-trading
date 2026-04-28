"""
Diagnostic 5: Win rate calculation verification.

Analyzes the raw OOS return distribution from the walk-forward backtest to
verify the win rate calculation and diagnose the signal structure.

Key questions:
1. Is win rate ~50% (symmetric distribution around zero)?
2. Are returns dominated by extreme events (fat tails)?
3. Is there a structural asymmetry in returns?
4. How does the return distribution explain the negative Sharpe?
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from quantcore.risk.metrics import sharpe_ratio, sortino_ratio, drawdown_stats
from scripts.real_data_backtest import STRATEGY_UNIVERSE, run_full_period_backtest, run_regime_backtest
from scripts.real_data_loader import load_universe


def analyze_return_distribution(
    returns: np.ndarray,
    label: str,
) -> dict:
    """Analyze the distribution of raw OOS returns."""
    n = len(returns)
    if n < 3:
        print(f"\n  {label}: insufficient returns ({n})")
        return {"n": n}

    mean_ret = float(np.mean(returns))
    median_ret = float(np.median(returns))
    std_ret = float(np.std(returns, ddof=1))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=True))  # excess kurtosis
    sharpe = sharpe_ratio(returns, annualize=False)

    n_pos = int((returns > 0).sum())
    n_neg = int((returns < 0).sum())
    n_zero = int((returns == 0).sum())
    win_rate = n_pos / n if n > 0 else 0.0

    # Sortino components
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0

    # Percentiles
    pctiles = {
        "p1": float(np.percentile(returns, 1)),
        "p5": float(np.percentile(returns, 5)),
        "p25": float(np.percentile(returns, 25)),
        "p50": float(np.percentile(returns, 50)),
        "p75": float(np.percentile(returns, 75)),
        "p95": float(np.percentile(returns, 95)),
        "p99": float(np.percentile(returns, 99)),
    }

    # Tail analysis
    threshold = 2.0 * std_ret if std_ret > 0 else 0.01
    n_positive_extreme = int((returns > threshold).sum())
    n_negative_extreme = int((returns < -threshold).sum())
    pos_extreme_pct = n_positive_extreme / n * 100
    neg_extreme_pct = n_negative_extreme / n * 100

    # Ratio of positive to negative total return
    total_pos = float(np.sum(returns[returns > 0]))
    total_neg = float(abs(np.sum(returns[returns < 0]))) if (returns < 0).any() else 0.0
    gain_loss_ratio = total_pos / total_neg if total_neg > 0 else float("inf")

    # Autocorrelation
    autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1]) if n > 1 else 0.0

    print(f"\n  {label}:")
    print(f"  {'=' * 50}")
    print(f"  N returns:           {n}")
    print(f"  Mean:                {mean_ret:+.6f}")
    print(f"  Median:              {median_ret:+.6f}")
    print(f"  Std:                 {std_ret:.6f}")
    print(f"  Skew:                {skew:+.4f}")
    print(f"  Excess Kurtosis:     {kurt:+.4f}")
    print(f"  Sharpe (daily):      {sharpe:+.4f}")
    print(f"  Win rate:            {win_rate:.2%}  ({n_pos}W / {n_neg}L / {n_zero}Z)")
    print(f"  Downside std:        {downside_std:.6f}")
    print(f"  Gain/Loss ratio:     {gain_loss_ratio:.3f}")
    print(f"  Autocorrelation:     {autocorr:+.4f}")

    print(f"\n  Percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    P{p:02d}: {pctiles[f'p{p}']:+.6f}")

    print(f"\n  Tails (>2sigma):")
    print(f"    Positive (>+{threshold:.4f}): {n_positive_extreme} ({pos_extreme_pct:.2f}%)")
    print(f"    Negative (<-{threshold:.4f}): {n_negative_extreme} ({neg_extreme_pct:.2f}%)")

    if n_positive_extreme > 0 and n_negative_extreme > 0:
        tail_ratio = n_positive_extreme / n_negative_extreme
        print(f"    Tail ratio (pos/neg): {tail_ratio:.3f}")

    # Normality test
    if n > 3:
        _, p_normal = stats.normaltest(returns)
        print(f"\n  Normality test p-value: {p_normal:.6f}")
        if p_normal < 0.05:
            print(f"  [INFO] Returns are NOT normally distributed (p={p_normal:.4f})")

    # Key findings
    findings = []

    if win_rate < 0.10:
        findings.append(f"Win rate {win_rate:.1%} is extremely low (< 10%)")
    elif win_rate < 0.30:
        findings.append(f"Win rate {win_rate:.1%} is low (10-30%)")
    elif win_rate > 0.70:
        findings.append(f"Win rate {win_rate:.1%} is high (> 70%)")

    if skew < -1.0:
        findings.append(f"Negative skew ({skew:.2f}) — strategy has left tail risk")
    elif skew > 1.0:
        findings.append(f"Positive skew ({skew:.2f}) — strategy has right tail upside")

    if kurt > 5:
        findings.append(f"Fat tails (kurtosis {kurt:.1f}) — extreme events dominate")
    elif kurt > 2:
        findings.append(f"Moderate tail thickness (kurtosis {kurt:.1f})")

    if gain_loss_ratio < 0.5:
        findings.append(f"Poor gain/loss ratio ({gain_loss_ratio:.2f}) — losses >> gains")
    elif gain_loss_ratio < 1.0:
        findings.append(f"Below-breakeven gain/loss ({gain_loss_ratio:.2f})")

    if abs(autocorr) > 0.10:
        findings.append(f"Non-trivial autocorrelation ({autocorr:+.3f})")

    return {
        "n": n,
        "mean": mean_ret,
        "median": median_ret,
        "std": std_ret,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "skew": skew,
        "kurtosis": kurt,
        "gain_loss_ratio": gain_loss_ratio,
        "findings": findings,
    }


def main():
    print("=" * 70)
    print("  DIAGNOSTIC 5: WIN RATE & RETURN DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # ── CTA Full Period ──
    print("\n" + "-" * 70)
    print("  CTA Full Period (2018-2024)")
    print("-" * 70)

    df_cta = load_universe("cta", "2018-01-01", "2024-12-31")
    cta_full = run_full_period_backtest("macro_cta_trend", df_cta)
    cta_rets = cta_full.get("oos_returns")

    cta_findings = []
    if cta_rets and len(cta_rets) > 3:
        cta_findings = analyze_return_distribution(np.array(cta_rets), "CTA Full Period").get("findings", [])
    else:
        print(f"  No returns: {cta_full.get('error')}")

    # ── CTA COVID Regime ──
    print("\n" + "-" * 70)
    print("  CTA COVID Regime (2020)")
    print("-" * 70)

    import pandas as pd

    df_covid = load_universe("cta", "2019-06-01", "2021-12-31")
    all_dates = sorted(df_covid["timestamp"].unique())
    covid_start = pd.Timestamp("2020-01-01")
    if hasattr(all_dates[0], "tz") and all_dates[0].tz is not None:
        covid_start = covid_start.tz_localize(all_dates[0].tz)
    warmup_idx = next(i for i, d in enumerate(all_dates) if d >= covid_start)

    covid_result = run_regime_backtest("macro_cta_trend", df_covid, warmup_idx)
    covid_rets = covid_result.get("oos_returns")

    covid_findings = []
    if covid_rets and len(covid_rets) > 3:
        covid_findings = analyze_return_distribution(np.array(covid_rets), "CTA COVID 2020").get("findings", [])
    else:
        print(f"  No returns: {covid_result.get('error')}")

    # ── Zero-return analysis: how many bars are ZERO (no signal) ──
    print("\n" + "-" * 70)
    print("  ZERO-RETURN ANALYSIS (bars with no active signal)")
    print("-" * 70)

    for label, rets in [
        ("CTA Full Period", cta_rets),
        ("CTA COVID", covid_rets),
    ]:
        if rets is None or len(rets) < 3:
            continue
        arr = np.array(rets)
        n_zero = int((arr == 0).sum())
        n_total = len(arr)
        n_nonzero = int((arr != 0).sum())
        pct_zero = n_zero / n_total * 100

        # Breakdown of return sources
        if n_nonzero > 0:
            nonzero = arr[arr != 0]
            pct_pos_nonzero = (nonzero > 0).sum() / n_nonzero * 100
            mean_nonzero = float(np.mean(nonzero))
            median_nonzero = float(np.median(nonzero))
            nonzero_sharpe = sharpe_ratio(nonzero, annualize=False) if len(nonzero) > 2 else 0
        else:
            pct_pos_nonzero = 0
            mean_nonzero = 0
            median_nonzero = 0
            nonzero_sharpe = 0

        print(f"\n  {label}:")
        print(f"    Total bars:        {n_total}")
        print(f"    Zero-return bars:  {n_zero} ({pct_zero:.1f}%)")
        print(f"    Non-zero bars:     {n_nonzero}")
        if n_nonzero > 0:
            print(f"    Non-zero win rate: {pct_pos_nonzero:.1f}%")
            print(f"    Non-zero mean:     {mean_nonzero:+.6f}")
            print(f"    Non-zero median:   {median_nonzero:+.6f}")
            print(f"    Non-zero Sharpe:   {nonzero_sharpe:+.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 5 SUMMARY")
    print("=" * 70)

    all_findings = cta_findings + covid_findings

    print(f"\n  Key findings:")
    for f in all_findings:
        print(f"    - {f}")

    # Win rate interpretation
    print(f"\n  Win rate check:")
    print(f"    CTA Full:  {cta_full.get('win_rate', 0):.2%}  "
          f"(reported by run_full_period_backtest)")

    if cta_rets and len(cta_rets) > 3:
        arr = np.array(cta_rets)
        computed_wr = float(np.mean(arr > 0))
        reported_wr = cta_full.get("win_rate", 0)
        wr_delta = abs(computed_wr - reported_wr)
        print(f"    Computed:  {computed_wr:.2%}")
        if wr_delta < 1e-10:
            print(f"    [PASS] Win rate matches: {wr_delta:.2e} delta")
        else:
            print(f"    [FAIL] Win rate MISMATCH: {wr_delta:.2e} delta")

    print()


if __name__ == "__main__":
    main()
