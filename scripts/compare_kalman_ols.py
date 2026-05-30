"""Compare OLS vs Kalman filter hedge ratios for pairs trading.

Runs the pairs trading strategy with both hedge ratio methods on the
same data and compares:
  - Spread stationarity (ADF test p-value)
  - Out-of-sample Sharpe of the resulting trading signals
  - Turnover (Kalman should reduce unnecessary rebalancing)

Usage:
    python scripts/compare_kalman_ols.py
    python scripts/compare_kalman_ols.py --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _load_spy_tlt(start: str, end: str) -> pd.DataFrame:
    """Load SPY and TLT data as a representative pair."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, progress=False)
    tlt = yf.download("TLT", start=start, end=end, progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)

    df = pd.DataFrame({
        "SPY": spy["Close"],
        "TLT": tlt["Close"],
    }).dropna()

    return df


def _compute_spread_ols(
    y: np.ndarray,
    x: np.ndarray,
    min_window: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spread using expanding-window OLS."""
    n = len(y)
    hedge_ratios = np.full(n, np.nan)
    spreads = np.full(n, np.nan)

    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    count = 0

    for i in range(n):
        if np.isnan(y[i]) or np.isnan(x[i]):
            continue

        count += 1
        sum_x += x[i]
        sum_y += y[i]
        sum_xx += x[i] ** 2
        sum_xy += x[i] * y[i]

        if count >= min_window:
            denom = count * sum_xx - sum_x ** 2
            if abs(denom) > 1e-10:
                hr = (count * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - hr * sum_x) / count
                hedge_ratios[i] = hr
                spreads[i] = y[i] - intercept - hr * x[i]
            elif i > 0:
                hedge_ratios[i] = hedge_ratios[i - 1]
                spreads[i] = spreads[i - 1]
        elif i > 0 and not np.isnan(hedge_ratios[i - 1]):
            hedge_ratios[i] = hedge_ratios[i - 1]
            spreads[i] = y[i] - hedge_ratios[i] * x[i]

    return hedge_ratios, spreads


def _compute_spread_kalman(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-4,
    Ve: float = 1e-1,
    warmup: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spread using Kalman filter."""
    from quantcore.signals.statistical.kalman_hedge import kalman_hedge_ratio_series

    betas, spreads, uncertainties = kalman_hedge_ratio_series(
        y, x, delta=delta, Ve=Ve, warmup=warmup,
    )
    return betas, spreads, uncertainties


def _trading_signals(
    zscore: np.ndarray,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
) -> np.ndarray:
    """Generate trading signals from z-score."""
    n = len(zscore)
    signal = np.zeros(n)
    position = 0.0

    for i in range(n):
        z = zscore[i]
        if np.isnan(z):
            signal[i] = position
            continue

        if abs(z) > stop:
            position = 0.0
        elif z > entry and position <= 0:
            position = -1.0
        elif z < -entry and position >= 0:
            position = 1.0
        elif abs(z) < exit:
            position = 0.0

        signal[i] = position

    return signal


def _rolling_zscore(spread: np.ndarray, lookback: int = 60) -> np.ndarray:
    """Rolling z-score of spread."""
    n = len(spread)
    result = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            m = np.mean(valid)
            s = np.std(valid, ddof=1)
            if s > 0 and not np.isnan(spread[i]):
                result[i] = (spread[i] - m) / s
    return result


def _compute_returns(
    signal: np.ndarray,
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    hedge_ratios: np.ndarray,
    tc_bps: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Compute strategy returns from signals."""
    n = len(signal)
    returns = np.zeros(n)
    turnover = 0.0

    for i in range(1, n):
        if np.isnan(hedge_ratios[i]) or np.isnan(hedge_ratios[i - 1]):
            continue

        # Daily return from the spread position
        y_ret = (y_prices[i] - y_prices[i - 1]) / max(y_prices[i - 1], 1e-10)
        x_ret = (x_prices[i] - x_prices[i - 1]) / max(x_prices[i - 1], 1e-10)

        # Spread return: long A, short beta*B
        spread_ret = y_ret - hedge_ratios[i] * x_ret
        returns[i] = signal[i] * spread_ret

        # Transaction cost on signal changes
        if signal[i] != signal[i - 1]:
            cost = abs(signal[i] - signal[i - 1]) * tc_bps / 10_000.0
            returns[i] -= cost
            turnover += abs(signal[i] - signal[i - 1])

    return returns, turnover


def _adf_test(spread: np.ndarray) -> float:
    """Augmented Dickey-Fuller test p-value."""
    from statsmodels.tsa.stattools import adfuller

    valid = spread[~np.isnan(spread)]
    if len(valid) < 50:
        return 1.0

    try:
        result = adfuller(valid, maxlag=12, autolag="AIC", regression="c")
        return float(result[1])
    except Exception:
        return 1.0


def run_comparison(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    oos_start: str = "2023-01-01",
) -> dict:
    """Run the OLS vs Kalman comparison."""
    print("Loading SPY/TLT data...")
    df = _load_spy_tlt(start, end)

    y = np.log(df["SPY"].values)
    x = np.log(df["TLT"].values)
    dates = df.index
    n = len(y)

    print(f"Data: {n} bars from {dates[0].date()} to {dates[-1].date()}")

    # Split: train on data before oos_start, test on rest
    oos_mask = dates >= pd.Timestamp(oos_start)
    train_mask = ~oos_mask

    # --- OLS ---
    print("\nComputing OLS hedge ratios...")
    t0 = time.time()
    ols_hr, ols_spread = _compute_spread_ols(y, x, min_window=252)
    ols_time = time.time() - t0

    # --- Kalman ---
    print("Computing Kalman hedge ratios...")
    t0 = time.time()
    kal_hr, kal_spread, kal_uncert = _compute_spread_kalman(
        y, x, delta=1e-4, Ve=1e-1, warmup=60,
    )
    kal_time = time.time() - t0

    # --- Stationarity tests ---
    ols_adf = _adf_test(ols_spread[oos_mask])
    kal_adf = _adf_test(kal_spread[oos_mask])

    # --- Trading signals ---
    ols_z = _rolling_zscore(ols_spread, lookback=60)
    kal_z = _rolling_zscore(kal_spread, lookback=60)

    ols_signal = _trading_signals(ols_z)
    kal_signal = _trading_signals(kal_z)

    # --- Returns (OOS only) ---
    spy_prices = df["SPY"].values
    tlt_prices = df["TLT"].values

    # Compute returns on full series, then slice to OOS
    ols_rets, ols_turn = _compute_returns(ols_signal, spy_prices, tlt_prices, ols_hr)
    kal_rets, kal_turn = _compute_returns(kal_signal, spy_prices, tlt_prices, kal_hr)

    # OOS slice
    oos_idx = np.where(oos_mask)[0]
    if len(oos_idx) == 0:
        return {"error": "No OOS data"}

    ols_oos = ols_rets[oos_idx]
    kal_oos = kal_rets[oos_idx]

    # --- Metrics ---
    def _sharpe(r: np.ndarray) -> float:
        r_clean = r[~np.isnan(r)]
        if len(r_clean) < 20 or np.std(r_clean, ddof=1) < 1e-12:
            return 0.0
        return float(np.mean(r_clean) / np.std(r_clean, ddof=1) * np.sqrt(252))

    def _max_dd(r: np.ndarray) -> float:
        eq = np.cumprod(1 + r[~np.isnan(r)])
        if len(eq) == 0:
            return 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
        return float(abs(np.min(dd)))

    ols_sharpe = _sharpe(ols_oos)
    kal_sharpe = _sharpe(kal_oos)
    ols_dd = _max_dd(ols_oos)
    kal_dd = _max_dd(kal_oos)

    # Turnover: normalize by number of OOS bars
    n_oos = len(oos_idx)
    ols_turn_norm = ols_turn / max(n_oos, 1) * 252  # annualized
    kal_turn_norm = kal_turn / max(n_oos, 1) * 252

    results = {
        "data_period": f"{start} to {end}",
        "oos_period": f"{oos_start} to {end}",
        "n_oos_bars": n_oos,
        "ols": {
            "sharpe": round(ols_sharpe, 4),
            "max_drawdown": round(ols_dd, 4),
            "annual_turnover": round(ols_turn_norm, 2),
            "adf_pvalue": round(ols_adf, 6),
            "computation_time_ms": round(ols_time * 1000, 1),
        },
        "kalman": {
            "sharpe": round(kal_sharpe, 4),
            "max_drawdown": round(kal_dd, 4),
            "annual_turnover": round(kal_turn_norm, 2),
            "adf_pvalue": round(kal_adf, 6),
            "computation_time_ms": round(kal_time * 1000, 1),
            "avg_beta_uncertainty": round(float(np.nanmean(kal_uncert[oos_mask])), 6),
        },
        "improvement": {
            "sharpe_delta": round(kal_sharpe - ols_sharpe, 4),
            "turnover_reduction_pct": round(
                (1 - kal_turn_norm / max(ols_turn_norm, 1e-10)) * 100, 1
            ),
            "stationarity_improvement": round(ols_adf - kal_adf, 6),
        },
        "recommendation": "kalman" if (kal_sharpe >= ols_sharpe or kal_turn_norm < ols_turn_norm) else "ols",
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OLS vs Kalman hedge ratios")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--oos-start", default="2023-01-01")
    args = parser.parse_args()

    results = run_comparison(args.start, args.end, args.oos_start)

    print("\n" + "=" * 60)
    print("  OLS vs KALMAN HEDGE RATIO COMPARISON")
    print("=" * 60)

    if "error" in results:
        print(f"  Error: {results['error']}")
        return

    print(f"\n  Data: {results['data_period']}")
    print(f"  OOS:  {results['oos_period']} ({results['n_oos_bars']} bars)")

    print(f"\n  {'Metric':<30} {'OLS':>12} {'Kalman':>12} {'Delta':>12}")
    print(f"  {'-'*66}")
    print(f"  {'OOS Sharpe':<30} {results['ols']['sharpe']:>+12.4f} {results['kalman']['sharpe']:>+12.4f} {results['improvement']['sharpe_delta']:>+12.4f}")
    print(f"  {'Max Drawdown':<30} {results['ols']['max_drawdown']:>12.4f} {results['kalman']['max_drawdown']:>12.4f} {results['kalman']['max_drawdown'] - results['ols']['max_drawdown']:>+12.4f}")
    print(f"  {'Annual Turnover':<30} {results['ols']['annual_turnover']:>12.2f} {results['kalman']['annual_turnover']:>12.2f} {results['improvement']['turnover_reduction_pct']:>+11.1f}%")
    print(f"  {'ADF p-value (spread)':<30} {results['ols']['adf_pvalue']:>12.6f} {results['kalman']['adf_pvalue']:>12.6f} {results['improvement']['stationarity_improvement']:>+12.6f}")

    print(f"\n  Recommendation: {results['recommendation'].upper()}")

    # Save report
    report_path = REPO_ROOT / "reports" / "tier1_kalman_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Kalman vs OLS Hedge Ratio Comparison",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Setup",
        "",
        f"- **Pair**: SPY / TLT (log prices)",
        f"- **Data period**: {results['data_period']}",
        f"- **OOS period**: {results['oos_period']}",
        f"- **OOS bars**: {results['n_oos_bars']}",
        f"- **Signal**: z-score entry at 2.0, exit at 0.5, stop at 4.0",
        f"- **Costs**: 5 bps per turnover",
        "",
        "## Results",
        "",
        "| Metric | OLS | Kalman | Delta |",
        "|--------|-----|--------|-------|",
        f"| OOS Sharpe | {results['ols']['sharpe']:+.4f} | {results['kalman']['sharpe']:+.4f} | {results['improvement']['sharpe_delta']:+.4f} |",
        f"| Max Drawdown | {results['ols']['max_drawdown']:.4f} | {results['kalman']['max_drawdown']:.4f} | {results['kalman']['max_drawdown'] - results['ols']['max_drawdown']:+.4f} |",
        f"| Annual Turnover | {results['ols']['annual_turnover']:.2f} | {results['kalman']['annual_turnover']:.2f} | {results['improvement']['turnover_reduction_pct']:+.1f}% |",
        f"| ADF p-value | {results['ols']['adf_pvalue']:.6f} | {results['kalman']['adf_pvalue']:.6f} | {results['improvement']['stationarity_improvement']:+.6f} |",
        f"| Avg beta uncertainty | N/A | {results['kalman'].get('avg_beta_uncertainty', 'N/A')} | — |",
        "",
        "## Conclusion",
        "",
        f"**Recommendation**: Use **{results['recommendation'].upper()}** hedge ratios.",
        "",
    ]

    if results["recommendation"] == "kalman":
        lines.extend([
            "The Kalman filter provides:",
            "- Time-varying hedge ratio that adapts to structural breaks",
            "- Built-in uncertainty measure (beta confidence interval)",
            f"- {'Lower' if results['kalman']['annual_turnover'] < results['ols']['annual_turnover'] else 'Similar'} turnover "
            f"({results['improvement']['turnover_reduction_pct']:+.1f}%)",
            "",
            "Setting `use_kalman: true` as default in `basket_spread_signal`.",
        ])
    else:
        lines.extend([
            "OLS performed comparably or better for this pair/period.",
            "The Kalman filter's added complexity did not provide material improvement.",
            "Keeping `use_kalman: false` as default.",
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
