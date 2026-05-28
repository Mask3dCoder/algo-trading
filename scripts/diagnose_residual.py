"""Diagnostic: trace factor_residual_v3 signal computation step-by-step.

Loads one month of data, runs the regression for each ETF, prints every
intermediate value to find the inversion bug causing 2.8% win rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]
START = "2019-06-01"
END = "2021-12-31"
REG_WINDOW = 252
RIDGE_ALPHA = 1e-6
RESIDUAL_WINDOW = 63
FACTOR_ASSETS = {"SPY", "TLT", "DBC", "GLD"}

def main():
    # 1. Download data
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True,
                        group_by="ticker", progress=False)
    closes: dict[str, pd.Series] = {}
    for t in TICKERS:
        closes[t] = data[t]["Close"].squeeze().astype(float)
    close_df = pd.DataFrame(closes).dropna()
    symbols_list = sorted(close_df.columns.tolist())
    n_assets = len(symbols_list)

    print(f"Symbols: {symbols_list}")
    print(f"Total bars: {len(close_df)}")
    print()

    # 2. Compute daily returns
    close_arr = close_df.values.astype(np.float64)
    rets = (close_arr[1:] - close_arr[:-1]) / np.maximum(close_arr[:-1], 1e-10)
    n_periods = rets.shape[0]

    # 3. Build factors (same as v3)
    sym_to_idx = {s: i for i, s in enumerate(symbols_list)}
    factor_defs = {
        "MKT": ("SPY", None, 1.0),
        "RATES": ("TLT", "SPY", 1.0),
        "INFL": ("DBC", None, 1.0),
        "SAFE": ("GLD", "TLT", 0.5),
    }
    factor_map = {}
    for fname, (sym1, sym2, scale) in factor_defs.items():
        idx1 = sym_to_idx.get(sym1, -1)
        idx2 = sym_to_idx.get(sym2) if sym2 else None
        if idx1 >= 0:
            factor_map[fname] = (idx1, idx2, scale)

    n_factors = len(factor_map)
    factors = np.zeros((n_periods, n_factors))
    for j, (fname, (idx1, idx2, scale)) in enumerate(sorted(factor_map.items())):
        if idx2 is not None:
            factors[:, j] = (rets[:, idx1] - rets[:, idx2]) * scale
        else:
            factors[:, j] = rets[:, idx1] * scale

    # 4. Pick a rebalance date: first bar after reg_window
    #    We'll analyze the residuals at bar = REG_WINDOW (first valid residual)
    t = REG_WINDOW  # Index of first valid residual

    print("=" * 100)
    print(f"DIAGNOSTIC: Residual computation at bar {t} (date ~ {close_df.index[t+1].date()})")
    print("=" * 100)

    # Run regression for each asset
    X_train = factors[t-REG_WINDOW:t]
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0, ddof=1)
    X_std_safe = np.maximum(X_std, 1e-10)
    X_train_std = (X_train - X_mean) / X_std_safe
    X_t_std = (factors[t] - X_mean) / X_std_safe

    I_reg = RIDGE_ALPHA * np.eye(n_factors)
    XtX_inv = np.linalg.inv(X_train_std.T @ X_train_std + I_reg)

    print(f"\n{'Asset':<8s} {'Actual Ret':>10s} {'Pred Ret':>10s} {'Residual':>10s} {'Signal':>8s} {'Dir':>6s}")
    print("-" * 60)

    signals = np.zeros(n_assets)
    asset_results = []

    for i, sym in enumerate(symbols_list):
        y_train = rets[t-REG_WINDOW:t, i]
        if np.std(y_train) < 1e-10:
            print(f"{sym:<8s} {'ZERO_VOL':>10s}")
            continue

        # BUGFIX: Add intercept by demeaning y_train
        y_mean = np.mean(y_train)
        y_train_dm = y_train - y_mean
        beta = XtX_inv @ (X_train_std.T @ y_train_dm)
        y_pred = float(X_t_std @ beta) + y_mean
        actual = float(rets[t, i])
        residual = actual - y_pred

        # Signal computation (same as v3 with bugfixes)
        if sym in FACTOR_ASSETS:
            sig = 0.0
        else:
            window = min(RESIDUAL_WINDOW, t)
            resid_window = rets[max(0, t-window):t, i]
            valid_std = np.std(resid_window)
            # BUGFIX: Use trailing average residual (smoothing)
            resid_vals = []
            for tt in range(max(0, t-5), t):
                if tt < REG_WINDOW:
                    continue
                X_tr_tmp = factors[tt-REG_WINDOW:tt]
                X_m_tmp = np.mean(X_tr_tmp, axis=0)
                X_s_tmp = np.maximum(np.std(X_tr_tmp, axis=0, ddof=1), 1e-10)
                X_tr_s_tmp = (X_tr_tmp - X_m_tmp) / X_s_tmp
                XtX_i_tmp = np.linalg.inv(X_tr_s_tmp.T @ X_tr_s_tmp + I_reg)
                y_tr_tmp = rets[tt-REG_WINDOW:tt, i]
                if np.std(y_tr_tmp) < 1e-10:
                    continue
                y_m = np.mean(y_tr_tmp)
                beta_tmp = XtX_i_tmp @ (X_tr_s_tmp.T @ (y_tr_tmp - y_m))
                y_p_tmp = float(((factors[tt] - X_m_tmp) / X_s_tmp) @ beta_tmp) + y_m
                resid_vals.append(float(rets[tt, i]) - y_p_tmp)

            if len(resid_vals) >= 2:
                sig = np.mean(resid_vals) / max(valid_std, 1e-10)
            else:
                sig = residual / max(valid_std, 1e-10) if valid_std > 1e-10 else 0.0

        signals[i] = sig
        direction = 1 if sig > 0 else -1

        print(f"{sym:<8s} {actual:>+10.6f} {y_pred:>+10.6f} {residual:>+10.6f} {sig:>+8.4f} {'LONG' if direction > 0 else 'SHORT':>6s}")

        asset_results.append({
            "symbol": sym,
            "actual": actual,
            "predicted": y_pred,
            "residual": residual,
            "signal": sig,
            "direction": direction,
            "beta": beta,
        })

    # 5. Check weight computation
    print("\n" + "=" * 60)
    print("WEIGHT COMPUTATION")
    print("=" * 60)

    sig_mean = np.mean(signals)
    sig_std = np.std(signals, ddof=1)
    raw_weights = (signals - sig_mean) / max(sig_std, 1e-10)

    # BUGFIX: Zero out factor asset weights (they have signal=0 but get pulled by z-score)
    for i, sym in enumerate(symbols_list):
        if sym in FACTOR_ASSETS:
            raw_weights[i] = 0.0

    # Vol scaling
    vol = np.std(rets[-20:], axis=0, ddof=1) * np.sqrt(252)
    vol = np.maximum(vol, 1e-10)
    vol_scaled = raw_weights / vol

    # Re-center only non-factor assets
    non_factor_mask = [s not in FACTOR_ASSETS for s in symbols_list]
    non_factor_vals = vol_scaled[non_factor_mask]
    centered = vol_scaled.copy()
    centered[non_factor_mask] -= np.mean(non_factor_vals)
    # Zero factor assets
    centered[~np.array(non_factor_mask)] = 0.0

    # Normalize
    max_abs = np.max(np.abs(centered))
    if max_abs > 1e-10:
        final_weights = centered / max_abs
    else:
        final_weights = centered

    print(f"\n{'Asset':<8s} {'Signal':>8s} {'Raw Wgt':>8s} {'Vol-Scaled':>10s} {'Final Wgt':>10s} {'Dir':>6s}")
    print("-" * 60)

    long_count = 0
    short_count = 0
    for i, sym in enumerate(symbols_list):
        dir_str = "LONG" if final_weights[i] > 0 else "SHORT"
        if final_weights[i] > 0:
            long_count += 1
        elif final_weights[i] < 0:
            short_count += 1
        print(f"{sym:<8s} {signals[i]:>+8.4f} {raw_weights[i]:>+8.4f} {vol_scaled[i]:>+10.6f} {final_weights[i]:>+10.6f} {dir_str:>6s}")

    print(f"\nLongs: {long_count}, Shorts: {short_count}, Neutrals: {n_assets - long_count - short_count}")

    # 6. Verify long is highest residual, short is lowest
    non_factor_results = [r for r in asset_results if r["symbol"] not in FACTOR_ASSETS]
    if non_factor_results:
        residuals_sorted = sorted(non_factor_results, key=lambda r: r["residual"], reverse=True)
        print(f"\n{'='*60}")
        print(f"NON-FACTOR ASSETS RANKED BY RESIDUAL")
        print(f"{'='*60}")
        print(f"{'Rank':<6s} {'Asset':<8s} {'Residual':>10s} {'Expected Dir':>14s}")
        print("-" * 40)
        for rank, r in enumerate(residuals_sorted):
            expected_dir = "LONG (top)" if rank < 2 else ("SHORT (bottom)" if rank >= len(residuals_sorted) - 2 else "NEUTRAL")
            print(f"{rank+1:<6d} {r['symbol']:<8s} {r['residual']:>+10.6f} {expected_dir:>14s}")

    # 7. Check for dollar neutrality (sum of weights should be ~0)
    print(f"\n{'='*60}")
    print(f"DOLLAR NEUTRALITY CHECK")
    print(f"{'='*60}")
    print(f"Sum of final weights: {np.sum(final_weights):+.8f} (should be ~0)")
    print(f"Sum of positive weights: {np.sum(final_weights[final_weights > 0])/2:.6f}")

    # 8. Check if signal direction is inverted
    # For the month starting at this rebalance, compute next-month returns
    if t + 21 < n_periods:
        next_month_rets = rets[t:t+21]
        actual_asset_returns = np.mean(next_month_rets, axis=0) * 21  # approx monthly

        print(f"\n{'='*60}")
        print(f"FOLLOWING MONTH RETURNS vs SIGNAL DIRECTION")
        print(f"{'='*60}")
        print(f"Checking if LONG assets actually outperform SHORT assets...")

        long_assets = [sym for i, sym in enumerate(symbols_list) if final_weights[i] > 0 and symbols_list[i] not in FACTOR_ASSETS]
        short_assets = [sym for i, sym in enumerate(symbols_list) if final_weights[i] < 0 and symbols_list[i] not in FACTOR_ASSETS]

        print(f"\nLong assets ({len(long_assets)}): {long_assets}")
        print(f"Short assets ({len(short_assets)}): {short_assets}")

        if long_assets and short_assets:
            long_rets = [actual_asset_returns[symbols_list.index(s)] for s in long_assets]
            short_rets = [actual_asset_returns[symbols_list.index(s)] for s in short_assets]
            print(f"Avg long return (next 21d): {np.mean(long_rets)*100:+.4f}%")
            print(f"Avg short return (next 21d): {np.mean(short_rets)*100:+.4f}%")
            print(f"Long-Short spread: {(np.mean(long_rets) - np.mean(short_rets))*100:+.4f}%")
            if np.mean(long_rets) < np.mean(short_rets):
                print("\n*** WARNING: Long assets underperform short! Signal is INVERTED! ***")
            elif np.mean(long_rets) > np.mean(short_rets):
                print("\nSignal direction CORRECT: Longs outperform shorts in following month.")

    # 9. Run the full signal pipeline as it would in the backtest
    print(f"\n{'='*60}")
    print(f"FULL SIGNAL HISTORY")
    print(f"{'='*60}")

    # Count how many times each non-factor asset is long vs short
    long_counts = {sym: 0 for sym in symbols_list if sym not in FACTOR_ASSETS}
    short_counts = {sym: 0 for sym in symbols_list if sym not in FACTOR_ASSETS}

    # Run every rebalance_freq days from reg_window to end
    rebalance_freq = 5
    for check_t in range(REG_WINDOW, n_periods, rebalance_freq):
        X_tr = factors[check_t-REG_WINDOW:check_t]
        X_m = np.mean(X_tr, axis=0)
        X_s = np.std(X_tr, axis=0, ddof=1)
        X_s = np.maximum(X_s, 1e-10)
        X_tr_s = (X_tr - X_m) / X_s
        X_c_s = (factors[check_t] - X_m) / X_s
        XtX_i = np.linalg.inv(X_tr_s.T @ X_tr_s + I_reg)

        sigs = np.zeros(n_assets)
        for i, sym in enumerate(symbols_list):
            y_tr = rets[check_t-REG_WINDOW:check_t, i]
            if np.std(y_tr) < 1e-10:
                continue
            y_m = np.mean(y_tr)
            beta = XtX_i @ (X_tr_s.T @ (y_tr - y_m))
            y_p = float(X_c_s @ beta) + y_m
            res = float(rets[check_t, i]) - y_p

            if sym in FACTOR_ASSETS:
                sigs[i] = 0.0
            else:
                wdw = min(RESIDUAL_WINDOW, check_t)
                rw = rets[max(0, check_t-wdw):check_t, i]
                s = np.std(rw)
                # BUGFIX: trailing average of residuals
                resid_vals = []
                for tt in range(max(REG_WINDOW, check_t-5), check_t):
                    X_tr2 = factors[tt-REG_WINDOW:tt]
                    X_m2 = np.mean(X_tr2, axis=0)
                    X_s2 = np.maximum(np.std(X_tr2, axis=0, ddof=1), 1e-10)
                    X_tr2_s = (X_tr2 - X_m2) / X_s2
                    XtX2_i = np.linalg.inv(X_tr2_s.T @ X_tr2_s + I_reg)
                    y_tr2 = rets[tt-REG_WINDOW:tt, i]
                    if np.std(y_tr2) < 1e-10:
                        continue
                    y_m2 = np.mean(y_tr2)
                    beta2 = XtX2_i @ (X_tr2_s.T @ (y_tr2 - y_m2))
                    y_p2 = float(((factors[tt] - X_m2) / X_s2) @ beta2) + y_m2
                    resid_vals.append(float(rets[tt, i]) - y_p2)
                if len(resid_vals) >= 2:
                    sigs[i] = np.mean(resid_vals) / max(s, 1e-10)
                else:
                    sigs[i] = res / max(s, 1e-10) if s > 1e-10 else 0.0

        for i, sym in enumerate(symbols_list):
            if sym not in FACTOR_ASSETS:
                if sigs[i] > 0:
                    long_counts[sym] += 1
                elif sigs[i] < 0:
                    short_counts[sym] += 1

    print(f"\n{'Symbol':<8s} {'Long count':>12s} {'Short count':>14s} {'Long%':>8s}")
    print("-" * 44)
    for sym in sorted(long_counts.keys()):
        total = long_counts[sym] + short_counts[sym]
        pct = long_counts[sym] / total * 100 if total > 0 else 0
        print(f"{sym:<8s} {long_counts[sym]:>12d} {short_counts[sym]:>14d} {pct:>7.1f}%")


if __name__ == "__main__":
    main()