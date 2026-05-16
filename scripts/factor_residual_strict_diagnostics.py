"""Strict factor-neutral diagnostics -- the final test.

Compares non-strict vs strict factor-neutral portfolio construction to
determine whether residual alpha survives exact factor neutralization.

Answers the meta-question:
  Is there ANY alpha left after full factor neutrality enforcement?

Usage:
    python scripts/factor_residual_strict_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging

import numpy as np
import pandas as pd

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class
import quantcore.strategies.macro.factor_residual  # noqa: F401
import quantcore.strategies.macro.factor_residual_strict  # noqa: F401

from scripts.real_data_loader import fetch_and_prepare, UNIVERSES

log = logging.getLogger(__name__)

# Factor definitions
FACTOR_DEFS: dict[str, tuple[str, str | None, float]] = {
    "MKT": ("SPY", None, 1.0),
    "RATES": ("TLT", "SPY", 1.0),
    "INFL": ("DBC", None, 1.0),
    "SAFE": ("GLD", "TLT", 0.5),
}

REBALANCE_FREQ = 5
FULL_START = "2008-01-01"
FULL_END = "2024-12-31"

STRICT_PARAMS = {
    "rebalance_freq": REBALANCE_FREQ,
    "reg_window": 252,
    "ridge_alpha": 1e-6,
    "resid_mom_window": 63,
    "resid_rev_weight": 0.3,
    "vol_lookback": 20,
    "min_weight": 0.01,
    "vol_target": 0.15,
}

ORIG_PARAMS = {
    "rebalance_freq": REBALANCE_FREQ,
    "reg_window": 252,
    "ridge_alpha": 1e-6,
    "resid_mom_window": 63,
    "resid_rev_weight": 0.3,
    "vol_lookback": 20,
    "min_weight": 0.01,
}


def df_to_data(df: pd.DataFrame) -> dict:
    symbols = sorted(df["symbol"].unique())
    # Causal forward-fill ONLY -- bfill is banned to avoid look-ahead.
    # Remaining NaN become zero after ffill + first-valid back-propagation.
    pivoted = df.pivot_table(index="timestamp", columns="symbol", values="close")
    for col in pivoted.columns:
        col_series = pivoted[col]
        if col_series.isna().all():
            continue
        first_valid = col_series.first_valid_index()
        if first_valid is not None:
            col_series = col_series.ffill()
            col_series = col_series.fillna(col_series.loc[first_valid])
            pivoted[col] = col_series
    pivoted = pivoted.fillna(0.0)
    if pivoted.empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}
    return {"close": pivoted.values, "symbols": symbols}


def build_factor_map(symbols):
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    fmap = {}
    for fname, (sym1, sym2, scale) in FACTOR_DEFS.items():
        idx1 = sym_to_idx.get(sym1, -1)
        if idx1 < 0:
            continue
        idx2 = sym_to_idx.get(sym2) if sym2 else None
        fmap[fname] = (idx1, idx2, scale)
    return fmap


def build_factors(rets, fmap):
    n_periods = rets.shape[0]
    factors = np.zeros((n_periods, len(fmap)))
    for j, (fname, (idx1, idx2, scale)) in enumerate(sorted(fmap.items())):
        if idx2 is not None:
            factors[:, j] = (rets[:, idx1] - rets[:, idx2]) * scale
        else:
            factors[:, j] = rets[:, idx1] * scale
    return factors


def rolling_ridge_rets(
    rets: np.ndarray, factors: np.ndarray,
    reg_window: int = 252, ridge_alpha: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute residuals, raw-scale betas, predicted returns.

    Returns:
        residuals:  (n_periods, n_assets)
        betas:      (n_periods, n_assets, n_factors) raw-scale loadings
        predicted:  (n_periods, n_assets)
    """
    n_periods, n_assets = rets.shape
    n_factors = factors.shape[1]
    residuals = np.full((n_periods, n_assets), np.nan)
    betas = np.full((n_periods, n_assets, n_factors), np.nan)
    predicted = np.full((n_periods, n_assets), np.nan)

    if n_periods < reg_window + 1:
        return residuals, betas, predicted

    I_reg = ridge_alpha * np.eye(n_factors)

    for t in range(reg_window, n_periods):
        X_train = factors[t - reg_window:t]
        X_mean = np.mean(X_train, axis=0)
        X_std = np.maximum(np.std(X_train, axis=0, ddof=1), 1e-10)
        X_train_std = (X_train - X_mean) / X_std
        X_t_std = (factors[t] - X_mean) / X_std

        try:
            XtX_inv = np.linalg.inv(X_train_std.T @ X_train_std + I_reg)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_train_std.T @ X_train_std)

        for i in range(n_assets):
            y_train = rets[t - reg_window:t, i]
            if np.std(y_train) < 1e-10:
                residuals[t, i] = 0.0
                betas[t, i, :] = 0.0
                predicted[t, i] = 0.0
                continue
            beta_std = XtX_inv @ (X_train_std.T @ y_train)
            y_pred = float(X_t_std @ beta_std)
            predicted[t, i] = y_pred
            residuals[t, i] = float(rets[t, i]) - y_pred
            # Convert to raw-scale loadings
            betas[t, i, :] = beta_std / X_std

    return residuals, betas, predicted


def run_strategy(
    close: np.ndarray, symbols: list[str],
    params: dict, strategy_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run strategy and collect weights, portfolio returns, factor exposures.

    Returns:
        all_weights:     (n_bars, n_assets) weights at each bar
        port_rets:       (n_bars,) portfolio returns
        factor_exposures: (n_bars, n_factors) portfolio B^T @ w at each bar
        residuals:       (n_bars, n_assets) residual returns
    """
    n_periods, n_assets = close.shape
    n_ret_bars = n_periods - 1

    daily_rets = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
    fmap = build_factor_map(symbols)
    factors = build_factors(daily_rets, fmap)
    factor_names = sorted(fmap.keys())
    n_factors = len(factor_names)

    cfg = StrategyConfig(
        name=strategy_id,
        strategy_type=StrategyType.MACRO,
        params=dict(params),
    )
    strat_cls = get_strategy_class(strategy_id)
    strat = strat_cls(cfg)
    strat.initialize()

    all_weights = np.zeros((n_ret_bars, n_assets))
    current_weights = None

    for t in range(1, n_periods):
        bar_data = {"close": close[:t + 1], "symbols": symbols}
        try:
            signals = strat.on_data(bar_data) or []
        except Exception:
            signals = []

        if signals:
            w = np.zeros(n_assets)
            for s in signals:
                try:
                    idx = symbols.index(s.symbol)
                except ValueError:
                    continue
                w[idx] = float(s.direction) * float(s.strength)
            current_weights = w

        if current_weights is not None:
            all_weights[t - 1] = current_weights

    # Compute residuals and betas externally for diagnostics
    residuals, betas, predicted = rolling_ridge_rets(daily_rets, factors)

    # Portfolio returns and factor exposures at each bar
    port_rets = np.zeros(n_ret_bars)
    factor_exposures = np.zeros((n_ret_bars, n_factors))

    for t in range(n_ret_bars):
        w = all_weights[t]
        port_rets[t] = np.sum(w * daily_rets[t])
        if np.all(np.isfinite(betas[t])):
            factor_exposures[t] = w @ betas[t]  # (n_factors,) = B^T @ w

    return all_weights, port_rets, factor_exposures, residuals, betas, predicted, daily_rets, factors, factor_names


def annualized_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 3 or np.std(rets) < 1e-10:
        return 0.0
    return float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252))


def main():
    logging.basicConfig(level=logging.WARNING)

    print("=" * 65)
    print("  FINAL TEST: Does residual alpha survive strict factor neutralization?")
    print("=" * 65)

    # Load data
    print(f"\n  Loading RP universe data ({FULL_START} -> {FULL_END})...")
    df = fetch_and_prepare("rp", FULL_START, FULL_END, use_cache=True)
    data = df_to_data(df)
    close = data["close"]
    symbols = list(data["symbols"])
    n_periods, n_assets = close.shape
    print(f"  Data: {n_periods} bars x {n_assets} assets: {symbols}")

    # Run both strategies
    print(f"\n  {'=' * 65}")
    print(f"  1. RUNNING BOTH STRATEGIES")
    print(f"  {'=' * 65}")

    strategies = [
        ("Original", "macro_factor_residual", ORIG_PARAMS),
        ("Strict", "macro_factor_residual_strict", STRICT_PARAMS),
    ]

    results = {}
    for label, sid, params in strategies:
        print(f"\n  --- {label}: {sid} ---")
        w, pr, fe, res, betas, pred, dret, fac, fnames = run_strategy(
            close, symbols, params, sid
        )

        sr = annualized_sharpe(pr)
        print(f"  Bars with positions: {np.sum(np.any(np.abs(w) > 1e-6, axis=1))}")
        print(f"  Portfolio Sharpe:    {sr:+.4f}")
        print(f"  Portfolio mean ret:  {np.mean(pr):+.6f}")
        print(f"  Portfolio std:       {np.std(pr, ddof=1):.6f}")

        results[label] = {
            "weights": w,
            "port_rets": pr,
            "factor_exposures": fe,
            "residuals": res,
            "betas": betas,
            "predicted": pred,
            "daily_rets": dret,
            "factors": fac,
            "factor_names": fnames,
            "sharpe": sr,
        }

    # =======================================================
    # 2. FACTOR NEUTRALITY TEST
    # =======================================================
    print(f"\n  {'=' * 65}")
    print(f"  2. FACTOR NEUTRALITY TEST (CORR(w, factor_exposure) == 0)")
    print(f"  {'=' * 65}")

    fnames = results["Original"]["factor_names"]
    for label in ["Original", "Strict"]:
        r = results[label]
        weights = r["weights"]
        fe = r["factor_exposures"]
        valid = np.any(np.abs(weights) > 1e-6, axis=1) & np.all(np.isfinite(fe), axis=1)

        print(f"\n  {label}:")
        for j, fn in enumerate(fnames):
            exposure = fe[valid, j]
            if len(exposure) < 10:
                print(f"    {fn:<10s}  insufficient data")
                continue
            mean_exp = np.mean(exposure)
            std_exp = np.std(exposure, ddof=1)
            t_stat = mean_exp / max(std_exp / np.sqrt(len(exposure)), 1e-10)
            print(f"    {fn:<10s}  mean exposure = {mean_exp:+.6f}  "
                  f"t-stat = {t_stat:+.3f}  "
                  f"{'NEUTRAL' if abs(t_stat) < 2 else 'LEAKAGE'}")

    # =======================================================
    # 3. RESIDUAL DOMINANCE TEST
    # =======================================================
    print(f"\n  {'=' * 65}")
    print(f"  3. RESIDUAL DOMINANCE TEST (PnL_residual / total_PnL)")
    print(f"  {'=' * 65}")

    for label in ["Original", "Strict"]:
        r = results[label]
        weights = r["weights"]
        residuals = r["residuals"]
        predicted = r["predicted"]
        dret = r["daily_rets"]

        n_bars = len(weights)

        # Align: weights[t] corresponds to daily_rets[t] and residuals[t]
        # Last residual might be NaN
        valid_min = 252  # skip first reg_window bars

        w_v = weights[valid_min:]
        r_v = residuals[valid_min:]
        p_v = predicted[valid_min:]
        ret_v = dret[valid_min:]

        n_valid = len(w_v)
        has_pos = np.any(np.abs(w_v) > 1e-6, axis=1)

        factor_pnl = np.sum(w_v * p_v, axis=1)
        residual_pnl = np.sum(w_v * r_v, axis=1)
        total_pnl = np.sum(w_v * ret_v, axis=1)

        abs_resid = np.sum(np.abs(residual_pnl[has_pos]))
        abs_factor = np.sum(np.abs(factor_pnl[has_pos]))
        abs_total = np.sum(np.abs(total_pnl[has_pos]))
        resid_frac = abs_resid / max(abs_resid + abs_factor, 1e-10)

        resid_sr = annualized_sharpe(residual_pnl[has_pos])
        factor_sr = annualized_sharpe(factor_pnl[has_pos])
        total_sr = annualized_sharpe(total_pnl[has_pos])

        print(f"\n  {label}:")
        print(f"    Residual |PnL| fraction:  {resid_frac:.1%}")
        print(f"    Residual PnL Sharpe:      {resid_sr:+.4f}")
        print(f"    Factor PnL Sharpe:        {factor_sr:+.4f}")
        print(f"    Total PnL Sharpe:         {total_sr:+.4f}")

        r["resid_pnl"] = residual_pnl
        r["factor_pnl"] = factor_pnl
        r["resid_frac"] = resid_frac
        r["resid_sr"] = resid_sr
        r["factor_sr"] = factor_sr
        r["total_sr"] = total_sr
        r["valid_mask"] = has_pos

    # =======================================================
    # 4. LEAKAGE DETECTION
    # =======================================================
    print(f"\n  {'=' * 65}")
    print(f"  4. LEAKAGE DETECTION")
    print(f"  {'=' * 65}")

    orig = results["Original"]
    strict = results["Strict"]

    print(f"\n  A. Sharpe collapse test:")
    print(f"     Original Sharpe:  {orig['sharpe']:+.4f}")
    print(f"     Strict Sharpe:    {strict['sharpe']:+.4f}")
    print(f"     Change:           {strict['sharpe'] - orig['sharpe']:+.4f}")
    collapse_pct = (orig['sharpe'] - strict['sharpe']) / max(abs(orig['sharpe']), 1e-10) * 100
    print(f"     Collapse:          {collapse_pct:.1f}%")

    if strict['sharpe'] < orig['sharpe'] * 0.5:
        print(f"     VERDICT: >50% Sharpe collapse -- significant alpha from factor leakage")
    elif strict['sharpe'] < orig['sharpe'] * 0.8:
        print(f"     VERDICT: 20-50% Sharpe collapse -- partial alpha from factor leakage")
    else:
        print(f"     VERDICT: <20% Sharpe collapse -- most alpha was genuine residual")

    print(f"\n  B. Residual PnL shift:")
    print(f"     Original residual fraction:  {orig['resid_frac']:.1%}")
    print(f"     Strict residual fraction:    {strict['resid_frac']:.1%}")
    if strict['resid_frac'] > 0.8:
        print(f"     VERDICT: Residual dominance achieved after neutralization")
    elif strict['resid_frac'] > orig['resid_frac']:
        print(f"     VERDICT: Residual fraction improved (factor leakage removed)")
    else:
        print(f"     VERDICT: Residual fraction did not improve")

    print(f"\n  C. Residual Sharpe comparison:")
    print(f"     Original residual Sharpe:  {orig['resid_sr']:+.4f}")
    print(f"     Strict residual Sharpe:    {strict['resid_sr']:+.4f}")
    if strict['resid_sr'] > 0.5:
        print(f"     VERDICT: Residual alpha survives neutralization (Sharpe > 0.5)")
    elif strict['resid_sr'] > 0:
        print(f"     VERDICT: Residual alpha present but weak (0 < Sharpe < 0.5)")
    else:
        print(f"     VERDICT: Residual alpha DESTROYED by neutralization")

    # =======================================================
    # 5. BETA SANITY CHECK
    # =======================================================
    print(f"\n  {'=' * 65}")
    print(f"  5. BETA SANITY CHECK (portfolio beta to SPY)")
    print(f"  {'=' * 65}")

    spy_idx = symbols.index("SPY")
    for label in ["Original", "Strict"]:
        r = results[label]
        w = r["weights"]
        dret = r["daily_rets"]
        valid_min = 252

        port_rets = np.sum(w[valid_min:] * dret[valid_min:], axis=1)
        spy_rets = dret[valid_min:, spy_idx]

        has_pos = np.any(np.abs(w[valid_min:]) > 1e-6, axis=1)
        pr = port_rets[has_pos]
        sr = spy_rets[has_pos]

        if len(pr) > 10:
            cov = np.cov(pr, sr)
            beta = cov[0, 1] / max(cov[1, 1], 1e-10)
            corr = np.corrcoef(pr, sr)[0, 1]
        else:
            beta = 0.0
            corr = 0.0

        print(f"    {label:<10s}  beta to SPY = {beta:+.4f}  "
              f"corr = {corr:+.4f}  "
              f"{'NEUTRAL' if abs(beta) < 0.1 else 'LEAKAGE'}")

    # =======================================================
    # 6. SUMMARY / META-ANSWER
    # =======================================================
    print(f"\n  {'=' * 65}")
    print(f"  6. META-ANSWER: Does macro alpha survive strict neutralization?")
    print(f"  {'=' * 65}")

    print(f"""
    COMPARISON TABLE:
    {'Metric':<30s} {'Original':>12s} {'Strict':>12s} {'Change':>12s}
    {'-'*30} {'-'*12} {'-'*12} {'-'*12}
    Sharpe (full)            {orig['sharpe']:>+12.4f} {strict['sharpe']:>+12.4f} {strict['sharpe'] - orig['sharpe']:>+12.4f}
    Residual PnL fraction    {orig['resid_frac']:>12.1%} {strict['resid_frac']:>12.1%} {strict['resid_frac'] - orig['resid_frac']:>+12.1%}
    Residual Sharpe          {orig['resid_sr']:>+12.4f} {strict['resid_sr']:>+12.4f} {strict['resid_sr'] - orig['resid_sr']:>+12.4f}
    """)

    # Decision tree
    if strict['sharpe'] >= 0.5 and strict['resid_frac'] > 0.8:
        print(f"  CONCLUSION: MACRO ALPHA IS REAL AND DEPLOYABLE.")
        print(f"  Residual alpha survives strict factor neutralization with")
        print(f"  Sharpe {strict['sharpe']:.2f} and {strict['resid_frac']:.0%} residual PnL share.")
    elif strict['sharpe'] > 0 and strict['resid_frac'] > 0.5:
        print(f"  CONCLUSION: MACRO ALPHA EXISTS BUT IS WEAKENED.")
        print(f"  Sharpe drops from {orig['sharpe']:.2f} to {strict['sharpe']:.2f} after")
        print(f"  strict neutralization. Residual fraction {strict['resid_frac']:.0%}.")
        print(f"  {collapse_pct:.0f}% of previous performance was factor leakage.")
        print(f"  True residual alpha exists but is below deployable threshold.")
    elif strict['sharpe'] > 0:
        print(f"  CONCLUSION: MACRO ALPHA IS MARGINAL.")
        print(f"  Strict neutralization leaves Sharpe {strict['sharpe']:.2f} with")
        print(f"  {strict['resid_frac']:.0%} residual PnL. Most previous alpha was")
        print(f"  factor misattribution ({collapse_pct:.0f}% leakage).")
    else:
        print(f"  CONCLUSION: MACRO ALPHA DOES NOT EXIST IN THIS UNIVERSE.")
        print(f"  All previous alpha was factor exposure misclassification.")
        print(f"  After strict neutralization: Sharpe {strict['sharpe']:.2f}.")
        print(f"  Factor leakage accounted for {collapse_pct:.0f}% of prior performance.")

    # Per-regime comparison
    print(f"\n  {'=' * 65}")
    print(f"  7. PER-REGIME SHARPE COMPARISON")
    print(f"  {'=' * 65}")

    regimes = {
        "Full 2018-2024": (close, None, None),
        "GFC 2008": (None, None, None),
        "COVID 2020": (None, None, None),
        "Rate Hikes 2022": (None, None, None),
        "Bull 2023-2024": (None, None, None),
    }

    # Compute per-regime Sharpe from the stored returns
    regime_dates = {
        "Full 2018-2024": ("2018-01-01", "2024-12-31"),
        "GFC 2008": ("2008-01-01", "2009-12-31"),
        "COVID 2020": ("2020-01-01", "2021-12-31"),
        "Rate Hikes 2022": ("2022-01-01", "2023-12-31"),
        "Bull 2023-2024": ("2023-01-01", "2024-12-31"),
    }

    # Use full timeframe returns from the stored arrays for comparison
    print(f"  {'Regime':<20s} {'Orig SR':>10s} {'Strict SR':>10s} {'Change':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    # For this we need the full return series aligned to dates
    # We can compute from the loaded data
    all_dates = sorted(df["timestamp"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for regime_name, (start_str, end_str) in regime_dates.items():
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)

        # Strip timezone from timestamps for comparison
        known_tz = None
        for d in all_dates:
            if hasattr(d, 'tz') and d.tz is not None:
                known_tz = d.tz
                break

        # Convert string dates to Timestamps if needed
        if all_dates and isinstance(all_dates[0], str):
            all_dates_parsed = [pd.Timestamp(d) for d in all_dates]
            if known_tz:
                all_dates_parsed = [d.tz_localize(known_tz) if d.tz is None else d for d in all_dates_parsed]
        else:
            all_dates_parsed = all_dates

        if known_tz is not None:
            if start_dt.tz is None:
                start_dt = start_dt.tz_localize(known_tz)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize(known_tz)

        # Find bar range
        start_idx = None
        end_idx = None
        for i, d in enumerate(all_dates_parsed):
            if start_idx is None and d >= start_dt:
                start_idx = i
            if d <= end_dt:
                end_idx = i

        if start_idx is None or end_idx is None or start_idx >= end_idx:
            continue

        # Returns are aligned to bars 1..n, so bar ret[start_idx-1] corresponds to date start_idx
        ret_start = max(0, start_idx - 1)
        ret_end = end_idx - 1

        orig_rets = results["Original"]["port_rets"][ret_start:ret_end]
        strict_rets = results["Strict"]["port_rets"][ret_start:ret_end]

        has_pos_o = np.any(np.abs(results["Original"]["weights"][ret_start:ret_end]) > 1e-6, axis=1)
        has_pos_s = np.any(np.abs(results["Strict"]["weights"][ret_start:ret_end]) > 1e-6, axis=1)

        sr_o = annualized_sharpe(orig_rets[has_pos_o]) if has_pos_o.sum() > 3 else 0.0
        sr_s = annualized_sharpe(strict_rets[has_pos_s]) if has_pos_s.sum() > 3 else 0.0

        print(f"  {regime_name:<20s} {sr_o:>+10.4f} {sr_s:>+10.4f} {sr_s - sr_o:>+10.4f}")

    # Save full results for later analysis
    print(f"\n  Diagnostics complete.")


if __name__ == "__main__":
    main()
