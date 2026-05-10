"""Deep diagnostics for macro_factor_residual strategy.

Replicates the factor model externally to collect betas, residuals, and
portfolio weight time series, then computes:

  A. Factor explanation ratio (R2 per asset)
  B. Residual autocorrelation (lag-1)
  C. Portfolio beta to SPY (neutrality check)
  D. Alpha decomposition (residual vs factor PnL fraction)
  E. Factor dominance check (no single factor >80%)

Usage:
    python scripts/factor_residual_diagnostics.py
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

from scripts.real_data_loader import fetch_and_prepare, UNIVERSES

log = logging.getLogger(__name__)

# -- Factor definitions (mirrors factor_residual.py) --------------------------
FACTOR_DEFS: dict[str, tuple[str, str | None, float]] = {
    "MKT": ("SPY", None, 1.0),
    "RATES": ("TLT", "SPY", 1.0),
    "INFL": ("DBC", None, 1.0),
    "SAFE": ("GLD", "TLT", 0.5),
}

# -- Config --------------------------------------------------------------------
REG_WINDOW = 252
RIDGE_ALPHA = 1e-6
REBALANCE_FREQ = 5
RESID_MOM_WINDOW = 63
RESID_REV_WEIGHT = 0.3
VOL_LOOKBACK = 20

FULL_START = "2008-01-01"
FULL_END = "2024-12-31"


def df_to_data(df: pd.DataFrame) -> dict:
    """Convert combined OHLCV DataFrame to strategy data dict."""
    symbols = sorted(df["symbol"].unique())
    pivoted = df.pivot_table(index="timestamp", columns="symbol", values="close")
    pivoted = pivoted.ffill().bfill().fillna(0.0)

    if pivoted.empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}

    data = {
        "close": pivoted.values,
        "symbols": symbols,
    }
    return data


def build_factor_map(symbols: list[str]) -> dict[str, tuple[int, int | None, float]]:
    """Map factor names to asset index pairs."""
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    fmap: dict[str, tuple[int, int | None, float]] = {}
    for fname, (sym1, sym2, scale) in FACTOR_DEFS.items():
        idx1 = sym_to_idx.get(sym1, -1)
        if idx1 < 0:
            continue
        idx2 = sym_to_idx.get(sym2) if sym2 else None
        fmap[fname] = (idx1, idx2, scale)
    return fmap


def build_factors(rets: np.ndarray, fmap: dict) -> np.ndarray:
    """Build factor matrix (n_periods, n_factors)."""
    n_periods = rets.shape[0]
    factors = np.zeros((n_periods, len(fmap)))
    for j, (fname, (idx1, idx2, scale)) in enumerate(sorted(fmap.items())):
        if idx2 is not None:
            factors[:, j] = (rets[:, idx1] - rets[:, idx2]) * scale
        else:
            factors[:, j] = rets[:, idx1] * scale
    return factors


def rolling_ridge_residuals(
    rets: np.ndarray, factors: np.ndarray, ridge_alpha: float = 1e-6, reg_window: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute residuals, betas, R2 per asset, and predicted returns.

    Returns:
        residuals:  (n_periods, n_assets) -- NaN for first reg_window bars
        betas:      (n_periods, n_factors, n_assets) -- NaN before reg_window
        r_squared:  (n_assets,) -- R2 for each asset over full period
        predicted:  (n_periods, n_assets) -- predicted returns
    """
    n_periods, n_assets = rets.shape
    n_factors = factors.shape[1]
    residuals = np.full((n_periods, n_assets), np.nan)
    betas = np.full((n_periods, n_factors, n_assets), np.nan)
    predicted = np.full((n_periods, n_assets), np.nan)

    if n_periods < reg_window + 1:
        return residuals, betas, np.zeros(n_assets), predicted

    I_reg = ridge_alpha * np.eye(n_factors)

    for t in range(reg_window, n_periods):
        X_train = factors[t - reg_window:t]
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0, ddof=1)
        X_std = np.maximum(X_std, 1e-10)
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
                betas[t, :, i] = 0.0
                predicted[t, i] = 0.0
                continue
            beta = XtX_inv @ (X_train_std.T @ y_train)
            betas[t, :, i] = beta
            y_pred = float(X_t_std @ beta)
            predicted[t, i] = y_pred
            residuals[t, i] = float(rets[t, i]) - y_pred

    # R2 per asset: 1 - var(residual) / var(return)
    r_squared = np.full(n_assets, np.nan)
    for i in range(n_assets):
        valid = ~np.isnan(residuals[:, i])
        if valid.sum() > 10:
            r_squared[i] = 1.0 - np.var(residuals[valid, i], ddof=1) / np.var(rets[valid, i], ddof=1)
        else:
            r_squared[i] = np.nan

    return residuals, betas, r_squared, predicted


def residual_autocorrelation(residuals: np.ndarray) -> np.ndarray:
    """Compute lag-1 autocorrelation of residuals per asset."""
    n_assets = residuals.shape[1]
    ac = np.full(n_assets, np.nan)
    for i in range(n_assets):
        r = residuals[:, i]
        valid = ~np.isnan(r)
        if valid.sum() < 10:
            continue
        rv = r[valid]
        ac[i] = np.corrcoef(rv[:-1], rv[1:])[0, 1]
    return ac


def compute_diagnostics(data: dict) -> dict:
    """Run full diagnostics on factor residual strategy."""
    close = data["close"]
    symbols = list(data["symbols"])
    n_periods, n_assets = close.shape

    print(f"\n  Data: {n_periods} bars x {n_assets} assets: {symbols}")
    print(f"  Reg window: {REG_WINDOW}, Ridge alpha: {RIDGE_ALPHA}")

    # -- Daily returns --
    daily_rets = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
    n_ret_bars = daily_rets.shape[0]
    print(f"  Return bars: {n_ret_bars}")

    # -- Factor map --
    fmap = build_factor_map(symbols)
    print(f"  Factors available: {list(fmap.keys())}")

    # -- Build factors --
    factors = build_factors(daily_rets, fmap)
    print(f"  Factor matrix: {factors.shape}")

    # -- Rolling ridge: residuals + betas --
    residuals, betas, r_squared, predicted = rolling_ridge_residuals(
        daily_rets, factors, RIDGE_ALPHA, REG_WINDOW
    )
    valid_residuals = residuals[REG_WINDOW:]  # first bar where we have residuals

    print(f"\n  -- A. Factor Explanation Ratio (R2) --")
    fname_width = max(len(s) for s in symbols)
    for i, s in enumerate(symbols):
        r2 = r_squared[i]
        print(f"    {s:<{fname_width}s}  R2 = {r2:.4f}  "
              f"(explained: {r2*100:.1f}%, residual: {(1-r2)*100:.1f}%)")

    # -- B. Residual autocorrelation --
    print(f"\n  -- B. Residual Autocorrelation (lag-1) --")
    autocorr = residual_autocorrelation(residuals)
    dw_mean = np.nanmean(autocorr)
    for i, s in enumerate(symbols):
        ac = autocorr[i]
        label = "<-- SIGNIFICANT" if abs(ac) > 0.1 else ""
        print(f"    {s:<{fname_width}s}  lag-1 rho = {ac:+.4f}  {label}")
    print(f"    {'Average':<{fname_width}s}  lag-1 rho = {dw_mean:+.4f}")

    # -- C + D + E: Run strategy to get weights, then decompose --
    print(f"\n  -- C/D/E. Portfolio Decomposition --")

    # Create and run strategy
    cfg = StrategyConfig(
        name="macro_factor_residual",
        strategy_type=StrategyType.MACRO,
        params={
            "rebalance_freq": REBALANCE_FREQ,
            "reg_window": REG_WINDOW,
            "ridge_alpha": RIDGE_ALPHA,
            "resid_mom_window": RESID_MOM_WINDOW,
            "resid_rev_weight": RESID_REV_WEIGHT,
            "vol_lookback": VOL_LOOKBACK,
            "min_weight": 0.01,
        },
    )

    strat_cls = get_strategy_class("macro_factor_residual")
    strat = strat_cls(cfg)
    strat.initialize()

    # Feed data bar by bar, collect weights
    all_weights: list[np.ndarray | None] = []  # one per return bar
    current_weights: np.ndarray | None = None

    for t in range(1, n_periods):
        bar_data = {
            "close": close[:t + 1],
            "symbols": symbols,
        }
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

        # For return bar t-1 (from close[t-1] to close[t]),
        # use weights that were active during that bar
        if current_weights is not None:
            all_weights.append(current_weights.copy())
        else:
            all_weights.append(None)

    # Convert to array (bars where we have both weights and residuals)
    w_array = np.array([w if w is not None else np.zeros(n_assets) for w in all_weights])

    # Align: weights shape (n_periods-1, n_assets), residuals shape (n_periods, n_assets)
    # residuals[t] corresponds to daily_rets[t] which is return from close[t] to close[t+1]
    # So residuals[t] aligns with daily_rets[t] aligns with all_weights[t]
    # We have residuals available from REG_WINDOW onwards
    valid_start = REG_WINDOW
    valid_mask = np.arange(len(all_weights)) >= valid_start

    w_valid = w_array[valid_mask]
    rets_valid = daily_rets[valid_mask]
    residuals_valid = residuals[valid_start:]
    predicted_valid = predicted[valid_start:]

    n_valid = np.sum(valid_mask)
    print(f"    Valid decomposition bars: {n_valid}")

    if n_valid < 10:
        print("    ERROR: insufficient valid bars")
        return {"error": "insufficient valid bars"}

    # Portfolio returns
    port_rets = np.sum(w_valid * rets_valid, axis=1)

    # -- C. Beta to SPY --
    spy_idx = symbols.index("SPY")
    spy_rets = rets_valid[:, spy_idx]
    cov = np.cov(port_rets, spy_rets)
    beta_spy = cov[0, 1] / max(cov[1, 1], 1e-10)
    print(f"\n    C. Portfolio Beta to SPY:")
    print(f"       beta = {beta_spy:+.4f}  {'PASS NEUTRAL' if abs(beta_spy) < 0.1 else 'FAIL NOT NEUTRAL'}")

    # -- D. Alpha decomposition --
    # Factor PnL = sum_i w_i * predicted_i
    # Residual PnL = sum_i w_i * residual_i
    factor_pnl = np.sum(w_valid * predicted_valid, axis=1)
    residual_pnl = np.sum(w_valid * residuals_valid, axis=1)
    total_pnl = port_rets

    # Residual fraction of absolute PnL
    abs_resid = np.sum(np.abs(residual_pnl))
    abs_factor = np.sum(np.abs(factor_pnl))
    abs_total = np.sum(np.abs(total_pnl))
    resid_frac = abs_resid / max(abs_resid + abs_factor, 1e-10)

    # Also compute as fraction of total PnL variance
    var_total = np.var(total_pnl)
    var_resid = np.var(residual_pnl)
    var_factor = np.var(factor_pnl)
    cov_resid_factor = np.cov(residual_pnl, factor_pnl)[0, 1]

    print(f"\n    D. Alpha Decomposition (residual vs factor PnL):")
    print(f"       Residual |Sigma(PnL)| fraction: {resid_frac:.1%}")
    print(f"       Residual PnL var fraction:   {var_resid / max(var_total, 1e-10):.1%}")
    print(f"       Factor PnL var fraction:     {var_factor / max(var_total, 1e-10):.1%}")
    print(f"       Residual PnL Sharpe:         {np.mean(residual_pnl) / max(np.std(residual_pnl), 1e-10) * np.sqrt(252):+.4f}")
    print(f"       Factor PnL Sharpe:           {np.mean(factor_pnl) / max(np.std(factor_pnl), 1e-10) * np.sqrt(252):+.4f}")
    print(f"       Total portfolio Sharpe:      {np.mean(total_pnl) / max(np.std(total_pnl), 1e-10) * np.sqrt(252):+.4f}")

    # -- E. Factor dominance --
    # Net exposure to each factor = sum_i w_i * beta_i_k
    # Factor contribution = net_exposure_k * factor_return_k
    n_factors = factors.shape[1]
    factor_names = sorted(fmap.keys())
    fcontrib = np.zeros((n_valid, n_factors))

    for t_idx in range(n_valid):
        w_t = w_valid[t_idx]
        for k in range(n_factors):
            # Net exposure to factor k: sum_i w_i[t] * beta_i_k[t]
            net_exp = np.sum(w_t * betas[valid_start + t_idx, k, :])
            # Contribution: net_exp * factor_return_k[t]
            fcontrib[t_idx, k] = net_exp * factors[valid_start + t_idx, k]

    abs_fcontrib = np.sum(np.abs(fcontrib), axis=0)
    total_abs_fcontrib = np.sum(abs_fcontrib)

    print(f"\n    E. Factor Dominance Check:")
    for k in range(n_factors):
        share = abs_fcontrib[k] / max(total_abs_fcontrib, 1e-10)
        print(f"       {factor_names[k]:<10s}  contribution = {share:.1%}  "
              f"cum PnL = {np.sum(fcontrib[:, k]):+.6f}")

    dominant_factor = np.argmax(abs_fcontrib)
    dominant_share = abs_fcontrib[dominant_factor] / max(total_abs_fcontrib, 1e-10)
    print(f"       Max single factor share: {dominant_share:.1%} "
          f"({'PASS' if dominant_share < 0.8 else 'FAIL'} threshold 80%)")

    # -- Per-asset PnL contribution --
    print(f"\n    Per-Asset PnL Contribution:")
    for i, s in enumerate(symbols):
        contrib = np.sum(w_valid[:, i] * rets_valid[:, i])
        abs_contrib = np.sum(np.abs(w_valid[:, i] * rets_valid[:, i]))
        avg_weight = np.mean(np.abs(w_valid[:, i]))
        print(f"       {s:<{fname_width}s}  cum PnL = {contrib:+.4f}  "
              f"avg |w| = {avg_weight:.4f}")

    # -- Summary --
    print(f"\n  {'=' * 55}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"  {'=' * 55}")

    passed_r2 = np.mean(r_squared) < 0.99  # should not explain everything
    passed_autocorr = np.nanmean(autocorr) < 0.1
    passed_beta_neutral = abs(beta_spy) < 0.1
    passed_resid_dominant = resid_frac > 0.5
    passed_factor_dominance = dominant_share < 0.8

    checks = [
        ("Factor R2 (not overfitted)", f"mean R2 = {np.mean(r_squared):.3f}", passed_r2),
        ("Residual autocorrelation", f"mean rho = {np.nanmean(autocorr):+.4f}", passed_autocorr),
        ("Beta neutrality (|beta_SPY| < 0.1)", f"beta = {beta_spy:+.4f}", passed_beta_neutral),
        ("Residual PnL dominant (>50%)", f"resid fraction = {resid_frac:.1%}", passed_resid_dominant),
        ("No factor > 80% dominance", f"max = {dominant_share:.1%}", passed_factor_dominance),
    ]

    for label, value, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    [{status}] {label:<40s}  {value}")

    n_pass = sum(1 for _, _, p in checks if p)
    print(f"\n    {n_pass}/{len(checks)} checks passed")

    return {
        "n_bars": n_periods,
        "n_assets": n_assets,
        "symbols": symbols,
        "r_squared": {s: float(r_squared[i]) for i, s in enumerate(symbols)},
        "residual_autocorrelation": {s: float(autocorr[i]) for i, s in enumerate(symbols)},
        "beta_to_spy": float(beta_spy),
        "residual_pnl_fraction": float(resid_frac),
        "residual_pnl_sharpe": float(np.mean(residual_pnl) / max(np.std(residual_pnl), 1e-10) * np.sqrt(252)),
        "factor_pnl_sharpe": float(np.mean(factor_pnl) / max(np.std(factor_pnl), 1e-10) * np.sqrt(252)),
        "portfolio_sharpe": float(np.mean(total_pnl) / max(np.std(total_pnl), 1e-10) * np.sqrt(252)),
        "max_factor_dominance": float(dominant_share),
        "dominant_factor": factor_names[dominant_factor],
        "checks_passed": n_pass,
        "checks_total": len(checks),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Ensure the rp universe data is loaded
    print("=" * 60)
    print("  Factor Residual Strategy -- Deep Diagnostics")
    print("=" * 60)

    print(f"\n  Loading RP universe data ({FULL_START} -> {FULL_END})...")
    df = fetch_and_prepare("rp", FULL_START, FULL_END, use_cache=True)
    print(f"  Loaded: {len(df)} bars x {df['symbol'].nunique()} symbols")

    data = df_to_data(df)
    close = data["close"]
    symbols = data["symbols"]
    print(f"  Close array: {close.shape}  Symbols: {symbols}")

    results = compute_diagnostics(data)

    if "error" not in results:
        print(f"\n  Diagnostics complete.")
    else:
        print(f"\n  Diagnostics FAILED: {results['error']}")


if __name__ == "__main__":
    main()
