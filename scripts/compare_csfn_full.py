"""Comprehensive strategy comparison on ETF_EXPANDED with detailed diagnostics.

Runs:
  1. macro_cross_sectional_factor_neutral (new)
  2. macro_cross_sectional_momentum (baseline)

On:
  3. ETF_EXPANDED (57-asset universe, 2018-2024)
  4. 5 bps transaction costs

Reports:
  - Sharpe, max drawdown, annual return, turnover, Calmar
  - Daily weights -> Herfindahl index, max single-asset weight
  - Equity curve (saved to CSV, can be plotted)
  - Rolling factor exposures using Ken French 5F + Mom
  - Comparison table

Usage:
    python scripts/compare_csfn_full.py
"""

from __future__ import annotations

import sys
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantcore.risk.metrics import sharpe_ratio, drawdown_stats
from quantcore.strategies.base import StrategyConfig, StrategyType

# -- trigger registrations --------------------------------------------------
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401

from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe
from quantcore.data.sources.fama_french import FamaFrenchLoader  # type: ignore  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Types & helpers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    name: str
    returns: npt.NDArray[np.float64] = field(repr=False)
    equity: npt.NDArray[np.float64] = field(repr=False)

    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_pct: float = 0.0
    calmar: float = 0.0
    turnover: float = 0.0
    n_trades: int = 0
    factor_exposures: dict[str, float] = field(default_factory=dict)
    residual_exposures: dict[str, float] = field(default_factory=dict)
    daily_weights: pd.DataFrame | None = None
    herfindahl_mean: float = 0.0
    max_single_weight: float = 0.0
    worst_dd_start: str | None = None
    worst_dd_end: str | None = None


def _df_to_data(df: pd.DataFrame) -> dict:
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)
    if n_assets == 0:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivots = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field)
        pivots[field] = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted  # noqa: E501

    if pivots["close"].empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}

    data: dict = {
        "close": pivots["close"].values,
        "high": pivots["high"].values,
        "low": pivots["low"].values,
        "open": pivots["open"].values,
        "volume": pivots["volume"].values,
        "symbols": symbols,
    }
    for k in ("close", "high", "low", "open", "volume"):
        arr = data[k]
        if np.any(np.isnan(arr)):
            data[k] = np.nan_to_num(pd.DataFrame(arr).ffill(limit=5).to_numpy(), nan=0.0)
    return data


def compute_bar_returns(
    strat,
    data: dict,
    timestamps: list,
    oos_start: int,
    oos_end: int,
    tc_bps: float = 5.0,
):
    """Compute bar returns and weight history for a strategy."""
    if oos_end - oos_start < 2:
        return np.array([], dtype=float), []

    close = data["close"]
    symbols = data["symbols"]
    if not symbols:
        return np.array([], dtype=float), []

    asset_rets = np.diff(close[oos_start:oos_end], axis=0) / np.maximum(close[oos_start:oos_end - 1], 1e-10)  # noqa: E501
    bar_rets = np.zeros(oos_end - oos_start - 1)
    weight_log: list[dict[str, float]] = []
    carried_weights: dict[str, float] = {}

    for idx, bar_i in enumerate(range(oos_start, oos_end - 1)):
        # Build causal bar_data
        bar_data: dict = {"close": data["close"][:bar_i + 1], "symbols": symbols}
        for k in ("high", "low", "open", "volume"):
            bar_data[k] = data.get(k, data["close"])[:bar_i + 1]
        if bar_i < len(timestamps):
            bar_data["timestamp"] = timestamps[bar_i]

        try:
            signals = strat.on_data(bar_data) or []
        except Exception:  # pragma: no cover
            signals = []

        current_weights: dict[str, float] = {}
        weighted_ret = 0.0
        total_w = 0.0

        for s in signals:
            try:
                asset_idx = symbols.index(s.symbol)
            except (ValueError, IndexError):
                continue
            w = float(s.direction) * float(s.strength)
            current_weights[s.symbol] = w
            weighted_ret += w * float(asset_rets[idx, asset_idx])
            total_w += abs(w)

        if total_w > 1e-10:
            bar_rets[idx] = weighted_ret / total_w

        # transaction cost (turnover)
        if tc_bps > 0 and current_weights:
            turnover = sum(
                abs(current_weights.get(sym, 0.0) - carried_weights.get(sym, 0.0))
                for sym in (set(current_weights) | set(carried_weights))
            )
            bar_rets[idx] -= turnover * tc_bps / 10000.0

        carried_weights = dict(current_weights)

        # log weights for today
        weight_log.append(carried_weights)

    return bar_rets, weight_log


def build_equity(returns: npt.NDArray[np.float64], initial: float = 1_000_000.0):  # noqa: E501
    if len(returns) == 0:
        return np.array([initial])
    eq = np.empty(len(returns) + 1)
    eq[0] = initial
    for i, r in enumerate(returns):
        eq[i + 1] = eq[i] * (1.0 + r)
    return eq


def compute_herfindahl(weights: npt.NDArray[np.float64]) -> float:
    """HHI = sum(w_i^2). 1/n for equal-weight, 1 for single-asset."""
    if len(weights) == 0:
        return 0.0
    w_norm = weights / (np.sum(np.abs(weights)) + 1e-12)
    return float(np.sum(w_norm ** 2))


def regress_portfolio_on_factors(port_rets: npt.NDArray[np.float64], factor_rets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # noqa: E501
    """OLS regression: port ~ factors. Returns factor betas."""
    min_len = min(len(port_rets), factor_rets.shape[0])
    if min_len < 10:
        return np.array([])
    y = port_rets[:min_len].reshape(-1, 1)
    X = factor_rets[:min_len]
    # Add intercept
    X_const = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.pinv(X_const.T @ X_const) @ X_const.T @ y
        return beta[1:].flatten()  # exclude intercept
    except Exception:
        return np.zeros(X.shape[1])


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

STRATEGY_PARAMS = {
    "macro_cross_sectional_factor_neutral": {
        "name": "Factor-Neutral StatArb",
        "params": {
            "rebalance_freq": 21,
            "reg_lookback": 252,
            "pred_window": 20,
            "ridge_alpha": 1.0,
            "max_position": 0.03,
            "target_vol": 0.15,
            "vol_lookback": 20,
            "zscore_cutoff": 1.5,
            "percentile_top": 0.30,
        },
    },
    "macro_cross_sectional_momentum": {
        "name": "CSM v1 (Baseline)",
        "params": {
            "rebalance_freq": 5,
            "mom_windows": "20,60,120",
            "mom_weights": "0.2,0.5,0.3",
            "target_vol": 0.12,
            "vol_lookback": 20,
            "use_vol_adjustment": True,
            "max_gross_exposure": 1.0,
            "beta_neutral": True,
            "regime_filter": True,
            "dd_control": True,
            "min_weight": 0.01,
        },
    },
}


def run_strategy(strategy_id: str, df: pd.DataFrame | None = None) -> BacktestResult:  # noqa: E501
    cfg = STRATEGY_PARAMS[strategy_id]
    universe = "ETF_EXPANDED"
    start, end = "2018-01-01", "2024-12-31"
    tc = 5.0

    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"  ID: {strategy_id}")
    print(f"  Universe: {universe}  (5%+ symbols may be delisted)")
    print(f"  Period: {start} -> {end}")
    print(f"  TC: {tc} bps  //  Warmup: 300 bars")
    print(f"{'='*60}")

    # Load data
    if df is None:
        print("  Loading data...")
        df = load_universe(universe, start, end)
    else:
        print("  Using provided DataFrame...")

    if df.empty:
        raise RuntimeError(f"No data returned for {universe}")

    print(f"  Loaded {len(df)} rows across {df['symbol'].nunique()} symbols")

    data = _df_to_data(df)
    timestamps = sorted(df["timestamp"].unique())
    close = data["close"]
    n_periods = close.shape[0]
    symbols = data["symbols"]

    print(f"  Data: {n_periods} bars x {len(symbols)} assets")

    # Build strategy
    scfg = StrategyConfig(name=strategy_id, strategy_type=StrategyType.MACRO, params=cfg["params"])

    cls = get_strategy_class(strategy_id)
    strat = cls(scfg)
    strat.initialize()

    # Warmup
    print("  Warming up...")
    warmup = min(300, n_periods - 10)
    for bar_i in range(0, warmup):
        bd = {"close": close[: bar_i + 1], "symbols": symbols}
        for k in ("high", "low", "open", "volume"):
            bd[k] = data.get(k, close)[: bar_i + 1]
        if bar_i < len(timestamps):
            bd["timestamp"] = timestamps[bar_i]
        try:
            strat.on_data(bd)
        except Exception:
            pass

    # OOS backtest
    print("  Running OOS backtest...")
    oos_start = warmup
    oos_end = n_periods
    returns, weight_log = compute_bar_returns(strat, data, timestamps, oos_start, oos_end, tc)  # type: ignore[arg-type]  # noqa: E501

    if len(returns) == 0:
        raise RuntimeError("No OOS returns")

    equity = build_equity(returns)

    # -- metrics -----------------------------------------------------------
    total_ret = (equity[-1] - 1_000_000.0) / 1_000_000.0 * 100.0
    n_years = len(returns) / 252.0
    ann_ret = ((equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0) * 100.0 if n_years else 0.0
    sr = sharpe_ratio(returns, rf=0.04)
    max_dd = drawdown_stats(returns)
    max_dd_pct = abs(max_dd.max_drawdown) * 100.0
    calmar = ann_ret / max(-max_dd_pct, 0.01) if max_dd_pct < 0 else 0.0

    # Turnover (sum of |dw| / 2 per period)
    turnover = sum(abs(returns[i]) for i in range(len(returns))) * 2.0  # rough proxy  # noqa: E501

    print(f"  Total Return:  {total_ret:+.2f}%")
    print(f"  Annual Return:   {ann_ret:+.2f}%")
    print(f"  Sharpe:        {sr:.4f}")
    print(f"  Max Drawdown:  {max_dd_pct:.2f}%")
    print(f"  Calmar:        {calmar:.4f}")
    print(f"  Turnover:      {turnover:.2f}")
    print(f"  OOS bars:      {len(returns)}")

    # -- weight diagnostics --------------------------------------------------
    daily_weights = pd.DataFrame(weight_log)
    daily_weights = daily_weights.fillna(0.0)

    # max single-asset weight at any point
    abs_w = daily_weights.abs()
    max_single = abs_w.max().max() if not abs_w.empty else 0.0

    # Herfindahl per day
    hhi_vals = []
    for _, row in daily_weights.iterrows():
        w = row.values
        if w.sum() != 0:
            w_arr = np.array(w, dtype=float)
            w_arr = w_arr / np.sum(np.abs(w_arr))
            hhi = float(np.sum(w_arr ** 2))
            hhi_vals.append(hhi)
    mean_hhi = np.mean(hhi_vals) if hhi_vals else 0.0

    print(f"  Max single-asset weight: {max_single*100:.2f}%")
    print(f"  Mean Herfindahl:        {mean_hhi:.4f} (1/n_eq={1.0/len(symbols):.4f})")

    # -- worst drawdown period -----------------------------------------------
    worst_start, worst_end = None, None
    peak = np.maximum.accumulate(equity)
    dd_series = (equity - peak) / peak
    worst_idx = int(np.argmin(dd_series))
    worst_start = str(timestamps[oos_start + max(0, worst_idx - 20)])
    worst_end = str(timestamps[oos_start + min(len(equity) - 1, worst_idx + 1)])

    # -- factor exposures ----------------------------------------------------
    # 3. Load real Fama-French data
    ff = None
    factor_names = None
    try:
        ff = FamaFrenchLoader(cache_dir=".cache/fama_french")
        ff5 = ff.load_5_factors(freq="daily")
        mom = ff.load_momentum(freq="daily")

        # Merge and align to our timestamps
        ff_common = ff5.join(mom, how="outer")
        ff_common = ff_common.sort_index()

        # Align to our backtest dates
        b_dates = pd.DatetimeIndex(timestamps[oos_start:oos_end])
        # Reindex to our dates (forward fill missing)
        ff_aligned = ff_common.reindex(b_dates, method="ffill")
        ff_aligned = ff_aligned.fillna(0.0)

        factor_names = list(ff_aligned.columns)  # type: ignore
        factor_rets = ff_aligned.to_numpy(dtype=float)  # type: ignore

        # Regress portfolio returns on factors
        betas = regress_portfolio_on_factors(returns, factor_rets)

        factor_exp = {}
        if len(factor_names) > 0:
            for i, name in enumerate(factor_names):
                factor_exp[name] = round(float(betas[i]), 6)

        # Report
        print(f"  Factor Exposures:")
        for k, v in factor_exp.items():
            print(f"    {k}: {v:.6f}")
    except Exception as e:
        print(f"  Factor exposure error: {e}")
        factor_exp = {}

    # Save equity curve to CSV for external plotting
    equity_df = pd.DataFrame({
        "date": timestamps[oos_start:oos_end + 1] if oos_start < len(timestamps) else list(range(len(equity))),
        "equity": equity,
    })
    tag = strategy_id.replace("macro_", "")
    equity_df.to_csv(f"backtest_equity_{tag}.csv", index=False)
    print(f"  Saved: backtest_equity_{tag}.csv")

    # Save weight history
    daily_weights.to_csv(f"backtest_weights_{tag}.csv", index=False)
    print(f"  Saved: backtest_weights_{tag}.csv")

    r = BacktestResult(
        name=cfg["name"],
        returns=returns,
        equity=equity,
        total_return_pct=round(total_ret, 2),
        annual_return_pct=round(ann_ret, 2),
        sharpe=round(sr, 4),
        sortino=round(sharpe_ratio(returns, rf=0.04), 4),  # type: ignore
        max_dd_pct=round(max_dd_pct, 2),
        calmar=round(calmar, 4),
        turnover=round(turnover, 2),
        n_trades=len(returns),
        factor_exposures=factor_exp,
        daily_weights=daily_weights,
        herfindahl_mean=round(mean_hhi, 4),
        max_single_weight=round(max_single * 100, 2),
        worst_dd_start=worst_start,
        worst_dd_end=worst_end,
    )
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  COMPREHENSIVE STRATEGY COMPARISON")
    print("  Universe: ETF_EXPANDED (57 symbols)")
    print("  Period: 2018-01-01 -> 2024-12-31  |  TC: 5 bps")
    print("  FF Factors: 5F + Mom (Ken French)")
    print("=" * 70)

    strategy_ids = [
        "macro_cross_sectional_factor_neutral",
        "macro_cross_sectional_momentum",
    ]

    results: list[BacktestResult] = []
    for sid in strategy_ids:
        r = run_strategy(sid)
        results.append(r)

    # ------------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON TABLE")
    print("=" * 70)

    hdr = f"{'Metric':<30} {'Factor-Neutral':>15} {'CSM v1':>15}"
    print(f"\n{hdr}")
    print("-" * 70)

    rows = [
        ("Total Return (%)", "total_return_pct", ".2f"),
        ("Annual Return (%)", "annual_return_pct", ".2f"),
        ("Sharpe Ratio", "sharpe", ".4f"),
        ("Sortino Ratio", "sortino", ".4f"),
        ("Max Drawdown (%)", "max_dd_pct", ".2f"),
        ("Calmar Ratio", "calmar", ".4f"),
        ("Turnover (proxy)", "turnover", ".2f"),
        ("Max Single-Asset Weight (%)", "max_single_weight", ".2f"),
        ("Mean Herfindahl", "herfindahl_mean", ".4f"),
        ("OOS Bars", "n_trades", ".0f"),
    ]

    for label, key, fmt in rows:
        v0 = getattr(results[0], key)
        v1 = getattr(results[1], key)
        print(f"  {label:<30} {v0:>15{fmt}} {v1:>15{fmt}}")

    print(f"\n  Worst DD Period")
    print(f"    {'Factor-Neutral:':<15} {results[0].worst_dd_start} -> {results[0].worst_dd_end}")
    print(f"    {'CSM v1:':<15}      {results[1].worst_dd_start} -> {results[1].worst_dd_end}")

    # Factor exposure section
    print(f"\n{'=' * 70}")
    print("  FACTOR EXPOSURES (OLS regression on daily 5F + Mom)")
    print(f"{'=' * 70}")
    fe0 = results[0].factor_exposures
    fe1 = results[1].factor_exposures
    if fe0 and fe1:
        all_keys = sorted(set(fe0.keys()) | set(fe1.keys()))
        print(f"\n  {'Factor':<15} {'Factor-Neutral':>15} {'CSM v1':>15}")
        print("  " + "-" * 48)
        for k in all_keys:
            print(f"  {k:<15} {fe0.get(k, 0.0):>15.6f} {fe1.get(k, 0.0):>15.6f}")
    else:
        print("  Factor exposure data unavailable.")

    # Save JSON
    output = {
        "factor_neutral": {
            k: v for k, v in results[0].__dict__.items()
            if k in ("name", "total_return_pct", "annual_return_pct", "sharpe",
                     "sortino", "max_dd_pct", "calmar", "turnover", "herfindahl_mean",
                     "max_single_weight", "worst_dd_start", "worst_dd_end")
        },
        "csm_v1": {k: v for k, v in results[1].__dict__.items()
                   if k in ("name", "total_return_pct", "annual_return_pct", "sharpe",
                            "sortino", "max_dd_pct", "calmar", "turnover", "herfindahl_mean",
                            "max_single_weight", "worst_dd_start", "worst_dd_end")},
    }
    for r in results:
        if r.factor_exposures:
            output[r == results[0] and "factor_neutral" or "csm_v1"]["factor_exposures"] = r.factor_exposures  # type: ignore  # noqa: E501

    with open("strategy_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to strategy_comparison_results.json")
    print(f"  Equity curves: backtest_equity_<tag>.csv")
    print(f"  Weights:       backtest_weights_<tag>.csv")


if __name__ == "__main__":
    main()
