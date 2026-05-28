"""Multi-strategy aggregator backtest — combines daily macro + intraday pairs.

Loads three positive-Sharpe strategies:
  1. macro_cross_sectional_momentum  (daily, Sharpe ~+0.85)
  2. macro_state_selective_momentum  (daily, Sharpe ~+0.81)
  3. pairs_trading_intraday          (5-min, Sharpe ~+1.075)

Runs each strategy once on the full period, aggregates to daily returns,
then performs walk-forward weight optimization on daily return series.
This avoids re-running expensive backtests per fold.

Target: combined OOS Sharpe > +1.0 with MaxDD improvement.

Usage:
    python scripts/run_full_aggregator.py
    python scripts/run_full_aggregator.py --start 2024-01-01 --end 2024-12-31
    python scripts/run_full_aggregator.py --method risk_parity
    python scripts/run_full_aggregator.py --n-folds 6
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from quantcore.risk.metrics import sharpe_ratio
from quantcore.strategy_factory.config_schema import StrategyFactoryConfig
from quantcore.strategy_factory.configurable_strategy import (
    ConfigurableStrategy,
    load_factory_config_by_name,
)
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe
from quantcore.data.intraday_loader import AlpacaIntradayLoader
from scripts.run_backtest_focused import (
    _df_to_data,
    build_equity_curve,
    max_drawdown_from_equity,
    compute_signal_returns_with_trades,
)
from scripts.run_strategy_factory_sweep import resample_ohlcv


# ── Strategy definitions ───────────────────────────────────────────────────

STRATEGY_DEFS: dict[str, dict] = {
    "macro_cross_sectional_momentum": {
        "type": "framework",
        "data_source": "daily",
        "universe": "expanded_15",
        "warmup_bars": 60,
    },
    "macro_state_selective_momentum": {
        "type": "framework",
        "data_source": "daily",
        "universe": "expanded_15",
        "warmup_bars": 60,
    },
    "pairs_trading_intraday": {
        "type": "factory",
        "data_source": "intraday",
        "universe": "expanded_15",
        "bar_size": "5min",
    },
}


# ── Single strategy backtest ───────────────────────────────────────────────

def run_single_strategy(
    name: str,
    defn: dict,
    start: str,
    end: str,
    tc_bps: float,
    rf_rate: float,
) -> dict | None:
    """Run one strategy over the full period. Returns metrics dict or None."""
    strat_type = defn.get("type", "factory")
    universe_name = defn["universe"]

    if universe_name not in UNIVERSES:
        print(f"  ERROR: universe '{universe_name}' not in UNIVERSES")
        return None

    # Load strategy instance
    if strat_type == "factory":
        cfg = load_factory_config_by_name(name)
        if cfg is None:
            print(f"  ERROR: factory config not found for '{name}'")
            return None
        warmup_bars = cfg.warmup_bars
        strat = ConfigurableStrategy(cfg)
    else:
        # Framework registry strategy
        try:
            cls = get_strategy_class(name)
        except KeyError:
            print(f"  ERROR: framework strategy '{name}' not in registry")
            return None
        fw_cfg = StrategyConfig(
            name=name,
            strategy_type=StrategyType.MACRO,
            params={},
        )
        strat = cls(fw_cfg)
        strat.initialize()
        warmup_bars = defn.get("warmup_bars", 200)

    # Load data
    try:
        if defn["data_source"] == "intraday":
            tickers = sorted(UNIVERSES[universe_name]["symbols"])
            loader = AlpacaIntradayLoader()
            df = loader.load_universe_range(tickers, start, end)
            bar_size = defn.get("bar_size", "5min")
            if bar_size != "1min" and not df.empty:
                rule_map = {"5min": "5min", "15min": "15min", "30min": "30min", "1h": "1h"}
                rule = rule_map.get(bar_size)
                if rule:
                    df = resample_ohlcv(df, rule)
        else:
            df = load_universe(universe_name, start, end)
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    if df.empty:
        print(f"  ERROR: empty dataframe")
        return None

    data = _df_to_data(df)
    close = data["close"]
    n_periods = close.shape[0]
    timestamps = sorted(df["timestamp"].unique())

    if n_periods < warmup_bars + 10:
        print(f"  SKIP: insufficient data ({n_periods} bars < {warmup_bars + 10} needed)")
        return None

    # Warmup
    print(f"  Warming up ({warmup_bars} bars)...")

    for bar_i in range(0, warmup_bars):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v
        if bar_i < len(timestamps):
            bar_data["timestamp"] = timestamps[bar_i]
        try:
            strat.on_data(bar_data)
        except Exception:
            pass

    # OOS backtest
    oos_start = warmup_bars
    oos_end = n_periods
    bar_returns, trades = compute_signal_returns_with_trades(
        strat, data, oos_start, oos_end,
        timestamps=timestamps, tc_bps=tc_bps,
    )

    if len(bar_returns) == 0:
        print(f"  ERROR: no OOS returns")
        return None

    # Bar-level metrics (for intraday, annualize differently)
    bars_per_year = 252 * 78 if defn["data_source"] == "intraday" else 252
    sr_bar = sharpe_ratio(bar_returns, rf=rf_rate)

    # Aggregate to daily FIRST, then clip extreme returns
    bar_ts = timestamps[oos_start:oos_start + len(bar_returns)]
    ts_idx = pd.DatetimeIndex(bar_ts[:len(bar_returns)])
    # Normalize timezone-aware timestamps to UTC date for consistent alignment
    if ts_idx.tz is not None:
        ts_idx = ts_idx.tz_convert("UTC").tz_localize(None)
    ts_series = pd.Series(bar_returns, index=ts_idx)
    daily_rets = ts_series.resample("D").apply(lambda x: (1 + x).prod() - 1).dropna()
    # Clip extreme daily returns — intraday compounding artifacts
    daily_rets = daily_rets.clip(-0.05, 0.05)

    # Compute equity and metrics from clipped daily returns
    equity = build_equity_curve(daily_rets.values, initial=1_000_000.0)
    total_return = (equity[-1] - 1_000_000.0) / 1_000_000.0 * 100.0
    max_dd = max_drawdown_from_equity(equity) * 100.0

    if len(daily_rets) < 30:
        print(f"  SKIP: only {len(daily_rets)} daily bars")
        return None

    daily_sr = sharpe_ratio(daily_rets.values, rf=rf_rate)

    print(f"  Bar-level Sharpe: {sr_bar:+.4f}")
    print(f"  Daily Sharpe:     {daily_sr:+.4f}")
    print(f"  Return:           {total_return:+.2f}%")
    print(f"  MaxDD:            {max_dd:.2f}%")
    print(f"  Daily bars:       {len(daily_rets)}")

    return {
        "status": "OK",
        "name": name,
        "bar_sharpe": sr_bar,
        "daily_sharpe": daily_sr,
        "return_pct": total_return,
        "max_dd_pct": max_dd,
        "n_trades": len(trades),
        "n_daily_bars": len(daily_rets),
        "data_source": defn["data_source"],
        "daily_returns": daily_rets.values,
        "daily_dates": list(daily_rets.index),
        # Unclipped daily returns for weight computation (true volatility)
        "daily_returns_raw": ts_series.resample("D").apply(lambda x: (1 + x).prod() - 1).dropna().values,
    }


# ── Walk-forward weight optimization ──────────────────────────────────────

def walk_forward_combine(
    daily_returns_dict: dict[str, np.ndarray],
    daily_dates_dict: dict[str, list],
    method: str = "risk_parity",
    n_folds: int = 4,
    rf_rate: float = 0.04,
    raw_returns_dict: dict[str, np.ndarray] | None = None,
) -> dict:
    """Walk-forward weight optimization on aligned daily returns.

    Aligns all strategies to common dates, splits into n_folds,
    trains weights on expanding window, tests on next fold.
    """
    # Find common dates
    all_dates = None
    strat_names = list(daily_returns_dict.keys())

    for name in strat_names:
        dates = pd.DatetimeIndex(daily_dates_dict[name])
        # Normalize timezone-aware to naive UTC dates for consistent alignment
        if dates.tz is not None:
            dates = dates.tz_convert("UTC").tz_localize(None)
        # Normalize to date-only (strip time component)
        dates = pd.DatetimeIndex(pd.Series(dates).dt.date)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    if all_dates is None or len(all_dates) < 60:
        return {"status": "ERROR", "reason": "insufficient common dates"}

    all_dates = all_dates.sort_values()

    # Build aligned return matrices (clipped for portfolio, raw for weights)
    aligned = {}
    aligned_raw = {}
    for name in strat_names:
        idx = pd.DatetimeIndex(daily_dates_dict[name])
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        idx = pd.DatetimeIndex(pd.Series(idx).dt.date)
        s = pd.Series(daily_returns_dict[name], index=idx)
        aligned[name] = s.reindex(all_dates).fillna(0.0).values
        if raw_returns_dict and name in raw_returns_dict:
            s_raw = pd.Series(raw_returns_dict[name], index=idx)
            aligned_raw[name] = s_raw.reindex(all_dates).fillna(0.0).values
        else:
            aligned_raw[name] = aligned[name]

    returns_matrix = np.column_stack([aligned[name] for name in strat_names])
    raw_matrix = np.column_stack([aligned_raw[name] for name in strat_names])
    n_days = returns_matrix.shape[0]
    n_strats = returns_matrix.shape[1]

    print(f"\n  Aligned: {n_strats} strategies x {n_days} common daily bars")

    # Walk-forward folds
    fold_size = n_days // n_folds
    if fold_size < 20:
        n_folds = max(1, n_days // 60)
        fold_size = n_days // n_folds

    rf_daily = (1.0 + rf_rate) ** (1.0 / 252.0) - 1.0
    combined_returns = []
    fold_metrics = []

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = min(test_start + fold_size, n_days) if fold_idx < n_folds - 1 else n_days

        # Training: all data before test period
        train_end = test_start
        min_train = 40  # minimum training bars

        if train_end < min_train:
            weights = np.ones(n_strats) / n_strats
        else:
            # Use raw (unclipped) returns for weight computation — true volatility
            train_data = raw_matrix[:train_end]
            weights = _compute_weights(train_data, method, rf_daily)

        # Test period
        test_data = returns_matrix[test_start:test_end]
        if len(test_data) == 0:
            continue

        oos_combined = test_data @ weights
        combined_returns.extend(oos_combined.tolist())

        fold_sr = sharpe_ratio(oos_combined, rf=rf_rate) if len(oos_combined) > 5 else 0.0
        fold_eq = build_equity_curve(oos_combined, initial=1_000_000.0)
        fold_ret = (fold_eq[-1] - 1_000_000.0) / 1_000_000.0 * 100.0
        fold_dd = max_drawdown_from_equity(fold_eq) * 100.0

        fold_start_date = str(all_dates[test_start].date())
        fold_end_date = str(all_dates[test_end - 1].date())

        print(f"  Fold {fold_idx + 1}: {fold_start_date} -> {fold_end_date} | "
              f"Sharpe={fold_sr:+.4f} Return={fold_ret:+.2f}% MaxDD={fold_dd:.2f}%")
        print(f"    Weights: {dict(zip(strat_names, [f'{w:.3f}' for w in weights]))}")

        fold_metrics.append({
            "fold": fold_idx + 1,
            "start": fold_start_date,
            "end": fold_end_date,
            "sharpe": round(fold_sr, 4),
            "return_pct": round(fold_ret, 2),
            "max_dd_pct": round(fold_dd, 2),
            "weights": {name: round(float(w), 4) for name, w in zip(strat_names, weights)},
            "n_days": len(oos_combined),
        })

    if not combined_returns:
        return {"status": "ERROR", "reason": "no OOS returns"}

    combined_arr = np.array(combined_returns)
    final_weights = weights  # last fold's weights

    return {
        "status": "OK",
        "combined_returns": combined_arr,
        "fold_metrics": fold_metrics,
        "n_folds": len(fold_metrics),
        "strategy_names": strat_names,
        "final_weights": {name: round(float(w), 4) for name, w in zip(strat_names, final_weights)},
    }


def _compute_weights(returns_matrix: np.ndarray, method: str, rf_daily: float) -> np.ndarray:
    """Compute portfolio weights for a training window."""
    n_strats = returns_matrix.shape[1]

    if method == "equal_weight":
        return np.ones(n_strats) / n_strats

    if method == "risk_parity":
        vols = np.std(returns_matrix, axis=0, ddof=1)
        vols = np.maximum(vols, 1e-10)
        inv_vol = 1.0 / vols
        return inv_vol / np.sum(inv_vol)

    if method == "hrp":
        from quantcore.portfolio.hrp import compute_hrp_weights
        df = pd.DataFrame(returns_matrix)
        weights_dict = compute_hrp_weights(df)
        w = np.array([weights_dict.get(i, 0.0) for i in range(n_strats)])
        total = w.sum()
        return w / total if total > 0 else np.ones(n_strats) / n_strats

    # mean_variance
    mu = np.mean(returns_matrix, axis=0) - rf_daily
    cov = np.cov(returns_matrix, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov += np.eye(n_strats) * 1e-6

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret / port_vol) * np.sqrt(252.0)

    x0 = np.ones(n_strats) / n_strats
    bounds = [(0.0, 1.0)] * n_strats
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    try:
        result = sp_optimize.minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        return result.x if result.success else x0
    except Exception:
        return x0


# ── Report generation ─────────────────────────────────────────────────────

def generate_report(
    standalone: dict[str, dict],
    wf_result: dict,
    method: str,
    rf_rate: float,
    output_path: Path,
) -> None:
    """Generate markdown report."""
    combined = wf_result["combined_returns"]
    combined_sharpe = sharpe_ratio(combined, rf=rf_rate)
    combined_eq = build_equity_curve(combined, initial=1_000_000.0)
    combined_dd = max_drawdown_from_equity(combined_eq)
    combined_ret = (combined_eq[-1] - 1_000_000.0) / 1_000_000.0 * 100.0

    lines = [
        "# Multi-Strategy Aggregator Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Strategy Components",
        "",
        "| Strategy | Data Source | Daily Sharpe | MaxDD% | Return% | Trades |",
        "|----------|------------|-------------|--------|---------|--------|",
    ]

    for name, m in standalone.items():
        lines.append(
            f"| {name} | {m.get('data_source', '?')} "
            f"| {m.get('daily_sharpe', 0):.4f} "
            f"| {m.get('max_dd_pct', 0):.2f} "
            f"| {m.get('return_pct', 0):+.2f} "
            f"| {m.get('n_trades', 0)} |"
        )

    lines.extend([
        "",
        "## Combined Portfolio (Walk-Forward OOS)",
        "",
        f"- **Combination method:** {method}",
        f"- **OOS Sharpe:** {combined_sharpe:.4f}",
        f"- **OOS Total Return:** {combined_ret:+.2f}%",
        f"- **OOS Max Drawdown:** {combined_dd * 100:.2f}%",
        f"- **OOS Days:** {len(combined)}",
        f"- **Walk-Forward Folds:** {wf_result.get('n_folds', 0)}",
        "",
        "## Final Weights",
        "",
    ])

    for name, w in wf_result.get("final_weights", {}).items():
        lines.append(f"- **{name}**: {w:.4f}")

    lines.append("")

    # Fold details
    folds = wf_result.get("fold_metrics", [])
    if folds:
        lines.append("## Walk-Forward Fold Results")
        lines.append("")
        lines.append("| Fold | Period | Sharpe | Return% | MaxDD% |")
        lines.append("|------|--------|--------|---------|--------|")
        for f in folds:
            lines.append(
                f"| {f['fold']} "
                f"| {f['start']} -> {f['end']} "
                f"| {f['sharpe']:.4f} "
                f"| {f['return_pct']:+.2f} "
                f"| {f['max_dd_pct']:.2f} |"
            )
        lines.append("")

    # Comparison
    best_standalone = max(standalone.values(), key=lambda m: m.get("daily_sharpe", -999))
    best_name = [k for k, v in standalone.items() if v is best_standalone][0]

    lines.extend([
        "## Comparison",
        "",
        f"- **Best standalone (daily):** {best_name} (Sharpe {best_standalone['daily_sharpe']:.4f})",
        f"- **Combined OOS Sharpe:** {combined_sharpe:.4f}",
        f"- **Improvement:** {combined_sharpe - best_standalone['daily_sharpe']:+.4f}",
        "",
    ])

    if combined_sharpe > 1.0:
        lines.append("**TARGET MET: Combined OOS Sharpe > +1.0**")
    else:
        lines.append(f"**Combined OOS Sharpe:** {combined_sharpe:.4f}")
    lines.append("")

    lines.extend([
        "## Assumptions",
        "",
        "- Daily macro strategies: yfinance daily OHLCV (adjusted)",
        "- Pairs intraday: Alpaca IEX 5-min bars, aggregated to daily returns",
        "- Walk-forward: expanding-window training, monthly OOS folds",
        "- Transaction cost: 5 bps per unit weight turnover",
        "- Risk-free rate: 4% annual",
        "- Sharpe: annualized via sqrt(252) on daily returns",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-strategy aggregator backtest")
    parser.add_argument("--start", default="2024-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date")
    parser.add_argument("--method", default="risk_parity",
                        choices=["mean_variance", "risk_parity", "equal_weight", "hrp"],
                        help="Weight combination method")
    parser.add_argument("--tc-bps", type=float, default=5.0, help="Transaction cost bps")
    parser.add_argument("--rf", type=float, default=0.04, help="Risk-free rate")
    parser.add_argument("--n-folds", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--output", default="reports/aggregator_results.md",
                        help="Output report path")
    args = parser.parse_args()

    print("=" * 60)
    print("  QUANTCORE — MULTI-STRATEGY AGGREGATOR BACKTEST")
    print("=" * 60)
    print(f"  Period:    {args.start} -> {args.end}")
    print(f"  Method:    {args.method}")
    print(f"  TC:        {args.tc_bps} bps")
    print(f"  RF:        {args.rf:.0%} annual")
    print(f"  Folds:     {args.n_folds}")
    print(f"  Strategies: {', '.join(STRATEGY_DEFS.keys())}")
    print("=" * 60)

    # ── Step 1: Run each strategy once ──
    print("\n" + "=" * 60)
    print("  STEP 1: Individual Strategy Backtests")
    print("=" * 60)

    standalone: dict[str, dict] = {}
    daily_returns_dict: dict[str, np.ndarray] = {}
    daily_dates_dict: dict[str, list] = {}

    for name, defn in STRATEGY_DEFS.items():
        print(f"\n[{name}]")
        result = run_single_strategy(name, defn, args.start, args.end, args.tc_bps, args.rf)

        if result is None:
            print(f"  FAILED")
            continue

        standalone[name] = result
        daily_returns_dict[name] = result["daily_returns"]
        daily_dates_dict[name] = result["daily_dates"]

    if len(daily_returns_dict) < 2:
        print("\nERROR: need at least 2 strategies with results")
        return 1

    # ── Step 2: Walk-forward combination ──
    print("\n" + "=" * 60)
    print("  STEP 2: Walk-Forward Weight Optimization")
    print("=" * 60)

    raw_returns_dict = {
        name: result.get("daily_returns_raw", result["daily_returns"])
        for name, result in standalone.items()
    }
    wf_result = walk_forward_combine(
        daily_returns_dict, daily_dates_dict,
        method=args.method,
        n_folds=args.n_folds,
        rf_rate=args.rf,
        raw_returns_dict=raw_returns_dict,
    )

    if wf_result["status"] != "OK":
        print(f"  ERROR: {wf_result.get('reason')}")
        return 1

    # ── Step 3: Final results ──
    combined = wf_result["combined_returns"]
    combined_sharpe = sharpe_ratio(combined, rf=args.rf)
    combined_eq = build_equity_curve(combined, initial=1_000_000.0)
    combined_dd = max_drawdown_from_equity(combined_eq) * 100.0
    combined_ret = (combined_eq[-1] - 1_000_000.0) / 1_000_000.0 * 100.0

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  OOS Combined Sharpe:  {combined_sharpe:+.4f}")
    print(f"  OOS Combined Return:  {combined_ret:+.2f}%")
    print(f"  OOS Combined MaxDD:   {combined_dd:.2f}%")
    print(f"  OOS Days:             {len(combined)}")

    best_standalone = max(standalone.values(), key=lambda m: m.get("daily_sharpe", -999))
    print(f"\n  Best standalone daily Sharpe: {best_standalone['daily_sharpe']:+.4f}")
    print(f"  Combined OOS Sharpe:          {combined_sharpe:+.4f}")
    print(f"  Improvement:                  {combined_sharpe - best_standalone['daily_sharpe']:+.4f}")

    if combined_sharpe > 1.0:
        print(f"\n  ** TARGET MET: Combined OOS Sharpe > +1.0 **")
    else:
        print(f"\n  Combined Sharpe: {combined_sharpe:+.4f}")

    # ── Write report ──
    output_path = REPO_ROOT / args.output
    generate_report(standalone, wf_result, args.method, args.rf, output_path)
    print(f"\nReport written to: {output_path}")

    # Save JSON
    json_path = REPO_ROOT / "reports" / "aggregator_results.json"
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "period": f"{args.start} -> {args.end}",
        "method": args.method,
        "n_folds": args.n_folds,
        "individual": {
            name: {k: v for k, v in m.items() if k not in ("daily_returns", "daily_dates")}
            for name, m in standalone.items()
        },
        "combined_oos_sharpe": round(combined_sharpe, 4),
        "combined_oos_return_pct": round(combined_ret, 2),
        "combined_oos_max_dd_pct": round(combined_dd, 2),
        "fold_metrics": wf_result["fold_metrics"],
        "final_weights": wf_result["final_weights"],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"JSON results saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
