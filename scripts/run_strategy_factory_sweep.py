"""Mass-validation sweep for strategy factory configs.

Loads all YAML configs from quantcore/strategy_factory/configs/, skips
configs with status != "active", runs backtest with signal-weighted returns
and transaction costs, logs results to experiments.csv, and produces
reports/factory_sweep.md.

Usage:
    python scripts/run_strategy_factory_sweep.py
    python scripts/run_strategy_factory_sweep.py --start 2018-01-01 --end 2024-12-31
    python scripts/run_strategy_factory_sweep.py --configs time_series_momentum carry
    python scripts/run_strategy_factory_sweep.py --data-source intraday --configs pairs_trading mean_reversion --start 2024-01-01 --end 2024-01-31
    python scripts/run_strategy_factory_sweep.py --data-source intraday --bar-size 15min --configs mean_reversion --start 2024-01-01 --end 2024-01-31
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from quantcore.risk.metrics import sharpe_ratio
from quantcore.strategy_factory.config_schema import StrategyFactoryConfig
from quantcore.strategy_factory.configurable_strategy import (
    ConfigurableStrategy,
    load_factory_config,
)
from scripts.real_data_loader import UNIVERSES, load_universe
from quantcore.data.intraday_loader import AlpacaIntradayLoader

# Re-use helpers from focused backtest
from scripts.run_backtest_focused import (
    _df_to_data,
    build_equity_curve,
    max_drawdown_from_equity,
    win_rate_from_trades,
    compute_signal_returns_with_trades,
)

CONFIGS_DIR = REPO_ROOT / "quantcore" / "strategy_factory" / "configs"
REPORTS_DIR = REPO_ROOT / "reports"
EXPERIMENTS_CSV = REPO_ROOT / "experiments.csv"

# Resample rule mapping for pandas resample()
RESAMPLE_RULES: dict[str, str] = {
    "1min": None,      # no resampling
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample long-format intraday OHLCV to a coarser bar size.

    Args:
        df: DataFrame with columns [symbol, timestamp, open, high, low, close, volume].
        rule: pandas resample rule — '5min', '15min', '30min', '1h'.

    Returns:
        Resampled DataFrame with same columns.
    """
    frames: list[pd.DataFrame] = []
    for symbol, group in df.groupby("symbol"):
        g = group.set_index("timestamp").sort_index()
        resampled = g.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
        resampled["symbol"] = symbol
        frames.append(resampled.reset_index())
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def discover_configs(
    configs_dir: Path,
    names: list[str] | None = None,
) -> list[StrategyFactoryConfig]:
    """Load all YAML configs, optionally filtered by name."""
    configs: list[StrategyFactoryConfig] = []
    for path in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = load_factory_config(path)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue
        if names and cfg.name not in names:
            continue
        configs.append(cfg)
    return configs


def run_single_backtest(
    cfg: StrategyFactoryConfig,
    start: str,
    end: str,
    tc_bps: float,
    risk_free_rate: float,
    data_source: str = "daily",
    bar_size: str = "5min",
) -> dict:
    """Run backtest for a single factory config. Returns metrics dict."""
    universe_name = cfg.universe
    if universe_name not in UNIVERSES:
        return {
            "config": cfg.name,
            "status": "SKIP",
            "reason": f"universe '{universe_name}' not in UNIVERSES",
        }

    print(f"  Loading {universe_name} data ({data_source}, bar={bar_size})...")
    try:
        if data_source == "intraday":
            tickers = sorted(UNIVERSES[universe_name]["symbols"])
            loader = AlpacaIntradayLoader()
            df = loader.load_universe_range(tickers, start, end)
            if bar_size != "1min" and not df.empty:
                rule = RESAMPLE_RULES.get(bar_size)
                if rule:
                    df = resample_ohlcv(df, rule)
                    print(f"    Resampled to {bar_size}: {len(df)} bars")
        else:
            df = load_universe(universe_name, start, end)
    except Exception as e:
        return {
            "config": cfg.name,
            "status": "ERROR",
            "reason": f"data load failed: {e}",
        }

    if df.empty:
        return {
            "config": cfg.name,
            "status": "ERROR",
            "reason": "empty dataframe",
        }

    data = _df_to_data(df)
    close = data["close"]
    n_periods = close.shape[0]
    timestamps = sorted(df["timestamp"].unique())

    if n_periods < cfg.warmup_bars + 10:
        return {
            "config": cfg.name,
            "status": "SKIP",
            "reason": f"insufficient data ({n_periods} bars < {cfg.warmup_bars + 10} needed)",
        }

    # Create strategy
    try:
        strat = ConfigurableStrategy(cfg)
    except Exception as e:
        return {
            "config": cfg.name,
            "status": "ERROR",
            "reason": f"strategy init failed: {e}",
        }

    # Warmup
    warmup_end = cfg.warmup_bars
    for bar_i in range(0, warmup_end):
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
    oos_start = warmup_end
    oos_end = n_periods

    oos_returns, trades = compute_signal_returns_with_trades(
        strat, data, oos_start, oos_end,
        timestamps=timestamps, tc_bps=tc_bps,
    )

    if len(oos_returns) == 0:
        return {
            "config": cfg.name,
            "status": "ERROR",
            "reason": "no OOS returns produced",
        }

    # Metrics
    equity = build_equity_curve(oos_returns, initial=1_000_000.0)
    total_return = (equity[-1] - 1_000_000.0) / 1_000_000.0 * 100.0
    sr = sharpe_ratio(oos_returns, rf=risk_free_rate)
    max_dd = max_drawdown_from_equity(equity) * 100.0
    wr = win_rate_from_trades(trades) * 100.0
    n_trades = len(trades)

    oos_start_ts = timestamps[oos_start] if oos_start < len(timestamps) else "?"
    oos_end_ts = timestamps[oos_end - 1] if oos_end - 1 < len(timestamps) else "?"

    return {
        "config": cfg.name,
        "type": cfg.type,
        "universe": universe_name,
        "status": "OK",
        "sharpe_ratio": round(sr, 4),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(wr, 1),
        "n_trades": n_trades,
        "oos_bars": len(oos_returns),
        "oos_period": f"{oos_start_ts} -> {oos_end_ts}",
        "final_equity": round(equity[-1], 2),
        "signal_formula": cfg.signal.formula,
        "rank_method": cfg.signal.rank or "none",
        "rebalance": cfg.rebalance,
        "market_neutral": cfg.market_neutral,
    }


def write_experiments_csv(results: list[dict], path: Path) -> None:
    """Append results to experiments.csv."""
    if not results:
        return
    # Collect all unique keys across all results (SKIP results have fewer fields)
    fieldnames = list(dict.fromkeys(k for r in results for k in r.keys()))
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def write_sweep_report(results: list[dict], path: Path, data_source: str = "daily", bar_size: str = "5min") -> None:
    """Generate markdown report."""
    ok = [r for r in results if r.get("status") == "OK"]
    skipped = [r for r in results if r.get("status") == "SKIP"]
    errors = [r for r in results if r.get("status") == "ERROR"]

    lines = [
        "# Strategy Factory Sweep Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total configs:** {len(results)}  ",
        f"**Tested:** {len(ok)}  ",
        f"**Skipped:** {len(skipped)}  ",
        f"**Errors:** {len(errors)}",
        "",
    ]

    # Results table
    if ok:
        ok_sorted = sorted(ok, key=lambda r: r.get("sharpe_ratio", -999), reverse=True)
        lines.append("## Results (sorted by Sharpe)")
        lines.append("")
        lines.append("| Config | Type | Universe | Sharpe | Return% | MaxDD% | WinRate% | Trades |")
        lines.append("|--------|------|----------|--------|---------|--------|----------|--------|")
        for r in ok_sorted:
            sr = r.get("sharpe_ratio", 0)
            flag = " +" if sr > 0 else ""
            lines.append(
                f"| {r['config']} | {r.get('type', '')} | {r.get('universe', '')} "
                f"| {sr:.4f}{flag} | {r.get('total_return_pct', 0):+.2f} "
                f"| {r.get('max_drawdown_pct', 0):.2f} "
                f"| {r.get('win_rate_pct', 0):.1f} "
                f"| {r.get('n_trades', 0)} |"
            )
        lines.append("")

    # Positive Sharpe highlight
    positive = [r for r in ok if r.get("sharpe_ratio", 0) > 0]
    if positive:
        lines.append("## Positive Sharpe Strategies")
        lines.append("")
        for r in sorted(positive, key=lambda r: r["sharpe_ratio"], reverse=True):
            lines.append(f"- **{r['config']}**: Sharpe {r['sharpe_ratio']:.4f}, "
                         f"Return {r['total_return_pct']:+.2f}%, "
                         f"MaxDD {r['max_drawdown_pct']:.2f}%")
        lines.append("")

    # Skipped / Errors
    if skipped:
        lines.append("## Skipped Configs")
        lines.append("")
        for r in skipped:
            lines.append(f"- {r['config']}: {r.get('reason', '')}")
        lines.append("")

    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- {r['config']}: {r.get('reason', '')}")
        lines.append("")

    # Summary
    lines.append("## Assumptions")
    lines.append("")
    if data_source == "intraday":
        lines.append(f"- Data: Alpaca Markets intraday bars (IEX feed), resampled to {bar_size}")
    else:
        lines.append("- Data: yfinance daily OHLCV (adjusted)")
    lines.append("- Transaction cost: 5 bps per unit weight turnover")
    lines.append("- Risk-free rate: 4% annual")
    lines.append("- Sharpe: annualized via sqrt(252)")
    lines.append("- Signal-weighted returns, no fill rejection")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy factory mass-validation sweep")
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date")
    parser.add_argument("--tc-bps", type=float, default=5.0, help="Transaction cost in bps")
    parser.add_argument("--rf", type=float, default=0.04, help="Risk-free rate (annual)")
    parser.add_argument("--configs", nargs="*", help="Specific config names to test")
    parser.add_argument("--include-requires-data", action="store_true",
                        help="Also test requires_data configs")
    parser.add_argument("--data-source", default="daily", choices=["daily", "intraday"],
                        help="Data source: daily (yfinance, default) or intraday (Alpaca 1min bars)")
    parser.add_argument("--bar-size", default="5min",
                        choices=["1min", "5min", "15min", "30min", "1h"],
                        help="Bar size for intraday data (default: 5min). Ignored for daily.")
    args = parser.parse_args()

    print("=" * 60)
    print("  STRATEGY FACTORY — MASS VALIDATION SWEEP")
    print("=" * 60)
    print(f"  Period:     {args.start} -> {args.end}")
    print(f"  Data:       {args.data_source}" + (f" ({args.bar_size})" if args.data_source == "intraday" else ""))
    print(f"  TC cost:    {args.tc_bps} bps")
    print(f"  Risk-free:  {args.rf:.0%} annual")
    print(f"  Configs:    {CONFIGS_DIR}")
    print()

    # Discover configs
    all_configs = discover_configs(CONFIGS_DIR, names=args.configs)
    print(f"Found {len(all_configs)} config(s).")

    # Filter by status
    active_configs = []
    skipped_configs = []
    for cfg in all_configs:
        if cfg.status == "active":
            active_configs.append(cfg)
        elif cfg.status == "requires_data" and args.include_requires_data:
            active_configs.append(cfg)
        else:
            skipped_configs.append(cfg)

    print(f"  Active:     {len(active_configs)}")
    print(f"  Skipped:    {len(skipped_configs)} (requires_data/requires_infra)")
    print()

    # Run backtests
    results: list[dict] = []

    # Record skipped
    for cfg in skipped_configs:
        results.append({
            "config": cfg.name,
            "status": "SKIP",
            "reason": f"status={cfg.status}: {cfg.requires or 'missing data/infra'}",
        })

    for i, cfg in enumerate(active_configs, 1):
        print(f"[{i}/{len(active_configs)}] Testing {cfg.name}...")
        print(f"  Signal: {cfg.signal.formula}, rank={cfg.signal.rank or 'none'}, "
              f"rebalance={cfg.rebalance}, neutral={cfg.market_neutral}")

        result = run_single_backtest(
            cfg, args.start, args.end, args.tc_bps, args.rf,
            data_source=args.data_source,
            bar_size=args.bar_size,
        )
        results.append(result)

        if result.get("status") == "OK":
            sr = result["sharpe_ratio"]
            flag = "POSITIVE" if sr > 0 else "NEGATIVE"
            print(f"  -> {flag}: Sharpe={sr:.4f}, Return={result['total_return_pct']:+.2f}%, "
                  f"MaxDD={result['max_drawdown_pct']:.2f}%")
        else:
            print(f"  -> {result['status']}: {result.get('reason', '')}")
        print()

    # Write outputs
    write_experiments_csv(results, EXPERIMENTS_CSV)
    print(f"Results appended to: {EXPERIMENTS_CSV}")

    write_sweep_report(results, REPORTS_DIR / "factory_sweep.md",
                       data_source=args.data_source, bar_size=args.bar_size)
    print(f"Report written to:   {REPORTS_DIR / 'factory_sweep.md'}")

    # Summary
    ok_results = [r for r in results if r.get("status") == "OK"]
    positive = [r for r in ok_results if r.get("sharpe_ratio", 0) > 0]
    print(f"\n{'=' * 60}")
    print(f"  SWEEP COMPLETE: {len(ok_results)} tested, {len(positive)} positive Sharpe")
    print(f"{'=' * 60}")

    if ok_results:
        print("\n  Ranked by Sharpe:")
        for r in sorted(ok_results, key=lambda r: r.get("sharpe_ratio", -999), reverse=True):
            sr = r["sharpe_ratio"]
            marker = "+" if sr > 0 else " "
            print(f"    {marker} {r['config']:35s}  Sharpe={sr:+.4f}  "
                  f"Return={r['total_return_pct']:+.2f}%  MaxDD={r['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()
