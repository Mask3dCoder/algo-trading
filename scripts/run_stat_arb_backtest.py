"""Statistical arbitrage backtest validation.

Pipeline:
  1. Load daily close prices for S&P 500 subset (2020-2024)
  2. Find cointegrated pairs on training period (2020-2021)
  3. Backtest mean reversion strategy on OOS period (2022-2024)
  4. Report: Sharpe, Max DD, Win Rate, Number of Trades

Outputs:
  - reports/stat_arb_validation.md
  - data/pairs_found.csv
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantcore.data import yfinance_loader
from quantcore.data.sources.historical_universe import HistoricalUniverseLoader
from quantcore.alpha.pairs_finder import PairsFinder
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.alpha.stat_arb_mean_reversion import StatArbMeanReversion
from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

UNIVERSE_START = "2019-01-01"  # extra history for z-score warmup
TRAIN_START = "2020-01-01"
TRAIN_END = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END = "2024-12-31"

HALF_SPREAD = 5e-4   # 5 bps
IMPACT = 1e-3         # 10 bps
PER_SHARE_COST = 0.005
MAX_NOTIONAL = 1_000_000.0

TOP_N_PAIRS = 20
MAX_TEST_PAIRS = None  # test all pairs (4851 for 99 stocks)

REPORTS_DIR = _ROOT / "reports"


# ── Data loading ───────────────────────────────────────────────────────

def load_universe_subset(n: int = 100) -> list[str]:
    """Load top N liquid S&P 500 stocks."""
    loader = HistoricalUniverseLoader()
    all_symbols = loader.get_constituents("SP500", date(2020, 1, 1))
    # Take first N (these are roughly alphabetical, not ideal but fine for now)
    return all_symbols[:n]


def load_daily_close(symbols: list[str]) -> pd.DataFrame:
    """Load daily close prices into a wide DataFrame."""
    _log.info("Loading daily close for %d symbols (%s -> %s)", len(symbols), UNIVERSE_START, OOS_END)
    data = yfinance_loader.load_multiple(symbols, UNIVERSE_START, OOS_END, interval="1d")
    _log.info("Loaded %d / %d symbols", len(data), len(symbols))

    frames = {}
    for sym, df in data.items():
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        frames[sym] = df["close"]

    prices = pd.DataFrame(frames)
    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.90))
    _log.info("Price matrix: %s", prices.shape)
    return prices


# ── Walk-forward backtest ──────────────────────────────────────────────

def run_backtest(
    strategy: StatArbMeanReversion,
    prices: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[np.ndarray, list[dict]]:
    """Run the stat arb backtest on daily data.

    Since we're using daily bars (not intraday), we simulate
    hourly rebalancing by checking every bar.

    Returns (daily_returns, trade_log).
    """
    mask = (prices.index >= start) & (prices.index <= end)
    oos_prices = prices.loc[mask]
    symbols = list(oos_prices.columns)

    if len(oos_prices) < 20:
        return np.array([]), []

    n_days = len(oos_prices)
    n_assets = len(symbols)

    # Build data dict for strategy
    close = oos_prices.values  # (n_days, n_assets)

    daily_returns: list[float] = []
    trade_log: list[dict] = []
    prev_weights = np.zeros(n_assets)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    for i in range(1, n_days):
        # Build data slice up to current bar
        data_slice = {
            "close": close[:i + 1],
            "symbols": symbols,
        }

        signals = strategy.on_data(data_slice)

        # Convert signals to weights
        target_weights = np.zeros(n_assets)
        for sig in signals:
            idx = sym_to_idx.get(sig.symbol)
            if idx is not None and sig.direction != 0:
                target_weights[idx] = sig.direction * sig.size / MAX_NOTIONAL
            elif idx is not None:
                target_weights[idx] = 0.0

        # Daily return from previous weights
        if i > 0:
            rets = (close[i] - close[i - 1]) / np.maximum(close[i - 1], 1e-10)
            day_ret = float(np.dot(prev_weights, rets))
            daily_returns.append(day_ret)

            # Transaction costs
            turnover = float(np.sum(np.abs(target_weights - prev_weights)))
            cost = turnover * (HALF_SPREAD + IMPACT)
            daily_returns[-1] -= cost

            if turnover > 1e-8:
                trade_log.append({
                    "day": i,
                    "date": str(oos_prices.index[i].date()),
                    "turnover": round(turnover, 4),
                    "cost": round(cost, 6),
                    "n_signals": len(signals),
                })

        prev_weights = target_weights.copy()

    return np.array(daily_returns), trade_log


def compute_metrics(returns: np.ndarray, trade_log: list[dict]) -> dict:
    """Compute backtest performance metrics."""
    if len(returns) == 0:
        return {"error": "no returns"}

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))
    annual_sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-12 else 0.0

    # Bootstrap CI
    block_size = max(5, int(len(returns) ** (1 / 3)))
    ci_lower, ci_upper = bootstrap_confidence_interval(
        returns, block_size=block_size, n_boot=2000, ci=0.90,
    )

    # DSR
    skew = float(np.mean((returns - mean_ret) ** 3) / (std_ret ** 3)) if std_ret > 1e-12 else 0.0
    kurt = float(np.mean((returns - mean_ret) ** 4) / (std_ret ** 4) - 3.0) if std_ret > 1e-12 else 3.0
    dsr_pval, dsr_stat = deflated_sharpe_ratio(
        observed_sharpe=annual_sharpe,
        num_trials=1,
        sample_length=len(returns),
        skewness=skew,
        kurtosis_=kurt,
    )

    # Max drawdown
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.abs(dd.min()))

    # Win rate
    win_rate = float(np.mean(returns > 0))

    # Number of trades (from trade log)
    n_trades = len(trade_log)

    # Avg daily turnover
    total_turnover = sum(t["turnover"] for t in trade_log)
    avg_turnover = total_turnover / max(n_trades, 1)

    return {
        "oos_sharpe": round(annual_sharpe, 3),
        "ci_90_lower": round(ci_lower, 3),
        "ci_90_upper": round(ci_upper, 3),
        "dsr_pvalue": round(dsr_pval, 4),
        "dsr_stat": round(dsr_stat, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": n_trades,
        "avg_turnover": round(avg_turnover, 4),
        "n_oos_days": len(returns),
    }


# ── Report generation ─────────────────────────────────────────────────

def generate_report(
    metrics: dict,
    pairs: list[dict],
    trade_log: list[dict],
    daily_returns: np.ndarray,
) -> str:
    lines = [
        "# Statistical Arbitrage Validation Report",
        "",
        f"**Date**: {date.today().isoformat()}  ",
        f"**Train Period**: {TRAIN_START} - {TRAIN_END}  ",
        f"**OOS Period**: {OOS_START} - {OOS_END}  ",
        f"**Costs**: {HALF_SPREAD*1e4:.0f}bps half-spread + {IMPACT*1e4:.0f}bps impact  ",
        f"**Pairs**: {len(pairs)} cointegrated pairs (top {TOP_N_PAIRS} by half-life)  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    if "error" not in metrics:
        verdict = "PAPER" if metrics["oos_sharpe"] > 1.0 else "RESEARCH" if metrics["oos_sharpe"] > 0.0 else "REJECT"
        lines += [
            f"| OOS Sharpe | {metrics['oos_sharpe']:.3f} |",
            f"| 90% CI | [{metrics['ci_90_lower']:.3f}, {metrics['ci_90_upper']:.3f}] |",
            f"| DSR p-value | {metrics['dsr_pvalue']:.4f} |",
            f"| Max Drawdown | {metrics['max_drawdown']:.2%} |",
            f"| Win Rate | {metrics['win_rate']:.1%} |",
            f"| Number of Trades | {metrics['n_trades']} |",
            f"| Avg Turnover/Trade | {metrics['avg_turnover']:.4f} |",
            f"| OOS Days | {metrics['n_oos_days']:,} |",
            f"| **Verdict** | **{verdict}** |",
        ]
    else:
        lines += [f"| Error | {metrics['error']} |"]

    lines += [
        "",
        "## Cointegrated Pairs (Top 20 by Half-Life)",
        "",
        "| # | Stock A | Stock B | Hedge Ratio | ADF p-val | Half-Life (days) |",
        "|---|---------|---------|-------------|-----------|------------------|",
    ]

    for i, p in enumerate(pairs[:20], 1):
        lines.append(
            f"| {i} | {p['stock_a']} | {p['stock_b']} | {p['hedge_ratio']:.4f} "
            f"| {p['adf_pvalue']:.4f} | {p['half_life']:.1f} |"
        )

    # Monthly returns breakdown
    if len(daily_returns) > 0 and "error" not in metrics:
        lines += [
            "",
            "## Monthly Returns",
            "",
            "| Month | Return | Sharpe |",
            "|-------|--------|--------|",
        ]
        # Approximate monthly grouping (21 trading days)
        n_months = len(daily_returns) // 21
        for m in range(n_months):
            start_idx = m * 21
            end_idx = min((m + 1) * 21, len(daily_returns))
            month_rets = daily_returns[start_idx:end_idx]
            month_mean = float(np.mean(month_rets))
            month_std = float(np.std(month_rets, ddof=1))
            month_sharpe = month_mean / month_std * np.sqrt(252) if month_std > 1e-12 else 0.0
            month_label = f"Month {m + 1}"
            lines.append(f"| {month_label} | {month_mean * 100:.2f}% | {month_sharpe:.2f} |")

    lines += [
        "",
        "## Methodology",
        "",
        "- **Pairs Selection**: Engle-Granger cointegration test (ADF p < 0.05)",
        "- **Half-Life**: Ornstein-Uhlenbeck process fit on spread",
        "- **Entry**: z-score > 2.0 (short spread) or < -2.0 (long spread)",
        "- **Exit**: z-score crosses 0.0",
        "- **Stop-Loss**: |z-score| > 4.0",
        "- **Position Sizing**: $50K per leg, max 10 simultaneous pairs",
        "- **Rebalancing**: Every bar (daily in this backtest)",
        "- **Cost Model**: 5 bps half-spread + 10 bps impact per trade",
        "",
        "## Known Limitations",
        "",
        "- Using daily bars (not intraday) — z-score signals are less granular",
        "- Survivorship bias: universe uses current S&P 500 constituents",
        "- No borrow cost for short leg",
        "- Pairs found on training period may decay in OOS",
        "- yfinance data only — no tick-level or order book validation",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    _log.info("=" * 60)
    _log.info("Statistical Arbitrage Backtest Validation")
    _log.info("=" * 60)

    # 1. Load universe
    symbols = load_universe_subset(n=100)
    prices = load_daily_close(symbols)
    _log.info("Universe: %d stocks, %d trading days", prices.shape[1], len(prices))

    # 2. Find pairs on training period
    _log.info("-" * 40)
    _log.info("Finding cointegrated pairs: %s -> %s", TRAIN_START, TRAIN_END)

    train_prices = prices.loc[(prices.index >= TRAIN_START) & (prices.index <= TRAIN_END)]
    finder = PairsFinder(
        lookback_days=min(504, len(train_prices)),
        max_pvalue=0.15,
        max_pairs=TOP_N_PAIRS,
        max_half_life=60.0,
        min_half_life=1.0,
    )
    pairs = finder.find_pairs(train_prices, max_test_pairs=MAX_TEST_PAIRS)
    _log.info("Found %d cointegrated pairs", len(pairs))

    if not pairs:
        _log.error("No cointegrated pairs found. Aborting.")
        return

    # Save pairs
    pairs_df = PairsFinder.pairs_to_dataframe(pairs)
    pairs_path = _ROOT / "data" / "pairs_found.csv"
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(pairs_path, index=False)
    _log.info("Saved pairs -> %s", pairs_path)

    # 3. Build strategy
    pairs_dicts = pairs_df.to_dict("records")
    config = StrategyConfig(
        name="stat_arb_mean_reversion",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=MAX_NOTIONAL,
        max_leverage=1.0,
        risk_budget_pct=0.02,
        params={
            "entry_threshold": 2.0,
            "exit_threshold": 0.0,
            "stop_threshold": 4.0,
            "lookback_zscore": 20,
            "max_pairs": TOP_N_PAIRS,
            "max_pair_notional": 50_000,
            "rebalance_freq": 1,  # daily since we're using daily bars
        },
    )

    strategy = StatArbMeanReversion(config, pairs=pairs_dicts)
    strategy.load_pairs(pairs_dicts)

    # 4. Backtest on OOS
    _log.info("-" * 40)
    _log.info("Backtesting OOS: %s -> %s", OOS_START, OOS_END)

    daily_returns, trade_log = run_backtest(strategy, prices, OOS_START, OOS_END)
    _log.info("OOS returns: %d days, %d trades", len(daily_returns), len(trade_log))

    # 5. Compute metrics
    metrics = compute_metrics(daily_returns, trade_log)

    if "error" not in metrics:
        _log.info(
            "Sharpe=%.3f, CI=[%.3f, %.3f], MaxDD=%.2f%%, WinRate=%.1f%%, Trades=%d",
            metrics["oos_sharpe"],
            metrics["ci_90_lower"],
            metrics["ci_90_upper"],
            metrics["max_drawdown"] * 100,
            metrics["win_rate"] * 100,
            metrics["n_trades"],
        )
    else:
        _log.error("Backtest error: %s", metrics["error"])

    # 6. Generate report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "stat_arb_validation.md"
    report = generate_report(metrics, pairs_dicts, trade_log, daily_returns)
    report_path.write_text(report, encoding="utf-8")
    _log.info("Report -> %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("STAT ARB RESULTS")
    print("=" * 70)
    if "error" not in metrics:
        verdict = "PAPER" if metrics["oos_sharpe"] > 1.0 else "RESEARCH" if metrics["oos_sharpe"] > 0.0 else "REJECT"
        print(f"  Sharpe:      {metrics['oos_sharpe']:+.3f}")
        print(f"  CI(90%):     [{metrics['ci_90_lower']:+.3f}, {metrics['ci_90_upper']:+.3f}]")
        print(f"  Max DD:      {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate:    {metrics['win_rate']:.1%}")
        print(f"  Trades:      {metrics['n_trades']}")
        print(f"  Verdict:     {verdict}")
    else:
        print(f"  ERROR: {metrics['error']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
