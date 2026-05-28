"""Run sentiment momentum backtest with walk-forward validation.

Tests the SentimentMomentumStrategy on 2022-2024 data using synthetic
sentiment (or real Alpaca News data if keys are configured).

Reports Sharpe, MaxDD, Win Rate, and optionally plugs into the
MultiStrategyAggregator if OOS Sharpe > 0.3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from quantcore.risk.metrics import sharpe_ratio
from quantcore.strategies.base import StrategyConfig, StrategyType

# Trigger @register decorators
import quantcore.alpha.sentiment_momentum  # noqa: F401
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.macro.state_selective_momentum  # noqa: F401

from quantcore.data.news_sentiment import NewsSentimentLoader
from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe


# ---------------------------------------------------------------------------
# Data helpers (from run_aggregator_backtest.py)
# ---------------------------------------------------------------------------

def _causal_ffill_2d(arr: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(arr)
    filled = df.ffill(limit=max_gap)
    fill_mask = df.isna().values & (~filled.isna().values)
    return filled.values, fill_mask


def _df_to_data(df: pd.DataFrame) -> dict:
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)
    if n_assets == 0:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivots: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field)
        pivots[field] = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted

    if pivots["close"].empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}

    data = {
        "close": pivots["close"].values,
        "high": pivots["high"].values,
        "low": pivots["low"].values,
        "open": pivots["open"].values,
        "volume": pivots["volume"].values,
        "symbols": symbols,
    }
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            df_filled, _ = _causal_ffill_2d(arr, max_gap=5)
            data[k] = np.nan_to_num(df_filled, nan=0.0)
    return data


# ---------------------------------------------------------------------------
# Signal-based return computation (from run_backtest_focused.py)
# ---------------------------------------------------------------------------

def compute_signal_returns(
    strat,
    data: dict,
    start: int,
    end: int,
    timestamps: list | None = None,
    tc_bps: float = 5.0,
) -> tuple[np.ndarray, list[dict]]:
    """Compute bar-level signal-weighted returns with trade tracking."""
    if end - start < 2:
        return np.array([], dtype=float), []

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)
    if n_assets == 0:
        return np.array([], dtype=float), []

    asset_rets = (close[start + 1:end] - close[start:end - 1]) / np.maximum(close[start:end - 1], 1e-10)

    bar_rets = np.zeros(end - start - 1)
    carried_weights: dict[str, float] = {}
    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for idx, bar_i in enumerate(range(start, end - 1)):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        if timestamps is not None and bar_i < len(timestamps):
            ts = timestamps[bar_i]
            if hasattr(strat, 'set_timestamp'):
                strat.set_timestamp(ts)
            bar_data["timestamp"] = ts

        try:
            signals = strat.on_data(bar_data) or []
        except Exception as e:
            import traceback
            print(f"  signal error bar {bar_i}: {e}")
            traceback.print_exc()
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        current_weights: dict[str, float] = {}
        current_signals: dict[str, dict] = {}

        for s in signals:
            weight = float(s.direction) * float(s.strength)
            if s.direction != 0:
                current_weights[s.symbol] = weight
                current_signals[s.symbol] = {
                    "direction": s.direction,
                    "strength": s.strength,
                }
            try:
                asset_idx = symbols.index(s.symbol)
            except (ValueError, IndexError):
                continue
            weighted_ret += weight * float(asset_rets[idx, asset_idx])
            total_weight += abs(float(s.strength))

        if total_weight > 1e-10:
            bar_rets[idx] = weighted_ret / total_weight

        # Transaction cost
        if tc_bps > 0 and current_weights:
            turnover = 0.0
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            for sym in all_syms:
                prev_w = carried_weights.get(sym, 0.0)
                curr_w = current_weights.get(sym, 0.0)
                turnover += abs(curr_w - prev_w)
            bar_rets[idx] -= turnover * tc_bps / 10000.0

        # Trade tracking
        if current_signals:
            for sym in list(positions.keys()):
                pos = positions[sym]
                exit_ts = timestamps[bar_i] if timestamps and bar_i < len(timestamps) else None
                exit_price = float(close[bar_i, symbols.index(sym)]) if bar_i < close.shape[0] else None
                if sym not in current_signals or current_signals[sym]["direction"] != pos["direction"]:
                    pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
                    pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
                    trades.append({
                        "symbol": sym, "direction": pos["direction"],
                        "entry_ts": pos["entry_ts"], "exit_ts": exit_ts,
                        "entry_price": pos["entry_price"], "exit_price": exit_price,
                        "pnl_abs": pnl_abs, "pnl_pct": pnl_pct,
                    })
                    positions.pop(sym)

            for sym in current_signals:
                if sym not in positions:
                    sig = current_signals[sym]
                    entry_price = float(close[bar_i, symbols.index(sym)]) if bar_i < close.shape[0] else None
                    entry_ts = timestamps[bar_i] if timestamps and bar_i < len(timestamps) else None
                    positions[sym] = {
                        "direction": sig["direction"],
                        "entry_ts": entry_ts,
                        "entry_price": entry_price,
                    }

        if current_weights:
            carried_weights = dict(current_weights)

    # Close remaining positions
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        exit_ts = timestamps[end - 1] if timestamps and end - 1 < len(timestamps) else None
        exit_price = float(close[end - 1, symbols.index(sym)]) if end - 1 < close.shape[0] else None
        pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
        pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
        trades.append({
            "symbol": sym, "direction": pos["direction"],
            "entry_ts": pos["entry_ts"], "exit_ts": exit_ts,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "pnl_abs": pnl_abs, "pnl_pct": pnl_pct,
        })

    return bar_rets, trades


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    strategy_factory,
    data: dict,
    timestamps: list,
    sentiment_df: pd.DataFrame,
    train_bars: int = 504,   # ~2 years
    test_bars: int = 252,    # ~1 year
    tc_bps: float = 5.0,
    warmup_bars: int = 60,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Run anchored walk-forward backtest.

    For each fold: train on [0, train_end], test on [train_end, train_end + test_bars].
    """
    n_bars = data["close"].shape[0]
    fold_results: list[dict] = []
    all_returns: list[np.ndarray] = []
    all_trades: list[dict] = []

    fold = 0
    test_start = train_bars
    while test_start + test_bars <= n_bars:
        test_end = test_start + test_bars

        print(f"  Fold {fold}: train [0,{test_start}] test [{test_start},{test_end}]")

        strat = strategy_factory()

        # Set sentiment data (full history available to strategy — no lookahead
        # because sentiment is computed from headlines published at/before each date)
        strat.set_sentiment_data(sentiment_df)

        # Warmup: feed train data
        for bar_i in range(0, min(test_start, n_bars - 1)):
            bar_data = {}
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

        # OOS test
        oos_rets, oos_trades = compute_signal_returns(
            strat, data, test_start, test_end,
            timestamps=timestamps, tc_bps=tc_bps,
        )

        if len(oos_rets) > 0:
            all_returns.append(oos_rets)
            all_trades.extend(oos_trades)

            sr = float(sharpe_ratio(oos_rets)) if len(oos_rets) > 1 else float("nan")
            fold_results.append({
                "fold": fold,
                "train_range": [0, test_start],
                "test_range": [test_start, test_end],
                "oos_sharpe": sr,
                "n_bars": len(oos_rets),
                "n_trades": len(oos_trades),
            })

        fold += 1
        test_start += test_bars

    combined = np.concatenate(all_returns) if all_returns else np.array([], dtype=float)
    return combined, all_trades, fold_results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bootstrap_sharpe_ci(
    returns: np.ndarray, rf: float = 0.04, n_boot: int = 2000, ci: float = 0.90,
) -> tuple[float, float]:
    if len(returns) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    sharpes = np.empty(n_boot)
    daily_rf = rf / 252.0
    for i in range(n_boot):
        sample = rng.choice(returns, size=len(returns), replace=True)
        excess = sample - daily_rf
        mu = np.mean(excess)
        sigma = np.std(excess, ddof=1)
        sharpes[i] = (mu / sigma * np.sqrt(252)) if sigma > 1e-10 else 0.0
    lo = float(np.percentile(sharpes, (1 - ci) / 2 * 100))
    hi = float(np.percentile(sharpes, (1 + ci) / 2 * 100))
    return lo, hi


def compute_metrics(
    returns: np.ndarray,
    trades: list[dict],
    rf_rate: float = 0.04,
) -> dict:
    if len(returns) == 0:
        return {}

    sr = float(sharpe_ratio(returns, rf=rf_rate))
    ci_lo, ci_hi = _bootstrap_sharpe_ci(returns, rf_rate)

    equity = np.cumprod(1 + returns)
    total_ret = (equity[-1] - 1.0) * 100.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd)) * 100.0

    pnls = [t["pnl_abs"] for t in trades if t.get("pnl_abs") is not None]
    win_rate = float(np.mean(np.array(pnls) > 0)) * 100.0 if pnls else 0.0

    return {
        "total_return_pct": round(total_ret, 2),
        "sharpe_ratio": round(sr, 4),
        "sharpe_ci_lower": round(ci_lo, 4),
        "sharpe_ci_upper": round(ci_hi, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_trades": len(trades),
        "n_bars": len(returns),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MACRO_UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC",
                   "EFA", "EEM", "HYG", "LQD", "USO", "SLV", "VNQ", "DIA", "IEF"]


def main() -> dict:
    # ── Configuration ──
    universe_name = "expanded_15"
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    tc_bps = 5.0
    rf_rate = 0.04
    train_bars = 504   # ~2 years
    test_bars = 252    # ~1 year

    symbols = UNIVERSES[universe_name]["symbols"]
    print("=" * 60)
    print("  SENTIMENT MOMENTUM BACKTEST")
    print("=" * 60)
    print(f"  Universe:    {universe_name} ({len(symbols)} symbols)")
    print(f"  Period:      {start_date} -> {end_date}")
    print(f"  TC:          {tc_bps} bps")
    print(f"  Risk-free:   {rf_rate:.0%} annual")
    print(f"  Train bars:  {train_bars} (~{train_bars // 21} months)")
    print(f"  Test bars:   {test_bars} (~{test_bars // 21} months)")
    print()

    # ── Load OHLCV data ──
    print("Loading OHLCV data...")
    df = load_universe(universe_name, start_date, end_date)
    if df.empty:
        print("ERROR: No data loaded.")
        return {}
    print(f"  {len(df)} rows, {df['symbol'].nunique()} symbols")

    data = _df_to_data(df)
    timestamps = sorted(df["timestamp"].unique())
    n_bars = data["close"].shape[0]
    print(f"  Data matrix: {n_bars} bars x {len(data['symbols'])} assets")

    # ── Load/generate sentiment data ──
    print("\nLoading sentiment data...")
    loader = NewsSentimentLoader(half_life_hours=24.0, avg_headlines_per_day=3.0)
    sentiment_df = loader.load(symbols, start_date, end_date)
    print(f"  {len(sentiment_df)} daily sentiment records")
    print(f"  Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    print(f"  Mean sentiment: {sentiment_df['sentiment_score'].mean():.4f}")
    print(f"  Std sentiment:  {sentiment_df['sentiment_score'].std():.4f}")

    # ── Run sentiment strategy ──
    print("\nRunning sentiment momentum strategy...")
    strategy_id = "sentiment_momentum"

    strat_params = {
        "sentiment_lookback": 5,
        "rebalance_freq": 1,       # daily rebalance
        "tercile_pct": 0.33,
        "max_strength": 1.0,
    }

    def strategy_factory():
        cfg = StrategyConfig(
            name=strategy_id,
            strategy_type=StrategyType.MACRO,
            params=strat_params,
        )
        cls = get_strategy_class(strategy_id)
        s = cls(cfg)
        s.initialize()
        return s

    if n_bars < train_bars + test_bars:
        print(f"  ERROR: Not enough data ({n_bars} bars) for train ({train_bars}) + test ({test_bars})")
        # Fall back to simple split
        train_bars = int(n_bars * 0.6)
        test_bars = n_bars - train_bars
        print(f"  Adjusting to: train={train_bars}, test={test_bars}")

    oos_returns, trades, fold_results = walk_forward_backtest(
        strategy_factory, data, timestamps, sentiment_df,
        train_bars=train_bars, test_bars=test_bars,
        tc_bps=tc_bps, warmup_bars=60,
    )

    if len(oos_returns) == 0:
        print("ERROR: No OOS returns produced.")
        return {}

    # ── Compute metrics ──
    metrics = compute_metrics(oos_returns, trades, rf_rate)

    print("\n" + "=" * 60)
    print("  SENTIMENT MOMENTUM RESULTS")
    print("=" * 60)
    print(f"  Total Return:     {metrics['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.4f}")
    print(f"  Sharpe 90% CI:    [{metrics['sharpe_ci_lower']:.4f}, {metrics['sharpe_ci_upper']:.4f}]")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:         {metrics['win_rate_pct']:.1f}%")
    print(f"  Trades:           {metrics['n_trades']}")
    print(f"  OOS Bars:         {metrics['n_bars']}")
    print("=" * 60)

    # ── Fold details ──
    print("\n  Walk-Forward Fold Details:")
    for fold in fold_results:
        sr_str = f"{fold['oos_sharpe']:.4f}" if not np.isnan(fold['oos_sharpe']) else "NaN"
        print(f"    Fold {fold['fold']}: train {fold['train_range']} test {fold['test_range']} "
              f"OOS Sharpe={sr_str} ({fold['n_bars']} bars, {fold['n_trades']} trades)")

    # ── Assumptions ──
    print("\n  ASSUMPTIONS:")
    print(f"    - Sentiment source:   Synthetic (VADER-scored templates)")
    print(f"    - Sentiment lookback: {strat_params['sentiment_lookback']} days")
    print(f"    - Rebalance:          Daily")
    print(f"    - Tercile:            Top {strat_params['tercile_pct']:.0%} long, bottom {strat_params['tercile_pct']:.0%} short")
    print(f"    - Transaction cost:   {tc_bps} bps per unit turnover")
    print(f"    - Dollar-neutral:     Yes (equal weight within terciles)")

    # ── Save results ──
    results = {
        "strategy": strategy_id,
        "config": {
            "universe": universe_name,
            "period": f"{start_date} -> {end_date}",
            "params": strat_params,
            "tc_bps": tc_bps,
            "rf_rate": rf_rate,
            "train_bars": train_bars,
            "test_bars": test_bars,
        },
        "metrics": metrics,
        "folds": fold_results,
    }

    out_path = REPO_ROOT / "results" / "sentiment_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # ── Check if we should plug into aggregator ──
    if metrics["sharpe_ratio"] > 0.3:
        print(f"\n  OOS Sharpe {metrics['sharpe_ratio']:.4f} > 0.3 — eligible for aggregator integration")
        results["aggregator_eligible"] = True
    else:
        print(f"\n  OOS Sharpe {metrics['sharpe_ratio']:.4f} <= 0.3 — not yet eligible for aggregator")
        results["aggregator_eligible"] = False

    return results


if __name__ == "__main__":
    results = main()
