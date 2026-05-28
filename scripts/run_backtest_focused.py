"""Run a focused backtest with realistic cost assumptions.

Uses the signal-based pipeline (`_compute_signal_returns` with trade tracking)
to produce: total return, annualized Sharpe, max drawdown, win rate, number of trades.

Supports any strategy that implements `on_data()`.
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

from quantcore.risk.metrics import sharpe_ratio, drawdown_stats
from quantcore.strategies.base import StrategyConfig, StrategyType

# Trigger @register decorators
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401

from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe


# ---------------------------------------------------------------------------
# DataFrame -> data dict (copied from real_data_backtest.py)
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
# Signal-based return computation with trade tracking
# ---------------------------------------------------------------------------

def compute_signal_returns_with_trades(
    strat,
    data: dict,
    start: int,
    end: int,
    timestamps: list | None = None,
    tc_bps: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    """Compute bar-level signal-weighted returns with full trade tracking.

    Returns (bar_returns, trades_list).
    """
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

    # Trade tracking state
    positions: dict[str, dict] = {}  # symbol -> {direction, entry_ts, entry_price}
    trades: list[dict] = []

    for idx, bar_i in enumerate(range(start, end - 1)):
        # Build bar_data (causal slice)
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
                    "metadata": getattr(s, 'metadata', {}),
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
            tc_cost = turnover * tc_bps / 10000.0
            bar_rets[idx] -= tc_cost

        # ---- Trade tracking ----
        if current_signals:
            # Check for exits (position no longer in current signals or direction flipped)
            for sym in list(positions.keys()):
                pos = positions[sym]
                exit_ts = timestamps[bar_i] if timestamps and bar_i < len(timestamps) else None
                exit_price = float(close[bar_i, symbols.index(sym)]) if bar_i < close.shape[0] else None
                if sym not in current_signals or current_signals[sym]["direction"] != pos["direction"]:
                    pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
                    pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
                    trades.append({
                        "symbol": sym,
                        "direction": pos["direction"],
                        "entry_ts": pos["entry_ts"],
                        "exit_ts": exit_ts,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl_abs": pnl_abs,
                        "pnl_pct": pnl_pct,
                    })
                    positions.pop(sym)

            # Check for new entries
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

    # Close remaining positions at end
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        exit_ts = timestamps[end - 1] if timestamps and end - 1 < len(timestamps) else None
        exit_price = float(close[end - 1, symbols.index(sym)]) if end - 1 < close.shape[0] else None
        pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
        pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
        trades.append({
            "symbol": sym,
            "direction": pos["direction"],
            "entry_ts": pos["entry_ts"],
            "exit_ts": exit_ts,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
        })

    return bar_rets, trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_equity_curve(returns: np.ndarray, initial: float = 1_000_000.0) -> np.ndarray:
    if len(returns) == 0:
        return np.array([initial])
    equity = np.empty(len(returns) + 1)
    equity[0] = initial
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def win_rate_from_trades(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    pnls = [t["pnl_abs"] for t in trades if t.get("pnl_abs") is not None]
    if not pnls:
        return 0.0
    return float(np.mean(np.array(pnls) > 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFENSE_STRATEGIES = {
    "macro_cross_sectional_factor_neutral": {
        "universe": "expanded_15",
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
        "universe": "expanded_15",
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


def main() -> dict:
    # ── Configuration ──
    strategy_id = "macro_cross_sectional_factor_neutral"
    strat_cfg = DEFENSE_STRATEGIES[strategy_id]
    universe_name = strat_cfg["universe"]
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    warmup_bars = 300  # ~14 months warmup for 252d reg_lookback
    cash = 1_000_000.0
    tc_bps = 5.0  # 5 bps round-trip turnover cost (commission + slippage proxy)
    risk_free_rate = 0.04  # 4% annual (approximate T-bill average 2018-2024)

    symbols = UNIVERSES[universe_name]["symbols"]
    print(f"Strategy:  {strategy_id}")
    print(f"Universe:  {universe_name} ({len(symbols)} symbols)")
    print(f"Period:    {start_date} -> {end_date}")
    print(f"Warmup:    {warmup_bars} bars (~{warmup_bars // 21} months)")
    print(f"TC cost:   {tc_bps} bps turnover")
    print(f"Risk-free: {risk_free_rate:.0%} annual")
    print()

    print("Loading data...")
    df = load_universe(universe_name, start_date, end_date)
    if df.empty:
        print("ERROR: No data loaded.")
        return {}
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols.\n")

    # Convert to data dict
    data = _df_to_data(df)
    close = data["close"]
    n_periods = close.shape[0]
    symbols_in_data = data["symbols"]
    print(f"Data matrix: {n_periods} bars x {len(symbols_in_data)} assets")

    # Timestamps for trade tracking
    timestamps = sorted(df["timestamp"].unique())

    # ── Setup strategy ──
    cfg = StrategyConfig(
        name=strategy_id,
        strategy_type=StrategyType.MACRO,
        params=strat_cfg["params"],
    )
    cls = get_strategy_class(strategy_id)
    strat = cls(cfg)
    strat.initialize()

    # ── Warmup (build strategy state, no returns collected) ──
    print("Running warmup...")
    try:
        strat_warmup = cls(cfg)
        strat_warmup.initialize()
        warmup_end = min(warmup_bars, n_periods - 10)
        # Feed warmup bars
        for bar_i in range(0, warmup_end):
            bar_data = {}
            for k, v in data.items():
                if k == "symbols":
                    bar_data[k] = v
                elif isinstance(v, np.ndarray) and v.ndim >= 1:
                    bar_data[k] = v[:bar_i + 1]
                else:
                    bar_data[k] = v
            if bar_i < len(timestamps):
                ts = timestamps[bar_i]
                if hasattr(strat_warmup, 'set_timestamp'):
                    strat_warmup.set_timestamp(ts)
                bar_data["timestamp"] = ts
            try:
                strat_warmup.on_data(bar_data)
            except Exception:
                pass
    except Exception as e:
        print(f"  Warmup warning: {e}")

    # ── OOS backtest with trade tracking ──
    print("Running OOS backtest with trade tracking...")
    oos_start = warmup_end
    oos_end = n_periods

    oos_returns, trades = compute_signal_returns_with_trades(
        strat_warmup, data, oos_start, oos_end,
        timestamps=timestamps, tc_bps=tc_bps,
    )

    if len(oos_returns) == 0:
        print("ERROR: No OOS returns produced.")
        return {}

    # ── Compute all metrics ──
    equity = build_equity_curve(oos_returns, initial=cash)
    total_return = (equity[-1] - cash) / cash * 100.0

    # Sharpe using quantcore's implementation (annualised, with risk-free rate)
    sr = sharpe_ratio(oos_returns, rf=risk_free_rate)
    max_dd = max_drawdown_from_equity(equity) * 100.0
    wr = win_rate_from_trades(trades) * 100.0
    n_trades = len(trades)

    # Also compute percentage of positive-return days
    pct_up_days = float(np.mean(oos_returns > 0)) * 100.0

    # Total transaction cost impact
    cum_return_no_tc = np.cumprod(1 + oos_returns)[-1] if len(oos_returns) > 0 else 1.0

    # ── Print results ──
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS — REALISTIC COST ASSUMPTIONS")
    print("=" * 60)
    print(f"  1. Total Return:        {total_return:+.2f}%")
    print(f"  2. Annualized Sharpe:    {sr:.4f}  (rf = {risk_free_rate:.0%} annual)")
    print(f"  3. Maximum Drawdown:     {max_dd:.2f}%")
    print(f"  4. Win Rate:             {wr:.1f}%  ({sum(1 for t in trades if t.get('pnl_abs', 0) > 0)}/{n_trades} trades)")
    print(f"  5. Number of Trades:     {n_trades}")
    print("-" * 60)
    print(f"  Final equity:           ${equity[-1]:,.2f}")
    print(f"  Up-day %:               {pct_up_days:.1f}%")
    print(f"  OOS bars:               {len(oos_returns)}")
    print(f"  OOS period:             {timestamps[oos_start] if oos_start < len(timestamps) else '?'} -> {timestamps[oos_end - 1] if oos_end - 1 < len(timestamps) else '?'}")
    print("=" * 60)

    # ── Assumptions ──
    print("\nASSUMPTIONS:")
    print(f"  - Data source:          yfinance daily OHLCV (adjusted)")
    print(f"  - Transaction cost:     {tc_bps} bps per unit of weight turnover (covers")
    print(f"                          commission + market impact for liquid ETFs)")
    print(f"  - Risk-free rate:       {risk_free_rate:.0%} annual (T-bill proxy)")
    print(f"  - Sharpe annualization: sqrt(252)")
    print(f"  - Fill model:           Signal-weighted; no fill rejection or partial fills")
    print(f"  - Short selling:        Allowed (no borrow cost beyond tc_bps)")
    print(f"  - Slippage:             Embedded in tc_bps (5 bps ~ IBKR Pro + impact)")
    print(f"  - Warmup:               {warmup_bars} bars excluded from performance")
    print(f"  - Drawdown:             Based on equity curve from signal-weighted returns")

    results = {
        "strategy": strategy_id,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sr, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(wr, 1),
        "n_trades": n_trades,
        "final_equity": round(equity[-1], 2),
        "oos_bars": len(oos_returns),
        "tc_bps": tc_bps,
        "risk_free_annual": risk_free_rate,
    }

    # Save equity curve
    eq_df = pd.DataFrame({
        "bar": np.arange(len(equity)),
        "equity": equity,
    })
    eq_df.to_csv("backtest_equity_curve.csv", index=False)
    print(f"\n  Equity curve saved to: backtest_equity_curve.csv")

    return results


if __name__ == "__main__":
    results = main()
    if results:
        print("\nRaw results JSON:")
        print(json.dumps(results, indent=2))
