"""Strict COVID-era diagnostic for Leader-Follower CTA model.

Isolates 2020-02-15 to 2020-06-30 window, computes all required metrics,
and outputs a structured report to answer:
1. Is flip frequency abnormally high?
2. Are most losses from short-duration trades?
3. Is the model entering too late/early/both?
4. Is whipsaw the dominant failure mode?

Usage:
    python scripts/covid_diagnostic.py
    python scripts/covid_diagnostic.py --strategy macro_cta_trend
    python scripts/covid_diagnostic.py --no-cache
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from datetime import timedelta

from scripts.real_data_backtest import (
    _df_to_data,
    _compute_signal_returns,
    _setup_strategy,
    STRATEGY_UNIVERSE,
    load_universe,
)
from quantcore.strategies.macro.cta_trend import CTATrendStrategy


def _holding_days(entry_ts, exit_ts) -> float:
    """Calculate holding period in days between two timestamps."""
    if entry_ts is None or exit_ts is None:
        return 0.0
    try:
        e = pd.Timestamp(entry_ts)
        x = pd.Timestamp(exit_ts)
        return (x - e).total_seconds() / (24 * 3600)  # 86400 seconds per day
    except Exception:
        return 0.0


def _strip_tz(ts):
    """Strip timezone info from timestamp if present."""
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    return t.replace(tzinfo=None) if t.tzinfo is not None else t


def run_covid_diagnostic(
    strategy_id: str = "macro_cta_trend",
    start_date: str = "2020-02-15",
    end_date: str = "2020-06-30",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run strict COVID-era diagnostic for Leader-Follower CTA model."""
    result: dict[str, Any] = {
        "covid_window": {"start": start_date, "end": end_date},
        "strategy": strategy_id,
        "flip_metrics": {},
        "trade_metrics": {},
        "holding_metrics": {},
        "whipsaw_metrics": {},
        "entry_timing": {},
        "trades": [],
        "fast_flips": [],
        "time_series": {},
        "error": None,
    }

    try:
        # 1. Load data for COVID window + warmup (200 days prior to build state)
        warmup_start = (pd.Timestamp(start_date) - timedelta(days=200)).strftime("%Y-%m-%d")
        df = load_universe(
            STRATEGY_UNIVERSE.get(strategy_id, "cta"),
            warmup_start,
            end_date,
            use_cache=use_cache,
        )
        if df.empty:
            result["error"] = "No data loaded for COVID window"
            return result

        # Filter to COVID window for analysis
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        covid_mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        covid_df = df[covid_mask].copy()
        if covid_df.empty:
            result["error"] = "No data in COVID window after filtering"
            return result

        # 2. Prepare data dict and timestamps
        data = _df_to_data(df)
        symbols = data["symbols"]
        close = data["close"]
        n_periods = close.shape[0]

        # Get timestamps from DataFrame (sorted), strip timezones
        all_ts = sorted(df["timestamp"].unique())
        all_ts = [_strip_tz(ts) for ts in all_ts]
        
        # Find start/end indices for COVID window
        covid_start_ts = pd.Timestamp(start_date)
        covid_end_ts = pd.Timestamp(end_date)
        covid_start_idx = next((i for i, ts in enumerate(all_ts) if ts >= covid_start_ts), 0)
        covid_end_idx = next((i for i, ts in enumerate(all_ts) if ts > covid_end_ts), n_periods - 1)

        # 3. Setup strategy and run with trade tracking
        strat = _setup_strategy(strategy_id)
        strat.initialize()

        # Reset diagnostics
        strat.reset_diagnostics()

        # Warmup: run up to COVID start to build state
        timestamps = all_ts  # full list of timestamps
        warmup_result = _compute_signal_returns(
            strat, data, 0, covid_start_idx, timestamps=timestamps, track_trades=False
        )
        warmup_rets = warmup_result[0] if isinstance(warmup_result, tuple) else warmup_result

        # Phase 1.6: Reset position tracking after warmup (prevents inherited state)
        strat.reset_diagnostics()

        # COVID window: run with trade tracking
        cov_rets, trades = _compute_signal_returns(
            strat, data, covid_start_idx, covid_end_idx, timestamps=timestamps, track_trades=True
        )

        # Get fast signal flips
        fast_flips = strat.get_fast_flips()
        # Filter flips to COVID window
        covid_flips = [
            f for f in fast_flips
            if f["timestamp"] is not None
            and _strip_tz(f["timestamp"]) is not None
            and _strip_tz(f["timestamp"]) >= covid_start_ts
            and _strip_tz(f["timestamp"]) <= covid_end_ts
        ]
        result["fast_flips"] = covid_flips

        # 4. Compute Flip Behavior Metrics (A)
        n_flips = len(covid_flips)
        n_days = (covid_end_ts - covid_start_ts).days
        n_months = n_days / 30.44
        flips_per_month = n_flips / n_months if n_months > 0 else 0.0
        # % of days with a flip
        flip_days = set()
        for f in covid_flips:
            try:
                flip_days.add(_strip_tz(f["timestamp"]).date())
            except Exception:
                pass
        pct_days_with_flip = len(flip_days) / n_days if n_days > 0 else 0.0

        result["flip_metrics"] = {
            "total_flips": n_flips,
            "flips_per_30d": round(flips_per_month, 2),
            "pct_days_with_flip": round(pct_days_with_flip * 100, 2),
            "n_days_in_window": n_days,
        }

        # 5. Compute Trade Quality Metrics (B) and Holding Characteristics (C)
        covid_trades = []  # Define in outer scope
        if not trades:
            result["trade_metrics"] = {"error": "No trades recorded in COVID window"}
            result["whipsaw_metrics"] = {}
            result["holding_metrics"] = {}
        else:
            # Filter trades to COVID window
            covid_trades = [
                t for t in trades
                if t["timestamp_entry"] is not None
                and _strip_tz(t["timestamp_entry"]) is not None
                and _strip_tz(t["timestamp_entry"]) >= covid_start_ts
                and _strip_tz(t["timestamp_entry"]) <= covid_end_ts
            ]
            result["trades"] = covid_trades

            pnl_list = [t["pnl_abs"] for t in covid_trades]
            holding_list = [t["holding_days"] for t in covid_trades]
            win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list) if pnl_list else 0.0

            result["trade_metrics"] = {
                "total_trades": len(covid_trades),
                "win_rate": round(win_rate * 100, 2),
                "avg_pnl_per_trade": round(np.mean(pnl_list) if pnl_list else 0.0, 4),
                "median_pnl_per_trade": round(np.median(pnl_list) if pnl_list else 0.0, 4),
                "pnl_percentiles": {
                    "p10": round(np.percentile(pnl_list, 10) if pnl_list else 0.0, 4),
                    "p25": round(np.percentile(pnl_list, 25) if pnl_list else 0.0, 4),
                    "p50": round(np.percentile(pnl_list, 50) if pnl_list else 0.0, 4),
                    "p75": round(np.percentile(pnl_list, 75) if pnl_list else 0.0, 4),
                    "p90": round(np.percentile(pnl_list, 90) if pnl_list else 0.0, 4),
                },
            }

            # Holding Characteristics (C)
            short_trades = [t for t in covid_trades if t["holding_days"] < 3]
            result["holding_metrics"] = {
                "avg_holding_days": round(np.mean(holding_list) if holding_list else 0.0, 2),
                "median_holding_days": round(np.median(holding_list) if holding_list else 0.0, 2),
                "pct_trades_under_3d": round(len(short_trades) / len(covid_trades) * 100 if covid_trades else 0.0, 2),
            }

            # 6. Whipsaw Detection (D)
            whipsaws = [
                t for t in covid_trades
                if t["holding_days"] <= 5 and t["pnl_abs"] < 0
            ]
            whipsaw_pnl = sum(t["pnl_abs"] for t in whipsaws)
            total_pnl = sum(t["pnl_abs"] for t in covid_trades)
            result["whipsaw_metrics"] = {
                "n_whipsaws": len(whipsaws),
                "pct_trades_whipsaw": round(len(whipsaws) / len(covid_trades) * 100 if covid_trades else 0.0, 2),
                "total_whipsaw_pnl": round(whipsaw_pnl, 4),
                "whipsaw_pnl_pct_of_total": round(whipsaw_pnl / total_pnl * 100 if total_pnl != 0 else 0.0, 2),
            }

        # 7. Entry Timing Analysis (Critical)
        # Find largest moves:
        # a) Crash: 2020-02-19 (S&P high) to 2020-03-23 (trough)
        # b) Rebound: 2020-03-23 to 2020-04-30 (recovery)
        crash_start = pd.Timestamp("2020-02-19")
        crash_end = pd.Timestamp("2020-03-23")
        rebound_start = crash_end
        rebound_end = pd.Timestamp("2020-04-30")

        # Get price data for a representative symbol (SPY or first symbol)
        rep_symbol = "SPY" if "SPY" in symbols else (symbols[0] if symbols else None)
        if rep_symbol:
            sym_idx = symbols.index(rep_symbol)
            # Get price array for this symbol
            sym_prices = close[:, sym_idx]
            # Get timestamps for the full period
            price_ts = all_ts[:len(sym_prices)]

            # Calculate crash move
            crash_start_idx = next((i for i, ts in enumerate(price_ts) if ts >= crash_start), None)
            crash_end_idx = next((i for i, ts in enumerate(price_ts) if ts >= crash_end), None)
            if crash_start_idx is not None and crash_end_idx is not None:
                crash_prices = sym_prices[crash_start_idx:crash_end_idx+1]
                crash_peak = sym_prices[crash_start_idx]
                crash_trough = np.min(crash_prices)
                full_crash_move = crash_peak - crash_trough

                # Find first strategy position during crash
                crash_trades = [
                    t for t in covid_trades
                    if t.get("symbol") == rep_symbol
                    and _strip_tz(t["timestamp_entry"]) >= crash_start
                    and _strip_tz(t["timestamp_entry"]) <= crash_end
                ]
                first_entry = None
                if crash_trades:
                    first_entry = min(crash_trades, key=lambda x: pd.Timestamp(x["timestamp_entry"]))
                    entry_idx = next((i for i, ts in enumerate(price_ts) if ts >= _strip_tz(first_entry["timestamp_entry"])), None)
                    if entry_idx is not None:
                        move_at_entry = crash_peak - sym_prices[entry_idx]
                        entry_delay_pct = (move_at_entry / full_crash_move * 100) if full_crash_move != 0 else 0.0
                    else:
                        entry_delay_pct = 0.0
                else:
                    entry_delay_pct = 100.0  # No entry during crash

                result["entry_timing"]["crash"] = {
                    "peak_date": str(crash_start.date()),
                    "trough_date": str(crash_end.date()),
                    "peak_price": float(crash_peak),
                    "trough_price": float(crash_trough),
                    "full_move": float(full_crash_move),
                    "first_entry_date": str(first_entry["timestamp_entry"]) if first_entry else None,
                    "entry_delay_pct": round(entry_delay_pct, 2),
                }

            # Calculate rebound move
            rebound_end_idx = next((i for i, ts in enumerate(price_ts) if ts >= rebound_end), None)
            rebound_start_idx = next((i for i, ts in enumerate(price_ts) if ts >= rebound_start), None)
            if rebound_start_idx is not None and rebound_end_idx is not None:
                rebound_prices = sym_prices[rebound_start_idx:rebound_end_idx+1]
                rebound_trough = sym_prices[rebound_start_idx]
                rebound_peak = np.max(rebound_prices)
                full_rebound_move = rebound_peak - rebound_trough

                # Find first strategy position during rebound
                rebound_trades = [
                    t for t in covid_trades
                    if t.get("symbol") == rep_symbol
                    and _strip_tz(t["timestamp_entry"]) >= rebound_start
                    and _strip_tz(t["timestamp_entry"]) <= rebound_end
                ]
                first_entry = None
                if rebound_trades:
                    first_entry = min(rebound_trades, key=lambda x: pd.Timestamp(x["timestamp_entry"]))
                    entry_idx = next((i for i, ts in enumerate(price_ts) if ts >= _strip_tz(first_entry["timestamp_entry"])), None)
                    if entry_idx is not None:
                        move_at_entry = sym_prices[entry_idx] - rebound_trough
                        entry_delay_pct = (move_at_entry / full_rebound_move * 100) if full_rebound_move != 0 else 0.0
                    else:
                        entry_delay_pct = 0.0
                else:
                    entry_delay_pct = 100.0

                result["entry_timing"]["rebound"] = {
                    "trough_date": str(rebound_start.date()),
                    "peak_date": str(rebound_end.date()),
                    "trough_price": float(rebound_trough),
                    "peak_price": float(rebound_peak),
                    "full_move": float(full_rebound_move),
                    "first_entry_date": str(first_entry["timestamp_entry"]) if first_entry else None,
                    "entry_delay_pct": round(entry_delay_pct, 2),
                }

        # 8. Time Series Data (for COVID window)
        if rep_symbol and 'sym_idx' in locals():
            # Get COVID window price data
            covid_price = sym_prices[covid_start_idx:covid_end_idx]
            covid_ts = all_ts[covid_start_idx:covid_end_idx]
            # Get fast signal for this symbol (simplified: use close price vs SMA 20)
            fast_signal = []
            for i in range(len(covid_price)):
                window = 20
                if i + covid_start_idx < window:
                    fast_signal.append(0.0)
                else:
                    sma = np.mean(sym_prices[max(0, covid_start_idx + i - window): covid_start_idx + i + 1])
                    fast_signal.append((covid_price[i] - sma) / sma if sma > 0 else 0.0)
            # Get position direction from trades (simplified)
            position = []
            current_pos = 0
            trade_idx = 0
            for ts in covid_ts:
                # Check if a trade starts at this timestamp
                while trade_idx < len(covid_trades) and _strip_tz(covid_trades[trade_idx]["timestamp_entry"]) <= ts:
                    current_pos = covid_trades[trade_idx]["position_direction"]
                    trade_idx += 1
                position.append(current_pos)

            result["time_series"] = {
                "timestamps": [str(ts) for ts in covid_ts],
                "price": [float(p) for p in covid_price],
                "fast_signal": [float(s) for s in fast_signal],
                "position": position,
            }

    except Exception as e:
        result["error"] = f"Diagnostic failed: {str(e)}"
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def print_diagnostic_report(result: dict[str, Any]) -> None:
    """Print structured diagnostic report for COVID window."""
    print("\n" + "=" * 80)
    print("COVID (2020-02-15 to 2020-06-30) LEADER-FOLLOWER DIAGNOSTIC REPORT")
    print("=" * 80)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    # Flip Metrics
    fm = result.get("flip_metrics", {})
    print("\n--- A. FLIP BEHAVIOR ---")
    print(f"Total flips: {fm.get('total_flips', 0)}")
    print(f"Flips per 30 days: {fm.get('flips_per_30d', 0)}")
    print(f"% of days with flip: {fm.get('pct_days_with_flip', 0)}%")

    # Trade Metrics
    tm = result.get("trade_metrics", {})
    print("\n--- B. TRADE QUALITY ---")
    if tm.get("error"):
        print(f"Error: {tm['error']}")
    else:
        print(f"Total trades: {tm.get('total_trades', 0)}")
        print(f"Win rate: {tm.get('win_rate', 0)}%")
        print(f"Avg PnL/trade: {tm.get('avg_pnl_per_trade', 0)}")
        print(f"Median PnL/trade: {tm.get('median_pnl_per_trade', 0)}")
        pcts = tm.get("pnl_percentiles", {})
        print(f"PnL percentiles: 10={pcts.get('p10',0)}, 25={pcts.get('p25',0)}, 50={pcts.get('p50',0)}, 75={pcts.get('p75',0)}, 90={pcts.get('p90',0)}")

    # Holding Metrics
    hm = result.get("holding_metrics", {})
    print("\n--- C. HOLDING CHARACTERISTICS ---")
    if hm:
        print(f"Avg holding period: {hm.get('avg_holding_days', 0)} days")
        print(f"Median holding period: {hm.get('median_holding_days', 0)} days")
        print(f"% trades <3 days: {hm.get('pct_trades_under_3d', 0)}%")

    # Whipsaw Metrics
    wm = result.get("whipsaw_metrics", {})
    print("\n--- D. WHIPSAW DETECTION (CRITICAL) ---")
    if wm:
        print(f"Number of whipsaws: {wm.get('n_whipsaws', 0)}")
        print(f"% of trades that are whipsaws: {wm.get('pct_trades_whipsaw', 0)}%")
        print(f"Total whipsaw PnL: {wm.get('total_whipsaw_pnl', 0)}")
        print(f"Whipsaw PnL as % of total: {wm.get('whipsaw_pnl_pct_of_total', 0)}%")

    # Entry Timing
    et = result.get("entry_timing", {})
    print("\n--- ENTRY TIMING ANALYSIS (CRITICAL) ---")
    if "crash" in et:
        crash = et["crash"]
        print(f"Crash (2020-02-19 to 2020-03-23):")
        print(f"  Peak: {crash['peak_price']} on {crash['peak_date']}")
        print(f"  Trough: {crash['trough_price']} on {crash['trough_date']}")
        print(f"  Full move: {crash['full_move']:.2f}")
        print(f"  First entry: {crash['first_entry_date']}")
        print(f"  Entry delay: {crash['entry_delay_pct']}% of move completed before entry")
    if "rebound" in et:
        rebound = et["rebound"]
        print(f"Rebound (2020-03-23 to 2020-04-30):")
        print(f"  Trough: {rebound['trough_price']} on {rebound['trough_date']}")
        print(f"  Peak: {rebound['peak_price']} on {rebound['peak_date']}")
        print(f"  Full move: {rebound['full_move']:.2f}")
        print(f"  First entry: {rebound['first_entry_date']}")
        print(f"  Entry delay: {rebound['entry_delay_pct']}% of move completed before entry")

    # Key Conclusions
    print("\n--- KEY CONCLUSIONS ---")
    flips_per_30d = fm.get("flips_per_30d", 0)
    pct_short = hm.get("pct_trades_under_3d", 0) if hm else 0
    pct_whipsaw = wm.get("pct_trades_whipsaw", 0) if wm else 0
    crash_delay = et.get("crash", {}).get("entry_delay_pct", 0)
    rebound_delay = et.get("rebound", {}).get("entry_delay_pct", 100)

    print(f"1. Flip frequency: {'ABNORMALLY HIGH' if flips_per_30d > 5 else 'NORMAL'} ({flips_per_30d} flips/30d)")
    print(f"2. Short-term churn: {'HIGH' if pct_short > 30 else 'NORMAL'} ({pct_short}% trades <3d)")
    print(f"3. Whipsaw dominance: {'YES' if pct_whipsaw > 20 else 'NO'} ({pct_whipsaw}% of trades)")
    print(f"4. Crash entry timing: {'LATE' if crash_delay > 50 else 'EARLY/ON TIME'} ({crash_delay}% move completed at entry)")
    print(f"5. Rebound entry timing: {'LATE' if rebound_delay > 50 else 'EARLY/ON TIME'} ({rebound_delay}% move completed at entry)")

    # Answer required questions
    print("\n--- ANSWERS TO REQUIRED QUESTIONS ---")
    print(f"Q1: Is flip frequency abnormally high? {'YES' if flips_per_30d > 5 else 'NO'} ({flips_per_30d} flips/30d)")
    print(f"Q2: Are most losses from short-duration trades? {'YES' if pct_short > 30 else 'NO'} ({pct_short}% trades <3d)")
    print(f"Q3: Is model entering too late/early/both? Crash: {'LATE' if crash_delay > 50 else 'EARLY/OK'}, Rebound: {'LATE' if rebound_delay > 50 else 'EARLY/OK'}")
    print(f"Q4: Is whipsaw dominant failure mode? {'YES' if pct_whipsaw > 20 else 'NO'} ({pct_whipsaw}% of trades)")
    print("=" * 80 + "\n")


def save_results(result: dict[str, Any], output_path: str = "results/covid_diagnostic.json") -> None:
    """Save diagnostic results to JSON."""
    path = Path(REPO_ROOT) / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types and timestamps
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (dict, list, str, int, float, type(None))):
            return obj
        return str(obj)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_serialize)
    print(f"\nResults saved to: {path}")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="COVID-era CTA diagnostic")
    parser.add_argument("--strategy", type=str, default="macro_cta_trend",
                        help="Strategy to diagnose (default: macro_cta_trend)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable parquet cache for yfinance")
    parser.add_argument("--output", type=str, default="results/covid_diagnostic.json",
                        help="Output JSON path (default: results/covid_diagnostic.json)")

    args = parser.parse_args()

    print("=" * 60)
    print("  COVID-19 Leader-Follower Diagnostic")
    print(f"  Strategy: {args.strategy}")
    print(f"  Window: 2020-02-15 to 2020-06-30")
    print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 60)

    result = run_covid_diagnostic(
        strategy_id=args.strategy,
        use_cache=not args.no_cache,
    )

    print_diagnostic_report(result)
    save_results(result, args.output)


if __name__ == "__main__":
    main()
