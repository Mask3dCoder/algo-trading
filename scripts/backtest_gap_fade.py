"""Backtest Opening Gap Fade strategy on 2024 daily data.

Uses daily OHLCV to simulate intraday gap-fade:
  - Entry: at daily open when |gap_pct| > threshold
  - Exit proxy: use daily close as proxy for 30-min post-open price
    (conservative — intraday reversion is typically stronger than full-day)
  - Retracement target: 50% of gap
  - Stop loss: 1.5% from entry

Usage:
    python scripts/backtest_gap_fade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from quantcore.risk.metrics import sharpe_ratio, drawdown_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "UNH", "JNJ",
    "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK", "LLY", "ABBV", "PFE",
    "KO", "AVGO", "PEP", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
    "DHR", "LIN", "TXN", "UPS", "RTX", "QCOM", "PM", "HON", "LOW", "AMGN",
    "INTU", "BA", "GE", "CAT", "SBUX", "ISRG", "AMD", "PLD", "BLK", "ADI",
]

GAP_THRESHOLD_PCT = 2.0
RETRACEMENT_TARGET = 0.50
STOP_LOSS_PCT = 1.5
MAX_POSITIONS = 10
MAX_POSITION_PCT = 0.05
MIN_PRICE = 10.0
INITIAL_CASH = 100_000.0
SLIPPAGE_BPS = 5
COMMISSION_PER_SHARE = 0.005


def fetch_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for each ticker. Returns {ticker: DataFrame}."""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and len(df) > 20:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except Exception as e:
            log.warning("Failed to fetch %s: %s", ticker, e)
    log.info("Fetched data for %d / %d tickers", len(data), len(tickers))
    return data


def run_backtest(
    data: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Run gap-fade backtest day by day."""
    # Build aligned date index
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    dates = sorted(d for d in all_dates if start_date <= d.strftime("%Y-%m-%d") <= end_date)

    if not dates:
        return {"error": "No trading dates in range"}

    cash = INITIAL_CASH
    positions: dict[str, dict] = {}  # ticker -> {entry_price, shares, direction, gap_pct}
    trades: list[dict] = []
    daily_equity: list[tuple[Any, float]] = []

    prev_close_map: dict[str, float] = {}

    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")

        # ── Mark-to-market ──────────────────────────────────────────────
        equity = cash
        for sym, pos in positions.items():
            if sym in data and dt in data[sym].index:
                row = data[sym].loc[dt]
                current_px = float(row["Open"]) * 0.7 + float(row["Close"]) * 0.3
            else:
                current_px = pos["entry_price"]
            pnl = (current_px - pos["entry_price"]) * pos["shares"] * pos["direction"]
            equity += pos["entry_price"] * abs(pos["shares"]) + pnl
        daily_equity.append((dt, equity))

        # ── Exit logic ──────────────────────────────────────────────────
        # Simulate 30-min post-open exit: blend open + close to approximate
        # early-session price.  Weight open=0.7, close=0.3 — captures the
        # partial reversion that typically occurs in the first 30 minutes.
        exit_syms = []
        for sym, pos in list(positions.items()):
            if sym not in data or dt not in data[sym].index:
                continue
            row = data[sym].loc[dt]
            day_open = float(row["Open"])
            day_close = float(row["Close"])
            day_high = float(row["High"])
            day_low = float(row["Low"])
            # 30-min proxy: mostly open, some intraday reversion toward close
            current_px = day_open * 0.7 + day_close * 0.3
            entry_px = pos["entry_price"]
            direction = pos["direction"]

            # Clamp exit within day's high/low
            current_px = max(day_low, min(day_high, current_px))

            # Retracement target
            gap_abs = abs(pos["gap_pct"]) / 100.0 * entry_px
            if direction == 1:  # long (fading negative gap)
                target = entry_px + gap_abs * RETRACEMENT_TARGET
                if current_px >= target:
                    exit_syms.append((sym, current_px, "retracement"))
                    continue
            else:  # short (fading positive gap)
                target = entry_px - gap_abs * RETRACEMENT_TARGET
                if current_px <= target:
                    exit_syms.append((sym, current_px, "retracement"))
                    continue

            # Stop loss
            stop_dist = entry_px * STOP_LOSS_PCT / 100.0
            if direction == 1 and current_px <= entry_px - stop_dist:
                exit_syms.append((sym, current_px, "stop_loss"))
            elif direction == -1 and current_px >= entry_px + stop_dist:
                exit_syms.append((sym, current_px, "stop_loss"))

        # Time exit: force close any remaining at 30-min proxy price
        for sym in list(positions.keys()):
            if sym not in [s for s, _, _ in exit_syms]:
                if sym in data and dt in data[sym].index:
                    row = data[sym].loc[dt]
                    exit_px = float(row["Open"]) * 0.7 + float(row["Close"]) * 0.3
                    exit_px = max(float(row["Low"]), min(float(row["High"]), exit_px))
                    exit_syms.append((sym, exit_px, "time_exit"))

        for sym, exit_px, reason in exit_syms:
            pos = positions.pop(sym)
            # Apply slippage on exit
            slip = exit_px * SLIPPAGE_BPS / 10000.0
            actual_exit = exit_px - slip * pos["direction"]
            pnl = (actual_exit - pos["entry_price"]) * pos["shares"] * pos["direction"]
            commission = COMMISSION_PER_SHARE * abs(pos["shares"])
            net_pnl = pnl - commission
            cash += pos["entry_price"] * abs(pos["shares"]) + net_pnl
            trades.append({
                "symbol": sym,
                "direction": "long" if pos["direction"] == 1 else "short",
                "entry_price": pos["entry_price"],
                "exit_price": actual_exit,
                "shares": pos["shares"],
                "pnl": net_pnl,
                "gap_pct": pos["gap_pct"],
                "reason": reason,
                "date": dt_str,
            })

        # ── Entry logic ─────────────────────────────────────────────────
        if len(positions) >= MAX_POSITIONS:
            prev_close_map.clear()
            for sym, df in data.items():
                if dt in df.index:
                    idx = df.index.get_loc(dt)
                    if idx > 0:
                        prev_close_map[sym] = float(df.iloc[idx - 1]["Close"])
            continue

        candidates = []
        for sym, df in data.items():
            if sym in positions:
                continue
            if dt not in df.index:
                continue

            idx = df.index.get_loc(dt)
            if idx < 1:
                continue

            prev_close = float(df.iloc[idx - 1]["Close"])
            today_open = float(df.loc[dt, "Open"])

            if prev_close < MIN_PRICE or today_open < MIN_PRICE:
                continue

            gap_pct = (today_open - prev_close) / prev_close * 100.0
            if abs(gap_pct) < GAP_THRESHOLD_PCT:
                continue

            candidates.append((sym, gap_pct, today_open))

        # Rank by absolute gap
        candidates.sort(key=lambda c: abs(c[1]), reverse=True)

        for sym, gap_pct, today_open in candidates:
            if len(positions) >= MAX_POSITIONS:
                break

            direction = -1 if gap_pct > 0 else 1  # fade the gap
            max_notional = equity * MAX_POSITION_PCT
            shares = int(max_notional / today_open)
            if shares <= 0:
                continue

            # Apply slippage on entry
            slip = today_open * SLIPPAGE_BPS / 10000.0
            entry_px = today_open + slip * direction

            cost = entry_px * shares
            commission = COMMISSION_PER_SHARE * shares
            if cost + commission > cash:
                continue

            cash -= cost + commission
            positions[sym] = {
                "entry_price": entry_px,
                "shares": shares,
                "direction": direction,
                "gap_pct": gap_pct,
            }

        # Update prev_close for next day
        prev_close_map.clear()
        for sym, df in data.items():
            if dt in df.index:
                prev_close_map[sym] = float(df.loc[dt, "Close"])

    # ── Compute metrics ─────────────────────────────────────────────────
    equity_series = pd.Series(
        [e for _, e in daily_equity],
        index=[d for d, _ in daily_equity],
    )
    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    ann_factor = 252 / len(returns) if len(returns) > 0 else 1
    ann_return = (1 + total_return) ** ann_factor - 1

    sr = sharpe_ratio(returns.values, rf=0.04) if len(returns) > 10 else 0.0
    dd = drawdown_stats(returns.values) if len(returns) > 1 else None

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0.0
    profit_factor = (sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses)) if losses else float("inf")

    return {
        "total_return": round(total_return * 100, 2),
        "annualized_return": round(ann_return * 100, 2),
        "sharpe_ratio": round(sr, 3),
        "max_drawdown_pct": round(dd.max_drawdown * 100, 2) if dd else 0.0,
        "total_trades": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "final_equity": round(equity_series.iloc[-1], 2),
        "start_equity": INITIAL_CASH,
    }


def main() -> None:
    log.info("Fetching S&P 500 daily data for 2024...")
    data = fetch_data(SP500_TICKERS, start="2023-11-01", end="2024-12-31")

    log.info("Running gap-fade backtest (2024-01-01 to 2024-12-31)...")
    results = run_backtest(data, start_date="2024-01-01", end_date="2024-12-31")

    print("\n" + "=" * 60)
    print("OPENING GAP FADE — 2024 BACKTEST RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)

    sr = results.get("sharpe_ratio", 0)
    if sr > 0.5:
        print(f"\n  Sharpe {sr} > 0.5 threshold — PROCEED to live deployment")
    else:
        print(f"\n  Sharpe {sr} <= 0.5 threshold — needs review before deployment")

    # Save results
    out_path = REPO_ROOT / "results" / "gap_fade_backtest_2024.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    import json
    main()
