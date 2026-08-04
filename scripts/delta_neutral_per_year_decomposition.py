#!/usr/bin/env python3
"""Delta Neutral VRP — Per-Year Walk-Forward Decomposition.

This is the SINGLE HIGHEST-VALUE ACTION identified by the Investment Committee:
determine whether delta_neutral_vrp survived 2022's inflation shock.

The walk-forward OOS Sharpe of +1.30 and MaxDD of 15.43% are aggregates
over 2020-2024.  Without per-year decomposition, we cannot assess whether
the 15.43% MaxDD was concentrated in 2022 (indicating regime vulnerability)
or distributed across years (indicating resilience).

Usage:
    python scripts/delta_neutral_per_year_decomposition.py

Outputs:
    reports/delta_neutral_per_year_decomposition.json
    reports/delta_neutral_per_year_decomposition.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Parameters (must match the validated strategy exactly) ────────────────────

UNDERLYING = "SPY"
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_PER_CONTRACT = 0.65
COMMISSION_PER_SHARE = 0.005
HEDGE_SLIPPAGE_PCT = 0.0005  # 5 bps
RISK_FREE_RATE = 0.04
RISK_PER_TRADE_PCT = 0.01
MAX_LOSS_MULT = 3.0
ENTRY_INTERVAL_DAYS = 21  # matched to the validated config
MAX_ACTIVE_POSITIONS = 2
PROFIT_TARGET_PCT = 0.50
DTE_ENTRY_MIN = 30
DTE_ENTRY_MAX = 45
DTE_EXIT = 14
DELTA_HEDGE_THRESHOLD = 5.0
IVR_ENTRY = 0.0  # unconditional entry
STRUCTURE = "straddle"

# Data range: need warmup from 2017 for IVR lookback, analysis covers 2020-2024
DATA_START = "2017-01-01"
ANALYSIS_START = "2020-01-01"
ANALYSIS_END = "2024-12-31"

# Per-year windows to decompose
YEAR_WINDOWS = {
    "2020": ("2020-01-01", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),  # THE CRITICAL YEAR
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
}

# Regime labels for context
REGIME_LABELS = {
    "2020": "COVID Shock + Recovery",
    "2021": "Post-COVID Bull / Easy Money",
    "2022": "Inflation Shock / Rate Hikes",
    "2023": "Recovery Bull / Soft Landing Hopes",
    "2024": "Compressed Volatility / AI Rally",
}


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_spy_data(start: str, end: str) -> tuple[pd.Series, pd.Series | None]:
    """Load SPY close prices and VIX from Yahoo Finance."""
    import yfinance as yf

    _log.info("Downloading SPY data (%s to %s)...", start, end)
    spy = yf.Ticker(UNDERLYING)
    hist = spy.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No SPY data returned")
    close = hist["Close"]
    close.index = pd.DatetimeIndex([d.date() for d in close.index])
    _log.info("Loaded %d SPY trading days", len(close))

    # Try to load VIX for more realistic IV
    _log.info("Downloading VIX data...")
    vix_ticker = yf.Ticker("^VIX")
    vix_hist = vix_ticker.history(start=start, end=end, auto_adjust=True)
    if not vix_hist.empty:
        vix = vix_hist["Close"]
        vix.index = pd.DatetimeIndex([d.date() for d in vix.index])
        _log.info("Loaded %d VIX trading days", len(vix))
    else:
        vix = None
        _log.warning("No VIX data — using synthetic IV only")

    return close, vix


# ── Per-Year Metrics ──────────────────────────────────────────────────────────


def compute_year_metrics(
    trades: list,
    equity_curve: list[tuple[date, float]],
    year_start: str,
    year_end: str,
    initial_capital: float,
    rf_annual: float = 0.04,
) -> dict:
    """Compute metrics for a single year from the full simulation output.

    Parameters
    ----------
    trades : list[DeltaNeutralTrade]
        All closed trades from the simulation.
    equity_curve : list[tuple[date, float]]
        Daily (date, equity) pairs from the simulation.
    year_start, year_end : str
        ISO date strings for the year window.
    initial_capital : float
        Starting capital for the full simulation (used to rebase).
    rf_annual : float
        Risk-free rate for excess Sharpe computation.

    Returns
    -------
    dict with per-year metrics.
    """
    ys = pd.Timestamp(year_start).date() if not isinstance(year_start, date) else year_start
    ye = pd.Timestamp(year_end).date() if not isinstance(year_end, date) else year_end

    # Filter equity curve to this year
    year_equity = [(d, v) for d, v in equity_curve if ys <= d <= ye]
    if len(year_equity) < 2:
        return {
            "year": year_start[:4],
            "sharpe": float("nan"),
            "sharpe_excess": float("nan"),
            "max_dd_pct": float("nan"),
            "total_return_pct": float("nan"),
            "annual_return_pct": float("nan"),
            "n_trades": 0,
            "win_rate_pct": float("nan"),
            "profit_factor": float("nan"),
            "avg_pnl_per_trade": float("nan"),
            "total_pnl": 0.0,
            "total_hedge_cost": 0.0,
            "n_trading_days": len(year_equity),
            "cumulative_gamma_cost": 0.0,
            "regime_label": REGIME_LABELS.get(year_start[:4], ""),
            "close_reasons": {},
            "trades_detail": [],
        }

    # Rebase equity to start at initial_capital for this year
    eq_start = year_equity[0][1]
    scale = initial_capital / eq_start if eq_start > 0 else 1.0
    dates_arr = [d for d, _ in year_equity]
    eq_arr = np.array([v * scale for _, v in year_equity])

    # Daily returns
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    daily_rets = np.clip(daily_rets, -1.0, 10.0)  # safety clamp

    # Sharpe (raw)
    mean_ret = float(np.mean(daily_rets))
    std_ret = float(np.std(daily_rets, ddof=1))
    if std_ret < 1e-10:
        sharpe = float("nan")
    else:
        sharpe = mean_ret / std_ret * np.sqrt(252)

    # Excess Sharpe (over risk-free)
    rf_daily = rf_annual / 252.0
    excess_rets = daily_rets - rf_daily
    std_excess = float(np.std(excess_rets, ddof=1))
    if std_excess < 1e-10:
        sharpe_excess = float("nan")
    else:
        sharpe_excess = float(np.mean(excess_rets)) / std_excess * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1.0) * 100.0
    max_dd_pct = float(np.abs(dd.min()))

    # Total return
    total_return_pct = (eq_arr[-1] / eq_arr[0] - 1.0) * 100.0

    # Annual return
    n_days = len(eq_arr)
    n_years = n_days / 252.0
    if n_years > 0 and (1 + total_return_pct / 100) > 0:
        annual_return_pct = ((1 + total_return_pct / 100) ** (1 / n_years) - 1) * 100.0
    else:
        annual_return_pct = 0.0

    # Filter trades closed in this year
    year_trades = [
        t for t in trades
        if t.closed and t.close_date is not None and ys <= t.close_date <= ye
    ]
    # Also trades opened in this year
    year_trades_opened = [
        t for t in trades
        if t.entry_date is not None and ys <= t.entry_date <= ye
    ]
    # Use close_date for primary attribution (matches P&L realization)
    year_trades_attr = year_trades if year_trades else year_trades_opened

    n_trades = len(year_trades_attr)
    winning_trades = [t for t in year_trades_attr if t.pnl > 0]
    win_rate_pct = (len(winning_trades) / n_trades * 100.0) if n_trades > 0 else float("nan")

    # Profit factor
    gross_profit = sum(t.pnl for t in year_trades_attr if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in year_trades_attr if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total P&L and hedge cost for this year
    total_pnl = sum(t.pnl for t in year_trades_attr)
    total_hedge_cost = sum(t.cumulative_hedge_cost for t in year_trades_attr)
    cumulative_gamma_cost = sum(getattr(t, "cumulative_gamma_cost", 0.0) for t in year_trades_attr)

    # Close reasons
    close_reasons: dict[str, int] = {}
    for t in year_trades_attr:
        reason = t.close_reason
        close_reasons[reason] = close_reasons.get(reason, 0) + 1

    # Per-trade detail
    trades_detail = []
    for t in year_trades_attr:
        trades_detail.append({
            "entry_date": str(t.entry_date),
            "close_date": str(t.close_date) if t.close_date else None,
            "strike_call": t.short_call_strike,
            "strike_put": t.short_put_strike,
            "contracts": t.contracts,
            "credit": round(t.credit, 4),
            "entry_iv": round(t.entry_iv, 4) if t.entry_iv else None,
            "pnl": round(t.pnl, 2),
            "close_reason": t.close_reason,
            "hedge_cost": round(t.cumulative_hedge_cost, 2),
        })

    avg_pnl = total_pnl / n_trades if n_trades > 0 else float("nan")

    return {
        "year": year_start[:4],
        "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else float("nan"),
        "sharpe_excess": round(sharpe_excess, 4) if np.isfinite(sharpe_excess) else float("nan"),
        "max_dd_pct": round(max_dd_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annual_return_pct": round(annual_return_pct, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate_pct, 1) if np.isfinite(win_rate_pct) else float("nan"),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("nan"),
        "avg_pnl_per_trade": round(avg_pnl, 2) if np.isfinite(avg_pnl) else float("nan"),
        "total_pnl": round(total_pnl, 2),
        "total_hedge_cost": round(total_hedge_cost, 2),
        "cumulative_gamma_cost": round(cumulative_gamma_cost, 2),
        "n_trading_days": n_days,
        "regime_label": REGIME_LABELS.get(year_start[:4], ""),
        "close_reasons": close_reasons,
        "trades_detail": trades_detail,
    }


# ── Main Execution ───────────────────────────────────────────────────────────


def run_decomposition() -> dict:
    """Run the full 2020-2024 simulation and decompose per-year."""
    from quantcore.alpha.delta_neutral import DeltaNeutralVRP

    # 1. Load data
    close, vix = load_spy_data(DATA_START, ANALYSIS_END)

    # 2. Create strategy instance (same params as validated backtest)
    _log.info("Initializing DeltaNeutralVRP strategy...")
    strategy = DeltaNeutralVRP(
        close=close,
        vix_series=vix,
        underlying=UNDERLYING,
        structure=STRUCTURE,
        dte_entry_min=DTE_ENTRY_MIN,
        dte_entry_max=DTE_ENTRY_MAX,
        dte_exit=DTE_EXIT,
        profit_target_pct=PROFIT_TARGET_PCT,
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        max_loss_mult=MAX_LOSS_MULT,
        entry_interval_days=ENTRY_INTERVAL_DAYS,
        ivr_entry=IVR_ENTRY,
        delta_hedge_threshold=DELTA_HEDGE_THRESHOLD,
        hedge_slippage_pct=HEDGE_SLIPPAGE_PCT,
        risk_free_rate=RISK_FREE_RATE,
        commission_per_contract=COMMISSION_PER_CONTRACT,
        commission_per_share=COMMISSION_PER_SHARE,
        use_enhanced_iv=True,
    )

    # 3. Run the full simulation (2017 data loaded for warmup,
    #    simulation starts trading after 60-day warmup)
    _log.info("Running full simulation (data covers %s to %s)...", DATA_START, ANALYSIS_END)
    all_trades, equity_curve, daily_returns, daily_turnover = strategy._simulate(
        close, strategy._iv,
        capital=INITIAL_CAPITAL,
    )
    _log.info("Simulation complete: %d total trades, %d equity points",
              len(all_trades), len(equity_curve))

    # 4. Compute aggregate metrics for the full 2020-2024 OOS period
    # First, filter equity curve to 2020-2024
    a_start = date(2020, 1, 1)
    a_end = date(2024, 12, 31)
    oos_equity = [(d, v) for d, v in equity_curve if a_start <= d <= a_end]

    aggregate = compute_year_metrics(
        all_trades, equity_curve,
        "2020-01-01", "2024-12-31",
        INITIAL_CAPITAL,
    )

    # 5. Compute per-year decomposition
    per_year = {}
    for year_label, (ys, ye) in YEAR_WINDOWS.items():
        _log.info("Computing metrics for %s (%s)...", year_label, REGIME_LABELS.get(year_label, ""))
        metrics = compute_year_metrics(
            all_trades, equity_curve, ys, ye, INITIAL_CAPITAL,
        )
        per_year[year_label] = metrics

        # Log key metrics immediately
        sharpe_str = f"{metrics['sharpe']:.2f}" if np.isfinite(metrics["sharpe"]) else "N/A"
        _log.info(
            "  %s: Sharpe=%s, MaxDD=%.2f%%, Trades=%d, WinRate=%.1f%%, P&L=$%.0f",
            year_label, sharpe_str, metrics["max_dd_pct"],
            metrics["n_trades"],
            metrics["win_rate_pct"] if np.isfinite(metrics["win_rate_pct"]) else 0,
            metrics["total_pnl"],
        )

    # 6. Build output
    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "strategy": "DeltaNeutralVRP",
        "structure": STRUCTURE,
        "parameters": {
            "underlying": UNDERLYING,
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "max_loss_mult": MAX_LOSS_MULT,
            "entry_interval_days": ENTRY_INTERVAL_DAYS,
            "max_active_positions": MAX_ACTIVE_POSITIONS,
            "profit_target_pct": PROFIT_TARGET_PCT,
            "dte_entry_min": DTE_ENTRY_MIN,
            "dte_entry_max": DTE_ENTRY_MAX,
            "dte_exit": DTE_EXIT,
            "commission_per_contract": COMMISSION_PER_CONTRACT,
            "commission_per_share": COMMISSION_PER_SHARE,
            "hedge_slippage_bps": HEDGE_SLIPPAGE_PCT * 10_000,
            "risk_free_rate": RISK_FREE_RATE,
            "ivr_entry": IVR_ENTRY,
            "delta_hedge_threshold": DELTA_HEDGE_THRESHOLD,
        },
        "aggregate_2020_2024": aggregate,
        "per_year": per_year,
        "audit_question": "Did delta_neutral_vrp survive 2022's inflation shock?",
        "commit_note": "No parameter changes. Same validated config. Decomposition only.",
    }

    return result


def generate_markdown_report(result: dict) -> str:
    """Generate a professional markdown report from the decomposition results."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    per_year = result["per_year"]
    agg = result["aggregate_2020_2024"]

    # Determine 2022 verdict
    y2022 = per_year.get("2022", {})
    sharpe_2022 = y2022.get("sharpe", float("nan"))
    maxdd_2022 = y2022.get("max_dd_pct", float("nan"))

    if np.isfinite(sharpe_2022) and sharpe_2022 < 0:
        verdict_2022 = "FAIL — Negative Sharpe indicates regime-picked artifact"
        verdict_emoji = "❌"
    elif np.isfinite(maxdd_2022) and maxdd_2022 > 20:
        verdict_2022 = "FAIL — MaxDD > 20% in 2022 indicates structural fragility"
        verdict_emoji = "❌"
    elif np.isfinite(sharpe_2022) and sharpe_2022 < 0.3:
        verdict_2022 = "MARGINAL — Sharpe barely positive in 2022; regime vulnerability"
        verdict_emoji = "⚠️"
    elif np.isfinite(sharpe_2022) and sharpe_2022 >= 0.3:
        verdict_2022 = "PASS — Positive Sharpe in 2022 supports cross-regime robustness"
        verdict_emoji = "✅"
    else:
        verdict_2022 = "INSUFFICIENT DATA — Cannot determine 2022 outcome"
        verdict_emoji = "⚠️"

    lines = [
        f"# Delta Neutral VRP — Per-Year Walk-Forward Decomposition",
        f"",
        f"**Generated**: {now}",
        f"**Purpose**: Answer the Investment Committee's critical question:",
        f"**Did delta_neutral_vrp survive 2022's inflation shock?**",
        f"",
        f"---",
        f"",
        f"## Parameters (Locked to Validated Config)",
        f"",
        f"| Parameter | Value |",
        f"|---|---|",
    ]
    for k, v in result["parameters"].items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"",
        f"## Aggregate OOS Performance (2020-2024)",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| OOS Sharpe | {agg['sharpe']:.4f} |" if np.isfinite(agg.get("sharpe", float("nan"))) else f"| OOS Sharpe | N/A |",
        f"| OOS Sharpe (excess) | {agg['sharpe_excess']:.4f} |" if np.isfinite(agg.get("sharpe_excess", float("nan"))) else f"| OOS Sharpe (excess) | N/A |",
        f"| Max Drawdown | {agg['max_dd_pct']:.2f}% |",
        f"| Total Return | {agg['total_return_pct']:.2f}% |",
        f"| Total Trades | {agg['n_trades']} |",
        f"| Win Rate | {agg['win_rate_pct']:.1f}% |" if np.isfinite(agg.get("win_rate_pct", float("nan"))) else f"| Win Rate | N/A |",
        f"| Profit Factor | {agg['profit_factor']:.2f} |" if np.isfinite(agg.get("profit_factor", float("nan"))) else f"| Profit Factor | N/A |",
        f"",
        f"## Per-Year Decomposition",
        f"",
        f"| Year | Regime | Sharpe | Sharpe (excess) | MaxDD | Return | Trades | Win Rate | P&L |",
        f"|---|---|---|---|---|---|---|---|---|",
    ]
    for year_label in ["2020", "2021", "2022", "2023", "2024"]:
        y = per_year.get(year_label, {})
        regime = y.get("regime_label", "")
        sharpe_s = f"{y['sharpe']:.2f}" if np.isfinite(y.get("sharpe", float("nan"))) else "N/A"
        sharpe_ex_s = f"{y['sharpe_excess']:.2f}" if np.isfinite(y.get("sharpe_excess", float("nan"))) else "N/A"
        maxdd_s = f"{y['max_dd_pct']:.2f}%"
        ret_s = f"{y['total_return_pct']:.2f}%"
        n_t = y.get("n_trades", 0)
        wr_s = f"{y['win_rate_pct']:.1f}%" if np.isfinite(y.get("win_rate_pct", float("nan"))) else "N/A"
        pnl_s = f"${y['total_pnl']:,.0f}"

        # Highlight 2022
        bold = "**" if year_label == "2022" else ""
        bold_end = "**" if year_label == "2022" else ""
        lines.append(
            f"| {bold}{year_label}{bold_end} | {regime} | {bold}{sharpe_s}{bold_end} | "
            f"{bold}{sharpe_ex_s}{bold_end} | {bold}{maxdd_s}{bold_end} | {ret_s} | "
            f"{n_t} | {wr_s} | {pnl_s} |"
        )

    lines += [
        f"",
        f"## 2022 Assessment (THE CRITICAL YEAR)",
        f"",
        f"**Verdict: {verdict_emoji} {verdict_2022}**",
        f"",
    ]

    if np.isfinite(sharpe_2022):
        lines += [
            f"### 2022 Details",
            f"",
            f"| Metric | Value | Gate | Pass? |",
            f"|---|---|---|---|",
            f"| 2022 Sharpe | {sharpe_2022:.4f} | ≥ 0 | {'✅' if sharpe_2022 >= 0 else '❌'} |",
            f"| 2022 MaxDD | {maxdd_2022:.2f}% | < 20% | {'✅' if maxdd_2022 < 20 else '❌'} |",
            f"| 2022 Trades | {y2022.get('n_trades', 0)} | — | — |",
            f"| 2022 Win Rate | {y2022.get('win_rate_pct', 0):.1f}% | — | — |" if np.isfinite(y2022.get("win_rate_pct", float("nan"))) else "",
            f"| 2022 P&L | ${y2022.get('total_pnl', 0):,.0f} | — | — |",
            f"| 2022 Profit Factor | {y2022.get('profit_factor', 0):.2f} | — | — |" if np.isfinite(y2022.get("profit_factor", float("nan"))) else "",
        ]

        # Trade-level detail for 2022
        trades_detail = y2022.get("trades_detail", [])
        if trades_detail:
            lines += [
                f"",
                f"### 2022 Trade-Level Detail",
                f"",
                f"| Entry | Close | Strike (C/P) | Contracts | Credit | IV | P&L | Reason |",
                f"|---|---|---|---|---|---|---|---|",
            ]
            for t in trades_detail:
                lines.append(
                    f"| {t['entry_date']} | {t['close_date']} | "
                    f"{t['strike_call']:.0f}/{t['strike_put']:.0f} | "
                    f"{t['contracts']} | {t['credit']:.2f} | "
                    f"{t['entry_iv']:.4f} | ${t['pnl']:,.0f} | {t['close_reason']} |"
                )
    else:
        lines.append(f"**Insufficient data to compute 2022 metrics.**")

    # Regime dependence assessment
    lines += [
        f"",
        f"## Regime Dependence Assessment",
        f"",
    ]
    sharpes = []
    for year_label in ["2020", "2021", "2022", "2023", "2024"]:
        y = per_year.get(year_label, {})
        s = y.get("sharpe", float("nan"))
        if np.isfinite(s):
            sharpes.append(s)

    if len(sharpes) >= 2:
        best_sharpe = max(sharpes)
        worst_sharpe = min(sharpes)
        if worst_sharpe > 0:
            stability_ratio = best_sharpe / worst_sharpe
            lines.append(f"- Best year Sharpe: {best_sharpe:.2f}")
            lines.append(f"- Worst year Sharpe: {worst_sharpe:.2f}")
            lines.append(f"- Stability ratio (best/worst): {stability_ratio:.1f}×")
            if stability_ratio > 5:
                lines.append(f"- **REGIME-PICKED**: Stability ratio > 5× — strategy is structurally dependent on favorable regimes")
            elif stability_ratio > 2:
                lines.append(f"- **MODERATE REGIME DEPENDENCE**: Stability ratio 2-5× — some regime vulnerability")
            else:
                lines.append(f"- **ACCEPTABLE**: Stability ratio < 2× — cross-regime robustness supported")
        else:
            lines.append(f"- Worst year Sharpe is negative ({worst_sharpe:.2f}) — **REGIME-PICKED** by definition")

    lines += [
        f"",
        f"## Committee Decision Framework",
        f"",
        f"Based on the 2022 result:",
        f"",
    ]
    if np.isfinite(sharpe_2022) and sharpe_2022 >= 0:
        lines += [
            f"→ **Proceed with paper trading.** 2022 Sharpe ≥ 0 supports cross-regime survivability.",
            f"→ Paper trading duration: 18-24 months minimum, 50+ trades required.",
            f"→ Kill switch: MaxDD > 20% or 5 consecutive losses or monthly Sharpe < -1.0.",
        ]
    else:
        lines += [
            f"→ **KILL the strategy immediately.** 2022 Sharpe < 0 confirms regime-picked artifact.",
            f"→ Do NOT proceed to paper trading. 18 months of paper trading would be wasted.",
            f"→ Transition to failure scenario: acquire options data, test HYPOTHESIS_002.",
        ]

    lines += [
        f"",
        f"---",
        f"",
        f"*Generated by `scripts/delta_neutral_per_year_decomposition.py`*",
        f"*No strategy parameters were modified. This is decomposition, not optimization.*",
        f"*Committee review date: 2026-06-23*",
    ]

    return "\n".join(lines)


def main() -> None:
    _log.info("=" * 70)
    _log.info("DELTA NEUTRAL VRP — PER-YEAR WALK-FORWARD DECOMPOSITION")
    _log.info("=" * 70)
    _log.info("")
    _log.info("CRITICAL QUESTION: Did delta_neutral_vrp survive 2022?")
    _log.info("  - If 2022 Sharpe < 0: KILL the strategy immediately")
    _log.info("  - If 2022 Sharpe >= 0: Proceed with paper trading")
    _log.info("")

    # Run decomposition
    result = run_decomposition()

    # 2022 verdict
    y2022 = result["per_year"].get("2022", {})
    sharpe_2022 = y2022.get("sharpe", float("nan"))
    maxdd_2022 = y2022.get("max_dd_pct", float("nan"))

    _log.info("")
    _log.info("=" * 70)
    _log.info("  2022 RESULTS (THE CRITICAL YEAR)")
    _log.info("=" * 70)
    if np.isfinite(sharpe_2022):
        _log.info("  2022 Sharpe:     %.4f", sharpe_2022)
        _log.info("  2022 MaxDD:      %.2f%%", maxdd_2022)
        _log.info("  2022 Trades:    %d", y2022.get("n_trades", 0))
        _log.info("  2022 P&L:        $%.0f", y2022.get("total_pnl", 0))
        _log.info("  2022 Win Rate:   %.1f%%", y2022.get("win_rate_pct", 0) if np.isfinite(y2022.get("win_rate_pct", float("nan"))) else 0)
    else:
        _log.info("  2022 Sharpe:     N/A (insufficient data)")
    _log.info("")

    if np.isfinite(sharpe_2022) and sharpe_2022 < 0:
        _log.info("  VERDICT: ❌ KILL — 2022 Sharpe < 0 confirms regime-picked artifact")
    elif np.isfinite(maxdd_2022) and maxdd_2022 > 20:
        _log.info("  VERDICT: ❌ KILL — 2022 MaxDD > 20%% confirms structural fragility")
    elif np.isfinite(sharpe_2022) and sharpe_2022 < 0.3:
        _log.info("  VERDICT: ⚠️  MARGINAL — 2022 Sharpe barely positive; proceed with caution")
    elif np.isfinite(sharpe_2022):
        _log.info("  VERDICT: ✅ PASS — 2022 Sharpe >= 0.3; proceed with paper trading")
    else:
        _log.info("  VERDICT: ⚠️  INSUFFICIENT DATA — Cannot determine 2022 outcome")

    _log.info("=" * 70)

    # Save JSON
    json_path = REPORTS_DIR / "delta_neutral_per_year_decomposition.json"
    # Convert NaN to None for JSON serialization
    def sanitize(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(sanitize(result), f, indent=2, default=str)
    _log.info("JSON saved to %s", json_path)

    # Save Markdown
    md = generate_markdown_report(result)
    md_path = REPORTS_DIR / "delta_neutral_per_year_decomposition.md"
    md_path.write_text(md, encoding="utf-8")
    _log.info("Markdown saved to %s", md_path)

    _log.info("")
    _log.info("Done. Review the 2022 column before making any paper trading decision.")


if __name__ == "__main__":
    main()
