"""Performance analytics: risk metrics, drawdown analysis, HTML reports.

Delegates core risk metrics to quantcore.risk.metrics — avoids duplication.
Adds backtest-specific: trade analytics, equity-based drawdown reporting, perf summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcore.core.constants import ANNUALIZATION_FACTOR
from quantcore.risk.metrics import (
    bootstrap_ci as _risk_bootstrap_ci,
)
from quantcore.risk.metrics import (
    calmar_ratio as _risk_calmar,
)
from quantcore.risk.metrics import (
    omega_ratio as _risk_omega,
)
from quantcore.risk.metrics import (
    sharpe_ratio as _risk_sharpe,
)
from quantcore.risk.metrics import (
    sortino_ratio as _risk_sortino,
)


@dataclass
class PerformanceReport:
    """Complete performance analytics for a backtest."""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    omega: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_hold_period: float = 0.0
    n_trades: int = 0
    drawdown_table: list[dict[str, Any]] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)


def compute_returns(equity: NDArray | pd.Series) -> NDArray:
    """Compute simple returns from equity curve."""
    equity = np.asarray(equity, dtype=float)
    return np.diff(equity) / equity[:-1]


# ---------------------------------------------------------------------------
# Delegated risk metrics (wrappers around quantcore.risk.metrics)
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: NDArray, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio — delegates to risk.metrics."""
    return _risk_sharpe(returns, rf=risk_free, annualize=True)


def sortino_ratio(returns: NDArray, risk_free: float = 0.0, target: float = 0.0) -> float:
    """Annualized Sortino ratio — delegates to risk.metrics."""
    return _risk_sortino(returns, mar=target, rf=risk_free, annualize=True)


def max_drawdown(equity: NDArray | pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.20 = 20% drawdown)."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    return float(np.max(drawdowns))


def calmar_ratio(returns: NDArray, equity: NDArray, risk_free: float = 0.0) -> float:
    """Calmar ratio — delegates to risk.metrics which computes max DD from returns."""
    return _risk_calmar(returns)


def omega_ratio(returns: NDArray, threshold: float = 0.0) -> float:
    """Omega ratio — delegates to risk.metrics."""
    return _risk_omega(returns, threshold=threshold)


def bootstrap_confidence(
    returns: NDArray,
    metric_fn,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval — delegates to risk.metrics."""
    return _risk_bootstrap_ci(returns, metric_fn, n_boot=n_boot, ci=confidence, seed=seed)


# ---------------------------------------------------------------------------
# Drawdown analysis (backtest-specific: works on equity curves)
# ---------------------------------------------------------------------------


def drawdown_analysis(equity: NDArray) -> list[dict[str, Any]]:
    """Compute top-10 drawdown episodes with start, trough, recovery, depth, duration."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak

    episodes: list[dict[str, Any]] = []
    in_drawdown = False
    dd_start = 0

    for i in range(len(dd)):
        if dd[i] > 0 and not in_drawdown:
            in_drawdown = True
            dd_start = i
        elif dd[i] == 0 and in_drawdown:
            in_drawdown = False
            trough_idx = dd_start + np.argmax(dd[dd_start:i])
            episodes.append({
                "start": dd_start,
                "trough": trough_idx,
                "recovery": i,
                "depth": float(dd[trough_idx]),
                "duration": i - dd_start,
            })

    # If still in drawdown at end
    if in_drawdown:
        trough_idx = dd_start + np.argmax(dd[dd_start:])
        episodes.append({
            "start": dd_start,
            "trough": trough_idx,
            "recovery": None,
            "depth": float(dd[trough_idx]),
            "duration": len(dd) - dd_start,
        })

    # Sort by depth (largest first) and take top 10
    episodes.sort(key=lambda x: x["depth"], reverse=True)
    return episodes[:10]


# ---------------------------------------------------------------------------
# Trade analytics
# ---------------------------------------------------------------------------


def trade_analytics(pnl_list: NDArray) -> dict[str, float]:
    """Compute trade-level analytics from a list of trade P&Ls."""
    pnls = np.asarray(pnl_list, dtype=float)
    if len(pnls) == 0:
        return {"win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "n_trades": 0}

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
    expectancy = float(np.mean(pnls))

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "n_trades": len(pnls),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_performance_report(
    equity: NDArray,
    returns: NDArray | None = None,
    trade_pnls: NDArray | None = None,
    risk_free: float = 0.0,
) -> PerformanceReport:
    """Compute the full performance report from equity curve and optional trade P&Ls."""
    equity = np.asarray(equity, dtype=float)
    if returns is None:
        returns = compute_returns(equity)
    r = np.asarray(returns, dtype=float)

    total_ret = float(equity[-1] / equity[0] - 1.0)
    n_years = len(r) / ANNUALIZATION_FACTOR
    annual_ret = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    dd_table = drawdown_analysis(equity)
    avg_dd = float(np.mean([e["depth"] for e in dd_table])) if dd_table else 0.0

    report = PerformanceReport(
        total_return=total_ret,
        annual_return=annual_ret,
        sharpe=sharpe_ratio(r, risk_free),
        sortino=sortino_ratio(r, risk_free),
        calmar=calmar_ratio(r, equity, risk_free),
        omega=omega_ratio(r),
        max_drawdown=max_drawdown(equity),
        avg_drawdown=avg_dd,
        drawdown_table=dd_table,
    )

    if trade_pnls is not None:
        ta = trade_analytics(trade_pnls)
        report.win_rate = ta["win_rate"]
        report.profit_factor = ta["profit_factor"]
        report.expectancy = ta["expectancy"]
        report.n_trades = int(ta["n_trades"])

    return report


def generate_html_report(report: PerformanceReport) -> str:
    """Generate an HTML performance report with Plotly charts."""
    return f"""<!DOCTYPE html>
<html><head><title>QuantCore Performance Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 2rem; }}
.metric {{ display: inline-block; background: #16213e; padding: 1rem 2rem; margin: 0.5rem; border-radius: 8px; }}
.metric .value {{ font-size: 1.5rem; font-weight: bold; color: #4ecca3; }}
.metric .label {{ font-size: 0.8rem; color: #999; }}
h1 {{ color: #4ecca3; }} h2 {{ color: #e94560; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #333; text-align: left; }}
th {{ color: #4ecca3; }}
</style></head><body>
<h1>QuantCore Performance Report</h1>
<div>
  <div class="metric"><div class="value">{report.total_return:.2%}</div><div class="label">Total Return</div></div>
  <div class="metric"><div class="value">{report.annual_return:.2%}</div><div class="label">Annual Return</div></div>
  <div class="metric"><div class="value">{report.sharpe:.2f}</div><div class="label">Sharpe</div></div>
  <div class="metric"><div class="value">{report.sortino:.2f}</div><div class="label">Sortino</div></div>
  <div class="metric"><div class="value">{report.calmar:.2f}</div><div class="label">Calmar</div></div>
  <div class="metric"><div class="value">{report.omega:.2f}</div><div class="label">Omega</div></div>
  <div class="metric"><div class="value">{report.max_drawdown:.2%}</div><div class="label">Max Drawdown</div></div>
  <div class="metric"><div class="value">{report.win_rate:.2%}</div><div class="label">Win Rate</div></div>
  <div class="metric"><div class="value">{report.profit_factor:.2f}</div><div class="label">Profit Factor</div></div>
  <div class="metric"><div class="value">{report.n_trades}</div><div class="label">Trades</div></div>
</div>
<h2>Drawdown Table</h2>
<table><tr><th>#</th><th>Depth</th><th>Duration</th></tr>
{" ".join(f'<tr><td>{i+1}</td><td>{e["depth"]:.2%}</td><td>{e["duration"]} bars</td></tr>' for i, e in enumerate(report.drawdown_table[:10]))}
</table></body></html>"""
