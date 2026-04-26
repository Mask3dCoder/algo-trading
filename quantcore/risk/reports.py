"""Risk report generation: daily, weekly, with HTML/Plotly output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from quantcore.core.types import PortfolioSnapshot
from quantcore.risk.metrics import compute_all_metrics
from quantcore.risk.var import cvar, historical_var


@dataclass
class DailyRiskReport:
    """Daily risk summary."""
    date: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_975: float = 0.0
    net_delta: float = 0.0
    gross_gamma: float = 0.0
    total_vega: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    limit_utilization: dict[str, float] = field(default_factory=dict)
    sharpe_rolling: float = 0.0
    positions_count: int = 0
    is_circuit_breaker: bool = False


@dataclass
class WeeklySummary:
    """Weekly risk & performance summary."""
    week_start: str = ""
    week_end: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_dd_week: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    top_contributors: list[tuple[str, float]] = field(default_factory=list)
    top_detractors: list[tuple[str, float]] = field(default_factory=list)
    var_violations: int = 0
    regime: str = "unknown"


def generate_daily_risk_report(
    portfolio: PortfolioSnapshot,
    returns_history: NDArray[np.floating],
    greeks: dict[str, float] | None = None,
    drawdown_current: float = 0.0,
    drawdown_max: float = 0.0,
) -> DailyRiskReport:
    """Generate a daily risk report."""
    r = np.asarray(returns_history, dtype=float).flatten()

    report = DailyRiskReport(
        date=str(portfolio.timestamp.date()) if portfolio.timestamp else "",
        pnl=float(portfolio.total_value),
        pnl_pct=float(r[-1] * 100) if len(r) > 0 else 0.0,
        positions_count=len(portfolio.positions),
    )

    # VaR
    if len(r) >= 20:
        var95 = historical_var(r, confidence=0.95)
        var99 = historical_var(r, confidence=0.99)
        report.var_95 = -var95.var
        report.var_99 = -var99.var
        report.cvar_975 = -cvar(r, confidence=0.975)

    # Greeks
    if greeks:
        report.net_delta = greeks.get("total_delta_dollar", 0.0)
        report.gross_gamma = greeks.get("total_gamma", 0.0)
        report.total_vega = greeks.get("total_vega", 0.0)

    # Drawdown
    report.current_drawdown = drawdown_current
    report.max_drawdown = drawdown_max

    # Rolling Sharpe
    if len(r) >= 63:
        from quantcore.risk.metrics import sharpe_ratio
        report.sharpe_rolling = sharpe_ratio(r[-63:])

    return report


def generate_weekly_summary(
    daily_returns: NDArray[np.floating],
    daily_pnl: dict[str, float] | None = None,
    attribution: dict[str, float] | None = None,
    regime: str = "unknown",
) -> WeeklySummary:
    """Generate a weekly performance summary."""
    r = np.asarray(daily_returns, dtype=float).flatten()

    if len(r) == 0:
        return WeeklySummary()

    metrics = compute_all_metrics(r)

    total_pnl = float(np.sum(r))
    pos_returns = r[r > 0]
    neg_returns = r[r < 0]

    win_rate = len(pos_returns) / len(r) if len(r) > 0 else 0.0
    avg_win = float(np.mean(pos_returns)) if len(pos_returns) > 0 else 0.0
    avg_loss = float(np.mean(neg_returns)) if len(neg_returns) > 0 else 0.0
    profit_factor = (
        abs(float(np.sum(pos_returns)) / float(np.sum(neg_returns)))
        if len(neg_returns) > 0 and float(np.sum(neg_returns)) != 0
        else float("inf") if len(pos_returns) > 0 else 0.0
    )

    # Top contributors/detractors
    top_contrib = []
    top_detract = []
    if attribution:
        sorted_attr = sorted(attribution.items(), key=lambda x: x[1])
        top_contrib = sorted_attr[-5:] if len(sorted_attr) >= 5 else sorted_attr
        top_detract = sorted_attr[:5] if len(sorted_attr) >= 5 else sorted_attr

    return WeeklySummary(
        pnl=total_pnl,
        pnl_pct=float(total_pnl * 100),
        sharpe=metrics.get("sharpe", 0.0),
        sortino=metrics.get("sortino", 0.0),
        calmar=metrics.get("calmar", 0.0),
        max_dd_week=float(np.min(np.minimum.accumulate(np.cumsum(r)))) if len(r) > 0 else 0.0,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        top_contributors=top_contrib,
        top_detractors=top_detract,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------


def _plotly_drawdown_chart(returns: NDArray[np.floating]) -> str:
    """Generate a Plotly drawdown chart HTML snippet."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p>Plotly not installed.</p>"

    r = np.asarray(returns, dtype=float).flatten()
    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)
    dd = cumulative - running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cumulative, mode="lines", name="Cumulative Returns",
        line=dict(color="blue", width=1),
    ))
    fig.add_trace(go.Scatter(
        y=running_max, mode="lines", name="Running Max",
        line=dict(color="gray", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        y=dd, mode="lines", name="Drawdown",
        fill="tozeroy", line=dict(color="red"),
    ))
    fig.update_layout(
        title="Equity Curve & Drawdown",
        xaxis_title="Days",
        yaxis_title="Return",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _plotly_var_chart(returns: NDArray[np.floating], var_series: NDArray[np.floating]) -> str:
    """Generate a Plotly VaR comparison chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p>Plotly not installed.</p>"

    r = np.asarray(returns, dtype=float).flatten()
    v = np.asarray(var_series, dtype=float).flatten()
    n = min(len(r), len(v))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=r[:n], mode="lines", name="Returns",
        line=dict(color="blue", width=0.5),
    ))
    fig.add_trace(go.Scatter(
        y=-v[:n], mode="lines", name="VaR (lower bound)",
        line=dict(color="red", dash="dot"),
    ))
    violations = r[:n] < -v[:n]
    viol_x = np.where(violations)[0]
    viol_y = r[:n][violations]
    if len(viol_x) > 0:
        fig.add_trace(go.Scatter(
            x=viol_x, y=viol_y, mode="markers",
            name="VaR Violations",
            marker=dict(color="red", size=6),
        ))
    fig.update_layout(
        title="Returns vs VaR",
        xaxis_title="Days",
        yaxis_title="Return",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _plotly_metrics_gauge(metrics: dict[str, float]) -> str:
    """Generate a Plotly metric gauge/bar chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p>Plotly not installed.</p>"

    gauge_metrics = ["sharpe", "sortino", "calmar", "omega"]
    values = [metrics.get(m, 0) for m in gauge_metrics]
    values = [v if v is not None and np.isfinite(v) else 0 for v in values]

    fig = go.Figure(data=[
        go.Bar(x=gauge_metrics, y=values, marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ])
    fig.update_layout(
        title="Risk Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=300,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_html_report(
    returns: NDArray[np.floating],
    var_series: NDArray[np.floating] | None = None,
    metrics: dict[str, float] | None = None,
    title: str = "QuantCore Risk Report",
) -> str:
    """Generate a complete HTML risk report with Plotly charts.

    Returns:
        HTML string.
    """
    r = np.asarray(returns, dtype=float).flatten()

    if metrics is None:
        metrics = compute_all_metrics(r)

    if var_series is None and len(r) >= 20:
        var_vals = np.full(len(r), float(historical_var(r, confidence=0.95).var))
    elif var_series is not None:
        var_vals = np.asarray(var_series, dtype=float).flatten()
    else:
        var_vals = np.zeros(len(r))

    dd_chart = _plotly_drawdown_chart(r)
    var_chart = _plotly_var_chart(r, var_vals)
    metrics_chart = _plotly_metrics_gauge(metrics)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; }}
        .chart-container {{ background: #16213e; border-radius: 8px; padding: 16px; margin: 16px 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
        .metric-card {{ background: #16213e; border-radius: 8px; padding: 12px; text-align: center; }}
        .metric-label {{ font-size: 0.8em; color: #888; text-transform: uppercase; }}
        .metric-value {{ font-size: 1.4em; font-weight: bold; color: #00d4ff; }}
        .warning {{ color: #ff6b6b; }}
        .good {{ color: #51cf66; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {"good" if metrics.get('sharpe', 0) > 1 else "warning"}">{metrics.get("sharpe", 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">{" "}{metrics.get("sortino", 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">{" "}{metrics.get("calmar", 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value warning">{metrics.get("max_drawdown", 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Skewness</div>
            <div class="metric-value">{" "}{metrics.get("skewness", 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Kurtosis</div>
            <div class="metric-value">{" "}{metrics.get("kurtosis", 0):.2f}</div>
        </div>
    </div>

    <div class="chart-container">{dd_chart}</div>
    <div class="chart-container">{var_chart}</div>
    <div class="chart-container">{metrics_chart}</div>

    <footer style="margin-top: 24px; font-size: 0.7em; color: #666;">
        QuantCore Risk Engine — Powered by Plotly
    </footer>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def generate_daily_and_weekly(
    portfolio: PortfolioSnapshot,
    returns_history: NDArray[np.floating],
    daily_returns: NDArray[np.floating],
    greeks: dict[str, float] | None = None,
    dd_current: float = 0.0,
    dd_max: float = 0.0,
    attribution: dict[str, float] | None = None,
) -> tuple[DailyRiskReport, WeeklySummary]:
    """Generate both daily and weekly reports."""
    daily = generate_daily_risk_report(
        portfolio, returns_history, greeks, dd_current, dd_max
    )
    weekly = generate_weekly_summary(
        daily_returns, attribution=attribution
    )
    return daily, weekly
