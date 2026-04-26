"""Portfolio dashboard panel: P&L chart, equity curve, positions table."""

from __future__ import annotations

from typing import Any


def build_portfolio_tab() -> Any:
    """Build the portfolio monitoring tab layout."""
    try:
        from dash import dcc, html
    except ImportError:
        return None

    return html.Div([
        html.H2("Portfolio Overview", style={"color": "#e0e0e0", "marginBottom": "16px"}),

        # Key metrics row
        html.Div([
            _metric_card("Portfolio Value", "$---", "portfolio-value"),
            _metric_card("Daily P&L", "$---", "daily-pnl"),
            _metric_card("Total Return", "---%", "total-return"),
            _metric_card("Cash", "$---", "cash-balance"),
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "16px", "marginBottom": "24px",
        }),

        # Equity curve chart
        html.Div([
            html.H3("Equity Curve", style={"color": "#ccc", "marginBottom": "8px"}),
            dcc.Graph(id="equity-curve-chart", config={"displayModeBar": False}),
        ], style={
            "backgroundColor": "#16213e", "borderRadius": "8px",
            "padding": "16px", "marginBottom": "16px",
        }),

        # Positions table and allocation
        html.Div([
            html.Div([
                html.H3("Positions", style={"color": "#ccc"}),
                html.Table(id="positions-table", style={"width": "100%", "color": "#e0e0e0", "borderCollapse": "collapse"}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
            html.Div([
                html.H3("Allocation", style={"color": "#ccc"}),
                dcc.Graph(id="allocation-pie", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px"}),
    ])


def _metric_card(title: str, value: str, element_id: str) -> Any:
    try:
        from dash import html
    except ImportError:
        return None
    return html.Div([
        html.Div(title, style={"fontSize": "12px", "color": "#888", "textTransform": "uppercase"}),
        html.Div(value, id=element_id, style={"fontSize": "22px", "fontWeight": "bold", "color": "#fff"}),
    ], style={
        "backgroundColor": "#16213e", "borderRadius": "8px",
        "padding": "16px", "textAlign": "center",
    })
