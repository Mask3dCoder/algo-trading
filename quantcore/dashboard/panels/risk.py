"""Risk dashboard panel: VaR gauges, Greeks heatmap, drawdown chart, correlations."""

from __future__ import annotations

from typing import Any


def build_risk_tab() -> Any:
    """Build the risk monitoring tab layout."""
    try:
        from dash import dcc, html
    except ImportError:
        return None

    return html.Div([
        html.H2("Risk Monitor", style={"color": "#e0e0e0", "marginBottom": "16px"}),

        # VaR cards
        html.Div([
            _var_card("VaR 95%", "var-95"),
            _var_card("VaR 99%", "var-99"),
            _var_card("CVaR 97.5%", "cvar-975"),
            _var_card("Max DD", "max-dd"),
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "16px", "marginBottom": "24px",
        }),

        # Risk charts row
        html.Div([
            html.Div([
                html.H3("VaR Violations", style={"color": "#ccc"}),
                dcc.Graph(id="var-violations-chart", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
            html.Div([
                html.H3("Drawdown", style={"color": "#ccc"}),
                dcc.Graph(id="drawdown-chart", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # Greeks heatmap and correlation matrix
        html.Div([
            html.Div([
                html.H3("Greeks", style={"color": "#ccc"}),
                dcc.Graph(id="greeks-heatmap", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
            html.Div([
                html.H3("Correlation Matrix", style={"color": "#ccc"}),
                dcc.Graph(id="correlation-matrix", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px"}),
    ])


def _var_card(label: str, element_id: str) -> Any:
    try:
        from dash import html
    except ImportError:
        return None
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": "#888", "textTransform": "uppercase"}),
        html.Div("$---", id=element_id, style={"fontSize": "22px", "fontWeight": "bold", "color": "#ff6b6b"}),
    ], style={
        "backgroundColor": "#16213e", "borderRadius": "8px",
        "padding": "16px", "textAlign": "center",
    })
