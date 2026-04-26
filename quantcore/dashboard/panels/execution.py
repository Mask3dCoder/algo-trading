"""Execution dashboard panel: VWAP performance, slippage histogram, fill quality, TCA."""

from __future__ import annotations

from typing import Any


def build_execution_tab() -> Any:
    """Build the execution quality monitoring tab layout."""
    try:
        from dash import dcc, html
    except ImportError:
        return None

    return html.Div([
        html.H2("Execution Monitor", style={"color": "#e0e0e0", "marginBottom": "16px"}),

        # Key execution metrics
        html.Div([
            _exec_metric("Avg Slippage", "--- bps", "avg-slippage"),
            _exec_metric("VWAP Performance", "--- bps", "vwap-perf"),
            _exec_metric("Fill Rate", "---%", "fill-rate"),
            _exec_metric("Implementation Shortfall", "--- bps", "impl-shortfall"),
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "16px", "marginBottom": "24px",
        }),

        # Slippage histogram and VWAP comparison
        html.Div([
            html.Div([
                html.H3("Slippage Distribution", style={"color": "#ccc"}),
                dcc.Graph(id="slippage-histogram", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
            html.Div([
                html.H3("VWAP vs Arrival Price", style={"color": "#ccc"}),
                dcc.Graph(id="vwap-comparison", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # TCA report section
        html.Div([
            html.H3("Transaction Cost Analysis", style={"color": "#ccc", "marginBottom": "8px"}),
            html.Table(id="tca-table", style={"width": "100%", "color": "#e0e0e0", "borderCollapse": "collapse"}),
        ], style={
            "backgroundColor": "#16213e", "borderRadius": "8px",
            "padding": "16px",
        }),
    ])


def _exec_metric(label: str, value: str, element_id: str) -> Any:
    try:
        from dash import html
    except ImportError:
        return None
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": "#888", "textTransform": "uppercase"}),
        html.Div(value, id=element_id, style={"fontSize": "22px", "fontWeight": "bold", "color": "#4ecdc4"}),
    ], style={
        "backgroundColor": "#16213e", "borderRadius": "8px",
        "padding": "16px", "textAlign": "center",
    })
