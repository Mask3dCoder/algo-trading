"""Market dashboard panel: regime bars, vol surface, term structure, sector heatmap."""

from __future__ import annotations

from typing import Any


def build_market_tab() -> Any:
    """Build the market data and regime tab layout."""
    try:
        from dash import dcc, html
    except ImportError:
        return None

    return html.Div([
        html.H2("Market Monitor", style={"color": "#e0e0e0", "marginBottom": "16px"}),

        # Regime indicators
        html.Div([
            html.Div(id="regime-indicator", children=[
                html.Div("Regime: Unknown", id="regime-label", style={"fontSize": "18px", "fontWeight": "bold"}),
                html.Div("Confidence: ---", id="regime-confidence", style={"fontSize": "14px", "color": "#888"}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "20px", "textAlign": "center",
            }),
        ], style={"marginBottom": "16px"}),

        # Vol surface and term structure
        html.Div([
            html.Div([
                html.H3("Volatility Surface", style={"color": "#ccc"}),
                dcc.Graph(id="vol-surface-3d", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "2",
            }),
            html.Div([
                html.H3("Term Structure", style={"color": "#ccc"}),
                dcc.Graph(id="term-structure-chart", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # Sector heatmap
        html.Div([
            html.H3("Sector Heatmap", style={"color": "#ccc", "marginBottom": "8px"}),
            dcc.Graph(id="sector-heatmap", config={"displayModeBar": False}),
        ], style={
            "backgroundColor": "#16213e", "borderRadius": "8px",
            "padding": "16px",
        }),
    ])
