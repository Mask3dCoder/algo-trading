"""Strategy dashboard panel: strategy P&L, active signals, recent trades."""

from __future__ import annotations

from typing import Any


def build_strategy_tab() -> Any:
    """Build the strategy monitoring tab layout."""
    try:
        from dash import dcc, html
    except ImportError:
        return None

    return html.Div([
        html.H2("Strategy Monitor", style={"color": "#e0e0e0", "marginBottom": "16px"}),

        # Strategy P&L cards
        html.Div(id="strategy-cards", style={
            "display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "16px", "marginBottom": "24px",
        }),

        # Active signals
        html.Div([
            html.H3("Active Signals", style={"color": "#ccc", "marginBottom": "8px"}),
            html.Table(id="signals-table", style={"width": "100%", "color": "#e0e0e0", "borderCollapse": "collapse"}),
        ], style={
            "backgroundColor": "#16213e", "borderRadius": "8px",
            "padding": "16px", "marginBottom": "16px",
        }),

        # Recent trades and performance
        html.Div([
            html.Div([
                html.H3("Recent Trades", style={"color": "#ccc"}),
                html.Table(id="trades-table", style={"width": "100%", "color": "#e0e0e0", "borderCollapse": "collapse"}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
            html.Div([
                html.H3("Strategy Performance", style={"color": "#ccc"}),
                dcc.Graph(id="strategy-perf-chart", config={"displayModeBar": False}),
            ], style={
                "backgroundColor": "#16213e", "borderRadius": "8px",
                "padding": "16px", "flex": "1",
            }),
        ], style={"display": "flex", "gap": "16px"}),
    ])
