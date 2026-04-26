"""Dash application skeleton: multi-tab monitoring dashboard with dark theme."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def create_app(title: str = "QuantCore Dashboard") -> Any:
    """Create the Dash application with dark theme and multi-tab layout."""
    try:
        import dash
        from dash import dcc, html
    except ImportError:
        return None

    app = dash.Dash(__name__, title=title, update_title=None)
    app._favicon = None

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(title, style={"margin": "0", "fontSize": "24px", "fontWeight": "600"}),
            html.Div([
                html.Span("Live", id="status-indicator", style={
                    "display": "inline-block", "width": "10px", "height": "10px",
                    "borderRadius": "50%", "backgroundColor": "#39ff14",
                    "marginRight": "8px",
                }),
                html.Span("System Healthy", id="status-text", style={"fontSize": "14px"}),
            ]),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "16px 24px",
            "backgroundColor": "#1a1a2e", "color": "#e0e0e0",
            "borderBottom": "1px solid #333",
        }),

        # Tab navigation
        dcc.Tabs(id="tabs", value="portfolio", children=[
            dcc.Tab(label="Portfolio", value="portfolio", style={"backgroundColor": "#16213e", "color": "#aaa", "border": "none"}, selected_style={"backgroundColor": "#0f3460", "color": "#fff", "border": "none"}),
            dcc.Tab(label="Strategies", value="strategy", style={"backgroundColor": "#16213e", "color": "#aaa", "border": "none"}, selected_style={"backgroundColor": "#0f3460", "color": "#fff", "border": "none"}),
            dcc.Tab(label="Risk", value="risk", style={"backgroundColor": "#16213e", "color": "#aaa", "border": "none"}, selected_style={"backgroundColor": "#0f3460", "color": "#fff", "border": "none"}),
            dcc.Tab(label="Market", value="market", style={"backgroundColor": "#16213e", "color": "#aaa", "border": "none"}, selected_style={"backgroundColor": "#0f3460", "color": "#fff", "border": "none"}),
            dcc.Tab(label="Execution", value="execution", style={"backgroundColor": "#16213e", "color": "#aaa", "border": "none"}, selected_style={"backgroundColor": "#0f3460", "color": "#fff", "border": "none"}),
            dcc.Tab(label="Alerts", value="alerts", style={"backgroundColor": "#16213e", "color": "#aaa", "border": "none"}, selected_style={"backgroundColor": "#0f3460", "color": "#fff", "border": "none"}),
        ]),

        # Tab content
        html.Div(id="tab-content", style={"padding": "24px", "backgroundColor": "#1a1a2e", "minHeight": "85vh"}),

        # Auto-refresh interval
        dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
    ], style={"fontFamily": "monospace", "backgroundColor": "#1a1a2e", "color": "#e0e0e0", "minHeight": "100vh"})

    return app


def run_dashboard(app: Any, host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    """Launch the dashboard server."""
    if app is not None:
        app.run(host=host, port=port, debug=debug)
