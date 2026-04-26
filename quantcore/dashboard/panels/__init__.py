"""Dashboard panel builders for each monitoring tab."""

from quantcore.dashboard.panels.portfolio import build_portfolio_tab
from quantcore.dashboard.panels.strategy import build_strategy_tab
from quantcore.dashboard.panels.risk import build_risk_tab
from quantcore.dashboard.panels.market import build_market_tab
from quantcore.dashboard.panels.execution import build_execution_tab

__all__ = [
    "build_portfolio_tab", "build_strategy_tab", "build_risk_tab",
    "build_market_tab", "build_execution_tab",
]
