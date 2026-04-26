"""Monitoring dashboard: portfolio, strategy, risk, market, execution panels."""

from quantcore.dashboard.app import create_app, run_dashboard
from quantcore.dashboard.alerts import (
    AlertSeverity, AlertChannel, AlertRule, Alert, AlertEngine,
    build_default_alert_rules, slack_notifier,
)
from quantcore.dashboard.panels.portfolio import build_portfolio_tab
from quantcore.dashboard.panels.strategy import build_strategy_tab
from quantcore.dashboard.panels.risk import build_risk_tab
from quantcore.dashboard.panels.market import build_market_tab
from quantcore.dashboard.panels.execution import build_execution_tab

__all__ = [
    "create_app", "run_dashboard",
    "AlertSeverity", "AlertChannel", "AlertRule", "Alert", "AlertEngine",
    "build_default_alert_rules", "slack_notifier",
    "build_portfolio_tab", "build_strategy_tab", "build_risk_tab",
    "build_market_tab", "build_execution_tab",
]
