"""Monitoring dashboard: portfolio, strategy, risk, market, execution panels."""

from quantcore.dashboard.alerts import (
    Alert,
    AlertChannel,
    AlertEngine,
    AlertRule,
    AlertSeverity,
    build_default_alert_rules,
    slack_notifier,
)
from quantcore.dashboard.app import create_app, run_dashboard
from quantcore.dashboard.panels.execution import build_execution_tab
from quantcore.dashboard.panels.market import build_market_tab
from quantcore.dashboard.panels.portfolio import build_portfolio_tab
from quantcore.dashboard.panels.risk import build_risk_tab
from quantcore.dashboard.panels.strategy import build_strategy_tab

__all__ = [
    "Alert",
    "AlertChannel",
    "AlertEngine",
    "AlertRule",
    "AlertSeverity",
    "build_default_alert_rules",
    "build_execution_tab",
    "build_market_tab",
    "build_portfolio_tab",
    "build_risk_tab",
    "build_strategy_tab",
    "create_app",
    "run_dashboard",
    "slack_notifier",
]
