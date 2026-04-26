"""Alert rules engine: configurable alerts with severity levels and notification routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    DASHBOARD = "dashboard"
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"


@dataclass
class AlertRule:
    """A single alert rule with condition and action."""
    rule_id: str = ""
    name: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.DASHBOARD])
    condition: Callable[[dict[str, Any]], bool] = lambda _: False
    cooldown_seconds: float = 300.0
    enabled: bool = True
    message_template: str = "{name} triggered: {details}"


@dataclass
class Alert:
    """A triggered alert."""
    alert_id: str = ""
    rule: AlertRule | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: datetime | None = None
    suppressed: bool = False


class AlertEngine:
    """Evaluate alert rules against system state and manage alert lifecycle."""

    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._alerts: list[Alert] = []
        self._last_fired: dict[str, datetime] = {}
        self._suppression_list: set[str] = set()
        self._handlers: dict[AlertChannel, list[Callable[[Alert], None]]] = {
            ch: [] for ch in AlertChannel
        }

    def add_rule(self, rule: AlertRule) -> None:
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        self._rules.pop(rule_id, None)

    def evaluate(self, state: dict[str, Any]) -> list[Alert]:
        """Evaluate all enabled rules against the current system state."""
        now = datetime.now(timezone.utc)
        fired: list[Alert] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.rule_id in self._suppression_list:
                continue

            last = self._last_fired.get(rule.rule_id)
            if last and (now - last).total_seconds() < rule.cooldown_seconds:
                continue

            try:
                if rule.condition(state):
                    alert = self._fire(rule, state)
                    fired.append(alert)
            except Exception:
                pass

        return fired

    def _fire(self, rule: AlertRule, state: dict[str, Any]) -> Alert:
        now = datetime.now(timezone.utc)
        alert_id = f"alert_{now.timestamp()}_{rule.rule_id}"
        message = rule.message_template.format(
            name=rule.name, details=str(state.get("details", "")),
        )
        alert = Alert(
            alert_id=alert_id, rule=rule, timestamp=now,
            severity=rule.severity, message=message, details=dict(state),
        )
        self._alerts.append(alert)
        self._last_fired[rule.rule_id] = now

        for channel in rule.channels:
            for handler in self._handlers.get(channel, []):
                try:
                    handler(alert)
                except Exception:
                    pass

        return alert

    def acknowledge(self, alert_id: str, user: str = "system") -> bool:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now(timezone.utc)
                return True
        return False

    def suppress_rule(self, rule_id: str, duration_seconds: float = 3600.0) -> None:
        self._suppression_list.add(rule_id)

    def unsuppress_rule(self, rule_id: str) -> None:
        self._suppression_list.discard(rule_id)

    def register_handler(self, channel: AlertChannel, handler: Callable[[Alert], None]) -> None:
        self._handlers.setdefault(channel, []).append(handler)

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        active = [a for a in self._alerts if not a.acknowledged and not a.suppressed]
        if severity:
            active = [a for a in active if a.severity == severity]
        return sorted(active, key=lambda a: a.timestamp, reverse=True)

    def get_recent_alerts(self, hours: float = 24.0) -> list[Alert]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self._alerts if a.timestamp >= cutoff]

    def clear_acknowledged(self) -> int:
        before = len(self._alerts)
        self._alerts = [a for a in self._alerts if not a.acknowledged]
        return before - len(self._alerts)

    @property
    def active_count(self) -> int:
        return len(self.get_active_alerts())

    @property
    def critical_count(self) -> int:
        return len(self.get_active_alerts(AlertSeverity.CRITICAL))


def build_default_alert_rules() -> list[AlertRule]:
    """Create a standard set of alert rules for the trading system."""
    return [
        AlertRule(
            rule_id="drawdown_10pct", name="Drawdown > 10%",
            description="Portfolio drawdown exceeds 10% from peak",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL],
            condition=lambda s: float(s.get("drawdown_pct", 0)) < -10.0,
        ),
        AlertRule(
            rule_id="var_breach", name="VaR Breach",
            description="Daily loss exceeds 95% VaR estimate",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.DASHBOARD],
            condition=lambda s: bool(s.get("var_breach", False)),
        ),
        AlertRule(
            rule_id="circuit_breaker", name="Circuit Breaker Active",
            description="Circuit breaker has been triggered",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL, AlertChannel.SLACK],
            condition=lambda s: bool(s.get("circuit_breaker_active", False)),
            cooldown_seconds=60.0,
        ),
        AlertRule(
            rule_id="concentration_25pct", name="Concentration > 25%",
            description="Single position exceeds 25% of portfolio",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.DASHBOARD],
            condition=lambda s: float(s.get("max_concentration", 0)) > 0.25,
        ),
        AlertRule(
            rule_id="correlation_break", name="Correlation Break",
            description="Significant change in correlation structure detected",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.DASHBOARD],
            condition=lambda s: bool(s.get("correlation_break", False)),
        ),
        AlertRule(
            rule_id="daily_loss_limit", name="Daily Loss Limit Exceeded",
            description="Daily P&L loss exceeds configured limit",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.DASHBOARD, AlertChannel.EMAIL],
            condition=lambda s: float(s.get("daily_loss_pct", 0)) > 3.0,
        ),
        AlertRule(
            rule_id="data_feed_stale", name="Data Feed Stale",
            description="Market data feed has not updated",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.DASHBOARD, AlertChannel.SLACK],
            condition=lambda s: float(s.get("data_stale_seconds", 0)) > 30.0,
            cooldown_seconds=60.0,
        ),
    ]


def slack_notifier(webhook_url: str) -> Callable[[Alert], None]:
    """Create a Slack notification handler for alerts."""
    import json
    import requests

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    def notify(alert: Alert) -> None:
        color = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}.get(
            alert.severity.value, "#cccccc",
        )
        payload = {
            "attachments": [{
                "color": color,
                "title": f"[{alert.severity.value.upper()}] {alert.message}",
                "text": json.dumps(alert.details, default=str),
                "ts": int(alert.timestamp.timestamp()),
            }]
        }
        try:
            session.post(webhook_url, json=payload, timeout=5)
        except Exception:
            pass
    return notify
