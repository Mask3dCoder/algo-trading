"""Strategy monitoring: signal tracking, regime detection, performance drift alerts."""

from quantcore.monitoring.alerting import (
    Alerter,
    AlertingConfig,
    AlertRecord,
    AlertSeverity,
    EmailConfig,
    SlackConfig,
)

__all__ = [
    "Alerter",
    "AlertingConfig",
    "AlertRecord",
    "AlertSeverity",
    "EmailConfig",
    "SlackConfig",
]
