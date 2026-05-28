"""Production alerting: file-based logging + optional Slack/email delivery.

Usage
-----
    alerter = Alerter.from_config(config_dict)
    alerter.send_alert("Circuit breaker triggered", severity="critical")
    alerter.send_daily_summary(equity=100_500, daily_pnl=250, positions=5)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRecord:
    """Single alert entry."""

    timestamp: str
    severity: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "severity": self.severity,
                "message": self.message,
                "metadata": self.metadata,
            },
            default=str,
        )


@dataclass
class SlackConfig:
    enabled: bool = False
    webhook_url: str = ""


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    username: str = ""
    password: str = ""


@dataclass
class AlertingConfig:
    enabled: bool = True
    file_path: str = "reports/alerts.log"
    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AlertingConfig:
        slack = SlackConfig(**d.get("slack", {}))
        email = EmailConfig(**d.get("email", {}))
        return cls(
            enabled=d.get("enabled", True),
            file_path=d.get("file_path", "reports/alerts.log"),
            slack=slack,
            email=email,
        )


class Alerter:
    """Multi-channel alert delivery.

    Always writes to the local alert log. Optionally sends to Slack/email.
    """

    def __init__(self, config: AlertingConfig) -> None:
        self.config = config
        self._ensure_log_dir()

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> Alerter:
        return cls(config=AlertingConfig.from_dict(d))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        message: str,
        severity: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send an alert to all configured channels."""
        if not self.config.enabled:
            return

        record = AlertRecord(
            timestamp=datetime.now(UTC).isoformat(),
            severity=severity,
            message=message,
            metadata=metadata or {},
        )

        # Always write to file
        self._write_file(record)

        # Optional channels
        if self.config.slack.enabled and self.config.slack.webhook_url:
            self._send_slack(record)

        if self.config.email.enabled and self.config.email.smtp_host:
            self._send_email(record)

        log.log(
            getattr(logging, severity.upper(), logging.INFO),
            "ALERT [%s] %s",
            severity.upper(),
            message,
        )

    def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        positions: int,
        circuit_breaker_state: str = "normal",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Send end-of-day summary."""
        meta: dict[str, Any] = {
            "equity": equity,
            "daily_pnl": daily_pnl,
            "positions": positions,
            "circuit_breaker": circuit_breaker_state,
        }
        if extra:
            meta.update(extra)

        pnl_sign = "+" if daily_pnl >= 0 else ""
        msg = (
            f"Daily Summary | Equity: ${equity:,.2f} | "
            f"P&L: {pnl_sign}${daily_pnl:,.2f} | "
            f"Positions: {positions} | CB: {circuit_breaker_state}"
        )
        self.send_alert(msg, severity="info", metadata=meta)

    # ------------------------------------------------------------------
    # File delivery (always on)
    # ------------------------------------------------------------------

    def _ensure_log_dir(self) -> None:
        Path(self.config.file_path).parent.mkdir(parents=True, exist_ok=True)

    def _write_file(self, record: AlertRecord) -> None:
        try:
            with open(self.config.file_path, "a", encoding="utf-8") as f:
                f.write(record.to_json() + "\n")
        except OSError as exc:
            log.warning("Failed to write alert to %s: %s", self.config.file_path, exc)

    # ------------------------------------------------------------------
    # Slack delivery (placeholder — requires requests)
    # ------------------------------------------------------------------

    def _send_slack(self, record: AlertRecord) -> None:
        try:
            import requests

            emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
                record.severity, "📢"
            )
            payload = {
                "text": f"{emoji} *{record.severity.upper()}* — {record.message}",
            }
            resp = requests.post(
                self.config.slack.webhook_url,
                json=payload,
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning("Slack webhook returned %d", resp.status_code)
        except ImportError:
            log.debug("requests not installed — Slack alert skipped")
        except Exception as exc:
            log.warning("Slack alert failed: %s", exc)

    # ------------------------------------------------------------------
    # Email delivery (placeholder — requires smtplib)
    # ------------------------------------------------------------------

    def _send_email(self, record: AlertRecord) -> None:
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(
                f"Severity: {record.severity}\n"
                f"Time: {record.timestamp}\n"
                f"Message: {record.message}\n"
                f"Metadata: {json.dumps(record.metadata, indent=2, default=str)}",
            )
            msg["Subject"] = f"[QuantCore {record.severity.upper()}] {record.message[:80]}"
            msg["From"] = self.config.email.from_addr
            msg["To"] = ", ".join(self.config.email.to_addrs)

            with smtplib.SMTP(self.config.email.smtp_host, self.config.email.smtp_port) as server:
                server.starttls()
                if self.config.email.username:
                    server.login(self.config.email.username, self.config.email.password)
                server.send_message(msg)
        except ImportError:
            log.debug("smtplib unavailable — email alert skipped")
        except Exception as exc:
            log.warning("Email alert failed: %s", exc)
