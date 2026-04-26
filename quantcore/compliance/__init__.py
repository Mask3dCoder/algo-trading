"""Compliance: regulatory reporting, position limits, best execution, audit trail."""

from quantcore.compliance.audit_log import (
    AuditEntry,
    AuditEventType,
    AuditLog,
    _compute_hash,
    tamper_detection_report,
)
from quantcore.compliance.limits import (
    ComplianceChecker,
    ComplianceResult,
    ComplianceStatus,
    check_best_execution,
    check_regulatory_limits,
)
from quantcore.compliance.reporting import (
    RegulatoryReport,
    generate_best_execution_report,
    generate_regulatory_report,
)
from quantcore.compliance.rules import (
    ComplianceRule,
    ComplianceRuleEngine,
    RuleSeverity,
    RuleType,
    RuleViolation,
    build_default_rules,
    check_reg_sho,
    detect_wash_sale,
)

__all__ = [
    "AuditEntry",
    "AuditEventType",
    "AuditLog",
    "ComplianceChecker",
    "ComplianceResult",
    "ComplianceRule",
    "ComplianceRuleEngine",
    "ComplianceStatus",
    "RegulatoryReport",
    "RuleSeverity",
    "RuleType",
    "RuleViolation",
    "_compute_hash",
    "build_default_rules",
    "check_best_execution",
    "check_reg_sho",
    "check_regulatory_limits",
    "detect_wash_sale",
    "generate_best_execution_report",
    "generate_regulatory_report",
    "tamper_detection_report",
]
