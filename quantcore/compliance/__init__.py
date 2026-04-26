"""Compliance: regulatory reporting, position limits, best execution, audit trail."""

from quantcore.compliance.limits import (
    ComplianceStatus, ComplianceResult, ComplianceChecker,
    check_regulatory_limits, check_best_execution,
)
from quantcore.compliance.reporting import (
    RegulatoryReport, generate_regulatory_report, generate_best_execution_report,
)
from quantcore.compliance.audit_log import (
    AuditEventType, AuditEntry, AuditLog, _compute_hash, tamper_detection_report,
)
from quantcore.compliance.rules import (
    RuleType, RuleSeverity, ComplianceRule, RuleViolation,
    ComplianceRuleEngine, build_default_rules,
    detect_wash_sale, check_reg_sho,
)

__all__ = [
    "ComplianceStatus", "ComplianceResult", "ComplianceChecker",
    "check_regulatory_limits", "check_best_execution",
    "RegulatoryReport", "generate_regulatory_report", "generate_best_execution_report",
    "AuditEventType", "AuditEntry", "AuditLog", "_compute_hash", "tamper_detection_report",
    "RuleType", "RuleSeverity", "ComplianceRule", "RuleViolation",
    "ComplianceRuleEngine", "build_default_rules",
    "detect_wash_sale", "check_reg_sho",
]
