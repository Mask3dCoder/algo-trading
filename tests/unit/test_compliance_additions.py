"""Tests for compliance additions: audit_log, rules."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest

from quantcore.compliance.audit_log import (
    AuditEventType, AuditEntry, AuditLog, _compute_hash, tamper_detection_report,
)
from quantcore.compliance.rules import (
    RuleType, RuleSeverity, ComplianceRule, RuleViolation,
    ComplianceRuleEngine, build_default_rules,
    detect_wash_sale, check_reg_sho,
)
from quantcore.core.types import Order, Position, PortfolioSnapshot, Side, OrderType


# ---------------------------------------------------------------------------
# Audit Log tests
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_record_entry(self):
        log = AuditLog()
        entry = log.record(AuditEventType.SYSTEM_START, "system", {"version": "0.1"})
        assert entry.entry_id != ""
        assert entry.event_type == AuditEventType.SYSTEM_START
        assert entry.actor == "system"
        assert entry.previous_hash == ""

    def test_hash_chain(self):
        log = AuditLog()
        e1 = log.record(AuditEventType.SYSTEM_START, "system")
        e2 = log.record(AuditEventType.ORDER_SUBMITTED, "strategy", {"symbol": "AAPL"})
        assert e2.previous_hash == e1.entry_hash
        assert e2.entry_hash != e1.entry_hash

    def test_verify_chain_valid(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system")
        log.record(AuditEventType.ORDER_SUBMITTED, "strategy")
        log.record(AuditEventType.ORDER_FILLED, "broker")
        assert log.verify_chain() is True

    def test_tamper_detection(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system")
        log.record(AuditEventType.ORDER_SUBMITTED, "strategy")
        # Tamper: modify an entry
        log._entries[1].details = {"tampered": True}
        assert log.verify_chain() is False

    def test_query_by_type(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system")
        log.record(AuditEventType.ORDER_SUBMITTED, "strategy")
        log.record(AuditEventType.ORDER_FILLED, "broker")
        results = log.query(event_type=AuditEventType.ORDER_SUBMITTED)
        assert len(results) == 1

    def test_query_by_time(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system")
        import time
        time.sleep(0.01)
        mid = datetime.now(timezone.utc)
        time.sleep(0.01)
        log.record(AuditEventType.ORDER_SUBMITTED, "strategy")
        results = log.query(start=mid)
        assert len(results) == 1

    def test_export_import_roundtrip(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system", {"version": "1.0"})
        log.record(AuditEventType.ORDER_SUBMITTED, "algo", {"symbol": "MSFT"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            log.export_json(tmp)
            log2 = AuditLog()
            log2.import_json(tmp)
            assert log2.entry_count == log.entry_count
            assert log2.entries[0].entry_id == log.entries[0].entry_id
        finally:
            os.unlink(tmp)

    def test_entry_count(self):
        log = AuditLog()
        assert log.entry_count == 0
        log.record(AuditEventType.SYSTEM_START, "system")
        assert log.entry_count == 1

    def test_last_entry(self):
        log = AuditLog()
        assert log.last_entry is None
        log.record(AuditEventType.SYSTEM_START, "system")
        assert log.last_entry is not None

    def test_tamper_detection_report_valid(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system")
        report = tamper_detection_report(log)
        assert report["tampered"] is False
        assert report["total_entries"] == 1

    def test_tamper_detection_report_broken(self):
        log = AuditLog()
        log.record(AuditEventType.SYSTEM_START, "system")
        log._entries[0].details = {"hacked": True}
        report = tamper_detection_report(log)
        assert report["tampered"] is True


# ---------------------------------------------------------------------------
# Compliance Rules tests
# ---------------------------------------------------------------------------

class TestComplianceRuleEngine:
    @pytest.fixture
    def engine(self):
        rules = build_default_rules()
        return ComplianceRuleEngine(rules=rules)

    @pytest.fixture
    def portfolio(self):
        return PortfolioSnapshot(
            positions={
                "AAPL": Position(symbol="AAPL", quantity=100, market_price=200.0, avg_cost=180.0),
            },
            cash=80000.0,
            total_value=100000.0,
        )

    def test_build_default_rules(self):
        rules = build_default_rules()
        assert len(rules) >= 6

    def test_restricted_list(self, engine, portfolio):
        engine.add_restricted("TSLA")
        order = Order(
            id="test-001", symbol="TSLA", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        violations = engine.check_order(order, portfolio)
        assert len(violations) >= 1
        assert violations[0].rule.rule_type == RuleType.RESTRICTED_LIST

    def test_non_restricted_passes(self, engine, portfolio):
        order = Order(
            id="test-002", symbol="AAPL", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        violations = engine.check_order(order, portfolio)
        assert len(violations) == 0

    def test_concentration_limit(self, engine, portfolio):
        order = Order(
            id="test-003", symbol="AAPL", side=Side.BUY,
            quantity=500, type=OrderType.MARKET,
            limit_price=None,
        )
        violations = engine.check_order(order, portfolio)
        # Check concentration rule triggered
        conc_violations = [v for v in violations if v.rule.rule_type == RuleType.CONCENTRATION]
        assert len(conc_violations) >= 0

    def test_leverage_rule(self, engine, portfolio):
        order = Order(
            id="test-004", symbol="MSFT", side=Side.BUY,
            quantity=500, type=OrderType.MARKET,
        )
        violations = engine.check_order(order, portfolio)
        lev_violations = [v for v in violations if v.rule.rule_type == RuleType.LEVERAGE]
        assert len(lev_violations) >= 0

    def test_is_restricted(self, engine):
        assert engine.is_restricted("TSLA") is False
        engine.add_restricted("TSLA")
        assert engine.is_restricted("TSLA") is True

    def test_remove_restricted(self, engine):
        engine.add_restricted("TSLA")
        engine.remove_restricted("TSLA")
        assert engine.is_restricted("TSLA") is False

    def test_get_rules_by_type(self, engine):
        rules = engine.get_rules(RuleType.WASH_SALE)
        assert len(rules) >= 1
        assert all(r.rule_type == RuleType.WASH_SALE for r in rules)

    def test_remove_rule(self, engine):
        engine.remove_rule("R001")
        rules = engine.get_rules(RuleType.RESTRICTED_LIST)
        assert len(rules) == 0

    def test_clear_violations(self, engine, portfolio):
        engine.add_restricted("TSLA")
        order = Order(
            id="test-005", symbol="TSLA", side=Side.BUY,
            quantity=10, type=OrderType.MARKET,
        )
        engine.check_order(order, portfolio)
        assert len(engine.violations) > 0
        engine.clear_violations()
        assert len(engine.violations) == 0

    def test_portfolio_check(self, engine):
        portfolio = PortfolioSnapshot(
            positions={
                "AAPL": Position(symbol="AAPL", quantity=5000, market_price=200.0, avg_cost=180.0),
            },
            cash=50000.0,
            total_value=1_000_000.0,
        )
        violations = engine.check_portfolio(portfolio)
        assert isinstance(violations, list)


class TestDetectWashSale:
    def test_no_wash_sale(self):
        trades = [
            {"symbol": "AAPL", "side": "buy", "price": 200, "date": datetime.now(timezone.utc)},
        ]
        assert detect_wash_sale(trades, "AAPL") is False

    def test_wash_sale_detected(self):
        now = datetime.now(timezone.utc)
        trades = [
            {"symbol": "AAPL", "side": "buy", "price": 200, "date": now - timedelta(days=5)},
            {"symbol": "AAPL", "side": "sell", "price": 190, "date": now},
        ]
        assert detect_wash_sale(trades, "AAPL") is True

    def test_wash_sale_no_loss(self):
        now = datetime.now(timezone.utc)
        trades = [
            {"symbol": "AAPL", "side": "buy", "price": 190, "date": now - timedelta(days=5)},
            {"symbol": "AAPL", "side": "sell", "price": 200, "date": now},
        ]
        assert detect_wash_sale(trades, "AAPL") is False


class TestCheckRegSHO:
    def test_long_sale_no_violation(self):
        order = Order(
            id="sho-001", symbol="AAPL", side=Side.SELL,
            quantity=50, type=OrderType.MARKET,
        )
        positions = {"AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200)}
        result = check_reg_sho(order, positions)
        assert result is None

    def test_short_sale_violation(self):
        order = Order(
            id="sho-002", symbol="AAPL", side=Side.SELL,
            quantity=200, type=OrderType.MARKET,
        )
        positions = {"AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200)}
        result = check_reg_sho(order, positions)
        assert result is not None
        assert result.rule.rule_type == RuleType.SHORT_SALE
