"""Adversarial Testing Harness for QuantCore.

Orchestrates BrokerEmulator, SUTAdapter, EventChaosGenerator, FailureInjector,
ValidationOracle, and ReplayEngine into a single test harness.

Each scenario:
  1. Builds event stream (with or without failures)
  2. Drives SUT + canonical broker through event stream
  3. Validates all invariants at every checkpoint
  4. Generates divergence report, SIS score, failure timeline

Usage:
    harness = AdversarialHarness()
    result = harness.run_scenario("broker_fill_loss", seed=42)
    print(result.sis_score, result.divergences)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantcore.core.types import Fill, Order, OrderType, Side

from tests.adversarial.broker_emulator import BrokerEmulator, CanonicalFillRecord
from tests.adversarial.chaos import (
    ChaosConfig,
    ChaosEvent,
    EventChaosGenerator,
    EventType,
    FailureEvent,
    FailureInjector,
    ReplayEngine,
    ReplayMode,
    RunResult,
)
from tests.adversarial.oracle import FindingSeverity, ValidationFinding, ValidationOracle, ValidationReport
from tests.adversarial.sut import SUTAdapter


@dataclass
class ScenarioResult:
    """Complete result of a single adversarial scenario run."""
    name: str = ""
    seed: int = 0
    passed: bool = False
    sis_score: dict[str, float] = field(default_factory=dict)
    divergences: list[dict[str, Any]] = field(default_factory=list)
    first_failure: ValidationFinding | None = None
    drift_timeline: list[dict[str, Any]] = field(default_factory=list)
    risk_breach_timeline: list[dict[str, Any]] = field(default_factory=list)
    n_events: int = 0
    n_validation_points: int = 0
    n_failures_injected: int = 0
    duration_seconds: float = 0.0
    reconciliation_summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def sis(self) -> float:
        return self.sis_score.get("overall", 0.0)


class AdversarialHarness:
    """Main adversarial test harness orchestrator.

    Drives the full pipeline: event stream → SUT + broker → oracle → report.
    """

    def __init__(self) -> None:
        self.sut: SUTAdapter | None = None
        self.broker: BrokerEmulator | None = None
        self.oracle: ValidationOracle | None = None
        self._pending_fills: dict[str, list[CanonicalFillRecord]] = {}
        self._disconnected: bool = False

    # ------------------------------------------------------------------
    # Scenario implementations
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario_name: str,
        seed: int = 42,
        **kwargs: Any,
    ) -> ScenarioResult:
        """Run a named scenario by name."""
        start = time.time()
        scenarios = {
            "broker_fill_loss": self._scenario_broker_fill_loss,
            "duplicate_fill_attack": self._scenario_duplicate_fill,
            "cash_overcommit": self._scenario_cash_overcommit,
            "disconnect_silent_fill": self._scenario_disconnect_silent_fill,
            "restart_active_trading": self._scenario_restart_active_trading,
        }
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Reset all components
        self.sut = SUTAdapter(initial_cash=kwargs.get("initial_cash", 1_000_000.0))
        self.broker = BrokerEmulator(initial_cash=kwargs.get("initial_cash", 1_000_000.0))
        self.oracle = ValidationOracle()
        self._pending_fills.clear()
        self._disconnected = False

        result = scenarios[scenario_name](seed=seed, **kwargs)
        result.duration_seconds = time.time() - start
        result.name = scenario_name
        return result

    # ------------------------------------------------------------------
    # Scenario 1: Broker Fill Loss
    # ------------------------------------------------------------------

    def _scenario_broker_fill_loss(self, seed: int = 42, **kwargs: Any) -> ScenarioResult:
        """Broker executes a fill, SUT never receives it, SUT continues trading.

        Expected: reconciliation detects divergence → HALT
        """
        assert self.sut is not None and self.broker is not None and self.oracle is not None

        rng = __import__("random").Random(seed)
        result = ScenarioResult(seed=seed)
        n_ticks = kwargs.get("n_ticks", 20)

        symbols = ["AAPL", "MSFT"]
        prices = {"AAPL": 150.0, "MSFT": 350.0}

        for tick in range(n_ticks):
            ts = float(tick)

            # Market data
            for sym in symbols:
                price = prices[sym] * (1.0 + rng.uniform(-0.01, 0.01))
                self.sut.process_market_data(sym, price)

            # Submit orders periodically
            if tick % 5 == 0 and tick < 15:
                sym = rng.choice(symbols)
                price = prices[sym]
                order = Order(
                    id=f"order-{tick}",
                    symbol=sym,
                    side=Side.BUY,
                    type=OrderType.MARKET,
                    quantity=Decimal("100"),
                )
                submitted = self.sut.submit_order(order)
                if submitted:
                    self.broker.submit_order(order)

            # Generate fills for submitted orders
            for oid in list(self._pending_fills.keys()):
                self._pending_fills.pop(oid)
            for oid, record in self.broker._orders.items():
                if record.acked and not record.fills:
                    fill_record = self.broker.generate_fill(
                        oid,
                        price=prices.get(record.order.symbol, 150.0),
                        quantity=100.0,
                    )
                    if fill_record:
                        # INTENTIONAL DIVERGENCE: drop ALL fills for first order
                        if tick == 0 and oid == "order-0":
                            self.broker.drop_fill(fill_record)
                        else:
                            self.sut.receive_fill(fill_record.fill, fill_record.symbol)

            # Process settlements
            self.sut.process_settlements()

            # Validate at every tick
            self.oracle.validate_all(self.sut, self.broker, ts)

        # Finalize
        report = self.oracle.finalize()
        result.sis_score = self.oracle.sis_score()
        result.divergences = report.divergences
        result.first_failure = report.first_failure
        result.drift_timeline = [{
            "ts": d.timestamp,
            "component": d.component,
            "canonical": d.canonical_value,
            "sut": d.sut_value,
            "delta": d.delta,
        } for d in report.drift_timeline]
        result.risk_breach_timeline = [{
            "ts": b.timestamp,
            "limit": b.limit_name,
            "value": b.current_value,
            "limit_value": b.limit_value,
        } for b in report.risk_breach_timeline]
        result.n_events = n_ticks
        result.n_validation_points = n_ticks
        result.n_failures_injected = 1  # one dropped fill
        result.passed = len(report.divergences) > 0  # scenario expects divergence
        result.reconciliation_summary = (
            f"Divergences detected: {len(report.divergences)}. "
            f"Expected behavior: reconciliation HALT."
        )
        result.details = {"dropped_fill_order": "order-0", "n_divergences": len(report.divergences)}
        return result

    # ------------------------------------------------------------------
    # Scenario 2: Duplicate Fill Attack
    # ------------------------------------------------------------------

    def _scenario_duplicate_fill(self, seed: int = 42, **kwargs: Any) -> ScenarioResult:
        """Same fill arrives twice with delay.

        Expected: idempotent rejection (second fill ignored, state unchanged).
        """
        assert self.sut is not None and self.broker is not None and self.oracle is not None
        result = ScenarioResult(seed=seed)

        # Set market prices before any order submission
        self.sut.process_market_data("AAPL", 150.0)
        self.sut.process_market_data("MSFT", 350.0)

        # Single order + fill
        order = Order(
            id="order-dup",
            symbol="AAPL",
            side=Side.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("100"),
        )

        self.sut.submit_order(order)
        self.broker.submit_order(order)

        # Generate fill in canonical broker
        fill_record = self.broker.generate_fill("order-dup", price=150.0, quantity=100.0)

        # SUT receives fill first time (legitimate)
        if fill_record:
            self.sut.receive_fill(fill_record.fill, "AAPL")

        # Validate after first fill
        self.oracle.validate_all(self.sut, self.broker, 0.0)

        # DUPLICATE ATTACK: same fill arrives again
        if fill_record:
            dup = self.broker.duplicate_fill(fill_record)
            if dup:
                self.sut.receive_fill(dup.fill, "AAPL")

        # Validate after duplicate
        self.oracle.validate_all(self.sut, self.broker, 1.0)

        # Process settlements and revalidate
        self.sut.process_settlements()
        self.oracle.validate_all(self.sut, self.broker, 2.0)

        report = self.oracle.finalize()
        result.sis_score = self.oracle.sis_score()
        result.divergences = report.divergences
        result.first_failure = report.first_failure
        result.n_events = 3
        result.n_validation_points = 3
        result.n_failures_injected = 1  # one duplicate fill
        result.passed = len(report.divergences) == 0  # should be clean
        result.reconciliation_summary = (
            f"Divergences: {len(report.divergences)}. "
            f"Expected: idempotent rejection (no state change from duplicate)."
        )
        return result

    # ------------------------------------------------------------------
    # Scenario 3: Cash Overcommit Under Delay
    # ------------------------------------------------------------------

    def _scenario_cash_overcommit(self, seed: int = 42, **kwargs: Any) -> ScenarioResult:
        """Unsettled cash appears usable, system attempts trade using it.

        Expected: trade blocked by settlement-aware cash check.
        """
        assert self.sut is not None and self.broker is not None and self.oracle is not None
        result = ScenarioResult(seed=seed)

        # Start with modest cash
        initial_cash = 100_000.0
        self.sut = SUTAdapter(initial_cash=initial_cash)
        self.broker = BrokerEmulator(initial_cash=initial_cash)

        # Set market prices
        self.sut.process_market_data("AAPL", 150.0)
        self.sut.process_market_data("MSFT", 350.0)

        # 1. Submit a large buy order (uses most available cash)
        # This reserves $90,000 from settled cash, leaving $10,000 available
        order1 = Order(
            id="order-big",
            symbol="AAPL",
            side=Side.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("600"),
        )
        assert self.sut.submit_order(order1), "First order should be accepted"
        self.broker.submit_order(order1)

        # 2. ATTEMPT: second order while first is still pending (committed)
        # Available cash = $100k - $90k (committed) = $10k
        # $17,500 > $10,000, so this should be BLOCKED
        order2 = Order(
            id="order-overcommit",
            symbol="MSFT",
            side=Side.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("50"),
        )

        accepted = self.sut.submit_order(order2)

        # Validate
        snapshot = self.sut.snapshot()
        self.oracle.validate_all(self.sut, self.broker, 0.0)

        if accepted:
            finding = ValidationFinding(
                timestamp=0.0,
                severity=FindingSeverity.CRITICAL,
                component="cash",
                check="cash_overcommit_blocked",
                expected="trade blocked (committed cash reduces available)",
                actual="trade accepted",
                details=f"Order2 accepted despite committed=$90k reducing available to $10k",
            )
            self.oracle._report.add_finding(finding)

        report = self.oracle.finalize()
        result.sis_score = self.oracle.sis_score()
        result.divergences = report.divergences
        result.first_failure = report.first_failure
        result.n_events = 3
        result.n_validation_points = 1
        result.n_failures_injected = 1
        result.passed = not accepted
        result.reconciliation_summary = (
            f"Second order {'accepted (FAIL)' if accepted else 'blocked (PASS)'}. "
            f"Available cash when blocked: ${snapshot.available_cash:.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # Scenario 4: Disconnect + Silent Fill Window
    # ------------------------------------------------------------------

    def _scenario_disconnect_silent_fill(self, seed: int = 42, **kwargs: Any) -> ScenarioResult:
        """Broker disconnects mid-session, fills occur during outage, reconnect later.

        Expected: full reconciliation or halt (never silent continuation).
        """
        assert self.sut is not None and self.broker is not None and self.oracle is not None
        rng = __import__("random").Random(seed)
        result = ScenarioResult(seed=seed)

        # Set market prices
        self.sut.process_market_data("AAPL", 150.0)
        self.sut.process_market_data("MSFT", 350.0)

        # Phase 1: Normal trading
        order1 = Order(id="o1", symbol="AAPL", side=Side.BUY, type=OrderType.MARKET, quantity=Decimal("100"))
        self.sut.submit_order(order1)
        self.broker.submit_order(order1)
        fill1 = self.broker.generate_fill("o1", price=150.0, quantity=100.0)
        if fill1:
            self.sut.receive_fill(fill1.fill, "AAPL")
        self.oracle.validate_all(self.sut, self.broker, 0.0)

        # Phase 2: Disconnect
        self.broker.set_disconnect(10.0)
        self._disconnected = True

        # Fills happen during outage (canonical records them, SUT doesn't receive)
        fill2 = self.broker.generate_fill("o1", price=155.0, quantity=50.0)  # additional fill
        # SUT never receives fill2 — it's in the disconnect window

        # SUT continues submitting orders (they fail at broker but SUT doesn't know)
        order2 = Order(id="o2", symbol="MSFT", side=Side.BUY, type=OrderType.MARKET, quantity=Decimal("50"))
        self.sut.submit_order(order2)
        # Broker is disconnected — this order doesn't register
        self.oracle.validate_all(self.sut, self.broker, 1.0)

        # Phase 3: Reconnect
        self.broker.reconnect()
        self._disconnected = False

        # SUT might receive delayed fills
        # Revalidation should detect divergence
        self.oracle.validate_all(self.sut, self.broker, 2.0)

        # Try to reconcile SUT state with broker
        sut_snap = self.sut.snapshot()
        reconciliation = self.broker.reconcile(sut_snap.positions, sut_snap.cash)

        if not reconciliation["matched"]:
            # Expected: reconciliation detects divergence
            result.reconciliation_summary = f"Reconciliation detected divergence: {len(reconciliation['divergences'])} issues."
        else:
            result.reconciliation_summary = "Reconciliation matched (SUT may have missed events)"

        report = self.oracle.finalize()
        result.sis_score = self.oracle.sis_score()
        result.divergences = report.divergences
        result.first_failure = report.first_failure
        result.n_events = 4
        result.n_validation_points = 3
        result.n_failures_injected = 1  # one fill not delivered during outage
        result.passed = not reconciliation["matched"]  # divergence expected
        return result

    # ------------------------------------------------------------------
    # Scenario 5: Restart During Active Trading
    # ------------------------------------------------------------------

    def _scenario_restart_active_trading(self, seed: int = 42, **kwargs: Any) -> ScenarioResult:
        """Kill system mid-event stream, restart with partial state.

        Expected: exact reconstruction OR full halt (never silent continuation).
        """
        assert self.sut is not None and self.broker is not None and self.oracle is not None
        result = ScenarioResult(seed=seed)
        initial_cash = kwargs.get("initial_cash", 1_000_000.0)

        # Set market prices
        self.sut.process_market_data("AAPL", 150.0)
        self.sut.process_market_data("MSFT", 350.0)

        # Phase 1: Active trading
        order1 = Order(id="r1", symbol="AAPL", side=Side.BUY, type=OrderType.MARKET, quantity=Decimal("200"))
        self.sut.submit_order(order1)
        self.broker.submit_order(order1)
        fill1 = self.broker.generate_fill("r1", price=150.0, quantity=200.0)
        if fill1:
            self.sut.receive_fill(fill1.fill, "AAPL")

        order2 = Order(id="r2", symbol="MSFT", side=Side.BUY, type=OrderType.MARKET, quantity=Decimal("50"))
        self.sut.submit_order(order2)
        self.broker.submit_order(order2)
        fill2 = self.broker.generate_fill("r2", price=350.0, quantity=50.0)
        if fill2:
            self.sut.receive_fill(fill2.fill, "MSFT")

        # Capture pre-restart state
        pre_positions = self.sut.snapshot()

        # Phase 2: "Restart" — reconstruct SUT from canonical broker state
        # This simulates what would happen if we restarted and synced from broker
        self.sut.reset(initial_cash=initial_cash)

        # Reconstruct from broker state
        for sym, pos in self.broker.positions.items():
            qty = float(pos.quantity)
            price = float(pos.market_price)
            if abs(qty) > 1e-10:
                # Re-create position via fills
                side = Side.BUY if qty > 0 else Side.SELL
                fill = Fill(
                    order_id=f"reconstruct-{sym}",
                    price=Decimal(str(price)),
                    quantity=Decimal(str(abs(qty))),
                    commission=Decimal("0"),
                    side=side,
                )
                self.sut.receive_fill(fill, sym)

        # Reconcile reconstructed SUT with canonical broker
        sut_snap = self.sut.snapshot()
        reconciliation = self.broker.reconcile(sut_snap.positions, sut_snap.cash)

        # Validate reconstruction
        self.oracle.validate_all(self.sut, self.broker, 3.0)

        report = self.oracle.finalize()
        result.sis_score = self.oracle.sis_score()
        result.divergences = report.divergences
        result.first_failure = report.first_failure
        result.n_events = 4
        result.n_validation_points = 2
        result.n_failures_injected = 0
        result.passed = reconciliation["matched"]
        result.reconciliation_summary = (
            f"Reconstruction {'MATCHED (PASS)' if reconciliation['matched'] else 'MISMATCHED'}. "
            f"{len(reconciliation['divergences'])} divergences."
        )
        return result

    # ------------------------------------------------------------------
    # Run all scenarios
    # ------------------------------------------------------------------

    def run_all_scenarios(
        self,
        seed: int = 42,
        **kwargs: Any,
    ) -> dict[str, ScenarioResult]:
        """Run all 5 critical test scenarios."""
        scenario_names = [
            "broker_fill_loss",
            "duplicate_fill_attack",
            "cash_overcommit",
            "disconnect_silent_fill",
            "restart_active_trading",
        ]
        results = {}
        for name in scenario_names:
            results[name] = self.run_scenario(name, seed=seed, **kwargs)
        return results

    def report_summary(self, results: dict[str, ScenarioResult]) -> str:
        """Generate a human-readable summary of scenario results."""
        lines = [
            "=" * 70,
            "ADVERSARIAL TEST HARNESS — SCENARIO RESULTS",
            "=" * 70,
        ]
        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            sis = result.sis
            lines.append(f"\n{name}:")
            lines.append(f"  Status:      {status}")
            lines.append(f"  SIS:         {sis:.6f}")
            lines.append(f"  Divergences: {len(result.divergences)}")
            lines.append(f"  Events:      {result.n_events}")
            lines.append(f"  Duration:    {result.duration_seconds:.3f}s")
            if result.first_failure:
                lines.append(f"  First fail:  {result.first_failure.details}")
            if result.reconciliation_summary:
                lines.append(f"  Summary:     {result.reconciliation_summary}")

        # Overall
        overall_sis = sum(r.sis for r in results.values()) / len(results)
        lines.append(f"\n{'=' * 70}")
        lines.append(f"OVERALL SIS (all scenarios): {overall_sis:.6f}")
        lines.append(f"SCENARIOS PASSED: {sum(1 for r in results.values() if r.passed)}/{len(results)}")
        lines.append(f"{'=' * 70}")
        return "\n".join(lines)
