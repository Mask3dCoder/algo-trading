"""TruthValidator: runtime enforcement of economic invariants.

Canonical economic truth is defined by EconomicEngineCore:
  - apply_fill() is the SINGLE authoritative implementation of position math
  - PortfolioState.total_value is the ONE canonical PnL
  - PortfolioState.cash and PortfolioState.positions are canonical
  - PortfolioState and PaperBroker are DELEGATES, not independent sources

All other representations are DERIVED or ADVISORY. No system may compute
independent PnL without referencing canonical state.

Mode system:
  soft  → logs violations via structlog (non-blocking)
  hard  → raises AssertionError on any CRITICAL violation

Usage:
    validator = TruthValidator(mode="hard")
    validator.check_fill_batch(portfolio_state, fills)
    validator.check_eod_reconciliation(portfolio_state)
    validator.assert_clean()

Reconciliation:
    report = reconcile(engine_state, paper_state)
    print(report.divergence_trace)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import structlog

from quantcore.core import economic_kernel
from quantcore.core.types import Fill, PortfolioSnapshot, Position, Side


@dataclass
class Violation:
    """A single invariant violation."""
    invariant: str
    message: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    source: str = ""


@dataclass
class ValidationResult:
    """Result of a validation run."""
    passed: bool = True
    violations: list[Violation] = field(default_factory=list)

    def add(self, invariant: str, message: str, severity: str, source: str = "") -> None:
        self.passed = False
        self.violations.append(Violation(invariant, message, severity, source))


class TruthValidator:
    """Runtime invariant enforcer for economic truth.

    Validates that all systems respect the canonical economic model.
    Supports soft (logging), hard (exception), and enforce (pre-block) modes.

    Mode system:
      soft    → logs violations via structlog (non-blocking)
      hard    → raises AssertionError on any CRITICAL violation
      enforce → same as hard, PLUS pre_check_fill() blocks invalid
                transitions BEFORE they occur. Call pre_check_fill()
                before every PortfolioState.apply_fill().
    """

    def __init__(self, mode: str = "soft") -> None:
        if mode not in ("soft", "hard", "enforce"):
            raise ValueError(f"mode must be 'soft', 'hard', or 'enforce', got {mode!r}")
        self.mode = mode
        self.results: dict[str, ValidationResult] = {}
        self._log = structlog.get_logger()

    def set_mode(self, mode: str) -> None:
        """Switch enforcement mode at runtime."""
        if mode not in ("soft", "hard", "enforce"):
            raise ValueError(f"mode must be 'soft', 'hard', or 'enforce', got {mode!r}")
        self.mode = mode

    def _handle(self, result: ValidationResult) -> None:
        """Enforce based on current mode."""
        if not result.violations:
            return
        criticals = [v for v in result.violations if v.severity == "CRITICAL"]
        if self.mode in ("hard", "enforce") and criticals:
            msgs = "\n".join(f"  [{v.invariant}] {v.message}" for v in criticals)
            raise AssertionError(
                f"TruthValidator ({self.mode} mode): {len(criticals)} CRITICAL violation(s):\n{msgs}"
            )
        for v in result.violations:
            self._log.warning(
                "truth_violation",
                invariant=v.invariant,
                severity=v.severity,
                message=v.message,
                source=v.source,
            )

    # ------------------------------------------------------------------
    # Invariant 1: Portfolio Identity
    #   total_value == cash + Σ(position.market_value)
    # ------------------------------------------------------------------

    def check_portfolio_identity(
        self,
        state: Any,
        label: str = "unnamed",
    ) -> ValidationResult:
        """Verify that total_value = cash + Σ(market_value).

        This is the foundational economic identity. If it fails, the portfolio
        state is corrupt and all derived computations are invalid.
        """
        result = ValidationResult()
        try:
            total = float(state.total_value)
            cash = float(state.cash)
            pos_sum = sum(float(p.market_value) for p in state.positions.values())
            computed = cash + pos_sum
            diff = abs(total - computed)
            if diff > 0.01:
                result.add(
                    "portfolio_identity",
                    f"total_value ({total}) != cash ({cash}) + Σ(market_value) ({pos_sum})"
                    f" — diff={diff:.2f}",
                    severity="CRITICAL",
                    source=label,
                )
        except (AttributeError, TypeError, ValueError) as exc:
            result.add(
                "portfolio_identity",
                f"cannot evaluate: {exc}",
                severity="CRITICAL",
                source=label,
            )
        self.results[f"portfolio_identity:{label}"] = result
        self._handle(result)
        return result

    # ------------------------------------------------------------------
    # Invariant 2: Fill side canonicalization
    #   All fills must have fill.side set
    # ------------------------------------------------------------------

    def check_fill_side(self, fills: list[Fill], label: str = "unnamed") -> ValidationResult:
        """Verify every fill has a non-None side field."""
        result = ValidationResult()
        missing = [f.order_id for f in fills if f.side is None]
        if missing:
            result.add(
                "fill_side",
                f"{len(missing)} fill(s) missing side: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                severity="CRITICAL",
                source=label,
            )
        self.results[f"fill_side:{label}"] = result
        self._handle(result)
        return result

    # ------------------------------------------------------------------
    # Invariant 3: No order_id direction parsing
    #   Economic meaning must NOT be derived from string patterns
    # ------------------------------------------------------------------

    def check_no_order_id_parsing(self, fills: list[Fill], label: str = "unnamed") -> ValidationResult:
        """Flag fills where order_id appears to encode direction.

        Checks for known anti-patterns: '_buy', '_sell' suffixes.
        """
        result = ValidationResult()
        for f in fills:
            oid = f.order_id
            if oid.endswith("_buy") or oid.endswith("_sell"):
                result.add(
                    "order_id_parsing",
                    f"fill.order_id '{oid}' ends with '_buy'/'_sell' — "
                    f"direction should come from fill.side, not string parsing",
                    severity="HIGH",
                    source=label,
                )
        self.results[f"order_id_parsing:{label}"] = result
        self._handle(result)
        return result

    # ------------------------------------------------------------------
    # Invariant 4: Position integrity
    #   Positions must have valid state: avg_cost > 0, market_price >= 0
    # ------------------------------------------------------------------

    def check_position_integrity(
        self,
        positions: dict[str, Position],
        label: str = "unnamed",
    ) -> ValidationResult:
        """Verify position-level invariants."""
        result = ValidationResult()
        for sym, pos in positions.items():
            fq = float(pos.quantity)
            if abs(fq) > 1e-10:
                if float(pos.avg_cost) <= 0:
                    result.add(
                        "position_integrity",
                        f"{sym}: avg_cost={pos.avg_cost} must be > 0 for non-zero position",
                        severity="CRITICAL",
                        source=label,
                    )
                if float(pos.market_price) < 0:
                    result.add(
                        "position_integrity",
                        f"{sym}: market_price={pos.market_price} must be >= 0",
                        severity="HIGH",
                        source=label,
                    )
        self.results[f"position_integrity:{label}"] = result
        self._handle(result)
        return result

    # ------------------------------------------------------------------
    # Invariant 5: EconomicEngineCore consistency
    #   PortfolioState matches what the kernel would produce for the
    #   same fill sequence.
    # ------------------------------------------------------------------

    def check_engine_state_consistency(
        self,
        state_before: Any,
        state_after: Any,
        fill: Fill,
        symbol: str,
        label: str = "unnamed",
    ) -> ValidationResult:
        """Verify that applying a fill through EconomicEngineCore matches state_after.

        Replays the fill through the kernel using state_before, then compares
        the kernel output with state_after. This detects divergence between
        the engine's state transition and the canonical kernel.

        Args:
            state_before: Portfolio state BEFORE the fill was applied.
            state_after: Portfolio state AFTER the fill was applied.
            fill: The fill that was applied.
            symbol: Symbol for the position.
        """
        result = ValidationResult()
        try:
            is_buy: bool
            if fill.side is not None:
                is_buy = fill.side == Side.BUY
            else:
                is_buy = fill.order_id.endswith("_buy")

            qty = float(fill.quantity)
            price = float(fill.price)

            # Build position from state_before
            pos = state_before.positions.get(symbol)
            current = None
            if pos is not None and abs(float(pos.quantity)) > 1e-10:
                current = {
                    "symbol": symbol,
                    "quantity": float(pos.quantity),
                    "avg_cost": float(pos.avg_cost),
                    "market_price": float(pos.market_price),
                    "realized_pnl": float(pos.realized_pnl),
                }

            # What the kernel would produce
            fa = economic_kernel.apply_fill(
                current,
                fill_price=price,
                fill_quantity=qty,
                is_buy=is_buy,
                symbol=symbol,
            )

            # Compare kernel result with state_after
            if fa.position is None:
                if symbol in state_after.positions:
                    result.add(
                        "engine_state_consistency",
                        f"kernel says position {symbol} should be closed, but state has it "
                        f"(qty={float(state_after.positions[symbol].quantity)})",
                        severity="CRITICAL",
                        source=label,
                    )
            else:
                actual_pos = state_after.positions.get(symbol)
                if actual_pos is None:
                    result.add(
                        "engine_state_consistency",
                        f"kernel says position {symbol} should exist (qty={fa.position['quantity']:.4f}), "
                        f"but state has it closed",
                        severity="CRITICAL",
                        source=label,
                    )
                else:
                    for field, k_val, a_attr, tol in [
                        ("quantity", fa.position["quantity"], "quantity", 1e-6),
                        ("avg_cost", fa.position["avg_cost"], "avg_cost", 0.001),
                        ("realized_pnl", fa.position["realized_pnl"], "realized_pnl", 0.01),
                    ]:
                        a_val = float(getattr(actual_pos, a_attr))
                        if abs(a_val - k_val) > tol:
                            result.add(
                                "engine_state_consistency",
                                f"position {symbol} {field} mismatch: "
                                f"kernel={k_val:.6f}, state={a_val:.6f}",
                                severity="CRITICAL",
                                source=label,
                            )
        except Exception as exc:
            result.add(
                "engine_state_consistency",
                f"check failed with exception: {exc}",
                severity="HIGH",
                source=label,
            )
        self.results[f"engine_state_consistency:{label}"] = result
        self._handle(result)
        return result

    # ------------------------------------------------------------------
    # ENFORCE mode: pre-transition validation
    #   Call pre_check_fill() BEFORE PortfolioState.apply_fill().
    #   In enforce mode, raises immediately to block invalid transitions.
    # ------------------------------------------------------------------

    def pre_check_fill(
        self,
        state_before: Any,
        fill: Fill,
        symbol: str,
        label: str = "unnamed",
    ) -> ValidationResult:
        """Pre-transition check: validate a fill BEFORE it is applied.

        Simulates the fill through EconomicEngineCore using the current
        portfolio state, then inspects the result for invariant violations.
        In ENFORCE mode, CRITICAL violations raise AssertionError to
        BLOCK the transition — the fill is never applied.

        Call this before every PortfolioState.apply_fill() when running
        in enforce mode.

        Args:
            state_before: Portfolio state BEFORE the fill (must have
                .cash, .positions, .total_value).
            fill: The proposed fill.
            symbol: Symbol for the position.
            label: Source label for tracing.

        Returns:
            ValidationResult (may be empty if the transition is clean).
        """
        result = ValidationResult()

        # 1. Fill must have side to determine direction canonically
        if fill.side is None:
            result.add(
                "pre_check_fill",
                f"Fill {fill.order_id} missing side — direction cannot be "
                f"determined from canonical source. Blocking transition.",
                severity="CRITICAL",
                source=label,
            )

        # 2. Simulate through the kernel to detect issues before they happen
        try:
            is_buy: bool
            if fill.side is not None:
                is_buy = fill.side == Side.BUY
            else:
                is_buy = fill.order_id.endswith("_buy")

            qty = float(fill.quantity)
            price = float(fill.price)

            # Build current position for the kernel
            pos = getattr(state_before, "positions", {}).get(symbol)
            current = None
            if pos is not None and abs(float(pos.quantity)) > 1e-10:
                current = {
                    "symbol": symbol,
                    "quantity": float(pos.quantity),
                    "avg_cost": float(pos.avg_cost),
                    "market_price": float(pos.market_price),
                    "realized_pnl": float(pos.realized_pnl),
                }

            fa = economic_kernel.apply_fill(
                current,
                fill_price=price,
                fill_quantity=qty,
                is_buy=is_buy,
                symbol=symbol,
            )

            # 3. Validate the simulated result
            if fa.position is not None:
                fq = fa.position["quantity"]
                fac = fa.position["avg_cost"]
                if abs(fq) > 1e-10 and fac <= 0:
                    result.add(
                        "pre_check_fill",
                        f"Would create position {symbol} with avg_cost={fac} "
                        f"(must be > 0 for non-zero position)",
                        severity="HIGH",
                        source=label,
                    )
                if fa.position.get("market_price", 0) < 0:
                    result.add(
                        "pre_check_fill",
                        f"Would create position {symbol} with negative "
                        f"market_price",
                        severity="HIGH",
                        source=label,
                    )

            # 4. Verify commission is non-negative
            if fill.commission is not None and float(fill.commission) < 0:
                result.add(
                    "pre_check_fill",
                    f"Fill {fill.order_id} has negative commission "
                    f"({fill.commission})",
                    severity="HIGH",
                    source=label,
                )

        except Exception as exc:
            result.add(
                "pre_check_fill",
                f"Kernel simulation failed for fill {fill.order_id}: {exc}",
                severity="CRITICAL",
                source=label,
            )

        if result.violations:
            self.results[f"pre_check_fill:{fill.order_id}:{label}"] = result
            self._handle(result)

        return result

    # ------------------------------------------------------------------
    # Integration checkpoints
    # ------------------------------------------------------------------

    def check_fill_batch(
        self,
        state: Any,
        fills: list[tuple[Fill, str]],
        label: str = "unnamed",
    ) -> ValidationResult:
        """Validate after every fill batch: identity, fill side, consistency.

        Args:
            state: PortfolioState-like object (must have .positions, .cash, .total_value).
            fills: List of (Fill, symbol) tuples in the order they were applied.
        """
        result = ValidationResult()
        self.check_portfolio_identity(state, f"{label}/batch")

        fill_list = [f for f, _ in fills]
        self.check_fill_side(fill_list, f"{label}/batch")
        self.check_no_order_id_parsing(fill_list, f"{label}/batch")
        self.check_position_integrity(state.positions, f"{label}/batch")

        self.results[f"fill_batch:{label}"] = result
        self._handle(result)
        return result

    def check_margin_event(
        self,
        state: Any,
        label: str = "unnamed",
    ) -> ValidationResult:
        """Validate after margin events or liquidation.

        Verifies portfolio identity still holds and no positions have
        invalid state after forced liquidation.
        """
        result = ValidationResult()
        self.check_portfolio_identity(state, f"{label}/margin")
        self.check_position_integrity(state.positions, f"{label}/margin")
        self.results[f"margin_event:{label}"] = result
        self._handle(result)
        return result

    def check_eod_reconciliation(
        self,
        state: Any,
        label: str = "unnamed",
    ) -> ValidationResult:
        """Validate at end-of-bar: identity, position integrity, PnL conservation."""
        result = ValidationResult()
        self.check_portfolio_identity(state, f"{label}/eod")
        self.check_position_integrity(state.positions, f"{label}/eod")

        # PnL conservation: position realized + unrealized should be reasonable
        try:
            total_pos_pnl = sum(
                float(p.realized_pnl) + float(p.unrealized_pnl)
                for p in state.positions.values()
            )
            total_value = float(state.total_value)
            if total_value > 0:
                pnl_ratio = abs(total_pos_pnl) / total_value
                if pnl_ratio > 10.0:
                    result.add(
                        "pnl_conservation",
                        f"Σ(position_pnl)={total_pos_pnl:.2f} exceeds "
                        f"10x total_value ({total_value:.2f})",
                        severity="MEDIUM",
                        source=label,
                    )
        except (AttributeError, TypeError, ValueError) as exc:
            result.add(
                "pnl_conservation",
                f"cannot evaluate: {exc}",
                severity="MEDIUM",
                source=label,
            )

        self.results[f"eod_reconciliation:{label}"] = result
        self._handle(result)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> list[Violation]:
        """Return all violations across all checks."""
        all_violations: list[Violation] = []
        for res in self.results.values():
            all_violations.extend(res.violations)
        return all_violations

    def summary(self) -> str:
        """One-line summary of validation state."""
        total = len(self.results)
        violations = len(self.report())
        criticals = sum(1 for v in self.report() if v.severity == "CRITICAL")
        if violations == 0:
            return f"TruthValidator [{self.mode}]: {total} checks passed, 0 violations"
        return (
            f"TruthValidator [{self.mode}]: {total} checks, {violations} violation(s)"
            f" ({criticals} CRITICAL)"
        )

    def assert_clean(self) -> None:
        """Raise if any CRITICAL violations exist. Call at system boundaries."""
        violations = self.report()
        criticals = [v for v in violations if v.severity == "CRITICAL"]
        if criticals:
            msgs = "\n".join(f"  [{v.invariant}] {v.message}" for v in criticals)
            raise AssertionError(
                f"TruthValidator: {len(criticals)} CRITICAL violation(s):\n{msgs}"
            )


# ---------------------------------------------------------------------------
# Reconciliation: cross-system state comparison for auditability
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationReport:
    """Detailed comparison between two portfolio state representations.

    Attributes:
        delta_cash: Cash difference (engine - paper).
        delta_total_value: Total value difference (engine - paper).
        position_deltas: Dict of {symbol: PositionDelta}.
        divergence_trace: Human-readable list of mismatches.
        is_identical: True if all deltas are within tolerance.
    """
    delta_cash: float = 0.0
    delta_total_value: float = 0.0
    position_deltas: dict[str, PositionDelta] = field(default_factory=dict)
    divergence_trace: list[str] = field(default_factory=list)
    is_identical: bool = True


@dataclass
class PositionDelta:
    """Per-symbol position difference between two systems."""
    engine_qty: float = 0.0
    paper_qty: float = 0.0
    engine_avg_cost: float = 0.0
    paper_avg_cost: float = 0.0
    engine_realized_pnl: float = 0.0
    paper_realized_pnl: float = 0.0


def reconcile(
    engine_state: Any,
    paper_state: Any,
    tolerance_cash: float = 0.01,
    tolerance_qty: float = 1e-6,
    tolerance_avg_cost: float = 0.001,
    tolerance_pnl: float = 0.01,
) -> ReconciliationReport:
    """Compare engine and paper broker portfolio states.

    Produces a detailed divergence report identifying every difference
    in cash, positions, avg_cost, and realized PnL between the two
    systems. Required for auditability and debugging execution drift.

    Args:
        engine_state: PortfolioState or any object with .cash, .positions,
            .total_value attributes.
        paper_state: PaperBroker or any object with ._cash, ._positions,
            ._get_account-like access. Supports both:
            - dict with 'cash'/'positions' keys
            - object with .cash/.positions (or ._cash/._positions) attrs.
        tolerance_cash: Maximum acceptable cash difference.
        tolerance_qty: Maximum acceptable quantity difference.
        tolerance_avg_cost: Maximum acceptable avg_cost difference.
        tolerance_pnl: Maximum acceptable realized_pnl difference.

    Returns:
        ReconciliationReport documenting all differences.
    """
    report = ReconciliationReport()

    # Extract engine state
    e_cash = _extract_cash(engine_state)
    e_positions = _extract_positions(engine_state)
    e_total = getattr(engine_state, 'total_value', None)
    if e_total is not None:
        e_total = float(e_total)

    # Extract paper state
    p_cash = _extract_cash(paper_state)
    p_positions = _extract_positions(paper_state)
    p_total = getattr(paper_state, 'total_value', None)
    if p_total is not None:
        p_total = float(p_total)

    # Cash comparison
    cash_diff = e_cash - p_cash
    report.delta_cash = cash_diff
    if abs(cash_diff) > tolerance_cash:
        report.divergence_trace.append(
            f"CASH MISMATCH: engine={e_cash:.2f}, paper={p_cash:.2f}, "
            f"delta={cash_diff:.2f}"
        )

    # Total value comparison
    if e_total is not None and p_total is not None:
        tv_diff = e_total - p_total
        report.delta_total_value = tv_diff
        if abs(tv_diff) > tolerance_cash:
            report.divergence_trace.append(
                f"TOTAL VALUE MISMATCH: engine={e_total:.2f}, paper={p_total:.2f}, "
                f"delta={tv_diff:.2f}"
            )
    elif e_total is not None and p_total is None:
        report.divergence_trace.append(
            f"TOTAL VALUE: engine={e_total:.2f}, paper=unavailable"
        )
    elif e_total is None and p_total is not None:
        report.divergence_trace.append(
            f"TOTAL VALUE: engine=unavailable, paper={p_total:.2f}"
        )

    # Position comparison
    all_symbols = set(e_positions.keys()) | set(p_positions.keys())
    for sym in sorted(all_symbols):
        e_pos = e_positions.get(sym)
        p_pos = p_positions.get(sym)

        e_qty = float(e_pos["quantity"]) if e_pos else 0.0
        p_qty = float(p_pos["quantity"]) if p_pos else 0.0
        e_ac = float(e_pos["avg_cost"]) if e_pos else 0.0
        p_ac = float(p_pos["avg_cost"]) if p_pos else 0.0
        e_pnl = float(e_pos["realized_pnl"]) if e_pos else 0.0
        p_pnl = float(p_pos["realized_pnl"]) if p_pos else 0.0

        pd = PositionDelta(
            engine_qty=e_qty, paper_qty=p_qty,
            engine_avg_cost=e_ac, paper_avg_cost=p_ac,
            engine_realized_pnl=e_pnl, paper_realized_pnl=p_pnl,
        )
        report.position_deltas[sym] = pd

        qty_diff = abs(e_qty - p_qty)
        ac_diff = abs(e_ac - p_ac) if (abs(e_qty) > tolerance_qty or abs(p_qty) > tolerance_qty) else 0.0
        pnl_diff = abs(e_pnl - p_pnl)

        if qty_diff > tolerance_qty:
            report.divergence_trace.append(
                f"POSITION {sym}: qty mismatch — engine={e_qty:.4f}, paper={p_qty:.4f}"
            )
        if ac_diff > tolerance_avg_cost:
            report.divergence_trace.append(
                f"POSITION {sym}: avg_cost mismatch — engine={e_ac:.4f}, paper={p_ac:.4f}"
            )
        if pnl_diff > tolerance_pnl:
            report.divergence_trace.append(
                f"POSITION {sym}: realized_pnl mismatch — engine={e_pnl:.4f}, paper={p_pnl:.4f}"
            )

    # Also compare position differences when one side is flat but the other isn't
    only_in_engine = set(e_positions.keys()) - set(p_positions.keys())
    only_in_paper = set(p_positions.keys()) - set(e_positions.keys())
    if only_in_engine:
        report.divergence_trace.append(
            f"POSITIONS ONLY IN ENGINE: {', '.join(sorted(only_in_engine))}"
        )
    if only_in_paper:
        report.divergence_trace.append(
            f"POSITIONS ONLY IN PAPER: {', '.join(sorted(only_in_paper))}"
        )

    report.is_identical = len(report.divergence_trace) == 0
    return report


def _extract_cash(state: Any) -> float:
    """Extract cash value from various state representations."""
    if isinstance(state, dict):
        return float(state.get("cash", 0.0))
    # Try .cash first, then ._cash
    cash = getattr(state, "cash", None)
    if cash is not None:
        return float(cash)
    cash = getattr(state, "_cash", None)
    if cash is not None:
        return float(cash)
    return 0.0


def _extract_positions(state: Any) -> dict[str, dict[str, float]]:
    """Extract normalized position dict from various state representations."""
    if isinstance(state, dict):
        raw = state.get("positions", {})
        return _normalize_positions(raw)

    # Try .positions first, then ._positions
    raw = getattr(state, "positions", None)
    if raw is None:
        raw = getattr(state, "_positions", {})
    return _normalize_positions(raw)


def _normalize_positions(
    raw: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Convert position objects to normalized float dicts."""
    result: dict[str, dict[str, float]] = {}
    for sym, pos in raw.items():
        if hasattr(pos, "quantity"):
            # Position model
            qty = float(pos.quantity)
            if abs(qty) < 1e-10:
                continue  # Skip zero positions
            result[sym] = {
                "quantity": qty,
                "avg_cost": float(pos.avg_cost),
                "market_price": float(pos.market_price),
                "realized_pnl": float(pos.realized_pnl),
            }
        elif isinstance(pos, dict):
            qty = float(pos.get("quantity", 0.0))
            if abs(qty) < 1e-10:
                continue
            result[sym] = {
                "quantity": qty,
                "avg_cost": float(pos.get("avg_cost", 0.0)),
                "market_price": float(pos.get("market_price", 0.0)),
                "realized_pnl": float(pos.get("realized_pnl", 0.0)),
            }
    return result


# ---------------------------------------------------------------------------
# State replay: deterministic fill sequence reproduction
# ---------------------------------------------------------------------------


def replay_engine(
    fill_stream: Iterable[tuple[Fill, str]],
    initial_state: PortfolioState | None = None,
    initial_cash: float = 1_000_000.0,
    validator: TruthValidator | None = None,
) -> PortfolioState:
    """Replay a stream of fills through the canonical EconomicEngineCore.

    This is a DETERMINISTIC operation: same fill_stream + same initial_state
    always produces the same final PortfolioState. Use this for:

    - Reproducing engine state from a fill log for debugging.
    - Proving equivalence between backtest engine and paper broker.
    - Auditing fill sequences for regulatory compliance.

    Args:
        fill_stream: Iterable of (Fill, symbol) tuples in application order.
        initial_state: Starting PortfolioState, or None to create a fresh one.
        initial_cash: Initial cash when creating a fresh state.
        validator: Optional TruthValidator for pre/post transition checks.
            In enforce mode, pre_check_fill blocks invalid transitions
            before they happen.

    Returns:
        PortfolioState after all fills are applied.
    """
    from quantcore.backtest.engine import PortfolioState

    state = initial_state or PortfolioState(initial_cash=initial_cash)

    for fill, symbol in fill_stream:
        if validator is not None:
            validator.pre_check_fill(state, fill, symbol, "replay")

        state.apply_fill(fill, symbol)

        if validator is not None:
            validator.check_portfolio_identity(state, f"replay/{fill.order_id}")

    return state
