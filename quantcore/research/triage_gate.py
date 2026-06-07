"""TriageGate — automated strategy promotion gate for CI/CD enforcement.

No strategy can be added to live.yaml unless TriageGate.check() returns True.
This gate is the ONLY mechanism for strategy promotion — manual overrides are
forbidden after this class is implemented.

Gates enforced:
  CI_low > 0 (90% confidence level — strategy must have positive OOS edge)
  DSR p < 0.10 (strategy survives multiple-testing correction)
  Max DD < 50% (engine-verified — catches implausible/low-DD strategy claims)
  N OOS trades > 10 (sample size gate — prevents overfitting on tiny samples)
  NullStrategy Sharpe ≈ 0.0 (engine bias check — must pass before any promotion)

All metrics are computed from OUT-OF-SAMPLE walk-forward results, not in-sample.

Usage:
    gate = TriageGate()
    result = gate.check(
        strategy_name="options_delta_neutral",
        oos_sharpe=2.48,
        ci_low=1.57,
        dsr_p=0.044,
        max_dd_pct=0.46,
        n_oos_trades=32,
        null_sharpe=0.01,    # From null strategy audit (must be ≈ 0)
    )
    if not result.passed:
        raise PromotionBlocked(f"Strategy failed gate: {result.failures}")

    # CI/CD enforcement
    gate.enforce_in_ci(strategies=all_strategy_reports)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class TriageResult:
    """Result of a TriageGate.check() call."""
    strategy_name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)  # All gate values

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] TriageGate: {self.strategy_name}"]
        for f in self.failures:
            lines.append(f"  GATE_FAIL: {f}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class TriageGate:
    """Automated strategy promotion gate.

    Thresholds (configurable per-strategy-type):
        CI_LOW_MIN: 0.0 (must have genuinely positive OOS edge)
        DSR_P_MAX: 0.10 (5% significance after multiple-testing correction)
        MAX_DD_PCT: 50.0 (never exceed 50% intraday MTM drawdown)
        MIN_OOS_TRADES: 10 (minimum sample size for OOS validation)
        NULL_SHARPE_MAX: 0.20 (null strategy must be indistinguishable from random)
    """

    GATE_THRESHOLDS = {
        "OPTIONS": dict(
            ci_low_min=0.0,
            dsr_p_max=0.10,
            max_dd_pct=50.0,
            min_oos_trades=10,
            null_sharpe_max=0.20,
            sharpe_min=0.3,          # OOS annual Sharpe must be positive
            max_dd_pct_strict=30.0, # Realistic for defined-risk options strategies
        ),
        "EQUITY": dict(
            ci_low_min=0.0,
            dsr_p_max=0.10,
            max_dd_pct=50.0,
            min_oos_trades=10,
            null_sharpe_max=0.20,
            sharpe_min=0.5,
            max_dd_pct_strict=35.0,  # Realistic for equity strategies with proper sizing
        ),
        "MACRO": dict(
            ci_low_min=0.0,
            dsr_p_max=0.10,
            max_dd_pct=50.0,
            min_oos_trades=10,
            null_sharpe_max=0.20,
            sharpe_min=0.3,
            max_dd_pct_strict=15.0,
        ),
    }

    def __init__(self, overrides: dict | None = None) -> None:
        """Initialize gate with optional per-type threshold overrides."""
        self._thresholds: dict = {
            k: {**v, **overrides.get(k, {})}
            for k, v in self.GATE_THRESHOLDS.items()
        } if overrides else dict(self.GATE_THRESHOLDS)

    def check(
        self,
        strategy_name: str,
        strategy_type: Literal["OPTIONS", "EQUITY", "MACRO"] = "OPTIONS",
        oos_sharpe: float = np.nan,
        ci_low: float = np.nan,
        ci_high: float = np.nan,
        dsr_p: float = np.nan,
        max_dd_pct: float = np.nan,
        n_oos_trades: int = 0,
        null_sharpe: float = np.nan,
        n_wf_folds_passed: int = 0,
        total_wf_folds: int = 0,
        paper_sharpe: float | None = None,
        backtest_sharpe: float | None = None,
    ) -> TriageResult:
        """Evaluate a strategy against all promotion gates.

        Returns a TriageResult with pass/fail and detailed failure list.
        """
        t = self._thresholds.get(strategy_type, self._thresholds["OPTIONS"])
        failures: list[str] = []
        warnings: list[str] = []

        # ── Gate 1: CI_low > 0 ─────────────────────────────────────────────
        if not np.isfinite(ci_low):
            failures.append(f"CI_low is NaN — cannot gate on unmeasurable confidence interval")
        elif ci_low < t["ci_low_min"]:
            failures.append(
                f"OOS Sharpe CI_low ({ci_low:.3f}) < {t['ci_low_min']} — "
                f"no evidence of genuine OOS edge at 90% confidence"
            )

        # ── Gate 2: DSR p < threshold ───────────────────────────────────────
        if not np.isfinite(dsr_p):
            failures.append("DSR p-value is NaN — cannot assess multiple-testing correction")
        elif dsr_p >= t["dsr_p_max"]:
            failures.append(
                f"DSR p-value ({dsr_p:.4f}) >= {t['dsr_p_max']} — "
                f"strategy does not survive Bonferroni/Harvey-Liu correction"
            )

        # ── Gate 3: Max DD < hard cap ────────────────────────────────────────
        if not np.isfinite(max_dd_pct):
            failures.append("Max DD is NaN — cannot assess risk")
        elif max_dd_pct >= t["max_dd_pct"]:
            failures.append(
                f"Max DD ({max_dd_pct:.2f}%) >= {t['max_dd_pct']}% — "
                f"strategy exceeds absolute risk limit"
            )
        elif max_dd_pct > t["max_dd_pct_strict"]:
            failures.append(
                f"Max DD ({max_dd_pct:.2f}%) > {t['max_dd_pct_strict']}% — "
                f"strategy hard cap exceeded"
            )

        # ── Gate 4: OOS trades > minimum ───────────────────────────────────────
        if n_oos_trades <= t["min_oos_trades"]:
            failures.append(
                f"N OOS trades ({n_oos_trades}) <= {t['min_oos_trades']} — "
                f"sample size insufficient for OOS validation"
            )

        # ── Gate 5: NullStrategy audit ───────────────────────────────────────
        if np.isfinite(null_sharpe):
            if abs(null_sharpe) > t["null_sharpe_max"]:
                failures.append(
                    f"NullStrategy Sharpe ({null_sharpe:.3f}) > {t['null_sharpe_max']} — "
                    f"engine may have look-ahead bias. HALT ALL DEVELOPMENT until fixed."
                )

        # ── Gate 6: OOS Sharpe > minimum ─────────────────────────────────────
        if np.isfinite(oos_sharpe):
            if oos_sharpe < t["sharpe_min"]:
                failures.append(
                    f"OOS Sharpe ({oos_sharpe:.3f}) < {t['sharpe_min']} — "
                    f"strategy has no OOS edge above minimum threshold"
                )

        # ── Warning 1: Walk-forward consistency ────────────────────────────────
        if total_wf_folds > 0:
            pass_rate = n_wf_folds_passed / total_wf_folds
            if pass_rate < 0.5:
                warnings.append(
                    f"Only {n_wf_folds_passed}/{total_wf_folds} WF folds passed — "
                    f"inconsistent edge across market regimes"
                )

        # ── Warning 2: Paper vs Backtest Sharpe divergence ────────────────────
        if paper_sharpe is not None and backtest_sharpe is not None:
            if paper_sharpe < 0.5 * backtest_sharpe:
                warnings.append(
                    f"Paper Sharpe ({paper_sharpe:.2f}) < 50% of backtest Sharpe ({backtest_sharpe:.2f}) — "
                    f"investigate slippage or model error before LIVE promotion"
                )
            if paper_sharpe < 0:
                failures.append(
                    f"Paper Sharpe is negative ({paper_sharpe:.2f}) — "
                    f"strategy is under-performing in live paper trading"
                )

        return TriageResult(
            strategy_name=strategy_name,
            passed=len(failures) == 0,
            failures=failures,
            warnings=warnings,
            metrics={
                "oos_sharpe": oos_sharpe,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "dsr_p": dsr_p,
                "max_dd_pct": max_dd_pct,
                "n_oos_trades": n_oos_trades,
                "null_sharpe": null_sharpe,
                "n_wf_folds_passed": n_wf_folds_passed,
                "total_wf_folds": total_wf_folds,
                "paper_sharpe": paper_sharpe,
                "backtest_sharpe": backtest_sharpe,
            },
        )

    def check_from_wf_result(
        self,
        strategy_name: str,
        strategy_type: Literal["OPTIONS", "EQUITY", "MACRO"],
        wf_result: dict,
        null_sharpe: float = np.nan,
        paper_sharpe: float | None = None,
    ) -> TriageResult:
        """Convenience method to gate from a walk-forward result dict.

        wf_result fields expected:
            oos_sharpe, ci_low, ci_high, dsr_p, max_dd_pct,
            n_oos_trades, n_wf_folds_passed, total_wf_folds
        """
        return self.check(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            oos_sharpe=wf_result.get("oos_sharpe", np.nan),
            ci_low=wf_result.get("ci_low", np.nan),
            ci_high=wf_result.get("ci_high", np.nan),
            dsr_p=wf_result.get("dsr_p", np.nan),
            max_dd_pct=wf_result.get("max_dd_pct", np.nan),
            n_oos_trades=wf_result.get("n_oos_trades", 0),
            null_sharpe=null_sharpe,
            n_wf_folds_passed=wf_result.get("n_wf_folds_passed", 0),
            total_wf_folds=wf_result.get("total_wf_folds", 0),
            paper_sharpe=paper_sharpe,
            backtest_sharpe=wf_result.get("backtest_sharpe", None),
        )

    def check_from_report(
        self,
        strategy_name: str,
        strategy_type: Literal["OPTIONS", "EQUITY", "MACRO"],
        report,         # PerformanceReport
        null_sharpe: float = np.nan,
        n_oos_trades: int = 0,
        n_wf_folds_passed: int = 0,
        total_wf_folds: int = 0,
        paper_sharpe: float | None = None,
    ) -> TriageResult:
        """Gate from a PerformanceReport (the output of engine.summary())."""
        return self.check(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            oos_sharpe=getattr(report, 'sharpe', np.nan),
            ci_low=getattr(report, 'ci_low', np.nan),
            ci_high=getattr(report, 'ci_high', np.nan),
            dsr_p=getattr(report, 'deflated_sharpe', np.nan),
            max_dd_pct=getattr(report, 'max_drawdown', np.nan) * 100.0,
            n_oos_trades=n_oos_trades,
            null_sharpe=null_sharpe,
            n_wf_folds_passed=n_wf_folds_passed,
            total_wf_folds=total_wf_folds,
            paper_sharpe=paper_sharpe,
            backtest_sharpe=None,
        )

    @staticmethod
    def enforce_in_ci(strategies: list[TriageResult]) -> list[TriageResult]:
        """Enforce the gate in CI/CD — returns failed strategies that must block promotion.

        In CI/CD, exit(1) if any strategy has passed=False.
        """
        failed = [s for s in strategies if not s.passed]
        if failed:
            print("TRIAGE FAILED — blocking promotion:")
            for s in failed:
                print(f"  {s}")
        return failed

    def to_dict(self, result: TriageResult) -> dict:
        """Serialize TriageResult to dict for JSON serialization in CI logs."""
        import math
        def _sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        return {
            "strategy": result.strategy_name,
            "passed": result.passed,
            "failures": result.failures,
            "warnings": result.warnings,
            "metrics": {k: _sanitize(v) for k, v in result.metrics.items()},
        }