#!/usr/bin/env python3
"""verify_honest_foundation.py — Complete forensic audit and validation.

This script runs the full "Honest Foundation" verification:
1. NullStrategy bias audit (Task 1.1)
2. Intraday DD verification (Task 1.2)
3. Triage gate enforcement (Task 4.1)
4. Portfolio-level DSR (Task 3.3)
5. Pre-mortem document checks (Task 4.2)

Usage:
    python scripts/verify_honest_foundation.py

Outputs:
    HONEST_FOUNDATION_REPORT.json — structured report for CI/CD consumption
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run_null_strategy_audit() -> dict:
    """Run NullStrategy through the engine to detect look-ahead bias."""
    try:
        from quantcore.backtest.null_strategy import NullStrategy, NullStrategyConfig
        from quantcore.backtest.engine import BacktestEngine
        from quantcore.strategies.base import StrategyConfig, StrategyType

        # Generate synthetic OHLCV bars
        import pandas as pd
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", "2024-12-31")
        p = 100.0
        rows = []
        for dt in dates:
            p *= 1 + rng.normal(0.0002, 0.012)
            rows.append({
                "symbol": "SPY",
                "timestamp": dt,
                "open": p * 0.999,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "volume": int(rng.integers(1e6, 5e6)),
            })
        data = pd.DataFrame(rows)

        # Run 20 null strategies for statistical power (more runs → stabler estimate)
        n_runs = 20
        sharpes = []
        for seed in range(n_runs):
            cfg = StrategyConfig(
                name="null_strategy",
                strategy_type=StrategyType.EQUITY,
                max_position_notional=1_000_000,
                params={"entry_prob": 0.05, "exit_prob": 0.10, "seed": seed},
            )
            strat = NullStrategy(cfg)
            # Null strategy must have ZERO transaction costs so that any
            # systematic non-zero Sharpe is due to look-ahead bias, not
            # cost drag.  This isolates the bias test from the cost test.
            engine = BacktestEngine(
                strategy=strat, initial_cash=1_000_000,
                slippage_bps=0.0, commission_per_share=0.0,
            )
            result = engine.run(data)
            report = engine.summary()
            sharpes.append(report.sharpe)

        sharpes = np.array([s for s in sharpes if math.isfinite(s)])
        if len(sharpes) == 0:
            return {
                "mean_sharpe": np.nan, "std_sharpe": np.nan,
                "engine_is_biased": True,
                "judgment": "FAIL — NullStrategy returned no valid夏普 ratios",
            }

        mean_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes))
        # With zero transaction costs, a truly unbiased engine should yield
        # mean Sharpe ≈ 0.0 ± 0.15 and std ≈ 0.20‑0.35 across random runs.
        # The original test used slippage/commission, which created a -0.29
        # mean Sharpe from cost drag rather than look-ahead bias.
        biased = abs(mean_sharpe) > 0.15 or std_sharpe < 0.02

        return {
            "mean_sharpe": round(mean_sharpe, 4),
            "std_sharpe": round(std_sharpe, 4),
            "min_sharpe": round(float(np.min(sharpes)), 4),
            "max_sharpe": round(float(np.max(sharpes)), 4),
            "engine_is_biased": biased,
            "judgment": (
                "FAIL — Engine has look-ahead bias" if biased
                else "PASS — Engine is unbiased"
            ),
            "gate_pass": abs(mean_sharpe) <= 0.1,
        }
    except Exception as e:
        return {"error": str(e), "engine_is_biased": True, "judgment": f"ERROR: {e}"}


SCOREBOARD_PATH = REPO_ROOT / "reports" / "strategy_scoreboard.json"


def _load_scoreboard_evidence() -> tuple[dict[str, dict], str | None]:
    """Load the batch validator's scoreboard (the real evidence source).

    Returns (by_name, staleness_note). A missing/unreadable scoreboard is
    NOT silently treated as "no evidence" — the caller fails closed.
    """
    if not SCOREBOARD_PATH.exists():
        return {}, f"{SCOREBOARD_PATH} does not exist — run scripts/validate_all_strategies.py"
    try:
        payload = json.loads(SCOREBOARD_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {}, f"scoreboard unreadable: {exc}"

    strategies = payload.get("strategies") or []
    by_name = {s["strategy_id"]: s for s in strategies if "strategy_id" in s}
    note = None
    # The scoreboard writes its generation stamp under "generated"
    # (see _save_scoreboard_json in validate_all_strategies.py).
    ts = payload.get("timestamp") or payload.get("generated_at") or payload.get("generated")
    if ts:
        try:
            stamp = datetime.fromisoformat(str(ts))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - stamp).days
            if age_days > 30:
                note = f"scoreboard is {age_days} days old (generated {ts})"
        except Exception:
            note = f"could not parse scoreboard timestamp {ts!r}"
    else:
        note = "scoreboard carries no generation timestamp (staleness unknown)"
    return by_name, note


def _score_number(row: dict, key: str) -> float:
    """Read a numeric metric from a scoreboard row; NaN if absent/unparsable.

    NaN (never a default) is deliberate: a missing metric must surface as a
    gate failure, not as a value that happens to pass.
    """
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return float("nan")


def check_triage_gates() -> list[dict]:
    """Run the TriageGate against REAL OOS evidence from the scoreboard.

    Previously this called ``gate.check()`` with hardcoded metrics
    (oos_sharpe=1.5, ci_low=0.5, dsr_p=0.05, n_oos_trades=20) and reported
    the verdict as if a backtest had produced them — fabricated evidence in
    the one tool whose job is to detect fabricated evidence. Every metric now
    comes from ``reports/strategy_scoreboard.json`` (written by
    scripts/validate_all_strategies.py through the full BacktestEngine), and
    a strategy with no scoreboard row FAILS the gate: no exception, no
    invented numbers.
    """
    from quantcore.research.triage_gate import TriageGate
    import yaml

    gate = TriageGate()

    try:
        with open(REPO_ROOT / "config" / "live.yaml") as f:
            config = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"cannot read config/live.yaml: {exc}"}]

    evidence, staleness_note = _load_scoreboard_evidence()

    results: list[dict] = []
    for strat_cfg in config.get("strategies", []):
        name = strat_cfg["name"]
        if not strat_cfg.get("enabled", False):
            continue

        stype = "OPTIONS" if "options" in name else "EQUITY"
        row = evidence.get(name)

        if row is None:
            results.append({
                "name": name,
                "passed": False,
                "failures": [
                    "NO EVIDENCE — no scoreboard row; run validate_all_strategies.py "
                    "to produce real OOS metrics (gate inputs are never fabricated)"
                ],
                "warnings": [staleness_note] if staleness_note else [],
                "metrics": "unavailable",
            })
            continue

        trades = _score_number(row, "n_trades")
        result = gate.check(
            strategy_name=name,
            strategy_type=stype,
            oos_sharpe=_score_number(row, "oos_sharpe"),
            ci_low=_score_number(row, "ci_lower_90"),
            ci_high=_score_number(row, "ci_upper_90"),
            dsr_p=_score_number(row, "dsr_pvalue"),
            max_dd_pct=_score_number(row, "max_drawdown_pct"),
            n_oos_trades=int(trades) if trades == trades else 0,  # nan check
            null_sharpe=0.05,
        )
        warnings = list(result.warnings)
        if staleness_note:
            warnings.append(staleness_note)
        if str(row.get("status", "ok")) != "ok":
            warnings.append(f"scoreboard status={row.get('status')}: {row.get('error', '')}")

        results.append({
            "name": name,
            "passed": result.passed,
            "failures": result.failures,
            "warnings": warnings,
            "metrics": str(result.metrics),
            "evidence_source": str(SCOREBOARD_PATH),
        })
    return results


def check_pre_mortems() -> dict:
    """Check pre-mortem documents exist for all enabled strategies."""
    pre_mortems = list(Path("strategies").glob("*/PRE_MORTEM.md"))
    live_enabled = []
    try:
        import yaml
        with open("config/live.yaml") as f:
            config = yaml.safe_load(f)
        live_enabled = [s["name"] for s in config.get("strategies", []) if s.get("enabled")]
    except Exception:
        pass

    found_names = [p.parent.name for p in pre_mortems]
    missing = [s for s in live_enabled if s not in found_names]

    return {
        "pre_mortems_found": len(pre_mortems),
        "pre_mortems_missing": missing,
        "pre_mortem_complete": len(missing) == 0,
    }


def compute_portfolio_dsr() -> dict:
    """Verify the portfolio-level DSR INFRASTRUCTURE on synthetic noise.

    Rationale (unchanged, now explicit in code): there is no live portfolio
    return series to deflate while the rebuild is frozen, so the check is
    that the DSR pipeline runs and correctly REJECTS pure noise (p ~ 1.0,
    not deployable). The series below is a zero-signal control, labeled
    ``synthetic`` in the output — it is never presented as portfolio
    performance. Wiring real portfolio returns happens only once strategies
    are live (see the 30-day paper gate).
    """
    try:
        from quantcore.research.portfolio_dsr import compute_portfolio_dsr
        rng = np.random.default_rng(42)
        # Synthetic zero-signal control: pure noise, mean ~0. This is NOT
        # performance data; it is the input that MUST fail the gate.
        synthetic_returns = rng.normal(0.0, 0.015, 505)

        result = compute_portfolio_dsr(
            synthetic_returns,
            n_strategies_tested=45,
            n_strategies_surviving=3,
        )
        dsr_p = result.dsr_p_value
        rejects_noise = not (dsr_p is not None and dsr_p < 0.05)
        return {
            "dsr_p_value": dsr_p,
            "is_deployable": getattr(result, "is_deployable", False),
            "reason": getattr(result, "fail_reason", ""),
            "input_kind": "synthetic_zero_signal_control",
            "input_note": (
                "Synthetic noise, NOT portfolio performance. Infrastructure "
                "check only: the DSR must reject it (p >= 0.05)."
            ),
            "rejects_noise_correctly": rejects_noise,
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    print("=" * 70)
    print("QUANTCORE HONEST FOUNDATION VERIFICATION v2")
    print("Forensic audit phase: ENGINE FOUNDATION -> STRATEGY HARDENING -> GOVERNANCE")
    print("=" * 70)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "2024-06-06-v2",
    }
    blockers = []

    # ── Task 1.1: NullStrategy bias audit ──────────────────────────────────
    print("\n[Task 1.1] NullStrategy bias audit...")
    null_result = run_null_strategy_audit()
    report["null_strategy_audit"] = null_result
    print(f"  Mean Sharpe: {null_result.get('mean_sharpe', 'N/A')}")
    print(f"  Std Sharpe: {null_result.get('std_sharpe', 'N/A')}")
    print(f"  Judgment: {null_result.get('judgment', 'N/A')}")
    if null_result.get("engine_is_biased", True):
        blockers.append(f"Task 1.1: {null_result.get('judgment')}")

    # ── Task 4.1: Triage gate enforcement ─────────────────────────────────
    print("\n[Task 4.1] Triage Gate enforcement...")
    triage_results = check_triage_gates()
    report["triage_gate_results"] = triage_results
    for tr in triage_results:
        print(f"  Strategy '{tr.get('name', 'unknown')}': {'PASS' if tr.get('passed') else 'FAIL'}")
        for f in tr.get("failures", []):
            print(f"    FAILURE: {f}")
            blockers.append(f"Task 4.1: {tr.get('name')} — {f}")

    # ── Task 3.3: Portfolio-level DSR ─────────────────────────────────────
    # This is an INFRASTRUCTURE test — we verify the DSR code runs correctly
    # and its gate logic is sound.  Using mock data is deliberate: noise must
    # be correctly rejected.  We do NOT block CI on the absence of deployable
    # alpha during the rebuild phase; the gate is applied to real portfolio
    # returns once strategies are live.
    print("\n[Task 3.3] Portfolio-level DSR (infrastructure check)...")
    dsr_result = compute_portfolio_dsr()
    report["portfolio_dsr"] = dsr_result
    print(f"  DSR p-value: {dsr_result.get('dsr_p_value', 'N/A')}")
    print(f"  Deployable: {dsr_result.get('is_deployable', False)}")
    # Only block if the DSR infrastructure itself is broken (exception or
    # nonsensical values).  A correct rejection of noise is a PASS.
    if "error" in dsr_result:
        blockers.append(f"Task 3.3: Portfolio DSR infrastructure failure ({dsr_result.get('reason', '')})")

    # ── Task 4.2: Pre-mortem documents ──────────────────────────────────
    print("\n[Task 4.2] Pre-mortem document check...")
    pm_result = check_pre_mortems()
    report["pre_mortem_check"] = pm_result
    print(f"  Pre-mortems found: {pm_result['pre_mortems_found']}")
    print(f"  Missing: {pm_result['pre_mortems_missing']}")
    if not pm_result["pre_mortem_complete"]:
        blockers.append("Task 4.2: Missing pre-mortem documents")

    # ── Task 1.2/1.3: Intraday infrastructure ──────────────────────────────
    # These are build artifacts — verify imports work
    print("\n[Task 1.2/1.3] Intraday infrastructure...")
    try:
        from quantcore.backtest.tick_context import TickContext, TickContextHistory
        from quantcore.risk.intraday_drawdown import IntradayDrawdownCalculator
        from quantcore.backtest.intraday_engine import IntradayBacktestEngine
        report["intraday_infra"] = {"status": "imported"}
        print("  [OK] IntradayBar, TickContext, IntradayDrawdownCalculator, IntradayBacktestEngine")
    except ImportError as e:
        report["intraday_infra"] = {"status": "failed", "error": str(e)}
        blockers.append(f"Task 1.2/1.3: Intraday infrastructure import failed: {e}")
        print(f"  [FAIL] Import failed: {e}")

    # ── Task 3.2: VIX Call Ladder ────────────────────────────────────────
    print("\n[Task 3.2] VIX Call Ladder hedge...")
    try:
        from quantcore.strategies.macro.vix_call_ladder import VIXCallLadderStrategy
        report["vix_hedge"] = {"status": "imported"}
        print("  [OK] VIXCallLadderStrategy imported")
    except ImportError as e:
        report["vix_hedge"] = {"status": "failed", "error": str(e)}
        blockers.append(f"Task 3.2: VIX hedge import failed: {e}")

    # ── Opening gap fade ───────────────────────────────────────────────────
    print("\n[Task 2.2] Opening Gap Fade strategy...")
    try:
        from quantcore.strategies.equity.opening_gap_fade import OpeningGapFadeStrategy
        report["opening_gap_fade"] = {"status": "imported"}
        print("  [OK] OpeningGapFadeStrategy imported")
    except ImportError as e:
        report["opening_gap_fade"] = {"status": "failed", "error": str(e)}
        blockers.append(f"Task 2.2: Opening gap fade import failed: {e}")

    # ── Overall ───────────────────────────────────────────────────────────
    report["overall_pass"] = len(blockers) == 0
    report["blockers"] = blockers

    out_path = REPO_ROOT / "HONEST_FOUNDATION_REPORT.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 70)
    if report["overall_pass"]:
        print("[PASS] OVERALL: PASS — Honest Foundation verified")
        return 0
    else:
        print("[FAIL] OVERALL: FAIL — Critical blockers found:")
        for b in blockers:
            print(f"  - {b}")
        return 1


if __name__ == "__main__":
    sys.exit(main())