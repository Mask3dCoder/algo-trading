from __future__ import annotations

from pathlib import Path

from scripts.validate_all_strategies import (
    StrategyResult,
    runtime_evidence_gate_failures,
)


def test_runtime_evidence_gate_blocks_live_strategy_with_weak_ci(tmp_path: Path) -> None:
    status_path = tmp_path / "strategy_status.yaml"
    status_path.write_text(
        "strategies:\n"
        "  weak_live:\n"
        "    phase: LIVE\n",
        encoding="utf-8",
    )

    result = StrategyResult(
        strategy_id="weak_live",
        universe="test",
        oos_sharpe=-0.25,
        ci_lower_90=-0.80,
        n_trades=12,
        status="ok",
    )

    failures = runtime_evidence_gate_failures([result], status_path, tmp_path / "missing_waivers.yaml")

    assert failures == [
        "weak_live (LIVE): CI lower bound -0.8000 <= 0. "
        "Add a documented waiver in config/validation_waivers.yaml only while under review."
    ]


def test_runtime_evidence_gate_allows_documented_active_waiver(tmp_path: Path) -> None:
    status_path = tmp_path / "strategy_status.yaml"
    waivers_path = tmp_path / "validation_waivers.yaml"
    status_path.write_text(
        "strategies:\n"
        "  waived_paper:\n"
        "    phase: PAPER\n",
        encoding="utf-8",
    )
    waivers_path.write_text(
        "waivers:\n"
        "  waived_paper:\n"
        "    gate: runtime_evidence_gate\n"
        "    expires_on: '2099-01-01'\n"
        "    reason: Temporary investigation waiver.\n",
        encoding="utf-8",
    )

    result = StrategyResult(
        strategy_id="waived_paper",
        universe="test",
        oos_sharpe=float("nan"),
        ci_lower_90=float("nan"),
        n_trades=0,
        status="ok",
    )

    assert runtime_evidence_gate_failures([result], status_path, waivers_path) == []
