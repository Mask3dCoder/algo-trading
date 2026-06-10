from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_preflight import expected_phase, resolve_config, run_preflight  # noqa: E402


def test_resolve_config_paper_flag() -> None:
    assert resolve_config(None, paper=True) == PROJECT_ROOT / "config" / "paper.yaml"


def test_expected_phase_for_paper_config() -> None:
    assert expected_phase(PROJECT_ROOT / "config" / "paper.yaml", paper=False) == "PAPER"


def test_run_preflight_accepts_real_paper_config() -> None:
    exit_code = run_preflight(PROJECT_ROOT / "config" / "paper.yaml", PROJECT_ROOT / "strategy_status.yaml")
    assert exit_code == 0


def test_run_preflight_rejects_bad_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "paper.yaml"
    status_path = tmp_path / "strategy_status.yaml"
    config_path.write_text(
        """
strategies:
  - name: equity_stat_arb
    enabled: true
    allocation_weight: 0.5
""",
        encoding="utf-8",
    )
    status_path.write_text(
        """
strategies:
  equity_stat_arb:
    phase: PAPER
""",
        encoding="utf-8",
    )

    exit_code = run_preflight(config_path, status_path)

    assert exit_code == 1
