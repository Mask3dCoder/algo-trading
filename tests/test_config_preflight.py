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


def test_run_preflight_rejects_lapsed_paper_batch(capsys) -> None:
    """The real paper config must NOT preflight clean.

    The Phase 4 window (2026-06-09 -> 2026-07-21) has elapsed and both its
    strategies are INACTIVE in strategy_status.yaml, so launching it would
    be launching a closed batch. Preflight is a launch gate: it must refuse.
    """
    exit_code = run_preflight(PROJECT_ROOT / "config" / "paper.yaml", PROJECT_ROOT / "strategy_status.yaml")

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "paper validation window ended 2026-07-21" in out
    assert "phase is 'INACTIVE'" in out


def test_run_preflight_accepts_consistent_paper_config(tmp_path: Path, capsys) -> None:
    """A paper config with a live window and matching PAPER phases passes."""
    config_path = tmp_path / "paper.yaml"
    status_path = tmp_path / "strategy_status.yaml"
    config_path.write_text(
        """
validation:
  start_date: 2099-01-01
  end_date: 2099-02-01
strategies:
  - name: some_paper_strategy
    enabled: true
    allocation_weight: 1.0
""",
        encoding="utf-8",
    )
    status_path.write_text(
        """
strategies:
  some_paper_strategy:
    phase: PAPER
""",
        encoding="utf-8",
    )

    exit_code = run_preflight(config_path, status_path)

    assert exit_code == 0
    assert "PASS: preflight checks passed" in capsys.readouterr().out


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
