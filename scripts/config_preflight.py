#!/usr/bin/env python3
"""Preflight checks for QuantCore live and paper configs.

Validates YAML syntax, strategy weights, runtime status gating, and kill-switch
history before launching a live or paper validation run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIVE_CONFIG = PROJECT_ROOT / "config" / "live.yaml"
DEFAULT_PAPER_CONFIG = PROJECT_ROOT / "config" / "paper.yaml"
DEFAULT_STATUS_PATH = PROJECT_ROOT / "strategy_status.yaml"
DEFAULT_KILL_SWITCH_LOGS = (
    PROJECT_ROOT / "reports" / "kill_switch_events.jsonl",
    PROJECT_ROOT / "reports" / "paper_kill_switch_events.jsonl",
)
DEFAULT_KILL_SWITCH_FILES = (
    PROJECT_ROOT / "KILL_SWITCH",
    PROJECT_ROOT / "PAPER_KILL_SWITCH",
)
WEIGHT_TOLERANCE = 1e-6


def resolve_config(config_arg: str | None, paper: bool) -> Path:
    """Resolve the config path requested by the CLI."""
    if config_arg:
        path = Path(config_arg)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if paper:
        return DEFAULT_PAPER_CONFIG
    return DEFAULT_LIVE_CONFIG


def expected_phase(config_path: Path, paper: bool) -> str:
    """Return the expected strategy_status.yaml phase for a config."""
    if paper or config_path.name == "paper.yaml":
        return "PAPER"
    return "LIVE"


def load_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load YAML and return ``(data, error)``."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return None, f"Config file not found: {path}"
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML in {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"Config root must be a mapping: {path}"
    return data, None


def load_status(path: Path) -> dict[str, Any]:
    """Load strategy status records."""
    if not path.exists():
        return {}
    data, error = load_yaml(path)
    if error or not isinstance(data, dict):
        return {}
    statuses = data.get("strategies", {})
    return statuses if isinstance(statuses, dict) else {}


def enabled_strategies(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled strategy records from a config."""
    strategies = config.get("strategies", [])
    if isinstance(strategies, dict):
        strategies = list(strategies.values())
    if not isinstance(strategies, list):
        return []
    return [s for s in strategies if isinstance(s, dict) and s.get("enabled", False)]


def strategy_weight(strategy: dict[str, Any]) -> float:
    """Return the allocation weight for a strategy record."""
    if "allocation_weight" in strategy:
        return float(strategy["allocation_weight"])
    return float(strategy.get("weight", 0.0))


def check_kill_switch_files() -> tuple[list[str], list[str]]:
    """Return missing/active kill-switch file errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    for path in DEFAULT_KILL_SWITCH_FILES:
        if path.exists():
            errors.append(f"Kill-switch file exists and must be cleared before launch: {path}")
    return errors, warnings


def check_kill_switch_logs() -> list[str]:
    """Warn if kill-switch history logs contain entries."""
    warnings: list[str] = []
    for path in DEFAULT_KILL_SWITCH_LOGS:
        if not path.exists():
            continue
        try:
            entries = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as exc:
            warnings.append(f"Could not read kill-switch log {path}: {exc}")
            continue
        if entries:
            warnings.append(f"Kill-switch history has {len(entries)} entries in {path}; review before launch")
    return warnings


def check_validation_window(config: dict[str, Any], expected: str) -> tuple[list[str], list[str]]:
    """Fail closed when the configured paper validation window has elapsed.

    The registry phase is only half of the launch gate; the configured
    window is the other half. Without this check a batch whose window
    ended months ago still preflights clean as long as the registry says
    PAPER — which is exactly how the Phase 4 batch (2026-06-09 ->
    2026-07-21) stayed launchable-looking long after it lapsed. A new
    window is a governance decision (FREEZE.md), not a config edit.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if expected != "PAPER":
        return errors, warnings

    validation = config.get("validation")
    if not isinstance(validation, dict):
        warnings.append("config has no 'validation' block; paper window cannot be verified")
        return errors, warnings

    end = validation.get("end_date")
    if not end:
        warnings.append("config validation block has no end_date; paper window cannot be verified")
        return errors, warnings

    try:
        end_date = date.fromisoformat(str(end))
    except ValueError:
        errors.append(f"config validation.end_date {end!r} is not an ISO date (YYYY-MM-DD)")
        return errors, warnings

    today = date.today()
    if end_date < today:
        errors.append(
            f"paper validation window ended {end_date.isoformat()} "
            f"({(today - end_date).days} days ago); launching requires a fresh window "
            "with explicit governance approval"
        )
    return errors, warnings


def run_preflight(config_path: Path, status_path: Path) -> int:
    """Run all preflight checks and return a process exit code."""
    errors: list[str] = []
    warnings: list[str] = []

    config, error = load_yaml(config_path)
    if error:
        errors.append(error)
        print("\n".join(f"ERROR: {item}" for item in errors))
        return 1

    status = load_status(status_path)
    expected = expected_phase(config_path, paper=config_path.name == "paper.yaml")
    strategies = enabled_strategies(config)

    if not strategies:
        errors.append("No enabled strategies found in config")

    weight_sum = 0.0
    for idx, strategy in enumerate(strategies, start=1):
        name = strategy.get("name")
        if not name:
            errors.append(f"Strategy #{idx} is missing a name")
            continue
        try:
            weight = strategy_weight(strategy)
        except (TypeError, ValueError):
            errors.append(f"Strategy {name} has an invalid weight")
            continue
        weight_sum += weight

        phase = status.get(name, {}).get("phase") if isinstance(status.get(name), dict) else None
        if phase != expected:
            errors.append(
                f"Strategy {name} is configured for {expected} but strategy_status.yaml phase is {phase!r}"
            )

    if abs(weight_sum - 1.0) > WEIGHT_TOLERANCE:
        errors.append(
            f"Enabled strategy weights sum to {weight_sum:.12f}, expected 1.0 "
            f"(tolerance {WEIGHT_TOLERANCE})"
        )
    else:
        warnings.append(f"Enabled strategy weights sum to {weight_sum:.12f}")

    window_errors, window_warnings = check_validation_window(config, expected)
    errors.extend(window_errors)
    warnings.extend(window_warnings)

    file_errors, file_warnings = check_kill_switch_files()
    errors.extend(file_errors)
    warnings.extend(file_warnings)
    warnings.extend(check_kill_switch_logs())

    print(f"Preflight: {config_path}")
    print(f"Expected strategy_status.yaml phase: {expected}")
    print(f"Enabled strategies: {len(strategies)}")
    print(f"Weight sum: {weight_sum:.12f}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1

    print("PASS: preflight checks passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate QuantCore launch configs")
    parser.add_argument("--config", type=Path, help="Config path; defaults to config/live.yaml")
    parser.add_argument("--paper", action="store_true", help="Read config/paper.yaml")
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS_PATH,
        help="strategy_status.yaml path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_config(str(args.config) if args.config else None, paper=args.paper)
    return run_preflight(config_path, args.status)


if __name__ == "__main__":
    sys.exit(main())
