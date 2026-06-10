#!/usr/bin/env python3
"""Run Phase 4 paper-monitoring checks and append a dated log.

Usage:
    python scripts/paper_daily_monitor.py

This wrapper is intentionally read-only:
- Runs the two required paper monitoring commands.
- Captures stdout/stderr.
- Appends the run output to reports/paper_monitoring_daily/.
- Exits non-zero if either command fails.

Example cron entry for weekdays at 16:30 ET:
    30 16 * * 1-5 cd /path/to/algo\\ sw && python scripts/paper_daily_monitor.py >> reports/paper_monitoring_daily/cron.log 2>&1
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "reports" / "paper_monitoring_daily"
COMMANDS = [
    [sys.executable, "scripts/strategy_risk_dashboard.py", "--paper"],
    [sys.executable, "scripts/correlation_monitor.py", "--paper"],
]


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"paper_daily_monitor_{stamp}.log"

    lines: list[str] = []
    lines.append(f"Paper daily monitor run: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Root: {ROOT}")
    lines.append("")

    failed = False
    for command in COMMANDS:
        command_label = " ".join(command)
        lines.append(f"$ {command_label}")
        lines.append("-" * 80)

        result = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.stdout:
            lines.append(result.stdout.rstrip())
        if result.stderr:
            lines.append(result.stderr.rstrip())
        lines.append(f"Exit code: {result.returncode}")
        lines.append("")

        if result.returncode != 0:
            failed = True

    if failed:
        lines.append("RESULT: one or more paper monitoring commands failed.")
    else:
        lines.append("RESULT: paper monitoring commands completed successfully.")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Paper daily monitor log: {log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
