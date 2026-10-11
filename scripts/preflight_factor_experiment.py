#!/usr/bin/env python3
"""Run the cross-sectional experiment preflight.

Verifies every prerequisite against real repository artifacts BEFORE any
result exists, and exits non-zero if the experiment cannot be run exactly as
pre-registered. This is deliberately fail-closed: an unverifiable prerequisite
is a blocker, not a warning.

Usage:
    python scripts/preflight_factor_experiment.py
    python scripts/preflight_factor_experiment.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quantcore.data.pit.market import default_market_data_dir  # noqa: E402
from quantcore.data.pit.market.store import ParquetPitStore  # noqa: E402
from quantcore.research.factor_preflight import (  # noqa: E402
    load_panel_symbols,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight the pre-registered experiment")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable report")
    args = parser.parse_args()

    symbols = load_panel_symbols(REPO_ROOT)
    store = ParquetPitStore(default_market_data_dir())
    report = run_preflight(store, symbols)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print("=" * 78)
        print("CROSS-SECTIONAL FACTOR EXPERIMENT - PREREQUISITE PREFLIGHT")
        print("Protocol: docs/cross_sectional_factor_protocol.md (locked 2026-08-04)")
        print("=" * 78)
        print(report.render())

    if not report.can_execute:
        print("\nBLOCKED: the experiment must not run until the failures above are resolved.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())