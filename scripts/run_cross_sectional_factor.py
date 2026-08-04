#!/usr/bin/env python3
"""Pre-registered cross-sectional factor experiment - CLI.

The decisive run is documented and locked in
``docs/cross_sectional_factor_protocol.md`` (commit ``ed52bae``). This CLI
executes exactly the three pre-registered variants against either the PIT
store (decisive) or the triage panel (smoke test only).

Examples
--------
    # Decisive run (requires a populated PIT store):
    python scripts/run_cross_sectional_factor.py --source pit

    # Triage smoke test (structural check only - never read the numbers):
    python scripts/run_cross_sectional_factor.py --source triage --triage --quick

The triage path writes to ``reports/cross_sectional_factor_triage_smoke.json``
and stamps every output with ``"triage": true`` so it can never be confused
with the decisive run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantcore.research.factor_experiment import (  # noqa: E402
    VARIANTS,
    ExperimentConfig,
    default_out_path,
    load_panel,
    run_experiment,
    write_results,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["pit", "triage"], default="pit")
    p.add_argument("--pit-dir", default=None, help="PIT store root (default data/pit/market)")
    p.add_argument("--triage-csv", default="data/sp500_pit_universe_2018_2024.csv")
    p.add_argument("--spy-csv", default="tests/fixtures/spy_raw_daily_2020_2024.csv",
                   help="Committed Yahoo SPY fixture for the triage benchmark")
    p.add_argument("--start", default=None, help="Window start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Window end (YYYY-MM-DD)")
    p.add_argument("--out", default=None, help="Output JSON path")
    p.add_argument("--quick", action="store_true",
                   help="Reduce bootstrap reps to 200 (smoke tests only)")
    p.add_argument("--triage", action="store_true",
                   help="Mark the run as TRIAGE; forces --source triage and labels output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.triage and args.source != "triage":
        args.source = "triage"

    cfg = ExperimentConfig(
        source=args.source,
        pit_dir=args.pit_dir,
        triage_csv=args.triage_csv,
        spy_csv=args.spy_csv,
        start=args.start,
        end=args.end,
        out=args.out,
        n_bootstrap=200 if args.quick else 2000,
        triage=args.triage,
    )

    if cfg.triage:
        banner = "=" * 72
        print(banner)
        print("  TRIAGE SMOKE RUN - structural check only.")
        print("  Output is NOT the decisive verdict and must not be read as one.")
        print(banner)

    panel = load_panel(cfg)
    print(
        f"panel: source={panel['source']} symbols={len(panel['symbols'])} "
        f"days={len(panel['dates'])} window={panel['window']['start']}..{panel['window']['end']}"
    )
    print(f"variants (frozen, n={len(VARIANTS)}): {', '.join(VARIANTS)}")

    results = run_experiment(panel, cfg)
    out_path = cfg.out or default_out_path(cfg)
    write_results(results, out_path)

    print(f"wrote {out_path}")
    for variant, res in results["variants"].items():
        verdict = res.get("verdict", res.get("error", "ERROR"))
        sharpe = res.get("oos_sharpe")
        dsr = res.get("dsr_p")
        print(f"  {variant:10s} verdict={verdict}  oos_sharpe={sharpe}  dsr_p={dsr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
