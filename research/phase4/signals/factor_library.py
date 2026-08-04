"""Phase 4 — Institutional Factor Library (Stage 10).

Builds a structured database of all evaluated factors:
  - JSON file per factor with full metadata
  - CSV index for rapid querying
  - Fields: ID, formula, IC, Rank IC, bootstrap CI, regime performance, capacity, status

This is the final, queryable artifact of the factor evaluation pipeline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FACTOR_LIBRARY_DIR = Path(__file__).resolve().parents[1] / "factor_library"


@dataclass
class FactorRecord:
    """Complete record for a single factor in the institutional library."""
    # Identity
    factor_id: str
    factor_name: str
    group: str                    # Momentum, Reversal, Risk, Quality, Volatility
    holding_days: int
    econ_rationale: str

    # Formula
    formula: str
    formula_code: str             # Python expression

    # Full-sample IC metrics
    pearson_ic: float
    spearman_ic: float
    icir: float
    hit_rate: float
    coverage: float
    persistence: float            # Signal autocorrelation (lag-1)

    # Statistical validation
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    t_stat: float
    t_pvalue: float
    permutation_pvalue: float
    reject_null: bool
    fdr_survives: bool            # Survived Benjamini-Hochberg correction

    # Regime performance
    regime_ics: dict = field(default_factory=dict)
    regime_robust: bool = False
    regime_stable: bool = False

    # Capacity & cost
    avg_adv_usd: float = 0.0
    max_capacity_usd: float = 0.0
    annual_turnover: float = 0.0
    annual_cost_bps: float = 0.0
    min_ic_to_survive: float = 0.0
    cost_viable: bool = False

    # Correlation
    max_corr_with: str = ""       # Factor name with highest |corr|
    max_corr_value: float = 0.0
    is_independent: bool = True   # Not flagged as redundant

    # Status
    promotion_decision: str = "PENDING"  # PROMOTE, REFINE, HOLD, ARCHIVE, REJECT
    last_updated: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Clean up NaN values for JSON serialization
        for key, val in d.items():
            if isinstance(val, float) and np.isnan(val):
                d[key] = None
        return d


def save_factor_record(
    record: FactorRecord,
    output_dir: Path | None = None,
) -> Path:
    """Save a single factor record as JSON."""
    if output_dir is None:
        output_dir = FACTOR_LIBRARY_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    record.last_updated = datetime.utcnow().isoformat() + "Z"
    filepath = output_dir / f"{record.factor_id}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, default=str)

    log.info("Saved factor record: %s → %s", record.factor_id, filepath)
    return filepath


def save_csv_index(
    records: list[FactorRecord],
    output_dir: Path | None = None,
) -> Path:
    """Save a CSV index of all factor records for rapid querying."""
    if output_dir is None:
        output_dir = FACTOR_LIBRARY_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        rows.append({
            "factor_id": r.factor_id,
            "factor_name": r.factor_name,
            "group": r.group,
            "pearson_ic": r.pearson_ic,
            "spearman_ic": r.spearman_ic,
            "icir": r.icir,
            "hit_rate": r.hit_rate,
            "reject_null": r.reject_null,
            "fdr_survives": r.fdr_survives,
            "regime_robust": r.regime_robust,
            "cost_viable": r.cost_viable,
            "is_independent": r.is_independent,
            "promotion_decision": r.promotion_decision,
            "max_capacity_usd": r.max_capacity_usd,
            "annual_cost_bps": r.annual_cost_bps,
        })

    df = pd.DataFrame(rows)
    filepath = output_dir / "factor_index.csv"
    df.to_csv(filepath, index=False)
    log.info("Saved CSV index: %d factors → %s", len(records), filepath)
    return filepath


def build_factor_library(
    factor_results: dict[str, dict],
    output_dir: Path | None = None,
) -> tuple[list[FactorRecord], Path]:
    """Build the complete institutional factor library from evaluation results.

    Args:
        factor_results: dict mapping factor_name → dict with all evaluation data.
            Expected keys: ic, rank_ic, icir, hit_rate, coverage, persistence,
            bootstrap_ci, stat_test, regime_ic, capacity, correlation, etc.
        output_dir: Override output directory (default: research/phase4/factor_library/).

    Returns:
        (list[FactorRecord], Path to CSV index).
    """
    if output_dir is None:
        output_dir = FACTOR_LIBRARY_DIR

    records: list[FactorRecord] = []

    # Factor metadata catalog (from extractor.py)
    from research.phase4.signals.extractor import FACTOR_CATALOG

    for factor_name, data in factor_results.items():
        meta = FACTOR_CATALOG.get(factor_name, {})

        record = FactorRecord(
            factor_id=factor_name,
            factor_name=factor_name,
            group=meta.get("group", data.get("group", "Unknown")),
            holding_days=meta.get("holding_days", 21),
            econ_rationale=meta.get("econ", data.get("econ", "")),
            formula=data.get("formula", ""),
            formula_code=data.get("formula_code", ""),
            # IC metrics
            pearson_ic=data.get("ic", 0.0),
            spearman_ic=data.get("rank_ic", 0.0),
            icir=data.get("icir", 0.0),
            hit_rate=data.get("hit_rate", 0.0),
            coverage=data.get("coverage", 0.0),
            persistence=data.get("persistence", 0.0),
            # Stats
            bootstrap_ci_lower=data.get("bootstrap_ci", (0, 0))[0],
            bootstrap_ci_upper=data.get("bootstrap_ci", (0, 0))[1],
            t_stat=data.get("t_stat", 0.0),
            t_pvalue=data.get("t_pvalue", 1.0),
            permutation_pvalue=data.get("permutation_pvalue", 1.0),
            reject_null=data.get("reject_null", False),
            fdr_survives=data.get("fdr_survives", False),
            # Regime
            regime_ics=data.get("regime_ics", {}),
            regime_robust=data.get("regime_robust", False),
            regime_stable=data.get("regime_stable", False),
            # Capacity
            avg_adv_usd=data.get("avg_adv_usd", 0.0),
            max_capacity_usd=data.get("max_capacity_usd", 0.0),
            annual_turnover=data.get("annual_turnover", 0.0),
            annual_cost_bps=data.get("annual_cost_bps", 0.0),
            min_ic_to_survive=data.get("min_ic", 0.0),
            cost_viable=data.get("cost_viable", False),
            # Correlation
            max_corr_with=data.get("max_corr_with", ""),
            max_corr_value=data.get("max_corr_value", 0.0),
            is_independent=data.get("is_independent", True),
            # Status
            promotion_decision=data.get("promotion_decision", "PENDING"),
        )
        records.append(record)

    # Save JSON files
    for record in records:
        save_factor_record(record, output_dir)

    # Save CSV index
    csv_path = save_csv_index(records, output_dir)

    log.info(
        "Factor library built: %d factors saved to %s",
        len(records), output_dir,
    )

    return records, csv_path
