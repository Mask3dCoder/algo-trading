"""Experiment tracking — persistent CSV-based log.

Every run that enters the framework is given a unique run ID and written
to ``experiments.csv`` in the project directory, preventing "p-hacking
by forgetting".
"""

from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CSV = Path("quantcore/research/experiments.csv")


@dataclass
class Experiment:
    """A single experiment record."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    strategy_name: str = "residual_momentum"
    parameters: dict[str, Any] = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""
    oos_sharpe: float = 0.0
    deflated_sharpe_pvalue: float = 1.0
    ci_90_lower: float = float("nan")
    status: str = "pending"
    notes: str = ""


def log_experiment(
    experiment: Experiment,
    csv_path: Path | str | None = None,
) -> str:
    """Append ``experiment`` to the experiment log CSV.

    Returns the experiment's unique ID.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment_id": experiment.experiment_id,
        "run_at": experiment.run_at,
        "strategy_name": experiment.strategy_name,
        "parameters": str(experiment.parameters),
        "start_date": experiment.start_date,
        "end_date": experiment.end_date,
        "oos_sharpe": experiment.oos_sharpe,
        "deflated_sharpe_pvalue": experiment.deflated_sharpe_pvalue,
        "ci_90_lower": experiment.ci_90_lower,
        "status": experiment.status,
        "notes": experiment.notes,
    }

    fieldnames = list(row.keys())
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return experiment.experiment_id


def load_experiments(csv_path: Path | str | None = None) -> pd.DataFrame:  # type: ignore[type-arg]
    """Load the full experiment CSV as a DataFrame."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.exists():
        return pd.DataFrame(columns=[
            "experiment_id", "run_at", "strategy_name", "parameters",
            "start_date", "end_date", "oos_sharpe", "deflated_sharpe_pvalue",
            "ci_90_lower", "status", "notes",
        ])
    return pd.read_csv(path)
