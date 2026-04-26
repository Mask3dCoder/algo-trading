"""Experiment tracking and comparison for strategy research."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class ExperimentConfig:
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    strategy_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    data_range: tuple[str, str] | None = None
    benchmark: str = "SPY"


@dataclass
class ExperimentResult:
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    trades_count: int = 0
    signals_count: int = 0
    runtime_seconds: float = 0.0
    error_message: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    git_commit: str | None = None


class ExperimentTracker:
    """Track, compare, and query strategy research experiments."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._experiments: dict[str, ExperimentResult] = {}
        self._storage_path = storage_path
        if storage_path and os.path.exists(storage_path):
            self._load()

    def create_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        exp = ExperimentResult(config=config)
        self._experiments[exp.experiment_id] = exp
        self._save()
        return exp

    def start_experiment(self, experiment_id: str) -> None:
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.RUNNING
            self._save()

    def complete_experiment(
        self,
        experiment_id: str,
        metrics: dict[str, float],
        trades_count: int = 0,
        signals_count: int = 0,
        runtime_seconds: float = 0.0,
        artifact_paths: list[str] | None = None,
    ) -> ExperimentResult | None:
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.COMPLETED
            exp.completed_at = datetime.now(timezone.utc)
            exp.metrics = metrics
            exp.trades_count = trades_count
            exp.signals_count = signals_count
            exp.runtime_seconds = runtime_seconds
            exp.artifact_paths = artifact_paths or []
            self._save()
        return exp

    def fail_experiment(self, experiment_id: str, error: str) -> None:
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.status = ExperimentStatus.FAILED
            exp.error_message = error
            self._save()

    def get_experiment(self, experiment_id: str) -> ExperimentResult | None:
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        strategy_name: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ExperimentResult]:
        results = list(self._experiments.values())
        if status:
            results = [e for e in results if e.status == status]
        if strategy_name:
            results = [e for e in results if e.config.strategy_name == strategy_name]
        if tags:
            results = [e for e in results if any(t in e.config.tags for t in tags)]
        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def compare_experiments(
        self, experiment_ids: list[str], metrics: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for eid in experiment_ids:
            exp = self._experiments.get(eid)
            if exp:
                metric_dict = exp.metrics
                if metrics:
                    metric_dict = {m: exp.metrics.get(m, float("nan")) for m in metrics}
                result[eid] = metric_dict
        return result

    def best_experiment(
        self, metric: str = "sharpe", higher_is_better: bool = True,
    ) -> ExperimentResult | None:
        completed = [e for e in self._experiments.values() if e.status == ExperimentStatus.COMPLETED]
        if not completed:
            return None
        sign = 1 if higher_is_better else -1
        return max(completed, key=lambda e: sign * e.metrics.get(metric, float("-inf")))

    def delete_experiment(self, experiment_id: str) -> None:
        self._experiments.pop(experiment_id, None)
        self._save()

    def export_csv(self, filepath: str, experiments: list[ExperimentResult] | None = None) -> None:
        import csv
        exps = experiments or list(self._experiments.values())
        if not exps:
            return
        all_metric_keys = set()
        for e in exps:
            all_metric_keys.update(e.metrics.keys())
        metric_keys = sorted(all_metric_keys)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["experiment_id", "name", "strategy", "status", "created_at"] + metric_keys
            writer.writerow(header)
            for e in exps:
                row = [e.experiment_id, e.config.name, e.config.strategy_name, e.status.value, e.created_at.isoformat()]
                row.extend(e.metrics.get(k, "") for k in metric_keys)
                writer.writerow(row)

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        sharpes = []
        for e in self._experiments.values():
            by_status[e.status.value] = by_status.get(e.status.value, 0) + 1
            if "sharpe" in e.metrics:
                sharpes.append(e.metrics["sharpe"])
        return {
            "total": len(self._experiments),
            "by_status": by_status,
            "best_sharpe": max(sharpes) if sharpes else 0.0,
            "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        }

    def _save(self) -> None:
        if not self._storage_path:
            return
        data = {}
        for eid, exp in self._experiments.items():
            data[eid] = {
                "experiment_id": exp.experiment_id,
                "config": {
                    "name": exp.config.name,
                    "description": exp.config.description,
                    "tags": exp.config.tags,
                    "strategy_name": exp.config.strategy_name,
                    "parameters": exp.config.parameters,
                    "data_range": list(exp.config.data_range) if exp.config.data_range else None,
                    "benchmark": exp.config.benchmark,
                },
                "status": exp.status.value,
                "created_at": exp.created_at.isoformat(),
                "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
                "metrics": exp.metrics,
                "trades_count": exp.trades_count,
                "signals_count": exp.signals_count,
                "runtime_seconds": exp.runtime_seconds,
                "error_message": exp.error_message,
                "artifact_paths": exp.artifact_paths,
                "git_commit": exp.git_commit,
            }
        parent = os.path.dirname(self._storage_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        if not self._storage_path or not os.path.exists(self._storage_path):
            return
        with open(self._storage_path) as f:
            raw = f.read().strip()
            if not raw:
                return
            data = json.loads(raw)
        for eid, d in data.items():
            cfg_dict = d["config"]
            config = ExperimentConfig(
                name=cfg_dict.get("name", ""),
                description=cfg_dict.get("description", ""),
                tags=cfg_dict.get("tags", []),
                strategy_name=cfg_dict.get("strategy_name", ""),
                parameters=cfg_dict.get("parameters", {}),
                data_range=tuple(cfg_dict["data_range"]) if cfg_dict.get("data_range") else None,
                benchmark=cfg_dict.get("benchmark", "SPY"),
            )
            exp = ExperimentResult(
                experiment_id=d["experiment_id"],
                config=config,
                status=ExperimentStatus(d.get("status", "draft")),
                created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(timezone.utc),
                completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
                metrics=d.get("metrics", {}),
                trades_count=d.get("trades_count", 0),
                signals_count=d.get("signals_count", 0),
                runtime_seconds=d.get("runtime_seconds", 0.0),
                error_message=d.get("error_message"),
                artifact_paths=d.get("artifact_paths", []),
                git_commit=d.get("git_commit"),
            )
            self._experiments[eid] = exp


def sensitivity_analysis(
    results: list[ExperimentResult], parameter_name: str, metric: str = "sharpe",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract parameter vs metric values across experiments."""
    params = []
    metrics = []
    for e in results:
        if parameter_name in e.config.parameters and metric in e.metrics:
            params.append(float(e.config.parameters[parameter_name]))
            metrics.append(e.metrics[metric])
    return np.array(params, dtype=float), np.array(metrics, dtype=float)


def find_best_parameters(
    results: list[ExperimentResult], metric: str = "sharpe", higher_is_better: bool = True,
) -> dict[str, Any]:
    """Return the parameters of the best-performing experiment by a metric."""
    if not results:
        return {}
    sign = 1 if higher_is_better else -1
    best = max(results, key=lambda e: sign * e.metrics.get(metric, float("-inf")))
    return dict(best.config.parameters)
