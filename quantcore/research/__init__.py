"""Research tools: alpha evaluation, experiment tracking, statistical testing."""

from quantcore.research.alpha_research import (
    DeflatedSharpeResult,
    deflated_sharpe,
    fdr_correction,
    haircut_sharpe,
    holm_correction,
    minimum_backtest_length,
    multiple_testing_summary,
    overfitting_probability,
    probabilistic_sharpe_ratio,
)
from quantcore.research.experiment_tracker import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTracker,
    find_best_parameters,
    sensitivity_analysis,
)

__all__ = [
    "DeflatedSharpeResult",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentTracker",
    "deflated_sharpe",
    "fdr_correction",
    "find_best_parameters",
    "haircut_sharpe",
    "holm_correction",
    "minimum_backtest_length",
    "multiple_testing_summary",
    "overfitting_probability",
    "probabilistic_sharpe_ratio",
    "sensitivity_analysis",
]
