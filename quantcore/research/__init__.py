"""Alpha research framework for institutional-grade strategy validation."""

from quantcore.research.alpha_research import (
    DeflatedSharpeResult,
    deflated_sharpe,
    fdr_correction,
    minimum_backtest_length,
    overfitting_probability,
    probabilistic_sharpe_ratio,
)
from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio
from quantcore.research.experiment_tracker import Experiment, load_experiments, log_experiment
from quantcore.research.walkforward import WalkForwardBacktester, WalkForwardResult

__all__ = [
    "DeflatedSharpeResult",
    "deflated_sharpe",
    "fdr_correction",
    "minimum_backtest_length",
    "overfitting_probability",
    "probabilistic_sharpe_ratio",
    "bootstrap_confidence_interval",
    "deflated_sharpe_ratio",
    "Experiment",
    "log_experiment",
    "load_experiments",
    "WalkForwardBacktester",
    "WalkForwardResult",
]
