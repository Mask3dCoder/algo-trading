"""Backtest bias detection: lookahead, survivorship, overfitting checks."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SurvivorReport:
    """Report on survivorship bias in backtest data."""
    total_symbols: int
    current_symbols: int
    delisted_included: int
    survivorship_bias_detected: bool


@dataclass
class OverfittingReport:
    """Report on overfitting risk."""
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    degradation_ratio: float
    overfitting_score: float
    is_overfitted: bool


def check_lookahead(code: str) -> list[str]:
    """Static analysis to detect lookahead bias patterns in strategy code.

    Scans for:
    - shift(-1) or shift(-n) — future data access
    - lookahead attribute access patterns
    - future column references
    """
    warnings: list[str] = []

    # Pattern 1: .shift(-1) or .shift(-n)
    shift_pattern = re.compile(r"\.shift\(\s*-\s*\d+\s*\)")
    matches = shift_pattern.findall(code)
    if matches:
        warnings.append(f"Future data access detected: {matches}. shift(-n) references future rows.")

    # Pattern 2: .iloc[n:] style forward references in strategy logic
    # Pattern 3: direct future column access like df['future_col']
    future_col_pattern = re.compile(r"['\"](?:future|next|forward|target|ahead)['\"]", re.IGNORECASE)
    future_matches = future_col_pattern.findall(code)
    if future_matches:
        warnings.append(f"Potential future column reference: {future_matches}")

    # AST-based check: look for comparisons with future index
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.attr, str) and "shift" in node.func.attr:
                        for arg in node.args:
                            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                                warnings.append("Negative shift detected at AST level — future data access")
    except SyntaxError:
        pass

    return warnings


def check_survivorship(
    current_universe: list[str],
    historical_universe: list[str],
    delisted_symbols: list[str],
) -> SurvivorReport:
    """Check if backtest universe includes delisted stocks (avoiding survivorship bias)."""
    current_set = set(current_universe)
    historical_set = set(historical_universe)
    delisted_set = set(delisted_symbols)

    included_delisted = len(current_set & delisted_set)
    missing_delisted = delisted_set - current_set

    return SurvivorReport(
        total_symbols=len(historical_set),
        current_symbols=len(current_set),
        delisted_included=included_delisted,
        survivorship_bias_detected=len(missing_delisted) > 0,
    )


def overfitting_score(
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
) -> OverfittingReport:
    """Compute overfitting score as the ratio of IS-to-OOS Sharpe degradation.

    A degradation_ratio > 2.0 suggests overfitting.
    """
    if in_sample_sharpe == 0:
        degradation = 0.0
    else:
        degradation = in_sample_sharpe / out_of_sample_sharpe if out_of_sample_sharpe != 0 else float("inf")

    score = abs(in_sample_sharpe - out_of_sample_sharpe) / max(abs(in_sample_sharpe), 1e-8)

    return OverfittingReport(
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=out_of_sample_sharpe,
        degradation_ratio=degradation,
        overfitting_score=score,
        is_overfitted=degradation > 2.0,
    )


def minimum_backtest_length(
    sharpe: float,
    n_trials: int = 1,
    target_pvalue: float = 0.05,
) -> float:
    """Minimum backtest length (in years) for statistical significance.

    Based on: Bailey & Lopez de Prado (2014) — Minimum Backtest Length.
    MinBTL = 1 + (2 * ln(n_trials) - ln(target_pvalue)) / (sharpe^2 * (1 - skew*kurt))
    Simplified version: MinBTL ≈ (2.8 * ln(n_trials) + 1) / sharpe^2
    """
    if sharpe <= 0:
        return float("inf")
    return max((2.8 * np.log(max(n_trials, 1)) + 1.0) / sharpe**2, 0.5)
