"""Risk-adjusted performance metrics.

All functions accept (returns: NDArray, ...) where returns are daily.
Values are annualized where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Core ratio metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: NDArray[np.floating],
    rf: float = 0.0,
    annualize: bool = True,
) -> float:
    """Sharpe ratio: (mean(R) - rf) / std(R), annualized."""
    r = np.asarray(returns, dtype=float).flatten()
    r_excess = r - rf / (TRADING_DAYS if annualize else 1.0)
    if len(r_excess) < 2 or np.std(r_excess) == 0:
        return 0.0
    sr = float(np.mean(r_excess) / np.std(r_excess, ddof=1))
    return sr * np.sqrt(TRADING_DAYS) if annualize else sr


def sortino_ratio(
    returns: NDArray[np.floating],
    mar: float = 0.0,
    rf: float | None = None,
    annualize: bool = True,
) -> float:
    """Sortino ratio: (mean(R) - MAR) / downside_std.

    If rf is provided, use rf as MAR. Otherwise use explicit MAR.
    """
    r = np.asarray(returns, dtype=float).flatten()
    target = (rf / TRADING_DAYS) if rf is not None else (mar / TRADING_DAYS if annualize else mar)

    r_excess_daily = r - target
    downside = r_excess_daily[r_excess_daily < 0]
    if len(downside) < 2:
        return 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0

    sr = float(np.mean(r_excess_daily) / downside_std)
    return sr * np.sqrt(TRADING_DAYS) if annualize else sr


def calmar_ratio(
    returns: NDArray[np.floating],
) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    r = np.asarray(returns, dtype=float).flatten()
    ann_ret = float(np.mean(r)) * TRADING_DAYS
    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float(np.abs((cumulative - running_max).min()))
    if max_dd == 0:
        return 0.0
    return ann_ret / max_dd


def omega_ratio(
    returns: NDArray[np.floating],
    threshold: float = 0.0,
) -> float:
    """Omega ratio: E[max(R - L, 0)] / E[max(L - R, 0)].

    Higher is better. Measures reward vs risk relative to a threshold L.
    """
    r = np.asarray(returns, dtype=float).flatten()
    upside = np.maximum(r - threshold, 0)
    downside = np.maximum(threshold - r, 0)
    up_sum = float(np.sum(upside))
    down_sum = float(np.sum(downside))
    if down_sum == 0:
        return float("inf") if up_sum > 0 else 1.0
    return up_sum / down_sum


def information_ratio(
    returns: NDArray[np.floating],
    benchmark_returns: NDArray[np.floating],
    annualize: bool = True,
) -> float:
    """Information Ratio: mean(R - B) / std(R - B)."""
    r = np.asarray(returns, dtype=float).flatten()
    b = np.asarray(benchmark_returns, dtype=float).flatten()
    n = min(len(r), len(b))
    active = r[:n] - b[:n]
    if len(active) < 2 or np.std(active) == 0:
        return 0.0
    ir = float(np.mean(active) / np.std(active, ddof=1))
    return ir * np.sqrt(TRADING_DAYS) if annualize else ir


def treynor_ratio(
    returns: NDArray[np.floating],
    market_returns: NDArray[np.floating],
    rf: float = 0.0,
    annualize: bool = True,
) -> float:
    """Treynor ratio: (mean(R) - rf) / beta."""
    r = np.asarray(returns, dtype=float).flatten()
    m = np.asarray(market_returns, dtype=float).flatten()
    n = min(len(r), len(m))
    r = r[:n]
    m = m[:n]

    r_excess = r - rf / (TRADING_DAYS if annualize else 1.0)
    m_excess = m - rf / (TRADING_DAYS if annualize else 1.0)

    cov = np.cov(r_excess, m_excess)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

    if beta == 0:
        return 0.0

    tr = float(np.mean(r_excess) / beta)
    return tr * TRADING_DAYS if annualize else tr


def jensens_alpha(
    returns: NDArray[np.floating],
    market_returns: NDArray[np.floating],
    rf: float = 0.0,
    annualize: bool = True,
) -> float:
    """Jensen's Alpha: R_p - [rf + beta * (R_m - rf)]."""
    r = np.asarray(returns, dtype=float).flatten()
    m = np.asarray(market_returns, dtype=float).flatten()
    n = min(len(r), len(m))
    r = r[:n]
    m = m[:n]

    rf_daily = rf / TRADING_DAYS
    r_excess = r - rf_daily
    m_excess = m - rf_daily

    cov = np.cov(r_excess, m_excess)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

    alpha_daily = float(np.mean(r_excess) - beta * np.mean(m_excess))
    return alpha_daily * TRADING_DAYS if annualize else alpha_daily


def m_squared(
    returns: NDArray[np.floating],
    market_returns: NDArray[np.floating],
    rf: float = 0.0,
) -> float:
    """M² (Modigliani): risk-adjusted return vs market.

    M² = rf + (R_p - rf) * (σ_m / σ_p)
    """
    r = np.asarray(returns, dtype=float).flatten()
    m = np.asarray(market_returns, dtype=float).flatten()
    n = min(len(r), len(m))

    r_ann = float(np.mean(r[:n])) * TRADING_DAYS
    m_ann = float(np.mean(m[:n])) * TRADING_DAYS
    sigma_p = float(np.std(r[:n], ddof=1)) * np.sqrt(TRADING_DAYS)
    sigma_m = float(np.std(m[:n], ddof=1)) * np.sqrt(TRADING_DAYS)

    if sigma_p == 0:
        return rf

    return rf + (r_ann - rf) * (sigma_m / sigma_p)


def sterling_ratio(
    returns: NDArray[np.floating],
) -> float:
    """Sterling ratio: annualized return / (max drawdown - 10%)."""
    r = np.asarray(returns, dtype=float).flatten()
    ann_ret = float(np.mean(r)) * TRADING_DAYS
    cumulative = np.cumsum(r)
    max_dd = float(np.abs((cumulative - np.maximum.accumulate(cumulative)).min()))
    denominator = max_dd + 0.10
    if denominator == 0:
        return 0.0
    return ann_ret / denominator


# ---------------------------------------------------------------------------
# Pain / Ulcer
# ---------------------------------------------------------------------------


def ulcer_index(returns: NDArray[np.floating]) -> float:
    """Ulcer Index: sqrt(mean(squared_drawdowns))."""
    r = np.asarray(returns, dtype=float).flatten()
    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)
    # Avoid division by zero
    drawdowns = np.where(running_max > 0, (cumulative - running_max) / running_max, 0.0)
    squared_drawdowns = drawdowns**2
    return float(np.sqrt(np.mean(squared_drawdowns)))


def pain_index(returns: NDArray[np.floating]) -> float:
    """Pain Index: mean(abs(drawdowns))."""
    r = np.asarray(returns, dtype=float).flatten()
    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = np.where(running_max > 0, np.abs(cumulative - running_max) / running_max, 0.0)
    return float(np.mean(drawdowns))


def pain_ratio(returns: NDArray[np.floating]) -> float:
    """Pain Ratio: annualized return / Pain Index."""
    r = np.asarray(returns, dtype=float).flatten()
    ann_ret = float(np.mean(r)) * TRADING_DAYS
    pi = pain_index(r)
    if pi == 0:
        return 0.0
    return ann_ret / pi


def burke_ratio(returns: NDArray[np.floating]) -> float:
    """Burke Ratio: excess return / sqrt(sum(squared_drawdowns))."""
    r = np.asarray(returns, dtype=float).flatten()
    ann_ret = float(np.mean(r)) * TRADING_DAYS
    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    denominator = float(np.sqrt(np.sum(np.maximum(drawdowns, 0)**2)))
    if denominator == 0:
        return 0.0
    return ann_ret / denominator


# ---------------------------------------------------------------------------
# Distributional statistics
# ---------------------------------------------------------------------------


def skewness(returns: NDArray[np.floating]) -> float:
    """Sample skewness."""
    r = np.asarray(returns, dtype=float).flatten()
    if len(r) < 3 or np.std(r) == 0:
        return 0.0
    return float(sp_stats.skew(r))


def kurtosis(returns: NDArray[np.floating]) -> float:
    """Excess kurtosis (normal = 0)."""
    r = np.asarray(returns, dtype=float).flatten()
    if len(r) < 4 or np.std(r) == 0:
        return 0.0
    return float(sp_stats.kurtosis(r))


def jarque_bera(returns: NDArray[np.floating]) -> tuple[float, float]:
    """Jarque-Bera normality test. Returns (stat, p_value)."""
    r = np.asarray(returns, dtype=float).flatten()
    if len(r) < 8:
        return 0.0, 1.0
    jb_stat, jb_pv = sp_stats.jarque_bera(r)
    return float(jb_stat), float(jb_pv)


# ---------------------------------------------------------------------------
# Drawdown stats
# ---------------------------------------------------------------------------


@dataclass
class DrawdownStats:
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # in days
    avg_drawdown_duration: float


def drawdown_stats(returns: NDArray[np.floating]) -> DrawdownStats:
    """Compute comprehensive drawdown statistics."""
    r = np.asarray(returns, dtype=float).flatten()
    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)

    dd = cumulative - running_max
    max_dd = float(dd.min())
    avg_dd = float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0

    # Drawdown durations
    in_dd = False
    dd_start = 0
    durations: list[int] = []
    for i in range(len(dd)):
        if not in_dd and dd[i] < 0:
            in_dd = True
            dd_start = i
        elif in_dd and dd[i] >= 0:
            durations.append(i - dd_start)
            in_dd = False
    if in_dd:
        durations.append(len(dd) - dd_start)

    max_dur = max(durations) if durations else 0
    avg_dur = float(np.mean(durations)) if durations else 0.0

    return DrawdownStats(
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=max_dur,
        avg_drawdown_duration=avg_dur,
    )


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------


def rolling(
    returns: NDArray[np.floating],
    metric_fn: Callable[[NDArray[np.floating]], float],
    window: int = 252,
) -> NDArray[np.floating]:
    """Compute a metric over a rolling window.

    Returns array of shape (n - window + 1,).
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)
    if n < window:
        return np.array([metric_fn(r)])

    results = np.zeros(n - window + 1)
    for i in range(n - window + 1):
        results[i] = metric_fn(r[i : i + window])
    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    returns: NDArray[np.floating],
    metric_fn: Callable[[NDArray[np.floating]], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a risk metric.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)
    if n < 10:
        val = metric_fn(r)
        return val, val

    rng = np.random.default_rng(seed)
    boot_results = np.zeros(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_results[i] = metric_fn(r[idx])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_results, alpha * 100))
    upper = float(np.percentile(boot_results, (1.0 - alpha) * 100))
    return lower, upper


# ---------------------------------------------------------------------------
# All metrics convenience
# ---------------------------------------------------------------------------


def compute_all_metrics(
    returns: NDArray[np.floating],
    market_returns: NDArray[np.floating] | None = None,
    rf: float = 0.0,
) -> dict[str, float]:
    """Compute all risk-adjusted metrics at once.

    Returns:
        dict mapping metric name → value.
    """
    r = np.asarray(returns, dtype=float).flatten()
    m = np.asarray(market_returns, dtype=float).flatten() if market_returns is not None else None

    metrics: dict[str, float] = {}

    # Core ratios
    metrics["sharpe"] = sharpe_ratio(r, rf)
    metrics["sortino"] = sortino_ratio(r, mar=0)
    metrics["calmar"] = calmar_ratio(r)
    metrics["omega"] = omega_ratio(r)
    metrics["sterling"] = sterling_ratio(r)
    metrics["ulcer_index"] = ulcer_index(r)
    metrics["pain_index"] = pain_index(r)
    metrics["pain_ratio"] = pain_ratio(r)
    metrics["burke_ratio"] = burke_ratio(r)

    # Distribution
    metrics["skewness"] = skewness(r)
    metrics["kurtosis"] = kurtosis(r)
    jb_stat, jb_pv = jarque_bera(r)
    metrics["jarque_bera_stat"] = jb_stat
    metrics["jarque_bera_pvalue"] = jb_pv

    # Drawdowns
    dd = drawdown_stats(r)
    metrics["max_drawdown"] = dd.max_drawdown
    metrics["avg_drawdown"] = dd.avg_drawdown
    metrics["max_dd_duration"] = float(dd.max_drawdown_duration)
    metrics["avg_dd_duration"] = dd.avg_drawdown_duration

    # Benchmark-relative (if available)
    if m is not None:
        metrics["information_ratio"] = information_ratio(r, m)
        metrics["treynor"] = treynor_ratio(r, m, rf)
        metrics["jensens_alpha"] = jensens_alpha(r, m, rf)
        metrics["m_squared"] = m_squared(r, m, rf)

    return metrics
