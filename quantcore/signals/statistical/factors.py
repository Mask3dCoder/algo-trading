"""Factor signals: value, quality, growth, sentiment, composite, IC analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Value Factor Signals
# ---------------------------------------------------------------------------


def _cross_sectional_zscore(values: NDArray) -> NDArray:
    """Cross-sectional z-score: normalize across assets at a point in time."""
    valid = values[~np.isnan(values)]
    if len(valid) < 5:
        return np.full_like(values, np.nan, dtype=float)
    m = np.mean(valid)
    s = np.std(valid, ddof=1)
    if s == 0:
        return np.full_like(values, 0.0, dtype=float)
    result = np.where(~np.isnan(values), (values - m) / s, np.nan)
    return np.clip(result / 3.0, -1.0, 1.0)


def pe_signal(pe_ratios: NDArray) -> NDArray:
    """P/E signal: low P/E = value (positive signal), high P/E = expensive (negative).

    Inverts the raw P/E since lower is better for value.
    """
    # Invert P/E to E/P (earnings yield) — higher is better
    with np.errstate(divide="ignore", invalid="ignore"):
        earnings_yield = np.where(
            (pe_ratios > 0) & ~np.isnan(pe_ratios),
            1.0 / pe_ratios,
            np.nan,
        )
    return _cross_sectional_zscore(earnings_yield)


def pb_signal(pb_ratios: NDArray) -> NDArray:
    """P/B signal: low P/B = value. Inverts to B/P (book-to-market)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        bm_ratio = np.where(
            (pb_ratios > 0) & ~np.isnan(pb_ratios),
            1.0 / pb_ratios,
            np.nan,
        )
    return _cross_sectional_zscore(bm_ratio)


def ev_ebitda_signal(ev_ebitda: NDArray) -> NDArray:
    """EV/EBITDA signal: low EV/EBITDA = value. Inverts to EBITDA/EV."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ebitda_yield = np.where(
            (ev_ebitda > 0) & ~np.isnan(ev_ebitda),
            1.0 / ev_ebitda,
            np.nan,
        )
    return _cross_sectional_zscore(ebitda_yield)


def fcf_yield_signal(fcf_yield: NDArray) -> NDArray:
    """FCF Yield signal: higher FCF yield = more value."""
    return _cross_sectional_zscore(fcf_yield)


# ---------------------------------------------------------------------------
# Quality Factor Signals
# ---------------------------------------------------------------------------


def roe_signal(roe: NDArray) -> NDArray:
    """ROE signal: higher ROE = better quality."""
    return _cross_sectional_zscore(roe)


def roic_signal(roic: NDArray) -> NDArray:
    """ROIC signal: higher ROIC = better capital allocation."""
    return _cross_sectional_zscore(roic)


def gross_margin_signal(gross_margin: NDArray) -> NDArray:
    """Gross margin signal: higher margin = competitive advantage."""
    return _cross_sectional_zscore(gross_margin)


def accruals_signal(accruals: NDArray) -> NDArray:
    """Accruals signal: lower accruals = higher earnings quality.

    Inverts since lower accruals is better.
    """
    return _cross_sectional_zscore(-accruals)


# ---------------------------------------------------------------------------
# Growth Factor Signals
# ---------------------------------------------------------------------------


def revenue_growth_signal(revenue_growth: NDArray) -> NDArray:
    """Revenue growth signal: higher growth = better."""
    return _cross_sectional_zscore(revenue_growth)


def earnings_growth_signal(earnings_growth: NDArray) -> NDArray:
    """Earnings growth signal: higher growth = better."""
    return _cross_sectional_zscore(earnings_growth)


# ---------------------------------------------------------------------------
# Sentiment Factor Signals
# ---------------------------------------------------------------------------


def analyst_revision_signal(revisions: NDArray) -> NDArray:
    """Analyst revision signal: positive revisions = bullish.

    revisions = (% upgrades - % downgrades) or net change in consensus.
    """
    return _cross_sectional_zscore(revisions)


def short_interest_signal(short_interest: NDArray) -> NDArray:
    """Short interest signal: high SI = bearish consensus.

    Inverts since lower SI is better.
    """
    return _cross_sectional_zscore(-short_interest)


# ---------------------------------------------------------------------------
# Composite Factor
# ---------------------------------------------------------------------------


def composite_factor(
    signals: list[NDArray],
    weights: list[float] | None = None,
) -> NDArray:
    """Combine multiple factor signals into a composite.

    Args:
        signals: list of factor signal arrays, all same shape.
        weights: optional weights (default: equal weight).

    Returns:
        Combined signal array in [-1, +1].
    """
    if not signals:
        return np.array([], dtype=float)

    n = len(signals[0])
    if weights is None:
        weights_array = np.ones(len(signals)) / len(signals)
    else:
        weights_array = np.array(weights, dtype=float)
        weights_array /= weights_array.sum()

    result = np.zeros(n, dtype=float)
    weight_sum = np.zeros(n, dtype=float)

    for sig, w in zip(signals, weights_array):
        valid = ~np.isnan(sig)
        result[valid] += sig[valid] * w
        weight_sum[valid] += w

    # Normalize by actual weight used (in case of NaN)
    mask = weight_sum > 0
    result[mask] /= weight_sum[mask]
    result[~mask] = np.nan

    return np.clip(result, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Factor IC (Information Coefficient)
# ---------------------------------------------------------------------------


def factor_ic(
    factor_scores: NDArray,
    forward_returns: NDArray,
    method: str = "spearman",
) -> tuple[float, float]:
    """Compute Information Coefficient of a factor.

    Args:
        factor_scores: (n_assets,) cross-sectional factor scores.
        forward_returns: (n_assets,) forward returns.
        method: "spearman" (rank IC) or "pearson".

    Returns:
        (ic, t_stat): IC value and approximate t-statistic.
    """
    valid = ~(np.isnan(factor_scores) | np.isnan(forward_returns))
    scores = factor_scores[valid]
    returns = forward_returns[valid]

    if len(scores) < 10:
        return 0.0, 0.0

    if method == "spearman":
        # Rank correlation
        rank_scores = _rank(scores)
        rank_returns = _rank(returns)
        ic = np.corrcoef(rank_scores, rank_returns)[0, 1]
    else:
        ic = np.corrcoef(scores, returns)[0, 1]

    if np.isnan(ic):
        return 0.0, 0.0

    # t-stat = IC * sqrt(N-2) / sqrt(1 - IC^2)
    n = len(scores)
    denom = np.sqrt(1.0 - ic ** 2) if abs(ic) < 1.0 else 1e-10
    t_stat = ic * np.sqrt(n - 2) / denom

    return float(ic), float(t_stat)


def _rank(x: NDArray) -> NDArray:
    """Assign ranks to values (1-based, average for ties)."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks


def rolling_factor_ic(
    factor_scores: NDArray,
    forward_returns: NDArray,
    lookback: int = 252,
    method: str = "spearman",
) -> NDArray:
    """Rolling IC time series for a factor.

    Args:
        factor_scores: (n_periods, n_assets) factor scores matrix.
        forward_returns: (n_periods, n_assets) forward returns matrix.
        lookback: rolling window for IC averaging.
        method: "spearman" or "pearson".

    Returns:
        (n_periods,) IC series.
    """
    n_periods = factor_scores.shape[0]
    ic_series = np.full(n_periods, np.nan, dtype=float)

    for t in range(n_periods):
        ic_val, _ = factor_ic(factor_scores[t, :], forward_returns[t, :], method)
        ic_series[t] = ic_val

    # Rolling average
    result = np.full(n_periods, np.nan, dtype=float)
    for i in range(lookback - 1, n_periods):
        window = ic_series[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            result[i] = np.mean(valid)

    return result


# ---------------------------------------------------------------------------
# Quintile Spread
# ---------------------------------------------------------------------------


def quintile_spread(
    factor_scores: NDArray,
    forward_returns: NDArray,
    n_quantiles: int = 5,
) -> tuple[float, list[float]]:
    """Compute quintile portfolio returns and top-bottom spread.

    Args:
        factor_scores: (n_assets,) cross-sectional scores.
        forward_returns: (n_assets,) forward returns.
        n_quantiles: number of quantile buckets.

    Returns:
        (spread, quintile_returns): top-bottom spread, list of quintile means.
    """
    valid = ~(np.isnan(factor_scores) | np.isnan(forward_returns))
    scores = factor_scores[valid]
    returns = forward_returns[valid]

    if len(scores) < n_quantiles * 3:
        return 0.0, [0.0] * n_quantiles

    # Sort by factor score
    order = np.argsort(scores)
    sorted_returns = returns[order]

    # Split into quantiles
    quantile_size = len(sorted_returns) // n_quantiles
    quintile_returns: list[float] = []

    for q in range(n_quantiles):
        start = q * quantile_size
        end = start + quantile_size if q < n_quantiles - 1 else len(sorted_returns)
        quintile_returns.append(float(np.mean(sorted_returns[start:end])))

    spread = quintile_returns[-1] - quintile_returns[0]
    return spread, quintile_returns
