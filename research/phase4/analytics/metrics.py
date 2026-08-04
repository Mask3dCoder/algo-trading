"""Phase 4 — Signal Analytics: IC Metrics (Stage 3).

Computes full predictive statistics for a (date × symbol) signal panel
against (date × symbol) forward returns.

All IC computations are done on the cross-sectional long format (stacked),
which is the institutional standard for panel factor research.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _stack_align(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Stack signal and forward returns into 1-D arrays, aligned by date and symbol."""
    # Stack to (date, symbol) long format
    s_stack = signal.stack(dropna=True).rename("signal")
    r_stack = fwd_returns.stack(dropna=True).rename("fwd_ret")
    merged = pd.concat([s_stack, r_stack], axis=1).dropna()
    dates = merged.index.get_level_values(0)
    return merged["signal"].values, merged["fwd_ret"].values, dates


def pearson_ic(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> float:
    """Overall Pearson IC across the full panel."""
    s, r, _ = _stack_align(signal, fwd_returns)
    if len(s) < 30:
        return float("nan")
    return float(np.corrcoef(s, r)[0, 1])


def spearman_ic(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> float:
    """Overall Spearman Rank IC across the full panel."""
    s, r, _ = _stack_align(signal, fwd_returns)
    if len(s) < 30:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(s, r)
    return float(rho)


def daily_ic_series(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """Compute IC each calendar date (cross-sectional IC time series).

    This is the institutional standard: IC is computed per-date across the
    cross-section of stocks, then averaged. This produces a time series
    of daily IC values that can be tested for persistence.
    """
    dates = signal.index.intersection(fwd_returns.index)
    ic_values: list[float] = []
    valid_dates: list[pd.Timestamp] = []

    for date in dates:
        s_row = signal.loc[date].dropna()
        r_row = fwd_returns.loc[date].dropna()
        common = s_row.index.intersection(r_row.index)
        if len(common) < 5:
            continue
        s_arr = s_row.loc[common].values
        r_arr = r_row.loc[common].values
        if method == "pearson":
            ic = float(np.corrcoef(s_arr, r_arr)[0, 1])
        else:
            ic, _ = scipy_stats.spearmanr(s_arr, r_arr)
            ic = float(ic)
        if not np.isnan(ic):
            ic_values.append(ic)
            valid_dates.append(date)

    return pd.Series(ic_values, index=valid_dates, name=f"ic_{method}")


def rolling_ic_series(
    ic_series: pd.Series,
    window: int = 126,
) -> pd.Series:
    """Rolling mean IC over a given window (in trading days)."""
    return ic_series.rolling(window, min_periods=max(20, window // 4)).mean()


def ic_information_ratio(ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC).

    Higher ICIR = more consistent signal. Institutional threshold: ICIR > 0.5.
    """
    mean = float(ic_series.mean())
    std  = float(ic_series.std())
    if std < 1e-10:
        return float("nan")
    return mean / std


def hit_rate(ic_series: pd.Series) -> float:
    """Fraction of dates where IC > 0."""
    if len(ic_series) == 0:
        return float("nan")
    return float((ic_series > 0).mean())


def ic_decay_curve(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    max_lag: int = 21,
) -> pd.Series:
    """IC as a function of forward holding period (lag 1 to max_lag days).

    Measures how quickly the signal's predictive power decays.
    A rapidly decaying curve suggests short holding period.
    A flat curve (same IC at lag 20 as lag 1) suggests the signal is slow.
    """
    ics: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        # Forward return at lag = compound return over lag days
        fwd_lag = (returns.shift(-lag).rolling(lag).sum())  # approximate
        # Actually: simply shift returns by lag for single-period IC
        fwd_lag = returns.shift(-lag)
        s, r, _ = _stack_align(signal, fwd_lag)
        if len(s) < 30:
            ics[lag] = float("nan")
        else:
            ics[lag] = float(np.corrcoef(s, r)[0, 1])
    return pd.Series(ics, name="ic_decay")


def quintile_return_analysis(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    n_quintiles: int = 5,
) -> dict:
    """Compute mean forward return by signal quintile.

    Monotonically increasing quintile returns (Q1 < Q2 < ... < Q5)
    indicate a consistently predictive signal across the distribution.
    """
    s, r, dates = _stack_align(signal, fwd_returns)
    if len(s) < n_quintiles * 10:
        return {"quintile_means": [], "spread": 0.0, "is_monotonic": False, "annualized_spread": 0.0}

    # Assign quintiles based on signal rank
    q_labels = pd.qcut(s, n_quintiles, labels=False, duplicates="drop")
    valid = q_labels is not None
    quintile_means = []
    for q in range(n_quintiles):
        mask = q_labels == q
        if mask.sum() > 0:
            quintile_means.append(float(np.mean(r[mask])))

    if len(quintile_means) < n_quintiles:
        return {"quintile_means": quintile_means, "spread": 0.0, "is_monotonic": False, "annualized_spread": 0.0}

    # Check monotonicity
    diffs = np.diff(quintile_means)
    is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    spread = quintile_means[-1] - quintile_means[0]

    return {
        "quintile_means": [round(x, 6) for x in quintile_means],
        "spread": round(spread, 6),
        "is_monotonic": is_monotonic,
        "annualized_spread": round(spread * 252, 4),
    }


def coverage(signal: pd.DataFrame) -> float:
    """Fraction of (date, symbol) cells with non-NaN signal values."""
    total = signal.size
    valid = signal.notna().sum().sum()
    return float(valid / total) if total > 0 else 0.0


def signal_persistence(signal: pd.DataFrame) -> float:
    """Mean autocorrelation of signal (lag-1), averaged across symbols.

    High persistence (>0.8) indicates low turnover. 
    Low persistence (<0.3) indicates frequent signal flipping (high cost).
    """
    autocorrs: list[float] = []
    for col in signal.columns:
        s = signal[col].dropna()
        if len(s) > 20:
            ac = float(s.autocorr(lag=1))
            if not np.isnan(ac):
                autocorrs.append(ac)
    return float(np.mean(autocorrs)) if autocorrs else float("nan")
