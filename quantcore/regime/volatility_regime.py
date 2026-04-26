"""Volatility regime classification: VIX, GARCH, realized vol percentile."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


class VolRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"
    EXTREME = "extreme"


@dataclass
class VolRegimeResult:
    regime: VolRegime
    regime_label: str
    current_vol: float
    threshold_low: float
    threshold_elevated: float
    threshold_crisis: float
    method: str
    percentile: float | None = None


# ---------------------------------------------------------------------------
# VIX-based regime
# ---------------------------------------------------------------------------


def vix_regime(vix_level: float) -> VolRegimeResult:
    """Classify volatility regime based on VIX level.

    <15   → Low vol
    15-25 → Normal
    25-35 → Elevated
    >35   → Crisis
    >60   → Extreme
    """
    if vix_level > 60:
        regime = VolRegime.EXTREME
    elif vix_level > 35:
        regime = VolRegime.CRISIS
    elif vix_level > 25:
        regime = VolRegime.ELEVATED
    elif vix_level >= 15:
        regime = VolRegime.NORMAL
    else:
        regime = VolRegime.LOW

    return VolRegimeResult(
        regime=regime,
        regime_label=regime.value,
        current_vol=vix_level,
        threshold_low=15.0,
        threshold_elevated=25.0,
        threshold_crisis=35.0,
        method="vix",
    )


# ---------------------------------------------------------------------------
# GARCH regime (Markov-switching GARCH)
# ---------------------------------------------------------------------------


def garch_regime(
    returns: NDArray[np.floating],
    lookback: int = 500,
) -> VolRegimeResult:
    """Classify volatility regime using GARCH conditional vol.

    Fits GARCH(1,1) and compares current conditional vol to its historical
    percentiles.

    Args:
        returns: Return series.
        lookback: Lookback window for percentile estimation.

    Returns:
        VolRegimeResult with regime classification.
    """
    try:
        from quantcore.math_models.statistics.garch import fit_garch
    except ImportError:
        return VolRegimeResult(
            regime=VolRegime.NORMAL,
            regime_label="normal",
            current_vol=0.0,
            threshold_low=0.15,
            threshold_elevated=0.25,
            threshold_crisis=0.35,
            method="garch_fallback",
        )

    r = np.asarray(returns, dtype=float).flatten()
    if len(r) < lookback:
        lookback = max(50, len(r))

    try:
        garch_result = fit_garch(r[-lookback:], p=1, q=1)
    except Exception:
        # Fallback: use historical vol percentile
        return vol_percentile_regime(r, window=lookback)

    cond_vol = garch_result.conditional_volatility
    current_vol = cond_vol[-1] * np.sqrt(252)
    hist_vol = cond_vol * np.sqrt(252)

    pct = sp_stats.percentileofscore(hist_vol, current_vol)

    if pct > 95:
        regime = VolRegime.EXTREME
    elif pct > 80:
        regime = VolRegime.CRISIS
    elif pct > 60:
        regime = VolRegime.ELEVATED
    elif pct > 20:
        regime = VolRegime.NORMAL
    else:
        regime = VolRegime.LOW

    return VolRegimeResult(
        regime=regime,
        regime_label=regime.value,
        current_vol=current_vol,
        threshold_low=float(np.percentile(hist_vol, 20)),
        threshold_elevated=float(np.percentile(hist_vol, 60)),
        threshold_crisis=float(np.percentile(hist_vol, 80)),
        method="garch",
        percentile=float(pct),
    )


# ---------------------------------------------------------------------------
# Realized vol percentile regime
# ---------------------------------------------------------------------------


def vol_percentile_regime(
    returns: NDArray[np.floating],
    window: int = 252,
) -> VolRegimeResult:
    """Classify regime by rolling realized vol percentile.

    Current realized vol vs its distribution over the lookback window.

    Args:
        returns: Return series.
        window: Lookback window.

    Returns:
        VolRegimeResult.
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < 21:
        return VolRegimeResult(
            regime=VolRegime.NORMAL,
            regime_label="normal",
            current_vol=float(np.std(r)) * np.sqrt(252),
            threshold_low=0.15,
            threshold_elevated=0.25,
            threshold_crisis=0.35,
            method="percentile",
        )

    # Current realized vol
    current_vol = float(np.std(r[-21:]) * np.sqrt(252))

    # Rolling realized vol series
    n_roll = min(window, len(r) - 21)
    rolling_vols = np.zeros(n_roll)
    for i in range(n_roll):
        start = max(0, len(r) - window + i - 20)
        end = max(21, len(r) - window + i)
        rolling_vols[i] = float(np.std(r[start:end]) * np.sqrt(252))

    pct = float(sp_stats.percentileofscore(rolling_vols, current_vol))

    if pct > 95:
        regime = VolRegime.EXTREME
    elif pct > 80:
        regime = VolRegime.CRISIS
    elif pct > 60:
        regime = VolRegime.ELEVATED
    elif pct > 20:
        regime = VolRegime.NORMAL
    else:
        regime = VolRegime.LOW

    return VolRegimeResult(
        regime=regime,
        regime_label=regime.value,
        current_vol=current_vol,
        threshold_low=float(np.percentile(rolling_vols, 20)) if len(rolling_vols) > 0 else 0.15,
        threshold_elevated=float(np.percentile(rolling_vols, 60)) if len(rolling_vols) > 0 else 0.25,
        threshold_crisis=float(np.percentile(rolling_vols, 80)) if len(rolling_vols) > 0 else 0.35,
        method="percentile",
        percentile=pct,
    )

