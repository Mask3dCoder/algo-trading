"""Hurst exponent estimation via R/S analysis and DFA."""

from __future__ import annotations

from numpy.typing import NDArray


def hurst_rs(series: NDArray) -> float:
    """Estimate Hurst exponent via R/S (rescaled range) analysis.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending/persistent
    """
    from quantcore.math_models.stochastic.fractional_bm import estimate_hurst_rs as _hrs
    return _hrs(series)


def hurst_dfa(series: NDArray) -> float:
    """Estimate Hurst exponent via Detrended Fluctuation Analysis."""
    from quantcore.math_models.stochastic.fractional_bm import estimate_hurst_dfa as _hdfa
    return _hdfa(series)
