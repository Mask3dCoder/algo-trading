"""Regime filter for pre-trade risk gating (native QuantCore).

A ``RegimeFilter`` maps a timestamp to a risk-on / risk-off boolean using a
precomputed, point-in-time macro series. Two native constructions are
supported:

* VIX term-structure slope: ``VIX_3M - VIX_Spot``. Contango (slope > 0) =>
  risk-on; backwardation (slope < 0) => risk-off (vol-crush defense).
* Yield-curve spread: ``10Y - 2Y`` (FRED ``T10Y2Y``). Inverted (< 0) =>
  risk-off; normal (> 0) => risk-on.

The series is supplied pre-loaded -- either from ``quantcore.alpha.fred_signals``
against real FRED data, or a synthetic series for offline harness runs. By
construction the filter only ever reads the last value at or before ``as_of``,
so it preserves the PIT boundary enforced by ``BacktestEngine``.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


class RegimeFilter:
    """Point-in-time risk-on / risk-off gate from a macro spread series."""

    def __init__(
        self,
        mode: str,
        series: pd.Series,
        threshold: float = 0.0,
        risk_off_when: str = "below",
    ) -> None:
        self.mode = mode
        self._series = series.sort_index()
        self.threshold = threshold
        self.risk_off_when = risk_off_when

    @classmethod
    def from_yield_curve(cls, series: pd.Series, threshold: float = 0.0) -> RegimeFilter:
        return cls("yield_curve", series, threshold, "below")

    @classmethod
    def from_vix_slope(cls, series: pd.Series, threshold: float = 0.0) -> RegimeFilter:
        return cls("vix_slope", series, threshold, "below")

    def is_risk_on(self, as_of: datetime) -> bool:
        # PIT: only the last observable value at or before as_of is readable.
        hist = self._series.loc[:pd.Timestamp(as_of)]
        if hist.empty:
            return True  # no macro data yet -> default risk-on (do no harm)
        value = float(hist.iloc[-1])
        if self.risk_off_when == "below":
            return value >= self.threshold
        return value <= self.threshold

    def label(self) -> str:
        return f"{self.mode}(thr={self.threshold})"
