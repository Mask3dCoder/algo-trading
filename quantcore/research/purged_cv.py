"""Purged, embargoed fold construction for time-series cross-validation.

Why this exists
---------------
FREEZE.md's unfreeze gate requires "walk-forward + purged k-fold" as a hard
gate. Walk-forward alone is not enough, and the difference is not academic.

A fold boundary on a return series is a lie about what the model knows. If a
training sample is dated ``t`` and its label is the return realised over
``[t, t + H]``, then a training sample dated one day before the OOS window
carries a label whose payoff is decided *inside* the OOS window. The fold looks
disjoint; the information is not. That leak is what makes a "too good" Sharpe
appear, and it is worst exactly where it matters most here: the decisive
experiment scores 12-month momentum, i.e. an H of roughly 252 trading days.

Two mechanisms close it:

* **Purge** — training samples whose label window would extend past the fold
  boundary are dropped, not merely moved.
* **Embargo** — an additional gap between the last usable training label and the
  first OOS observation, covering serial correlation and labelling frictions
  (López de Prado's purging and embargo).

With ``label_horizon_days = 0`` this reduces to the un-purged anchored
walk-forward, so existing callers keep their exact behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    """One anchored fold with its leak boundaries made explicit."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # effective (purged + clamped) end
    train_end_raw: pd.Timestamp      # unpurged boundary, kept for reporting
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    label_horizon_days: int = 0
    embargo_days: int = 0

    @property
    def purged_days(self) -> int:
        """Days removed because a training label would reach the OOS window.

        This is the CONFIGURED removal, not the calendar distance: a boundary
        landing on a weekend also shortens the realised gap, and crediting that
        to leakage purging would overstate how much was purged.
        """
        return int(self.label_horizon_days)

    @property
    def realised_gap_days(self) -> int:
        """Actual calendar distance between usable training end and OOS start.

        Includes purge, embargo, and any weekend/holiday clamp.
        """
        return int((self.oos_start - self.train_end).days)

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "train_end_raw": self.train_end_raw,
            "oos_start": self.oos_start,
            "oos_end": self.oos_end,
            "purged_days": self.purged_days,
            "realised_gap_days": self.realised_gap_days,
            "embargo_days": self.embargo_days,
        }

    def assert_no_label_leakage(self, label_horizon_days: int) -> None:
        """Raise if any training label could be realised inside the OOS window.

        This is the invariant the gate is about, so it is expressed as an
        assertion rather than left to review.
        """
        label_end = self.train_end + pd.Timedelta(days=label_horizon_days)
        if label_end >= self.oos_start:
            raise ValueError(
                f"fold {self.index} leaks: a training label realised up to "
                f"{label_end.date()} overlaps an OOS window starting "
                f"{self.oos_start.date()}"
            )


def purged_train_end(
    oos_start: pd.Timestamp,
    *,
    label_horizon_days: int = 0,
    embargo_days: int = 0,
) -> pd.Timestamp:
    """Last timestamp whose label is fully known strictly before ``oos_start``.

    The extra day matters and is easy to get wrong. A training sample dated
    ``train_end`` with horizon ``H`` has its label realised on
    ``train_end + H``. Setting ``train_end = oos_start - H`` therefore puts the
    label's resolution date exactly ON the first OOS observation, which is a
    leak: the label is paid off by a price from the test window. The boundary
    must be strictly earlier, hence the ``+ 1``.

    With ``label_horizon_days=0`` this returns exactly ``oos_start - 1 day``,
    which is the unpurged walk-forward boundary, so callers that never ask for a
    purge keep their previous behaviour bit for bit.
    """
    if label_horizon_days < 0 or embargo_days < 0:
        raise ValueError("label_horizon_days and embargo_days must be >= 0")
    return oos_start - pd.Timedelta(days=label_horizon_days + embargo_days + 1)


def generate_purged_folds(
    dates: pd.DatetimeIndex,  # type: ignore[type-arg]
    *,
    train_years: int,
    test_years: int,
    label_horizon_days: int = 0,
    embargo_days: int = 0,
    validate: bool = True,
) -> list[PurgedFold]:
    """Anchored expanding-window folds with purging and embargo applied.

    Parameters
    ----------
    dates
        The full evaluation calendar, ascending.
    label_horizon_days
        Forward reach of a training label. For 12-month momentum use ~252
        trading days expressed in calendar days, or pass the trading-day count
        with ``trading_days`` semantics via :func:`label_horizon_from_bars`.
    embargo_days
        Extra calendar-day gap after the last usable training label.
    validate
        Assert the no-leakage invariant on every fold before returning.
    """
    if len(dates) == 0:
        return []

    start_date, end_date = dates[0], dates[-1]
    folds: list[PurgedFold] = []
    oos_start = start_date + pd.DateOffset(years=train_years)
    idx = 0

    while oos_start <= end_date:
        idx += 1
        oos_end = min(
            oos_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1),
            end_date,
        )
        train_end_raw = oos_start - pd.Timedelta(days=1)
        train_end = purged_train_end(
            oos_start, label_horizon_days=label_horizon_days, embargo_days=embargo_days
        )

        # Clamp the effective end onto real observations, and drop the fold
        # entirely if purging left no usable training history.
        usable = dates[dates <= train_end]
        if len(usable) == 0:
            oos_start += pd.DateOffset(years=1)
            continue
        effective_end = usable[-1]

        folds.append(
            PurgedFold(
                index=idx,
                train_start=start_date,
                train_end=effective_end,
                train_end_raw=min(train_end_raw, end_date),
                oos_start=oos_start,
                oos_end=min(oos_end, end_date),
                label_horizon_days=label_horizon_days,
                embargo_days=embargo_days,
            )
        )
        oos_start += pd.DateOffset(years=1)

    if validate:
        for fold in folds:
            fold.assert_no_label_leakage(label_horizon_days)
    return folds


def label_horizon_from_bars(bars: int, bars_per_year: int = 252) -> int:
    """Convert a label horizon in trading bars to calendar days.

    Calendar days are over-approximated by ~1.41x the bar count, so the purge
    is conservative rather than tight: over-purging costs statistical power,
    under-purging costs validity.
    """
    if bars <= 0:
        return 0
    return round(bars * 365.25 / bars_per_year)
