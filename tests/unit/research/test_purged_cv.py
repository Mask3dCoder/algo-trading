"""Tests for purged/embargoed fold construction and its enforcement.

These pin the invariant behind FREEZE.md unfreeze gate #3: a training label
must not be realised inside the OOS window. A fold boundary drawn on the return
series alone does not give that, because a label dated one day before the
boundary is decided by prices inside the window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from itertools import pairwise

from quantcore.research.purged_cv import (  # type: ignore[import-not-found]
    PurgedFold,
    generate_purged_folds,
    label_horizon_from_bars,
    purged_train_end,
)


def _calendar(start: str = "2005-01-03", periods: int = 6000) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods, tz="UTC")


def test_no_purge_matches_legacy_boundary() -> None:
    """label_horizon_days=0 must reduce to the unpurged oos_start - 1 day boundary.

    Both the boundary and the clamp to a real observation are part of the
    legacy behaviour, so the comparison is against the clamped value: a
    boundary landing on a weekend resolves to the preceding business day.
    """
    dates = _calendar()
    folds = generate_purged_folds(dates, train_years=5, test_years=1, label_horizon_days=0)
    assert folds
    for f in folds:
        # No configured purge means no purge is CLAIMED, even though the
        # boundary may still be clamped back to a preceding business day.
        assert f.purged_days == 0
        assert f.train_end <= f.oos_start - pd.Timedelta(days=1)
        assert f.realised_gap_days <= 3  # purge(0) + embargo(0) + weekend clamp


def test_label_horizon_purges_training_window() -> None:
    dates = _calendar()
    horizon = label_horizon_from_bars(252)
    folds = generate_purged_folds(dates, train_years=5, test_years=1, label_horizon_days=horizon)

    assert folds
    for f in folds:
        assert f.purged_days > 0
        # A label on the last usable training day resolves strictly before OOS.
        assert f.train_end + pd.Timedelta(days=horizon) < f.oos_start


def test_embargo_adds_gap_on_top_of_purge() -> None:
    dates = _calendar()
    base = generate_purged_folds(dates, train_years=5, test_years=1, label_horizon_days=100)
    with_embargo = generate_purged_folds(
        dates, train_years=5, test_years=1, label_horizon_days=100, embargo_days=10
    )
    gap = with_embargo[0].oos_start - with_embargo[0].train_end
    base_gap = base[0].oos_start - base[0].train_end
    assert gap >= base_gap + pd.Timedelta(days=9)


def test_momentum_label_horizon_is_purged() -> None:
    """12-month momentum: H ~= 365 calendar days, the case that motivated this."""
    dates = _calendar()
    folds = generate_purged_folds(dates, train_years=5, test_years=1, label_horizon_days=365)
    for f in folds:
        assert f.train_end <= f.oos_start - pd.Timedelta(days=365)
        f.assert_no_label_leakage(365)


def test_assertion_catches_an_unpurged_fold() -> None:
    """The guard must actually fail on the leaky shape, not merely exist."""
    leaky = PurgedFold(
        index=1,
        train_start=pd.Timestamp("2010-01-01", tz="UTC"),
        train_end=pd.Timestamp("2014-12-31", tz="UTC"),
        train_end_raw=pd.Timestamp("2014-12-31", tz="UTC"),
        oos_start=pd.Timestamp("2015-01-01", tz="UTC"),
        oos_end=pd.Timestamp("2015-12-31", tz="UTC"),
    )
    with pytest.raises(ValueError, match="leaks"):
        leaky.assert_no_label_leakage(365)


def test_negative_horizon_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        purged_train_end(pd.Timestamp("2020-01-01", tz="UTC"), label_horizon_days=-1)


def test_folds_are_non_overlapping_and_ordered() -> None:
    dates = _calendar()
    folds = generate_purged_folds(
        dates, train_years=5, test_years=1, label_horizon_days=label_horizon_from_bars(252)
    )
    assert len(folds) >= 2
    for a, b in pairwise(folds):
        assert a.oos_end < b.oos_start
        assert a.train_start <= a.train_end <= a.oos_start


def test_folds_dropped_when_purge_consumes_training_history() -> None:
    """Purging must not emit a fold whose training set is empty."""
    dates = _calendar(periods=1500)  # ~6 years of business days
    folds = generate_purged_folds(dates, train_years=5, test_years=1, label_horizon_days=365)
    for f in folds:
        assert f.train_end > f.train_start


# --- walk-forward integration ----------------------------------------------
class _LeakyStrategy:
    """No training_end parameter: the purge cannot be honoured."""

    def run_returns_backtest(self, *, start, end, training_start, **kwargs):
        raise AssertionError("not called")


class _PurgeAwareStrategy(_LeakyStrategy):
    def run_returns_backtest(self, *, start, end, training_start, training_end=None, **kwargs):
        raise AssertionError("not called")


def test_walkforward_detects_strategy_that_cannot_purge(caplog) -> None:
    from quantcore.research.walkforward import WalkForwardBacktester

    wf = WalkForwardBacktester(
        _LeakyStrategy(), train_years=2, test_years=1, label_horizon_days=365
    )
    assert wf._supports_training_end() is False
    with caplog.at_level("WARNING"):
        wf._warn_if_purge_unsupported()
    assert "purge_unsupported" in caplog.text
    assert "optimistically biased" in caplog.text


def test_walkforward_detects_purge_aware_strategy() -> None:
    from quantcore.research.walkforward import WalkForwardBacktester

    wf = WalkForwardBacktester(_PurgeAwareStrategy(), train_years=2, test_years=1, label_horizon_days=365)
    assert wf._supports_training_end() is True


def test_walkforward_silent_when_no_purge_requested(caplog) -> None:
    from quantcore.research.walkforward import WalkForwardBacktester

    wf = WalkForwardBacktester(_LeakyStrategy(), train_years=2, test_years=1)
    with caplog.at_level("WARNING"):
        wf._warn_if_purge_unsupported()
    assert "purge_unsupported" not in caplog.text


def test_walkforward_folds_are_purged() -> None:
    from quantcore.research.walkforward import WalkForwardBacktester

    wf = WalkForwardBacktester(
        _PurgeAwareStrategy(), train_years=5, test_years=1, label_horizon_days=365
    )
    folds = wf.generate_folds(_calendar())
    assert folds
    for f in folds:
        assert f["purged_days"] > 0
        assert f["train_end"] + pd.Timedelta(days=365) < f["oos_start"]
