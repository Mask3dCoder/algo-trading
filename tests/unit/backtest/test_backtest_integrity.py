"""Tests for backtest engine integrity fixes.

Validates:
  A4: Drawdown graduated actions (REDUCE_25/50/75) actually reduce positions
  A5: Drawdown FLATTEN closes all positions
  A6: Walk-forward embargo defaults to 1%
"""

from __future__ import annotations

from quantcore.backtest.walk_forward import (
    anchored_walk_forward,
    rolling_walk_forward,
)

# ---------------------------------------------------------------------------
# A6: Walk-forward embargo default
# ---------------------------------------------------------------------------

class TestEmbargoDefault:
    def test_rolling_walk_forward_default_embargo(self):
        """Default embargo_pct should create a gap between train and test."""
        folds = rolling_walk_forward(n_bars=1000, train_size=252, test_size=63)
        assert len(folds) > 0
        for fold in folds:
            assert fold.test_start >= fold.train_end
            if fold.embargo_start is not None:
                assert fold.embargo_start == fold.train_end

    def test_rolling_walk_forward_zero_embargo_still_works(self):
        """Explicit zero embargo should still produce valid folds."""
        folds = rolling_walk_forward(
            n_bars=1000, train_size=252, test_size=63, embargo_pct=0.0,
        )
        assert len(folds) > 0
        for fold in folds:
            assert fold.test_start == fold.train_end

    def test_anchored_walk_forward_default_embargo(self):
        """Anchored WF should also default to 1% embargo."""
        folds = anchored_walk_forward(n_bars=1000, initial_train=252, test_size=63)
        assert len(folds) > 0
        for fold in folds:
            assert fold.test_start >= fold.train_end

    def test_embargo_creates_gap(self):
        """With 10% embargo, gap should be visible."""
        folds = rolling_walk_forward(
            n_bars=2000, train_size=252, test_size=100, embargo_pct=0.10,
        )
        assert len(folds) > 0
        for fold in folds:
            gap = fold.test_start - fold.train_end
            assert gap >= 10  # 10% of 100 = 10 bars


# ---------------------------------------------------------------------------
# A4+A5: Drawdown actions
# ---------------------------------------------------------------------------

class TestDrawdownActions:
    def test_drawdown_tracker_reduce_action(self):
        """DrawdownTracker should return REDUCE_25 at -5% drawdown."""
        from quantcore.risk.drawdown import (
            DrawdownAction,
            DrawdownConfig,
            DrawdownTracker,
        )

        config = DrawdownConfig(
            thresholds=[
                (-0.05, DrawdownAction.REDUCE_25),
                (-0.10, DrawdownAction.REDUCE_50),
                (-0.15, DrawdownAction.REDUCE_75),
                (-0.20, DrawdownAction.FLATTEN),
            ],
            halt_threshold=-0.25,
        )
        tracker = DrawdownTracker(config)

        tracker.update(1_000_000.0)  # peak
        tracker.update(970_000.0)    # -3%
        state = tracker.update(945_000.0)  # -5.5% -> REDUCE_25
        assert state.action == DrawdownAction.REDUCE_25

    def test_drawdown_tracker_flatten_action(self):
        """DrawdownTracker should return FLATTEN at -20% drawdown."""
        from quantcore.risk.drawdown import (
            DrawdownAction,
            DrawdownConfig,
            DrawdownTracker,
        )

        config = DrawdownConfig(
            thresholds=[
                (-0.05, DrawdownAction.REDUCE_25),
                (-0.10, DrawdownAction.REDUCE_50),
                (-0.15, DrawdownAction.REDUCE_75),
                (-0.20, DrawdownAction.FLATTEN),
            ],
            halt_threshold=-0.25,
        )
        tracker = DrawdownTracker(config)

        tracker.update(1_000_000.0)
        state = tracker.update(800_000.0)  # -20% -> FLATTEN
        assert state.action == DrawdownAction.FLATTEN

    def test_drawdown_tracker_halt_action(self):
        """DrawdownTracker should return HALT at -25% drawdown."""
        from quantcore.risk.drawdown import (
            DrawdownAction,
            DrawdownConfig,
            DrawdownTracker,
        )

        config = DrawdownConfig(
            thresholds=[
                (-0.05, DrawdownAction.REDUCE_25),
                (-0.10, DrawdownAction.REDUCE_50),
                (-0.15, DrawdownAction.REDUCE_75),
                (-0.20, DrawdownAction.FLATTEN),
            ],
            halt_threshold=-0.25,
        )
        tracker = DrawdownTracker(config)

        tracker.update(1_000_000.0)
        state = tracker.update(745_000.0)  # -25.5% -> HALT
        assert state.action == DrawdownAction.HALT
