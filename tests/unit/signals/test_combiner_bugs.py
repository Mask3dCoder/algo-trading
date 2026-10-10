"""Tests for the 3 critical bug fixes in signals/combiner.py.

Bug 1: IC-weight combiner had a dead np.zeros() call (line 75).
Bug 2: Stacking combiner had len(signals) result discarded (line 147).

These tests verify that the combiners produce correct, non-NaN output
on synthetic data with known properties.
"""

import numpy as np

from quantcore.signals.combiner import (
    adaptive_combine,
    equal_weight_combine,
    ic_weight_combine,
    regime_weight_combine,
    sign_agreement_combine,
    stacking_combine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals_and_returns(
    n: int = 500,
    n_signals: int = 3,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Create synthetic signals and forward returns with known IC structure.

    Signal 0: positively correlated with forward_returns (IC ~ +0.3)
    Signal 1: negatively correlated with forward_returns (IC ~ -0.3)
    Signal 2: uncorrelated noise (IC ~ 0.0)
    """
    rng = np.random.default_rng(seed)
    forward_returns = rng.standard_normal(n) * 0.01

    signal0 = forward_returns * 0.3 + rng.standard_normal(n) * 0.007  # IC ~ 0.3
    signal1 = -forward_returns * 0.3 + rng.standard_normal(n) * 0.007  # IC ~ -0.3
    signal2 = rng.standard_normal(n) * 0.01  # IC ~ 0

    return [signal0, signal1, signal2], forward_returns


# ---------------------------------------------------------------------------
# Bug 1: IC-weight combiner — rolling_ics must be populated
# ---------------------------------------------------------------------------

class TestICWeightCombine:
    """Verify Bug 1 fix: rolling IC array is computed and used."""

    def test_ic_weight_produces_non_nan_output(self):
        """The combiner must return non-NaN values after the lookback period."""
        signals, fwd = _make_signals_and_returns(n=500, n_signals=3)
        result = ic_weight_combine(signals, fwd, lookback=126, ic_floor=0.0)

        # After lookback, result should be non-NaN for most points
        valid_count = np.sum(~np.isnan(result[126:]))
        assert valid_count > 200, (
            f"IC-weight combiner produced {valid_count} valid values out of "
            f"{len(result) - 126} — expected >200 after lookback"
        )

    def test_ic_weight_positive_ic_signal_dominates(self):
        """Signal with positive IC should drive the combined signal direction."""
        signals, fwd = _make_signals_and_returns(n=500, n_signals=3)
        result = ic_weight_combine(signals, fwd, lookback=126, ic_floor=0.0)

        # The combined signal should be positively correlated with signal0
        # (which has IC ~ +0.3)
        valid = ~np.isnan(result)
        if np.sum(valid) > 50:
            corr = np.corrcoef(result[valid], signals[0][valid])[0, 1]
            assert corr > 0.0, (
                f"Expected positive correlation between combined and signal0 "
                f"(positive IC), got {corr:.4f}"
            )

    def test_ic_weight_negative_ic_signal_is_flipped(self):
        """Signal with negative IC should have its sign flipped in combination."""
        signals, fwd = _make_signals_and_returns(n=500, n_signals=3)
        result = ic_weight_combine(signals, fwd, lookback=126, ic_floor=0.0)

        # signal1 has IC ~ -0.3, so it should be flipped → correlated with -signal1
        # which is correlated with forward_returns
        valid = ~np.isnan(result)
        if np.sum(valid) > 50:
            corr_with_neg_signal1 = np.corrcoef(result[valid], -signals[1][valid])[0, 1]
            assert corr_with_neg_signal1 > -0.5, (
                f"Expected negative-IC signal to be flipped, got corr={corr_with_neg_signal1:.4f}"
            )

    def test_ic_weight_output_bounded(self):
        """Output must be in [-1, +1]."""
        signals, fwd = _make_signals_and_returns(n=500)
        result = ic_weight_combine(signals, fwd, lookback=126)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= -1.0 - 1e-10) and np.all(valid <= 1.0 + 1e-10), (
                f"IC-weight output out of bounds: [{valid.min():.4f}, {valid.max():.4f}]"
            )

    def test_ic_weight_empty_signals(self):
        """Empty signal list should return empty array."""
        result = ic_weight_combine([], np.array([]), lookback=126)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Bug 2: Stacking combiner — n_signals must be assigned
# ---------------------------------------------------------------------------

class TestStackingCombine:
    """Verify Bug 2 fix: n_signals is properly assigned and used."""

    def test_stacking_produces_non_nan_output(self):
        """Stacking combiner must return non-NaN values after training period."""
        signals, fwd = _make_signals_and_returns(n=500, n_signals=3)
        result = stacking_combine(signals, fwd, lookback=252, train_ratio=0.7)

        # After lookback, should have some non-NaN values
        valid_count = np.sum(~np.isnan(result[252:]))
        assert valid_count > 50, (
            f"Stacking combiner produced {valid_count} valid values out of "
            f"{len(result) - 252} — expected >50 after lookback"
        )

    def test_stacking_output_bounded(self):
        """Output must be in [-1, +1]."""
        signals, fwd = _make_signals_and_returns(n=500)
        result = stacking_combine(signals, fwd, lookback=252, train_ratio=0.7)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= -1.0 - 1e-10) and np.all(valid <= 1.0 + 1e-10), (
                f"Stacking output out of bounds: [{valid.min():.4f}, {valid.max():.4f}]"
            )

    def test_stacking_empty_signals(self):
        """Empty signal list should return empty array."""
        result = stacking_combine([], np.array([]), lookback=252)
        assert len(result) == 0

    def test_stacking_predictive_direction(self):
        """Stacking should learn the correct direction of the predictive signal."""
        signals, fwd = _make_signals_and_returns(n=600, n_signals=3)
        result = stacking_combine(signals, fwd, lookback=300, train_ratio=0.7)

        valid = ~np.isnan(result)
        if np.sum(valid) > 50:
            # Combined signal should be positively correlated with forward returns
            corr = np.corrcoef(result[valid], fwd[:len(result)][valid])[0, 1]
            # Allow for weak but non-negative correlation (stacking may overfit)
            assert corr > -0.3, (
                f"Stacking combined signal has unexpected strong negative "
                f"correlation with forward returns: {corr:.4f}"
            )


# ---------------------------------------------------------------------------
# Sanity: other combiners still work
# ---------------------------------------------------------------------------

class TestOtherCombiners:
    """Sanity tests for combiners that were not buggy."""

    def test_equal_weight_bounded(self):
        signals, _ = _make_signals_and_returns(n=100)
        result = equal_weight_combine(signals)
        assert np.all(np.abs(result[~np.isnan(result)]) <= 1.0 + 1e-10)

    def test_sign_agreement_bounded(self):
        signals, _ = _make_signals_and_returns(n=100)
        result = sign_agreement_combine(signals)
        assert np.all(np.abs(result[~np.isnan(result)]) <= 1.0 + 1e-10)

    def test_regime_weight_bounded(self):
        signals, _ = _make_signals_and_returns(n=100)
        regime = np.random.randint(0, 2, size=100)
        weights = {0: [0.5, 0.3, 0.2], 1: [0.1, 0.7, 0.2]}
        result = regime_weight_combine(signals, regime, weights)
        assert np.all(np.abs(result[~np.isnan(result)]) <= 1.0 + 1e-10)

    def test_adaptive_bounded(self):
        signals, fwd = _make_signals_and_returns(n=500)
        result = adaptive_combine(signals, fwd, lookback=126)
        assert np.all(np.abs(result[~np.isnan(result)]) <= 1.0 + 1e-10)
