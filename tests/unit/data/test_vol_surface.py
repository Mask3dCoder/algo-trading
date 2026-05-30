"""Unit tests for volatility surface model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestVolatilitySurface:
    """Tests for VolatilitySurface class."""

    def test_atm_iv_at_moneyness_1(self):
        """ATM IV (moneyness=1.0) should be close to the base ATM IV."""
        from quantcore.data.vol_surface import VolatilitySurface

        vs = VolatilitySurface(atm_iv=0.20, vix_level=20.0)
        iv = vs.iv_at_strike_dte(1.0, 30)

        # ATM should be close to the input ATM IV
        assert abs(iv - 0.20) < 0.05

    def test_otm_put_higher_than_atm(self):
        """OTM puts should have higher IV than ATM (positive skew)."""
        from quantcore.data.vol_surface import VolatilitySurface

        vs = VolatilitySurface(atm_iv=0.20, vix_level=20.0)
        atm_iv = vs.iv_at_strike_dte(1.0, 30)
        otm_put_iv = vs.iv_at_strike_dte(0.90, 30)  # 10% OTM put

        assert otm_put_iv > atm_iv

    def test_otm_call_lower_than_atm(self):
        """OTM calls should have lower IV than ATM (negative skew for calls)."""
        from quantcore.data.vol_surface import VolatilitySurface

        vs = VolatilitySurface(atm_iv=0.20, vix_level=20.0)
        atm_iv = vs.iv_at_strike_dte(1.0, 30)
        otm_call_iv = vs.iv_at_strike_dte(1.10, 30)  # 10% OTM call

        # OTM call IV should be lower than ATM
        assert otm_call_iv < atm_iv

    def test_higher_vix_more_skew(self):
        """Higher VIX should produce more pronounced skew."""
        from quantcore.data.vol_surface import VolatilitySurface

        vs_low = VolatilitySurface(atm_iv=0.15, vix_level=15.0)
        vs_high = VolatilitySurface(atm_iv=0.30, vix_level=30.0)

        # Get OTM put IV for both
        skew_low = vs_low.iv_at_strike_dte(0.90, 30) - vs_low.iv_at_strike_dte(1.0, 30)
        skew_high = vs_high.iv_at_strike_dte(0.90, 30) - vs_high.iv_at_strike_dte(1.0, 30)

        # Higher VIX → more skew
        assert skew_high > skew_low

    def test_short_dte_higher_iv(self):
        """Short-dated options should have higher IV when ATM != long-run."""
        from quantcore.data.vol_surface import VolatilitySurface

        # Use ATM IV above long-run to see term structure effect
        vs = VolatilitySurface(atm_iv=0.30, vix_level=30.0, long_run_iv=0.20)
        iv_7d = vs.iv_at_strike_dte(1.0, 7)
        iv_90d = vs.iv_at_strike_dte(1.0, 90)

        # Short-dated should be higher (closer to elevated ATM IV)
        assert iv_7d >= iv_90d

    def test_iv_bounded(self):
        """IV should always be within reasonable bounds."""
        from quantcore.data.vol_surface import VolatilitySurface

        vs = VolatilitySurface(atm_iv=0.20, vix_level=20.0)

        # Test extreme moneyness
        for m in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
            for dte in [1, 7, 30, 90, 365]:
                iv = vs.iv_at_strike_dte(m, dte)
                assert 0.04 <= iv <= 1.50, f"IV {iv} out of bounds for m={m}, dte={dte}"

    def test_smile_shape(self):
        """Smile should be convex (U-shaped)."""
        from quantcore.data.vol_surface import VolatilitySurface

        vs = VolatilitySurface(atm_iv=0.20, vix_level=20.0)
        strikes, ivs = vs.iv_smile(spot=100.0, dte=30, n_strikes=21)

        # ATM IV should be the minimum (or near minimum)
        atm_idx = len(strikes) // 2
        atm_iv = ivs[atm_idx]

        # At least some OTM puts should have higher IV
        assert any(iv > atm_iv for iv in ivs[:atm_idx])


class TestSkewAdjustment:
    """Tests for _skew_adjustment function."""

    def test_monotonic_in_vix(self):
        """Skew adjustment should increase with VIX level."""
        from quantcore.data.vol_surface import _skew_adjustment

        prev = 0
        for vix in [10, 15, 20, 25, 30, 35, 40, 50]:
            adj = _skew_adjustment(vix)
            assert adj >= prev, f"Skew not monotonic at VIX={vix}"
            prev = adj

    def test_reasonable_range(self):
        """Skew adjustment should be in a reasonable range."""
        from quantcore.data.vol_surface import _skew_adjustment

        for vix in [10, 15, 20, 25, 30, 35, 40, 50]:
            adj = _skew_adjustment(vix)
            assert 0.005 <= adj <= 0.15, f"Skew {adj} out of range at VIX={vix}"


class TestGenerateEnhancedSyntheticIV:
    """Tests for generate_enhanced_synthetic_iv function."""

    def test_returns_series_with_correct_index(self):
        """Returns a Series with the same index as input."""
        from quantcore.data.vol_surface import generate_enhanced_synthetic_iv

        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        close = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)

        iv = generate_enhanced_synthetic_iv(close, lookback=60)

        assert isinstance(iv, pd.Series)
        assert len(iv) == len(close)
        assert iv.index.equals(close.index)

    def test_iv_bounded(self):
        """IV should be bounded between 0.05 and 0.80."""
        from quantcore.data.vol_surface import generate_enhanced_synthetic_iv

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        close = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)

        iv = generate_enhanced_synthetic_iv(close, lookback=60)

        valid = iv.dropna()
        assert valid.min() >= 0.05
        assert valid.max() <= 0.80

    def test_no_nans_after_warmup(self):
        """No NaN values after the lookback warmup period."""
        from quantcore.data.vol_surface import generate_enhanced_synthetic_iv

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        close = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)

        iv = generate_enhanced_synthetic_iv(close, lookback=60)

        # After warmup, no NaN
        assert not iv.iloc[60:].isna().any()

    def test_enhanced_different_from_flat(self):
        """Enhanced model should produce different IV than flat model."""
        from quantcore.data.vol_surface import generate_enhanced_synthetic_iv

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        close = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100, index=dates)

        iv_enhanced = generate_enhanced_synthetic_iv(close, lookback=60, use_skew=True)
        iv_flat = generate_enhanced_synthetic_iv(close, lookback=60, use_skew=False)

        # They should differ (not identical)
        diff = (iv_enhanced - iv_flat).dropna().abs()
        assert diff.mean() > 0.001, "Enhanced and flat models produce identical IV"
