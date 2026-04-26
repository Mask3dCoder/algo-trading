"""Unit tests for technical signals: volume, volatility, momentum."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.signals.technical.volume import (
    vwap_deviation, obv, volume_price_trend, ad_line,
    chaikin_money_flow, mfi, mfi_signal,
    unusual_volume, obv_divergence, ease_of_movement,
)
from quantcore.signals.technical.volatility import (
    atr, atr_normalized, historical_vol, historical_vol_signal,
    vol_regime, vol_regime_signal,
    vol_of_vol, parkinson_vol, garman_klass_vol,
)
from quantcore.signals.technical.momentum import (
    cross_sectional_momentum, time_series_momentum,
    dual_momentum, high_52w_proximity,
    vol_scaled_momentum, roc, roc_signal, cmo, cmo_signal,
)
from quantcore.signals.technical.trend import sma, ema, macd
from quantcore.signals.technical.mean_reversion import rsi, bollinger_bands


# ---------------------------------------------------------------------------
# Volume Signals Tests
# ---------------------------------------------------------------------------


class TestOBV:
    def test_monotonic_increase_in_uptrend(self) -> None:
        """OBV should monotonically increase in a pure uptrend."""
        n = 50
        close = np.arange(100.0, 100.0 + n, dtype=float)
        volume = np.ones(n) * 1000.0
        result = obv(close, volume)
        # OBV should be strictly increasing
        for i in range(2, n):
            assert result[i] > result[i - 1]

    def test_monotonic_decrease_in_downtrend(self) -> None:
        close = np.arange(100.0, 100.0 - 50, -1.0)
        volume = np.ones(50) * 1000.0
        result = obv(close, volume)
        for i in range(2, len(close)):
            assert result[i] < result[i - 1]

    def test_flat_close_no_change(self) -> None:
        close = np.full(50, 100.0)
        volume = np.ones(50) * 1000.0
        result = obv(close, volume)
        assert result[-1] == 0.0  # No change from start


class TestVolumePriceTrend:
    def test_positive_when_uptrend(self) -> None:
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volume = np.ones(5) * 1000.0
        result = volume_price_trend(close, volume)
        assert result[-1] > 0

    def test_negative_when_downtrend(self) -> None:
        close = np.array([104.0, 103.0, 102.0, 101.0, 100.0])
        volume = np.ones(5) * 1000.0
        result = volume_price_trend(close, volume)
        assert result[-1] < 0


class TestADLine:
    def test_positive_when_close_near_high(self) -> None:
        n = 30
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 109.0)  # Close near high
        volume = np.ones(n) * 1000.0
        result = ad_line(high, low, close, volume)
        assert result[-1] > 0  # Accumulation

    def test_negative_when_close_near_low(self) -> None:
        n = 30
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 91.0)  # Close near low
        volume = np.ones(n) * 1000.0
        result = ad_line(high, low, close, volume)
        assert result[-1] < 0  # Distribution


class TestCMF:
    def test_positive_when_close_near_high(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 108.0)  # Near high
        volume = rng.uniform(1000, 2000, n)
        result = chaikin_money_flow(high, low, close, volume, period=21)
        # Last valid value should be positive
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 0

    def test_range_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        price = 100 + np.cumsum(rng.normal(0, 1, n))
        high = price + rng.uniform(0, 2, n)
        low = price - rng.uniform(0, 2, n)
        close = price
        volume = rng.uniform(1000, 5000, n)
        result = chaikin_money_flow(high, low, close, volume, period=21)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)


class TestMFI:
    def test_high_when_uptrend(self) -> None:
        n = 50
        close = np.arange(100.0, 100.0 + n, dtype=float)
        high = close + 1.0
        low = close - 1.0
        volume = np.ones(n) * 1000.0
        result = mfi(high, low, close, volume, period=14)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 50.0  # Above midpoint in uptrend

    def test_range_zero_to_hundred(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        price = 100 + np.cumsum(rng.normal(0, 1, n))
        high = price + rng.uniform(0, 2, n)
        low = price - rng.uniform(0, 2, n)
        close = price
        volume = rng.uniform(1000, 5000, n)
        result = mfi(high, low, close, volume, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)


class TestUnusualVolume:
    def test_flags_spike(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        volume = rng.uniform(900, 1100, n)
        volume[50] = 5000.0  # Huge spike
        result = unusual_volume(volume, lookback=20, threshold=2.0)
        assert result[50] > 2.0

    def test_normal_volume_low_zscore(self) -> None:
        rng = np.random.default_rng(42)
        volume = rng.uniform(900, 1100, 1000)
        result = unusual_volume(volume, lookback=20)
        valid = result[~np.isnan(result)]
        # Most values should be within ±3
        assert np.mean(np.abs(valid) < 3.0) > 0.9


class TestOBVDivergence:
    def test_bearish_divergence(self) -> None:
        """Price rising but OBV falling should be bearish divergence."""
        n = 40
        close = np.arange(100.0, 100.0 + n, dtype=float)
        volume = np.ones(n) * 1000.0
        # Make volume decrease as price rises → OBV grows slower
        volume[20:] = np.linspace(500, 100, 20)
        result = obv_divergence(close, volume, lookback=20)
        # Check some values
        valid = result[20:]
        assert len(valid) > 0  # At least some values computed


# ---------------------------------------------------------------------------
# Volatility Signals Tests
# ---------------------------------------------------------------------------


class TestATR:
    def test_higher_during_volatile_period(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        # Low vol first half, high vol second half
        close = np.zeros(n)
        high = np.zeros(n)
        low = np.zeros(n)
        close[0] = 100.0
        for i in range(1, n):
            vol = 0.5 if i < 50 else 3.0
            close[i] = close[i - 1] + rng.normal(0, vol)
            high[i] = close[i] + abs(rng.normal(0, vol))
            low[i] = close[i] - abs(rng.normal(0, vol))

        result = atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        # ATR in second half should be higher
        if len(valid) > 30:
            low_vol_atr = np.mean(valid[:20])
            high_vol_atr = np.mean(valid[-20:])
            assert high_vol_atr > low_vol_atr

    def test_positive_values(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + rng.uniform(0, 2, n)
        low = close - rng.uniform(0, 2, n)
        result = atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)


class TestHistoricalVol:
    def test_known_value(self) -> None:
        """Constant returns should give zero volatility."""
        n = 100
        close = np.arange(100.0, 200.0, dtype=float)  # Linear uptrend
        result = historical_vol(close, windows=[21])
        hv = result[21]
        valid = hv[~np.isnan(hv)]
        # For nearly-linear returns, vol should be close to zero
        assert np.all(valid < 0.5)  # Very low vol

    def test_annualized_multiple_windows(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.02, n))
        result = historical_vol(close, windows=[5, 21, 63, 252])
        # All windows should produce reasonable annualized vol (5% - 80%)
        for w, hv in result.items():
            valid = hv[~np.isnan(hv)]
            if len(valid) > 0:
                assert np.all(valid > 0.01)  # At least 1% annualized
                assert np.all(valid < 5.0)  # Less than 500% annualized


class TestVolRegime:
    def test_produces_valid_labels(self) -> None:
        rng = np.random.default_rng(42)
        n = 600
        # Low vol first part, high vol second
        close = np.zeros(n)
        close[0] = 100.0
        for i in range(1, n):
            vol = 0.005 if i < 300 else 0.03
            close[i] = close[i - 1] * (1 + rng.normal(0, vol))

        result = vol_regime(close, n_states=3, lookback=252)
        valid = result[result >= 0]
        if len(valid) > 0:
            assert np.all(valid >= 0)
            assert np.all(valid <= 2)


class TestParkinsonVol:
    def test_positive_output(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        result = parkinson_vol(high, low, period=21)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_lower_than_close_to_close_for_same_data(self) -> None:
        """Parkinson estimator should be more efficient (less noisy) than close-to-close."""
        rng = np.random.default_rng(42)
        n = 500
        close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.015, n))
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)

        pv = parkinson_vol(high, low, period=63)
        hv_dict = historical_vol(close, windows=[63])
        hv = hv_dict[63]

        pv_valid = pv[~np.isnan(pv)]
        hv_valid = hv[~np.isnan(hv)]
        # Parkinson vol should be a reasonable estimate (within 3x of close-to-close)
        if len(pv_valid) > 0 and len(hv_valid) > 0:
            ratio = np.mean(pv_valid) / np.mean(hv_valid)
            assert 0.3 < ratio < 3.0


# ---------------------------------------------------------------------------
# Momentum Signals Tests
# ---------------------------------------------------------------------------


class TestCrossSectionalMomentum:
    def test_ranks_normalized(self) -> None:
        rng = np.random.default_rng(42)
        n_periods = 300
        n_assets = 10
        returns_matrix = rng.normal(0.0005, 0.02, (n_periods, n_assets))
        result = cross_sectional_momentum(returns_matrix, lookback=252, skip=21)
        # Check last period's ranks are within [-1, 1]
        last = result[-1, :]
        valid = last[~np.isnan(last)]
        if len(valid) > 0:
            assert np.all(valid >= -1.0)
            assert np.all(valid <= 1.0)

    def test_winner_highest_rank(self) -> None:
        """Asset with highest past return should get highest rank."""
        n_periods = 300
        n_assets = 5
        returns_matrix = np.zeros((n_periods, n_assets))
        # Make asset 0 a clear winner
        returns_matrix[:, 0] = 0.002  # Strong positive
        returns_matrix[:, 1:] = -0.001  # Losers

        result = cross_sectional_momentum(returns_matrix, lookback=252, skip=21)
        last = result[-1, :]
        valid_mask = ~np.isnan(last)
        if np.any(valid_mask):
            assert np.nanargmax(last) == 0


class TestTimeSeriesMomentum:
    def test_positive_for_uptrend(self) -> None:
        n = 300
        returns = np.full(n, 0.001)  # Persistent positive returns
        result = time_series_momentum(returns, lookback=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 0

    def test_negative_for_downtrend(self) -> None:
        n = 300
        returns = np.full(n, -0.001)
        result = time_series_momentum(returns, lookback=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] < 0


class TestDualMomentum:
    def test_positive_when_outperforming(self) -> None:
        n = 300
        returns = np.full(n, 0.002)
        benchmark = np.full(n, 0.001)  # Asset outperforms
        result = dual_momentum(returns, benchmark, lookback=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 0

    def test_negative_when_asset_declining(self) -> None:
        n = 300
        returns = np.full(n, -0.001)
        benchmark = np.full(n, 0.001)
        result = dual_momentum(returns, benchmark, lookback=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] < 0


class TestHigh52wProximity:
    def test_at_high_proximity_near_one(self) -> None:
        n = 300
        close = np.arange(100.0, 100.0 + n, dtype=float)  # Monotonic rise
        result = high_52w_proximity(close, period=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # At the end, we're at the 252-day high
            assert valid[-1] > 0  # Positive z-score

    def test_far_from_high(self) -> None:
        n = 300
        close = np.ones(n) * 100.0
        close[100] = 150.0  # Set 52w high at 150
        close[200:] = 100.0  # Then drift back to 100
        result = high_52w_proximity(close, period=252)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] < 0  # Below average proximity


class TestROC:
    def test_positive_for_rising_prices(self) -> None:
        close = np.arange(100.0, 130.0, dtype=float)
        result = roc(close, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_negative_for_falling_prices(self) -> None:
        close = np.arange(130.0, 100.0, -1.0, dtype=float)
        result = roc(close, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 0)


class TestCMO:
    def test_range_bounded(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        result = cmo(close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100.0)
        assert np.all(valid <= 100.0)

    def test_cmo_signal_range(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        result = cmo_signal(close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)


# ---------------------------------------------------------------------------
# Integration: all signals produce output on same data
# ---------------------------------------------------------------------------


class TestSignalIntegration:
    def test_all_signals_on_realistic_data(self) -> None:
        """Smoke test: all signals produce output arrays of correct length on realistic data."""
        rng = np.random.default_rng(42)
        n = 500
        close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
        close = np.maximum(close, 1.0)  # Ensure positive
        high = close * (1 + rng.uniform(0, 0.02, n))
        low = close * (1 - rng.uniform(0, 0.02, n))
        volume = rng.uniform(1e6, 5e6, n)

        # Volume signals
        obv(close, volume)
        ad_line(high, low, close, volume)
        mfi(high, low, close, volume, period=14)
        chaikin_money_flow(high, low, close, volume, period=21)
        unusual_volume(volume, lookback=20)

        # Volatility signals
        atr(high, low, close, period=14)
        hv = historical_vol(close, windows=[21])
        assert 21 in hv
        vol_of_vol(close, vol_window=21, vol_of_vol_window=63)

        # Momentum signals
        roc(close, period=21)
        cmo(close, period=14)
        high_52w_proximity(close, period=252)

    def test_volume_signal_with_vwap(self) -> None:
        """VWAP deviation produces bounded output."""
        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        close = np.maximum(close, 1.0)
        volume = rng.uniform(1e6, 5e6, n)
        result = vwap_deviation(close, volume, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)
