"""Unit tests for data quality engines: outlier detection, missing data, stale quotes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcore.data.quality.missing_data import (
    em_impute,
    forward_fill,
    impute,
    interpolate_missing,
)
from quantcore.data.quality.outlier_detection import (
    detect_all,
    grubbs_test,
    iqr_outliers,
    return_outliers,
    zscore_outliers,
)
from quantcore.data.quality.stale_quotes import cross_validate, detect_stale, majority_vote


class TestOutlierDetection:
    def test_zscore_detects_injected_outliers(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 1000)
        series[50] = 10.0   # Inject outlier
        series[200] = -8.0  # Inject outlier
        mask = zscore_outliers(series, threshold=4.0)
        assert mask[50] == True
        assert mask[200] == True

    def test_zscore_low_false_positive_rate(self) -> None:
        rng = np.random.default_rng(42)
        clean = rng.normal(0, 1, 10000)
        mask = zscore_outliers(clean, threshold=4.0)
        false_pos_rate = np.mean(mask)
        assert false_pos_rate < 0.02  # <2% false positive

    def test_grubbs_finds_single_outlier(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 100).tolist()
        series[25] = 20.0  # Extreme outlier
        indices = grubbs_test(np.array(series), alpha=0.05)
        assert 25 in indices

    def test_iqr_detects_outliers(self) -> None:
        series = np.concatenate([np.random.default_rng(42).normal(0, 1, 100), [10.0, -10.0]])
        mask = iqr_outliers(series, factor=1.5)
        assert mask[-2] == True
        assert mask[-1] == True

    def test_return_outliers(self) -> None:
        prices = np.cumsum(np.random.default_rng(42).normal(0, 1, 100)) + 100
        prices[50] = prices[49] * 1.5  # 50% jump
        mask = return_outliers(prices, max_sigma=10)
        assert mask[50] == True

    def test_detect_all_runs(self) -> None:
        series = np.random.default_rng(42).normal(0, 1, 200)
        series[100] = 15.0
        report = detect_all(series)
        assert np.any(report.combined_mask)


class TestMissingData:
    def test_forward_fill(self) -> None:
        s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0], index=pd.date_range("2024-01-01", periods=5))
        filled = forward_fill(s, max_gap=3)
        assert filled.iloc[1] == 1.0
        assert filled.iloc[2] == 1.0
        assert filled.iloc[3] == 1.0

    def test_forward_fill_respects_max_gap(self) -> None:
        s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0],
                       index=pd.date_range("2024-01-01", periods=7))
        filled = forward_fill(s, max_gap=3)
        assert np.isnan(filled.iloc[5])  # Shouldn't fill beyond max_gap

    def test_interpolate(self) -> None:
        s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0])
        filled = interpolate_missing(s, method="linear")
        assert abs(filled.iloc[2] - 3.0) < 0.01

    def test_em_impute(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (50, 3))
        matrix[10, 0] = np.nan
        matrix[20, 1] = np.nan
        matrix[30, 2] = np.nan
        result = em_impute(matrix)
        assert not np.any(np.isnan(result))

    def test_impute_with_report(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        filled, report = impute(s, method="ffill")
        assert report.n_filled == 2


class TestStaleQuotes:
    def test_detect_stale(self) -> None:
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 105.0])
        mask = detect_stale(prices, max_unchanged=5)
        assert np.any(mask)  # Should flag at least the later stale points

    def test_cross_validate_discrepancy(self) -> None:
        a = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        b = np.array([100.0, 101.0, 115.0, 103.0, 104.0])  # Big discrepancy at index 2
        report = cross_validate(a, b, threshold=0.01)
        assert report.n_discrepancies >= 1

    def test_majority_vote(self) -> None:
        a = np.array([100.0, 101.0, 102.0])
        b = np.array([100.0, 101.0, 102.0])
        c = np.array([100.0, 101.0, 50.0])  # Outlier source
        result = majority_vote({"a": a, "b": b, "c": c})
        assert result[2] == 102.0  # Median of (102, 102, 50) is 102
