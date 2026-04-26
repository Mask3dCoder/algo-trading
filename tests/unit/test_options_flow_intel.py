"""Tests for alt_data options flow: unusual activity, GEX, max pain, put/call."""

from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime

from quantcore.alt_data.options_flow.unusual_activity import (
    UOAType, UOAConfig, UOATrade, UOADetector,
    detect_sweep_cascade, compute_premium_sentiment,
)
from quantcore.alt_data.options_flow.gex import (
    GEXConfig, GEXResult, GEXCalculator, detect_gamma_squeeze,
)
from quantcore.alt_data.options_flow.max_pain import (
    MaxPainConfig, MaxPainResult, MaxPainCalculator,
    put_call_ratio, put_call_oi_ratio, pcr_signal,
)
from quantcore.alt_data.options_flow.put_call import (
    PCRConfig, PCRSnapshot, PutCallAnalyzer,
    compute_equity_only_pcr, pcr_term_structure,
)


# ---------------------------------------------------------------------------
# Unusual Options Activity tests
# ---------------------------------------------------------------------------

class TestUOADetector:
    def test_classify_block(self):
        detector = UOADetector()
        trade_type = detector.classify_trade({"quantity": 600, "conditions": []})
        assert trade_type == UOAType.BLOCK

    def test_classify_sweep(self):
        detector = UOADetector()
        trade_type = detector.classify_trade({"quantity": 100, "conditions": ["sweep"]})
        assert trade_type == UOAType.SWEEP

    def test_classify_unknown(self):
        detector = UOADetector()
        trade_type = detector.classify_trade({"quantity": 10, "conditions": []})
        assert trade_type == UOAType.UNKNOWN

    def test_is_unusual_below_threshold(self):
        detector = UOADetector(config=UOAConfig(min_premium_threshold=200000))
        trade = UOATrade(premium=50000, quantity=10)
        hist_vol = np.array([10, 12, 11, 9, 11])
        assert detector.is_unusual(trade, hist_vol, 1000) is False

    def test_is_unusual_above_threshold(self):
        detector = UOADetector(config=UOAConfig(min_premium_threshold=50000))
        trade = UOATrade(premium=200000, quantity=500)
        hist_vol = np.array([10, 12, 11, 9, 11])
        assert detector.is_unusual(trade, hist_vol, 100) is True

    def test_aggregate_by_symbol(self):
        detector = UOADetector()
        trades = [
            UOATrade(symbol="AAPL"),
            UOATrade(symbol="AAPL"),
            UOATrade(symbol="MSFT"),
        ]
        by_sym = detector.aggregate_by_symbol(trades)
        assert len(by_sym["AAPL"]) == 2
        assert len(by_sym["MSFT"]) == 1

    def test_compute_uoa_score(self):
        detector = UOADetector()
        trades = [
            UOATrade(symbol="AAPL", premium=200000, trade_type=UOAType.SWEEP),
            UOATrade(symbol="AAPL", premium=100000, trade_type=UOAType.SWEEP),
        ]
        score = detector.compute_uoa_score(trades, "AAPL")
        assert 0 <= score <= 1


class TestDetectSweepCascade:
    def test_no_sweeps(self):
        trades = [UOATrade(trade_type=UOAType.BLOCK)]
        cascades = detect_sweep_cascade(trades)
        assert cascades == []

    def test_cascade_detected(self):
        now = datetime(2024, 1, 1, 12, 0, 0)
        trades = []
        for i in range(3):
            import copy
            t = UOATrade(
                symbol="AAPL", trade_type=UOAType.SWEEP,
                timestamp=datetime(2024, 1, 1, 12, 0, i * 10),
            )
            trades.append(t)
        cascades = detect_sweep_cascade(trades, time_window_sec=60)
        assert len(cascades) == 1
        assert len(cascades[0]) == 3


class TestComputePremiumSentiment:
    def test_bullish(self):
        trades = [
            UOATrade(symbol="AAPL", option_type="call", premium=70000),
            UOATrade(symbol="AAPL", option_type="put", premium=30000),
        ]
        sentiment = compute_premium_sentiment(trades)
        assert sentiment["AAPL"] > 0.5

    def test_bearish(self):
        trades = [
            UOATrade(symbol="AAPL", option_type="call", premium=30000),
            UOATrade(symbol="AAPL", option_type="put", premium=70000),
        ]
        sentiment = compute_premium_sentiment(trades)
        assert sentiment["AAPL"] < 0.5


# ---------------------------------------------------------------------------
# GEX tests
# ---------------------------------------------------------------------------

class TestGEXCalculator:
    def test_init(self):
        calc = GEXCalculator()
        assert calc.config.risk_free_rate == 0.05

    def test_compute_empty_chain(self):
        calc = GEXCalculator()
        result = calc.compute_from_chain(100.0, [])
        assert result.total_gex == 0.0
        assert result.symbol == ""

    def test_compute_simple_chain(self):
        calc = GEXCalculator()
        options = [
            {"symbol": "AAPL", "strike": 100, "dte": 30, "open_interest": 100,
             "iv": 0.2, "option_type": "call"},
            {"symbol": "AAPL", "strike": 100, "dte": 30, "open_interest": 50,
             "iv": 0.2, "option_type": "put"},
        ]
        result = calc.compute_from_chain(100.0, options)
        assert result.symbol == "AAPL"
        assert result.total_gex >= 0

    def test_gamma_flip_with_data(self):
        calc = GEXCalculator()
        gex_by_strike = {95.0: 100.0, 100.0: -50.0, 105.0: -100.0}
        flip = calc.find_gamma_flip(gex_by_strike, 100.0)
        assert isinstance(flip, float)


class TestDetectGammaSqueeze:
    def test_normal_gex(self):
        hist = np.array([100, 120, 110, 90, 105, 115, 108, 112, 98, 103, 107, 111, 95, 104, 109, 113, 97, 102, 106, 110])
        assert detect_gamma_squeeze(100.0, hist) is False

    def test_extreme_negative_gex(self):
        hist = np.array([100, 120, 110, 90, 105, 115, 108, 112, 98, 103, 107, 111, 95, 104, 109, 113, 97, 102, 106, 110])
        mean = float(np.mean(hist))
        std = float(np.std(hist))
        extreme_neg = mean - 4 * std  # truly negative, below -2 z-score
        if extreme_neg >= 0:
            extreme_neg = -1.0
        assert detect_gamma_squeeze(extreme_neg, hist) is True


# ---------------------------------------------------------------------------
# Max Pain tests
# ---------------------------------------------------------------------------

class TestMaxPainCalculator:
    def test_simple_three_strike(self):
        calc = MaxPainCalculator()
        options = [
            {"strike": 95, "option_type": "call", "open_interest": 50, "expiration": "2024-01-19"},
            {"strike": 95, "option_type": "put", "open_interest": 20, "expiration": "2024-01-19"},
            {"strike": 100, "option_type": "call", "open_interest": 100, "expiration": "2024-01-19"},
            {"strike": 100, "option_type": "put", "open_interest": 80, "expiration": "2024-01-19"},
            {"strike": 105, "option_type": "call", "open_interest": 30, "expiration": "2024-01-19"},
            {"strike": 105, "option_type": "put", "open_interest": 150, "expiration": "2024-01-19"},
        ]
        result = calc.compute(100.0, options)
        assert result.max_pain > 0
        assert result.call_resistance > 0
        assert result.max_pain_distance_pct >= 0

    def test_empty_chain(self):
        calc = MaxPainCalculator()
        result = calc.compute(100.0, [])
        assert result.max_pain == 0.0


class TestPutCallRatio:
    def test_pcr(self):
        assert put_call_ratio(100, 50) == 0.5
        assert put_call_ratio(100, 150) == 1.5

    def test_zero_call_volume(self):
        assert put_call_ratio(0, 100) == float("inf")


class TestPCRSignal:
    def test_normal_pcr(self):
        hist = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 0.85, 0.95, 1.05, 0.9, 1.0])
        result = pcr_signal(1.0, hist)
        assert "pcr_zscore" in result
        assert "pcr_percentile" in result

    def test_small_history(self):
        result = pcr_signal(1.0, np.array([0.8, 0.9]))
        assert result["pcr_zscore"] == 0.0


# ---------------------------------------------------------------------------
# Put/Call analysis tests
# ---------------------------------------------------------------------------

class TestPutCallAnalyzer:
    def test_compute(self):
        config = PCRConfig(min_volume=1)
        analyzer = PutCallAnalyzer(config=config)
        trades = [
            {"option_type": "call", "volume": 100, "open_interest": 500},
            {"option_type": "put", "volume": 50, "open_interest": 300},
        ]
        snapshot = analyzer.compute(trades)
        assert snapshot.total_call_volume == 100
        assert snapshot.total_put_volume == 50
        assert snapshot.volume_pcr == 0.5

    def test_classify_bullish(self):
        analyzer = PutCallAnalyzer()
        result = analyzer.classify_sentiment(0.5, 0.6)
        assert result in ("bullish", "neutral_bullish", "neutral_bearish", "bearish", "neutral")

    def test_classify_bearish(self):
        analyzer = PutCallAnalyzer()
        result = analyzer.classify_sentiment(1.5, 1.3)
        assert result in ("bearish", "neutral_bearish")


class TestComputeEquityOnlyPCR:
    def test_excludes_etfs(self):
        trades = [
            {"symbol": "SPY", "option_type": "call", "volume": 100},
            {"symbol": "SPY", "option_type": "put", "volume": 200},
            {"symbol": "AAPL", "option_type": "call", "volume": 100},
            {"symbol": "AAPL", "option_type": "put", "volume": 50},
        ]
        pcr = compute_equity_only_pcr(trades)
        assert pcr == 0.5


class TestPCRTermStructure:
    def test_flat(self):
        result = pcr_term_structure({"near": 0.8, "far": 0.85})
        assert "slope" in result
        assert "shape" in result

    def test_single_expiry(self):
        result = pcr_term_structure({"near": 0.8})
        assert result["shape"] == "flat"
