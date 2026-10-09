"""Regression tests for compliance reporting (Fix 2).

Tests:
- generate_regulatory_report: slippage, best_exec_compliant, flags fields
- generate_regulatory_report: silent TCA failure when trades are dicts (known gap)
- generate_regulatory_report: flags for drawdown and leverage thresholds
- generate_regulatory_report: VaR backtesting (Basel traffic light)
- generate_best_execution_report: slippage calculation and venue breakdown
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.compliance.reporting import (
    RegulatoryReport,
    generate_best_execution_report,
    generate_regulatory_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, seed: int = 42) -> list[float]:
    rng = np.random.RandomState(seed)
    return rng.normal(0.0004, 0.01, n).tolist()


def _make_var_series(returns: list[float], confidence: float = 0.99) -> list[float]:
    """VaR series slightly larger than losses at confidence level."""
    r = np.array(returns)
    threshold = float(-np.percentile(-r, confidence * 100))
    return [threshold * 1.1] * len(returns)


def _make_trades(n: int = 5) -> list[dict]:
    """List of trade dicts (the format generate_regulatory_report expects)."""
    return [
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100 + i * 10,
            "price": 150.0 + i,
            "benchmark_price": 150.0 + i - 0.5,
            "venue": "ARCA",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# generate_regulatory_report tests
# ---------------------------------------------------------------------------

class TestRegulatoryReport:
    """Tests for generate_regulatory_report output fields."""

    def test_returns_regulatory_report_instance(self):
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
        )
        assert isinstance(report, RegulatoryReport)

    def test_avg_slippage_bps_field_exists(self):
        """Fix 2 regression: avg_slippage_bps must be present."""
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
            trades=_make_trades(),
        )
        assert hasattr(report, "avg_slippage_bps")
        assert isinstance(report.avg_slippage_bps, float)

    def test_best_exec_compliant_field_exists(self):
        """Fix 2 regression: best_exec_compliant must be present."""
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
            trades=_make_trades(),
        )
        assert hasattr(report, "best_exec_compliant")
        assert isinstance(report.best_exec_compliant, bool)

    def test_flags_field_exists(self):
        """Fix 2 regression: flags must be a list."""
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
        )
        assert hasattr(report, "flags")
        assert isinstance(report.flags, list)

    def test_tca_dict_type_mismatch_silent_failure(self):
        """Known gap: tca_strategy expects list[TCAResult], gets list[dict].

        The except clause on reporting.py:149 silently catches the crash,
        leaving avg_slippage_bps at 0.0. This test documents that behavior.
        """
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
            trades=_make_trades(),  # list[dict], not list[TCAResult]
        )
        # Due to the type mismatch, tca_strategy crashes silently
        # avg_slippage_bps stays at default 0.0
        assert report.avg_slippage_bps == 0.0
        assert report.best_exec_compliant is True  # default, not actually checked

    def test_no_trades_no_slippage(self):
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
            trades=None,
        )
        assert report.avg_slippage_bps == 0.0
        assert report.n_trades == 0

    def test_max_drawdown_flag(self):
        """Max drawdown > 20% sets a flag."""
        # Use strongly negative returns to produce >20% drawdown
        returns = [-0.02] * 50 + [-0.03] * 20 + [0.01] * 182
        report = generate_regulatory_report(
            returns=returns,
            var_series=[0.01] * len(returns),
        )
        drawdown_flags = [f for f in report.flags if "drawdown" in f.lower()]
        assert len(drawdown_flags) >= 1

    def test_leverage_flag(self):
        """Leverage > 3.0x sets a flag."""
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
            positions={"AAPL": 500_000.0, "GOOG": 400_000.0},
            portfolio_value=200_000.0,  # gross = 900k / 200k = 4.5x
        )
        leverage_flags = [f for f in report.flags if "leverage" in f.lower()]
        assert len(leverage_flags) >= 1

    def test_var_violations_recommendation(self):
        """Var violations > 10 triggers a recommendation."""
        returns = _make_returns()
        var_series = [0.001] * len(returns)  # very small VaR → many violations
        report = generate_regulatory_report(
            returns=returns,
            var_series=var_series,
        )
        var_recs = [r for r in report.recommendations if "var" in r.lower() or "recalib" in r.lower()]
        assert len(var_recs) >= 1

    def test_turnover_computed(self):
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
            trades=_make_trades(3),
            portfolio_value=1_000_000.0,
        )
        assert report.n_trades == 3
        assert report.turnover > 0.0

    def test_risk_metrics_populated(self):
        report = generate_regulatory_report(
            returns=_make_returns(),
            var_series=_make_var_series(_make_returns()),
        )
        assert report.volatility > 0.0
        assert isinstance(report.sharpe_ratio, float)


# ---------------------------------------------------------------------------
# generate_best_execution_report tests
# ---------------------------------------------------------------------------

class TestBestExecutionReport:
    """Tests for generate_best_execution_report (independent of tca_strategy)."""

    def test_no_trades_returns_compliant(self):
        result = generate_best_execution_report([])
        assert result["compliant"] is True
        assert result["message"] == "No trades to analyze"

    def test_slippage_bps_computed(self):
        trades = [
            {"price": 100.5, "benchmark_price": 100.0, "venue": "ARCA"},
            {"price": 200.3, "benchmark_price": 200.0, "venue": "NYSE"},
        ]
        result = generate_best_execution_report(trades)
        assert result["n_trades"] == 2
        assert result["avg_slippage_bps"] > 0.0
        assert "ARCA" in result["venues"]
        assert "NYSE" in result["venues"]

    def test_compliant_when_slippage_low(self):
        trades = [
            {"price": 100.01, "benchmark_price": 100.0, "venue": "ARCA"},
        ]
        result = generate_best_execution_report(trades)
        assert result["compliant"] is True

    def test_non_compliant_when_slippage_high(self):
        # slippage = (105 - 100) / 100 * 10000 = 500 bps → > 50 bps
        trades = [
            {"price": 105.0, "benchmark_price": 100.0, "venue": "DARK"},
        ]
        result = generate_best_execution_report(trades)
        assert result["compliant"] is False
        assert any("slippage" in f.lower() for f in result["flags"])

    def test_venue_breakdown(self):
        trades = [
            {"price": 100.1, "benchmark_price": 100.0, "venue": "ARCA"},
            {"price": 100.2, "benchmark_price": 100.0, "venue": "ARCA"},
            {"price": 100.5, "benchmark_price": 100.0, "venue": "NYSE"},
        ]
        result = generate_best_execution_report(trades)
        assert result["venues"]["ARCA"]["n_trades"] == 2
        assert result["venues"]["NYSE"]["n_trades"] == 1

    def test_missing_benchmark_uses_price(self):
        trades = [
            {"price": 100.0, "venue": "ARCA"},  # no benchmark_price
        ]
        result = generate_best_execution_report(trades)
        # benchmark_price defaults to price → slippage = 0
        assert result["avg_slippage_bps"] == 0.0

    def test_missing_venue_defaults_unknown(self):
        trades = [
            {"price": 100.5, "benchmark_price": 100.0},
        ]
        result = generate_best_execution_report(trades)
        assert "unknown" in result["venues"]
