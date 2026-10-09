"""Phase 8d: Regression tests for /execution/tca/trade and /execution/tca/strategy.

Verifies that the TCA API endpoints correctly:
  1. Accept JSON payloads and return bps-denominated results (Fix 3)
  2. Convert input dicts to TCAResult objects (Fix 3 type mismatch)
  3. Pass through correct slippage/cost values matching the corrected tca.py (Fix 1)
"""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from quantcore.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /execution/tca/trade
# ---------------------------------------------------------------------------
class TestTCATradeEndpoint:
    """Validate the /execution/tca/trade endpoint returns correct bps values."""

    def test_single_fill_returns_200(self) -> None:
        resp = client.post(
            "/execution/tca/trade",
            params={"benchmark_price": 100.0, "side": "buy"},
            json={"execution_prices": [100.10], "quantities": [100]},
        )
        assert resp.status_code == 200

    def test_single_fill_slippage_bps(self) -> None:
        """One fill at 100.10 vs benchmark 100.00 → 10 bps slippage."""
        resp = client.post(
            "/execution/tca/trade",
            params={"benchmark_price": 100.0, "side": "buy"},
            json={"execution_prices": [100.10], "quantities": [100]},
        )
        data = resp.json()
        assert math.isclose(data["slippage_bps"], 10.0, abs_tol=1e-6)
        assert math.isclose(data["total_cost_bps"], 10.0, abs_tol=1e-6)

    def test_multi_fill_weighted_average(self) -> None:
        """Two fills at different prices → weighted average slippage in bps."""
        resp = client.post(
            "/execution/tca/trade",
            params={"benchmark_price": 100.0, "side": "buy"},
            json={
                "execution_prices": [100.05, 100.15],
                "quantities": [200, 100],
            },
        )
        data = resp.json()
        # weighted_avg_fill = (100.05*200 + 100.15*100) / 300 = 100.0833...
        expected_avg = (100.05 * 200 + 100.15 * 100) / 300
        expected_bps = (expected_avg - 100.0) / 100.0 * 10_000
        assert math.isclose(data["slippage_bps"], expected_bps, abs_tol=1e-4)
        assert math.isclose(data["total_cost_bps"], expected_bps, abs_tol=1e-4)

    def test_sell_slippage_direction(self) -> None:
        """Sell order filling below benchmark → positive slippage (cost)."""
        resp = client.post(
            "/execution/tca/trade",
            params={"benchmark_price": 100.0, "side": "sell"},
            json={"execution_prices": [99.90], "quantities": [100]},
        )
        data = resp.json()
        assert data["slippage_bps"] > 0
        assert math.isclose(data["slippage_bps"], 10.0, abs_tol=1e-6)

    def test_benchmark_price_in_response(self) -> None:
        resp = client.post(
            "/execution/tca/trade",
            params={"benchmark_price": 100.0, "side": "buy"},
            json={"execution_prices": [100.05], "quantities": [100]},
        )
        data = resp.json()
        assert data["benchmark_price"] == 100.0


# ---------------------------------------------------------------------------
# /execution/tca/strategy
# ---------------------------------------------------------------------------
class TestTCAStrategyEndpoint:
    """Validate the /execution/tca/strategy endpoint aggregates trades correctly.

    Fix 3: endpoint converts input dicts to TCAResult objects before calling
    tca_strategy(), ensuring correct type semantics.
    """

    def _make_trade(self, symbol: str = "AAPL", **overrides) -> dict:
        """Build a minimal TCAResult-compatible dict."""
        base = {
            "symbol": symbol,
            "side": "buy",
            "shares": 100.0,
            "arrival_price": 100.0,
            "avg_fill_price": 100.05,
            "market_vwap": 100.02,
            "total_cost_bps": 5.0,
            "spread_cost_bps": 1.0,
            "impact_cost_bps": 3.0,
            "timing_cost_bps": 1.0,
            "opportunity_cost_bps": 0.0,
            "total_cost_dollars": 5.0,
            "strategy": "test",
            "venue": "NYSE",
            "time_of_day": "10:30:00",
            "timestamp": "2025-01-01T10:30:00",
        }
        base.update(overrides)
        return base

    def test_returns_200(self) -> None:
        resp = client.post(
            "/execution/tca/strategy",
            params={"benchmark": "arrival"},
            json=[self._make_trade()],
        )
        assert resp.status_code == 200

    def test_aggregates_single_trade(self) -> None:
        """Single trade: strategy summary n_trades=1, avg_cost_bps=5.0."""
        resp = client.post(
            "/execution/tca/strategy",
            params={"benchmark": "arrival"},
            json=[self._make_trade()],
        )
        data = resp.json()
        assert data["n_trades"] == 1
        assert math.isclose(data["avg_cost_bps"], 5.0, abs_tol=1e-6)

    def test_aggregates_multiple_trades(self) -> None:
        """Two trades at 5.0 and 15.0 bps → avg 10.0 bps."""
        trades = [
            self._make_trade(total_cost_bps=5.0),
            self._make_trade(total_cost_bps=15.0),
        ]
        resp = client.post(
            "/execution/tca/strategy",
            params={"benchmark": "arrival"},
            json=trades,
        )
        data = resp.json()
        assert data["n_trades"] == 2
        assert math.isclose(data["avg_cost_bps"], 10.0, abs_tol=1e-6)

    def test_strategy_summary_fields_present(self) -> None:
        """Response contains all StrategySummary fields."""
        resp = client.post(
            "/execution/tca/strategy",
            params={"benchmark": "arrival"},
            json=[self._make_trade()],
        )
        data = resp.json()
        required_fields = [
            "strategy_name",
            "n_trades",
            "total_notional",
            "total_cost_bps",
            "avg_cost_bps",
            "spread_cost_bps",
            "impact_cost_bps",
            "timing_cost_bps",
            "opportunity_cost_bps",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_benchmark_param_controls_strategy_name(self) -> None:
        resp = client.post(
            "/execution/tca/strategy",
            params={"benchmark": "vwap"},
            json=[self._make_trade()],
        )
        data = resp.json()
        assert data["strategy_name"] == "vwap"
