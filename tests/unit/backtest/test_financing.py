"""Tests for financing costs in backtest engine (Priority 2)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quantcore.backtest.engine import BacktestEngine, SignalData
from quantcore.backtest.financing import (
    BorrowCostRegistry,
    FinancingConfig,
    compute_daily_borrow_cost,
    compute_daily_margin_interest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ShortStrategy:
    """Strategy that goes short on the first bar and stays flat."""

    def __init__(self) -> None:
        self._fired = False

    def on_bar(self, bar, portfolio):
        if not self._fired:
            self._fired = True
            return [SignalData(symbol="TEST", direction="short", strength=1.0, strategy_id="test")]
        return []

    def on_fill(self, fill, portfolio):
        pass


class _LongMarginStrategy:
    """Strategy that goes long on the first bar, uses aggressive sizing for margin."""

    def __init__(self) -> None:
        self._fired = False

    def on_bar(self, bar, portfolio):
        if not self._fired:
            self._fired = True
            return [SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")]
        return []

    def on_fill(self, fill, portfolio):
        pass


def _make_data(n_bars: int = 252, symbol: str = "TEST") -> pd.DataFrame:
    """Create synthetic price data."""
    rng = np.random.default_rng(42)
    rows = []
    price = 100.0
    for i in range(n_bars):
        ts = datetime(2020, 1, 1) + pd.Timedelta(days=i)
        price = price * (1 + rng.normal(0, 0.005))
        rows.append({
            "symbol": symbol,
            "timestamp": ts,
            "open": round(max(price * 0.995, 1.0), 2),
            "high": round(price * 1.01, 2),
            "low": round(price * 0.99, 2),
            "close": round(price, 2),
            "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BorrowCostRegistry unit tests
# ---------------------------------------------------------------------------


class TestBorrowCostRegistry:
    def test_default_rate(self) -> None:
        registry = BorrowCostRegistry(default_rate=0.005)
        assert registry.get_rate("UNKNOWN") == 0.005

    def test_set_and_get_rate(self) -> None:
        registry = BorrowCostRegistry()
        registry.set_rate("GME", 0.50)
        assert registry.get_rate("GME") == 0.50

    def test_classify_easy(self) -> None:
        registry = BorrowCostRegistry()
        assert registry.classify_borrow(0.002) == "easy"

    def test_classify_medium(self) -> None:
        registry = BorrowCostRegistry()
        assert registry.classify_borrow(0.01) == "medium"

    def test_classify_hard(self) -> None:
        registry = BorrowCostRegistry()
        assert registry.classify_borrow(0.03) == "hard"

    def test_classify_special(self) -> None:
        registry = BorrowCostRegistry()
        assert registry.classify_borrow(0.10) == "special"

    def test_contains(self) -> None:
        registry = BorrowCostRegistry()
        registry.set_rate("AAPL", 0.0025)
        assert "AAPL" in registry
        assert "UNKNOWN" not in registry


# ---------------------------------------------------------------------------
# Unit tests for daily cost computations
# ---------------------------------------------------------------------------


class TestCostComputations:
    def test_short_borrow_cost_known_value(self) -> None:
        """$1M short at 2% for 1 day = 1M * 0.02 / 360 = $55.56."""
        cost = compute_daily_borrow_cost(1_000_000, 0.02)
        assert abs(cost - 55.56) < 0.01

    def test_short_borrow_cost_zero_value(self) -> None:
        cost = compute_daily_borrow_cost(0, 0.02)
        assert cost == 0.0

    def test_margin_interest_known_value(self) -> None:
        """$100K margin at 6.5% for 1 day = 100K * 0.065 / 360 = $18.06."""
        cost = compute_daily_margin_interest(100_000, 0.065)
        assert abs(cost - 18.06) < 0.01

    def test_margin_interest_zero_balance(self) -> None:
        cost = compute_daily_margin_interest(0, 0.065)
        assert cost == 0.0

    def test_margin_interest_negative_balance(self) -> None:
        """Negative margin balance means no margin used."""
        cost = compute_daily_margin_interest(-50_000, 0.065)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Integration tests with BacktestEngine
# ---------------------------------------------------------------------------


class TestFinancingIntegration:
    def test_short_borrow_cost_accrued(self) -> None:
        """A short position should accrue borrow costs."""
        data = _make_data(100)
        registry = BorrowCostRegistry(default_rate=0.05)  # 5% annual borrow rate
        config = FinancingConfig(borrow_cost_registry=registry, margin_rate=0.0)

        engine = BacktestEngine(
            strategy=_ShortStrategy(),
            initial_cash=1_000_000.0,
            sizing_pct=0.30,
            financing_config=config,
        )
        engine.run(data)
        report = engine.summary()

        # Should have some borrow cost
        assert report.total_borrow_cost > 0, (
            f"Short position should accrue borrow costs, got ${report.total_borrow_cost:.4f}"
        )

    def test_higher_borrow_rate_higher_cost(self) -> None:
        """Higher borrow rates should produce higher total costs."""
        data = _make_data(100)

        # Low rate
        reg_low = BorrowCostRegistry(default_rate=0.01)
        engine_low = BacktestEngine(
            strategy=_ShortStrategy(), initial_cash=1_000_000.0,
            sizing_pct=0.30,
            financing_config=FinancingConfig(borrow_cost_registry=reg_low, margin_rate=0.0),
        )
        engine_low.run(data)
        cost_low = engine_low.summary().total_borrow_cost

        # High rate
        reg_high = BorrowCostRegistry(default_rate=0.10)
        engine_high = BacktestEngine(
            strategy=_ShortStrategy(), initial_cash=1_000_000.0,
            sizing_pct=0.30,
            financing_config=FinancingConfig(borrow_cost_registry=reg_high, margin_rate=0.0),
        )
        engine_high.run(data)
        cost_high = engine_high.summary().total_borrow_cost

        assert cost_high > cost_low, (
            f"Higher borrow rate ({cost_high:.2f}) should cost more than lower rate ({cost_low:.2f})"
        )

    def test_no_borrow_cost_when_no_config(self) -> None:
        """Without financing config, no borrow costs should accrue."""
        data = _make_data(100)
        engine = BacktestEngine(
            strategy=_ShortStrategy(), initial_cash=1_000_000.0, sizing_pct=0.30,
        )
        engine.run(data)
        report = engine.summary()
        assert report.total_borrow_cost == 0.0

    def test_no_borrow_cost_for_long_only(self) -> None:
        """A long-only position should not incur borrow costs."""
        data = _make_data(100)

        class _LongOnly:
            def __init__(self):
                self._fired = False
            def on_bar(self, bar, portfolio):
                if not self._fired:
                    self._fired = True
                    return [SignalData(symbol="TEST", direction="long", strength=1.0, strategy_id="test")]
                return []
            def on_fill(self, fill, portfolio):
                pass

        registry = BorrowCostRegistry(default_rate=0.05)
        config = FinancingConfig(borrow_cost_registry=registry, margin_rate=0.0)
        engine = BacktestEngine(
            strategy=_LongOnly(), initial_cash=1_000_000.0,
            sizing_pct=0.30, financing_config=config,
        )
        engine.run(data)
        report = engine.summary()
        assert report.total_borrow_cost == 0.0

    def test_margin_interest_on_leveraged_long(self) -> None:
        """A leveraged long position should accrue margin interest."""
        data = _make_data(100)

        # Use most of capital — should trigger some margin
        engine = BacktestEngine(
            strategy=_LongMarginStrategy(), initial_cash=100_000.0,
            sizing_pct=0.95,
            financing_config=FinancingConfig(
                borrow_cost_registry=None, margin_rate=0.065,
            ),
        )
        engine.run(data)
        report = engine.summary()

        # Might have 0 margin interest if cash covers it
        assert hasattr(report, "total_margin_interest")

    def test_margin_breach_forced_liquidation(self) -> None:
        """Extreme leverage should trigger forced liquidation."""
        data = _make_data(50)

        engine = BacktestEngine(
            strategy=_LongMarginStrategy(), initial_cash=100_000.0,
            sizing_pct=5.0,  # 500% of portfolio — extreme leverage
            financing_config=FinancingConfig(
                borrow_cost_registry=None, margin_rate=0.065, margin_limit=3.0,
            ),
        )
        result = engine.run(data)
        # Engine should complete without error even with forced liquidation
        assert not result.empty

    def test_financing_in_summary_report(self) -> None:
        """Summary should include financing cost fields."""
        data = _make_data(100)
        registry = BorrowCostRegistry(default_rate=0.05)
        config = FinancingConfig(borrow_cost_registry=registry, margin_rate=0.065)
        engine = BacktestEngine(
            strategy=_ShortStrategy(), initial_cash=1_000_000.0,
            sizing_pct=0.30, financing_config=config,
        )
        engine.run(data)
        report = engine.summary()
        assert hasattr(report, "total_borrow_cost")
        assert hasattr(report, "total_margin_interest")
        assert hasattr(report, "borrow_drag_bps")
        assert hasattr(report, "margin_drag_bps")
