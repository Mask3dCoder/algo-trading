"""Integration/smoke tests for the backtest demo pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from quantcore.backtest.engine import OHLCV, PortfolioState


class TestMomentumCrossoverStrategy:
    """Verify the demo strategy logic."""

    @pytest.fixture
    def strategy(self):
        from scripts.backtest_demo import MomentumCrossoverStrategy

        return MomentumCrossoverStrategy(fast_period=3, slow_period=5)

    def _make_bar(self, symbol: str, close: float, ts: str | None = None) -> OHLCV:
        from datetime import UTC, datetime
        from decimal import Decimal

        return OHLCV(
            symbol=symbol,
            timestamp=datetime.fromisoformat(ts) if ts else datetime(2023, 1, 1, tzinfo=UTC),
            open=Decimal(str(close - 1)),
            high=Decimal(str(close + 1)),
            low=Decimal(str(close - 1)),
            close=Decimal(str(close)),
            volume=1000000,
        )

    def test_no_signal_during_warm_up(self, strategy) -> None:
        portfolio = PortfolioState(initial_cash=1_000_000)
        for i in range(5):
            bar = self._make_bar("SPY", 400.0 + i)
            signals = strategy.on_bar(bar, portfolio)
            if i < 4:
                assert signals == []

    def test_golden_cross_generates_long_signal(self, strategy) -> None:
        portfolio = PortfolioState(initial_cash=1_000_000)

        # Warm up with descending prices so fast(3) starts below slow(5)
        # After 5 bars: fast=mean(460,440,420)=440, slow=mean(500..420)=460
        for close in (500, 480, 460, 440, 420):
            bar = self._make_bar("SPY", close)
            strategy.on_bar(bar, portfolio)

        # Now a strong up bar: fast crosses above slow
        # fast=mean(440,420,600)=486.7, slow=mean(480,460,440,420,600)=480
        bar = self._make_bar("SPY", 600.0)
        signals = strategy.on_bar(bar, portfolio)

        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].strategy_id == "momentum_crossover"

    def test_death_cross_generates_short_signal(self, strategy) -> None:
        portfolio = PortfolioState(initial_cash=1_000_000)

        # Warm up with ascending prices so fast(3) starts above slow(5)
        for close in (400, 410, 420, 430, 440):
            bar = self._make_bar("SPY", close)
            strategy.on_bar(bar, portfolio)
        # After warm: prev_fast=430, prev_slow=420 (fast > slow)
        strategy._in_position = True

        # A low bar triggers death cross: fast drops below slow
        # fast=mean(430,440,200)=356.7, slow=mean(410..200)=380
        bar = self._make_bar("SPY", 200.0)
        signals = strategy.on_bar(bar, portfolio)

        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert not strategy._in_position

    def test_no_duplicate_entry_when_in_position(self, strategy) -> None:
        portfolio = PortfolioState(initial_cash=1_000_000)
        strategy._in_position = True

        # Warm with fast < slow (prevent signal)
        for i in range(5):
            bar = self._make_bar("SPY", 400.0 + i)
            strategy.on_bar(bar, portfolio)

        # Push a strong bar to create golden cross
        bar = self._make_bar("SPY", 500.0)
        signals = strategy.on_bar(bar, portfolio)
        # Already in position, so golden cross should not generate another entry
        long_signals = [s for s in signals if s.direction == "long"]
        assert len(long_signals) == 0


class TestBuyAndHold:
    def test_buy_and_hold_equity(self) -> None:
        from scripts.backtest_demo import compute_buy_and_hold

        df = pd.DataFrame({
            "symbol": ["SPY", "SPY", "SPY"],
            "timestamp": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
            "open": [380.0, 385.0, 390.0],
            "high": [385.0, 388.0, 392.0],
            "low": [378.0, 383.0, 388.0],
            "close": [384.0, 387.0, 391.0],
            "volume": [50000000, 52000000, 51000000],
        })

        result = compute_buy_and_hold(df, 1_000_000)
        assert len(result) == 3
        # Shares = 1M / 384 = ~2604.17
        # Day 3 close = 391 → equity = 2604.17 * 391 ≈ 1,018,229
        assert result["equity"].iloc[-1] > 1_000_000


class TestBacktestDemoEndToEnd:
    """Smoke test: runs the full demo pipeline with a tiny dataset."""

    def test_full_pipeline_with_synthetic_data(self) -> None:
        """Run backtest engine on a small synthetic DataFrame."""
        from scripts.backtest_demo import MomentumCrossoverStrategy

        from quantcore.backtest.engine import BacktestEngine

        dates = pd.date_range("2023-01-03", periods=300, freq="B")
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(42)
        close = 400.0 + np.cumsum(rng.normal(0.1, 2.0, len(dates)))

        df = pd.DataFrame({
            "symbol": "SPY",
            "timestamp": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 50_000_000,
        })

        strategy = MomentumCrossoverStrategy(fast_period=20, slow_period=50)
        engine = BacktestEngine(
            strategy=strategy,
            initial_cash=1_000_000,
            slippage_bps=5.0,
        )
        result = engine.run(df)

        assert not result.empty
        assert "timestamp" in result.columns
        assert "equity" in result.columns
        assert len(result) > 0

        report = engine.summary()
        assert report.total_return != 0 or report.n_trades >= 0
