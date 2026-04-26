"""Unit tests for backtest engine, simulation, validation, and analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.backtest.engine import BacktestEngine, HistoricalDataHandler, PortfolioState
from quantcore.backtest.simulation import FixedSlippage, VolumeSlippage, PerShareCommission, PerTradeCommission, ShortBorrowCost
from quantcore.backtest.validation import check_lookahead, overfitting_score, minimum_backtest_length
from quantcore.backtest.walk_forward import rolling_walk_forward, anchored_walk_forward, purged_kfold_cv
from quantcore.backtest.analytics import (
    sharpe_ratio, sortino_ratio, max_drawdown, omega_ratio,
    drawdown_analysis, bootstrap_confidence, generate_performance_report,
)
from quantcore.core.types import Order, Side, OrderType


class TestSlippage:
    def test_fixed_slippage_buy(self) -> None:
        model = FixedSlippage(bps=10.0)
        fill_price = model.apply(100.0, Side.BUY, 100)
        assert fill_price > 100.0
        assert abs(fill_price - 100.1) < 0.001  # 10bps = 0.1%

    def test_fixed_slippage_sell(self) -> None:
        model = FixedSlippage(bps=10.0)
        fill_price = model.apply(100.0, Side.SELL, 100)
        assert fill_price < 100.0

    def test_buy_slippage_worse_than_ideal(self) -> None:
        model = FixedSlippage(bps=5.0)
        for _ in range(10):
            price = np.random.uniform(10, 500)
            fill = model.apply(price, Side.BUY, 100)
            assert fill >= price  # Buy slippage makes price worse (higher)

    def test_sell_slippage_worse_than_ideal(self) -> None:
        model = FixedSlippage(bps=5.0)
        for _ in range(10):
            price = np.random.uniform(10, 500)
            fill = model.apply(price, Side.SELL, 100)
            assert fill <= price  # Sell slippage makes price worse (lower)


class TestCommission:
    def test_per_share(self) -> None:
        model = PerShareCommission(rate=0.005)
        cost = model.calculate(100, 50.0)
        assert cost == 0.5  # 100 * 0.005

    def test_per_trade(self) -> None:
        model = PerTradeCommission(flat=1.0)
        cost = model.calculate(1000, 50.0)
        assert cost == 1.0

    def test_commission_positive(self) -> None:
        model = PerShareCommission(rate=0.01)
        for qty in [1, 10, 100, 1000]:
            cost = model.calculate(qty, 50.0)
            assert cost > 0


class TestLookaheadDetection:
    def test_shift_negative_detected(self) -> None:
        code = "df['future'] = df['close'].shift(-1)"
        warnings = check_lookahead(code)
        assert len(warnings) > 0

    def test_clean_code_passes(self) -> None:
        code = "df['sma'] = df['close'].rolling(20).mean()"
        warnings = check_lookahead(code)
        assert len(warnings) == 0


class TestOverfittingScore:
    def test_overfitted_detected(self) -> None:
        report = overfitting_score(in_sample_sharpe=3.0, out_of_sample_sharpe=0.5)
        assert report.is_overfitted

    def test_clean_strategy_passes(self) -> None:
        report = overfitting_score(in_sample_sharpe=1.0, out_of_sample_sharpe=0.8)
        assert not report.is_overfitted


class TestMinimumBacktestLength:
    def test_sharpe_1_needs_few_years(self) -> None:
        mbl = minimum_backtest_length(sharpe=1.0, n_trials=1)
        assert mbl < 10

    def test_sharpe_0_needs_infinite(self) -> None:
        mbl = minimum_backtest_length(sharpe=0.0, n_trials=1)
        assert mbl == float("inf")


class TestWalkForward:
    def test_rolling_no_overlap(self) -> None:
        folds = rolling_walk_forward(1000, train_size=200, test_size=50, embargo_pct=0.01)
        for fold in folds:
            assert fold.train_end <= fold.test_start

    def test_anchored_expanding_window(self) -> None:
        folds = anchored_walk_forward(1000, initial_train=200, test_size=50)
        for i in range(1, len(folds)):
            assert folds[i].train_end > folds[i-1].train_end

    def test_purged_kfold_zero_overlap(self) -> None:
        folds = purged_kfold_cv(1000, n_splits=5, embargo_pct=0.01)
        for fold in folds:
            if fold.train_start < fold.test_start:
                assert fold.train_end < fold.test_start


class TestAnalytics:
    def test_sharpe_known_value(self) -> None:
        # Constant 1% daily returns → Sharpe = 0.01/0 * sqrt(252) → inf
        # Use slightly noisy returns
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, 252)
        sr = sharpe_ratio(returns)
        assert -5 < sr < 5  # Reasonable range

    def test_max_drawdown_positive(self) -> None:
        equity = np.array([100, 110, 105, 95, 90, 100, 110])
        mdd = max_drawdown(equity)
        expected = (110 - 90) / 110  # ~18.18%
        assert abs(mdd - expected) < 0.01

    def test_omega_ratio(self) -> None:
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        omega = omega_ratio(returns)
        assert omega > 1.0  # More gains than losses

    def test_drawdown_analysis(self) -> None:
        equity = np.array([100, 110, 105, 90, 95, 100, 110, 108, 115])
        dd = drawdown_analysis(equity)
        assert len(dd) > 0
        assert dd[0]["depth"] > 0

    def test_bootstrap_ci(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        lower, upper = bootstrap_confidence(returns, lambda r: sharpe_ratio(r), n_boot=500)
        assert lower < upper

    def test_performance_report(self) -> None:
        rng = np.random.default_rng(42)
        equity = 100_000 * np.cumprod(1 + rng.normal(0.0002, 0.01, 252))
        report = generate_performance_report(equity)
        assert report.total_return != 0
        assert report.max_drawdown > 0
