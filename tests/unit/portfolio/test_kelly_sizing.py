"""Unit tests for fractional Kelly position sizing."""

from __future__ import annotations

import numpy as np


class TestComputeKellyFromTrades:
    """Tests for compute_kelly_from_trades function."""

    def test_basic_computation(self):
        """Kelly fraction computed correctly from trade records."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_trades

        trades = [
            {"pnl_pct": 0.02}, {"pnl_pct": 0.03}, {"pnl_pct": -0.01},
            {"pnl_pct": 0.01}, {"pnl_pct": -0.02}, {"pnl_pct": 0.04},
            {"pnl_pct": 0.02}, {"pnl_pct": -0.01}, {"pnl_pct": 0.03},
            {"pnl_pct": 0.01}, {"pnl_pct": -0.01}, {"pnl_pct": 0.02},
            {"pnl_pct": 0.03}, {"pnl_pct": -0.02}, {"pnl_pct": 0.01},
            {"pnl_pct": 0.02}, {"pnl_pct": -0.01}, {"pnl_pct": 0.03},
            {"pnl_pct": 0.01}, {"pnl_pct": -0.01},
        ]

        stats = compute_kelly_from_trades(trades, fraction=0.25)

        assert stats.win_rate > 0.5
        assert stats.payoff_ratio > 0
        assert stats.kelly_fraction >= 0
        assert stats.fractional_kelly == stats.kelly_fraction * 0.25
        assert stats.n_trades == 20

    def test_all_winners(self):
        """All winning trades -> maximum Kelly."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_trades

        trades = [{"pnl_pct": 0.01} for _ in range(30)]
        stats = compute_kelly_from_trades(trades)

        assert stats.win_rate == 1.0
        assert stats.kelly_fraction == 1.0

    def test_all_losers(self):
        """All losing trades -> zero Kelly."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_trades

        trades = [{"pnl_pct": -0.01} for _ in range(30)]
        stats = compute_kelly_from_trades(trades)

        assert stats.win_rate == 0.0
        assert stats.kelly_fraction == 0.0

    def test_insufficient_trades(self):
        """Returns conservative defaults when too few trades."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_trades

        trades = [{"pnl_pct": 0.01}] * 5
        stats = compute_kelly_from_trades(trades, min_trades=20)

        assert stats.kelly_fraction == 0.0  # conservative

    def test_fraction_clamping(self):
        """Fractional Kelly is clamped to [0, 1]."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_trades

        trades = [{"pnl_pct": 0.05}] * 30  # all winners
        stats = compute_kelly_from_trades(trades, fraction=0.5)

        assert 0 <= stats.fractional_kelly <= 1.0


class TestComputeKellyFromReturns:
    """Tests for compute_kelly_from_returns function."""

    def test_basic_computation(self):
        """Kelly fraction computed from return series."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_returns

        np.random.seed(42)
        returns = np.random.randn(200) * 0.01 + 0.001  # slightly positive mean

        stats = compute_kelly_from_returns(returns)

        assert stats.win_rate > 0
        assert stats.n_trades == 200
        assert stats.kelly_fraction >= 0

    def test_positive_edge(self):
        """Positive-edge strategy has positive Kelly."""
        from quantcore.portfolio.kelly_sizing import compute_kelly_from_returns

        # Win rate 60%, avg win 2%, avg loss 1%
        returns = np.concatenate([
            np.random.randn(120) * 0.005 + 0.02,  # winners
            np.random.randn(80) * 0.005 - 0.01,   # losers
        ])

        stats = compute_kelly_from_returns(returns, fraction=0.25)

        assert stats.kelly_fraction > 0
        assert stats.edge > 0


class TestKellySizer:
    """Tests for KellySizer class."""

    def test_fit_from_trades(self):
        """Fitting from trades stores stats."""
        from quantcore.portfolio.kelly_sizing import KellySizer

        sizer = KellySizer(fraction=0.25)
        trades = [{"pnl_pct": 0.02}] * 15 + [{"pnl_pct": -0.01}] * 10

        stats = sizer.fit_from_trades("test_strategy", trades)

        assert sizer.is_fitted
        assert sizer.get_stats("test_strategy") is not None

    def test_fit_from_returns(self):
        """Fitting from returns stores stats."""
        from quantcore.portfolio.kelly_sizing import KellySizer

        sizer = KellySizer(fraction=0.25)
        returns = np.random.randn(200) * 0.01

        stats = sizer.fit_from_returns("test_strategy", returns)

        assert sizer.is_fitted
        assert stats.n_trades == 200

    def test_position_size(self):
        """Position size is fraction of equity."""
        from quantcore.portfolio.kelly_sizing import KellySizer

        sizer = KellySizer(fraction=0.25, max_kelly=0.20)

        # Fit with positive-edge data
        returns = np.concatenate([
            np.random.randn(120) * 0.005 + 0.02,
            np.random.randn(80) * 0.005 - 0.01,
        ])
        sizer.fit_from_returns("strat_a", returns)

        size = sizer.position_size("strat_a", equity=1_000_000)

        assert size > 0
        assert size <= 1_000_000 * 0.20  # max_kelly

    def test_unfitted_strategy_default(self):
        """Unfitted strategy returns 2% default."""
        from quantcore.portfolio.kelly_sizing import KellySizer

        sizer = KellySizer()
        size = sizer.position_size("unknown", equity=1_000_000)

        assert size == 1_000_000 * 0.02

    def test_portfolio_weights(self):
        """Portfolio weights are computed for all fitted strategies."""
        from quantcore.portfolio.kelly_sizing import KellySizer

        sizer = KellySizer(fraction=0.25)

        sizer.fit_from_returns("a", np.random.randn(100) * 0.01 + 0.002)
        sizer.fit_from_returns("b", np.random.randn(100) * 0.01 - 0.001)

        weights = sizer.portfolio_weights(1_000_000)

        assert "a" in weights
        assert "b" in weights
        assert sum(weights.values()) <= 1_000_000


class TestRollingKellySizer:
    """Tests for RollingKellySizer class."""

    def test_update_and_compute(self):
        """Update with returns and compute Kelly."""
        from quantcore.portfolio.kelly_sizing import RollingKellySizer

        sizer = RollingKellySizer(lookback=100, fraction=0.25)

        np.random.seed(42)
        for _ in range(150):
            sizer.update("strat", np.random.randn() * 0.01 + 0.001)

        stats = sizer.current_kelly("strat")
        assert stats.n_trades == 100  # trimmed to lookback

    def test_position_size(self):
        """Position size is computed from rolling window."""
        from quantcore.portfolio.kelly_sizing import RollingKellySizer

        sizer = RollingKellySizer(lookback=100, fraction=0.25)

        for _ in range(150):
            sizer.update("strat", np.random.randn() * 0.01 + 0.002)

        size = sizer.position_size("strat", equity=1_000_000)
        assert 0 < size <= 200_000  # max 20% Kelly

    def test_empty_history(self):
        """Empty history returns zero Kelly."""
        from quantcore.portfolio.kelly_sizing import RollingKellySizer

        sizer = RollingKellySizer()
        stats = sizer.current_kelly("unknown")

        assert stats.kelly_fraction == 0.0
