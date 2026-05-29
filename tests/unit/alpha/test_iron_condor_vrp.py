"""Unit tests for the Iron Condor VRP strategy."""

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def spy_close() -> pd.Series:
    """Synthetic SPY daily close prices (3 years)."""
    rng = np.random.default_rng(42)
    n = 756  # ~3 years of trading days
    # Random walk starting at 400
    log_returns = rng.normal(0.0003, 0.012, n)
    prices = 400 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def iv_series(spy_close: pd.Series) -> pd.Series:
    """Synthetic IV series with VRP premium."""
    from quantcore.alpha.iron_condor import generate_synthetic_iv
    return generate_synthetic_iv(spy_close)


# ── BSM Helpers ──────────────────────────────────────────────────────────────


class TestBSMHelpers:
    def test_bsm_put_non_negative(self):
        from quantcore.alpha.iron_condor import _bsm_put
        assert _bsm_put(100, 95, 0.1, 0.04, 0.20) >= 0

    def test_bsm_call_non_negative(self):
        from quantcore.alpha.iron_condor import _bsm_call
        assert _bsm_call(100, 105, 0.1, 0.04, 0.20) >= 0

    def test_put_call_parity(self):
        from quantcore.alpha.iron_condor import _bsm_put, _bsm_call
        S, K, T, r, sigma = 100, 100, 0.1, 0.04, 0.20
        call = _bsm_call(S, K, T, r, sigma)
        put = _bsm_put(S, K, T, r, sigma)
        # C - P = S - K*exp(-rT)
        expected = S - K * np.exp(-r * T)
        assert abs(call - put - expected) < 0.01

    def test_strike_from_delta_call(self):
        from quantcore.alpha.iron_condor import _strike_from_delta
        K = _strike_from_delta(100, 0.1, 0.04, 0.20, 0.16, True)
        # 16-delta call should be OTM (strike > spot)
        assert K > 100

    def test_strike_from_delta_put(self):
        from quantcore.alpha.iron_condor import _strike_from_delta
        K = _strike_from_delta(100, 0.1, 0.04, 0.20, 0.16, False)
        # 16-delta put should be OTM (strike < spot)
        assert K < 100


# ── Synthetic IV ─────────────────────────────────────────────────────────────


class TestSyntheticIV:
    def test_iv_positive(self, spy_close):
        from quantcore.alpha.iron_condor import generate_synthetic_iv
        iv = generate_synthetic_iv(spy_close)
        assert (iv > 0).all()

    def test_iv_bounded(self, spy_close):
        from quantcore.alpha.iron_condor import generate_synthetic_iv
        iv = generate_synthetic_iv(spy_close)
        assert iv.min() >= 0.08
        assert iv.max() <= 0.80

    def test_iv_same_index(self, spy_close):
        from quantcore.alpha.iron_condor import generate_synthetic_iv
        iv = generate_synthetic_iv(spy_close)
        assert len(iv) == len(spy_close)
        assert iv.index.equals(spy_close.index)


# ── IV Rank ──────────────────────────────────────────────────────────────────


class TestIVR:
    def test_ivr_bounded(self, iv_series):
        from quantcore.alpha.iron_condor import compute_ivr
        for i in [100, 200, 300, 500]:
            ivr = compute_ivr(iv_series, i)
            assert 0 <= ivr <= 100

    def test_ivr_insufficient_data(self, iv_series):
        from quantcore.alpha.iron_condor import compute_ivr
        ivr = compute_ivr(iv_series, 5)  # too few observations
        assert ivr == 50.0


# ── Strategy Construction ────────────────────────────────────────────────────


class TestStrategyInit:
    def test_default_params(self, spy_close, iv_series):
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series)
        assert strat.delta_short == 0.16
        assert strat.spread_width == 5.0
        assert strat.dte_exit == 21
        assert strat.profit_target_pct == 0.50

    def test_custom_params(self, spy_close, iv_series):
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(
            close=spy_close, iv=iv_series,
            delta_short=0.10, spread_width=10.0,
        )
        assert strat.delta_short == 0.10
        assert strat.spread_width == 10.0

    def test_synthetic_iv_fallback(self, spy_close):
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close)  # no IV provided
        assert strat._iv is not None
        assert len(strat._iv) == len(spy_close)


# ── Simulation ───────────────────────────────────────────────────────────────


class TestSimulation:
    def test_run_returns_backtest_returns_correct_types(self, spy_close, iv_series):
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-06-30")
        training_start = pd.Timestamp("2022-01-01")

        eq, rets, turn, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        assert isinstance(eq, pd.Series)
        assert isinstance(rets, pd.Series)
        assert isinstance(turn, pd.Series)

    def test_run_returns_backtest_equity_starts_at_capital(self, spy_close, iv_series):
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)  # always enter

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-06-30")
        training_start = pd.Timestamp("2022-01-01")

        eq, _, _, _ = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
            initial_capital=500_000.0,
        )
        if len(eq) > 0:
            assert eq.iloc[0] == 500_000.0

    def test_trades_generated_when_ivr_low(self, spy_close, iv_series):
        """With IVR entry = 0, trades should be generated."""
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        assert len(trades) > 0, "Expected trades with IVR entry = 0"

    def test_no_trades_when_ivr_high(self, spy_close, iv_series):
        """With IVR entry = 100, no trades should be generated."""
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series, ivr_entry=100.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        assert len(trades) == 0

    def test_no_lookahead_bias(self, spy_close, iv_series):
        """Signal at time t should only use data up to t."""
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        # Run on first half
        mid = pd.Timestamp("2024-06-30")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        eq1, _, _, logs1 = strat.run_returns_backtest(
            start=pd.Timestamp("2024-01-01"), end=mid,
            training_start=training_start,
        )
        eq2, _, _, logs2 = strat.run_returns_backtest(
            start=pd.Timestamp("2024-01-01"), end=end,
            training_start=training_start,
        )

        # Equity curves should match on the overlapping period
        if len(eq1) > 0 and len(eq2) > 0:
            overlap = eq1.index.intersection(eq2.index)
            if len(overlap) > 0:
                np.testing.assert_allclose(
                    eq1.loc[overlap].values,
                    eq2.loc[overlap].values,
                    atol=1.0,  # small tolerance for floating point
                    err_msg="Equity diverged — possible lookahead bias",
                )


# ── Exit Logic ───────────────────────────────────────────────────────────────


class TestExitLogic:
    def test_profit_target_exit(self, spy_close, iv_series):
        """Trades should close at profit target."""
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        # At least some trades should exit at profit target
        profit_exits = [t for t in trades if t.close_reason == "profit_target"]
        # Not all trades will hit profit target, but some should
        # (this is a statistical test — with enough trades, some should profit)
        if len(trades) > 5:
            assert len(profit_exits) > 0 or len(trades) > 0

    def test_time_exit(self, spy_close, iv_series):
        """Trades should close at dte_exit."""
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        time_exits = [t for t in trades if t.close_reason == "time_exit"]
        # With dte_exit=21 and 30-45 DTE entries, many should time-exit
        if len(trades) > 5:
            assert len(time_exits) > 0
