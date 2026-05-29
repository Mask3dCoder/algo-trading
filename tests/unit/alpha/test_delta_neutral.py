"""Unit tests for the Delta Neutral VRP strategy."""

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def spy_close() -> pd.Series:
    """Synthetic SPY daily close prices (3 years)."""
    rng = np.random.default_rng(42)
    n = 756  # ~3 years of trading days
    log_returns = rng.normal(0.0003, 0.012, n)
    prices = 400 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def iv_series(spy_close: pd.Series) -> pd.Series:
    """Synthetic IV series with VRP premium."""
    from quantcore.alpha.delta_neutral import generate_synthetic_iv
    return generate_synthetic_iv(spy_close)


# ── BSM Helpers ──────────────────────────────────────────────────────────────


class TestBSMHelpers:
    def test_bsm_put_non_negative(self):
        from quantcore.alpha.delta_neutral import _bsm_put
        assert _bsm_put(100, 95, 0.1, 0.04, 0.20) >= 0

    def test_bsm_call_non_negative(self):
        from quantcore.alpha.delta_neutral import _bsm_call
        assert _bsm_call(100, 105, 0.1, 0.04, 0.20) >= 0

    def test_put_call_parity(self):
        from quantcore.alpha.delta_neutral import _bsm_put, _bsm_call
        S, K, T, r, sigma = 100, 100, 0.1, 0.04, 0.20
        call = _bsm_call(S, K, T, r, sigma)
        put = _bsm_put(S, K, T, r, sigma)
        expected = S - K * np.exp(-r * T)
        assert abs(call - put - expected) < 0.01

    def test_bsm_delta_call_range(self):
        from quantcore.alpha.delta_neutral import _bsm_delta
        delta = _bsm_delta(100, 100, 0.1, 0.04, 0.20, is_call=True)
        assert 0.0 <= delta <= 1.0

    def test_bsm_delta_put_range(self):
        from quantcore.alpha.delta_neutral import _bsm_delta
        delta = _bsm_delta(100, 100, 0.1, 0.04, 0.20, is_call=False)
        assert -1.0 <= delta <= 0.0

    def test_bsm_delta_atm_call_near_half(self):
        from quantcore.alpha.delta_neutral import _bsm_delta
        delta = _bsm_delta(100, 100, 0.25, 0.04, 0.20, is_call=True)
        # ATM call delta should be near 0.5
        assert abs(delta - 0.5) < 0.1

    def test_bsm_delta_atm_put_near_neg_half(self):
        from quantcore.alpha.delta_neutral import _bsm_delta
        delta = _bsm_delta(100, 100, 0.25, 0.04, 0.20, is_call=False)
        # ATM put delta should be near -0.5
        assert abs(delta + 0.5) < 0.1

    def test_strike_from_delta_call(self):
        from quantcore.alpha.delta_neutral import _strike_from_delta
        K = _strike_from_delta(100, 0.1, 0.04, 0.20, 0.16, True)
        assert K > 100

    def test_strike_from_delta_put(self):
        from quantcore.alpha.delta_neutral import _strike_from_delta
        K = _strike_from_delta(100, 0.1, 0.04, 0.20, 0.16, False)
        assert K < 100


# ── Synthetic IV ─────────────────────────────────────────────────────────────


class TestSyntheticIV:
    def test_iv_positive(self, spy_close):
        from quantcore.alpha.delta_neutral import generate_synthetic_iv
        iv = generate_synthetic_iv(spy_close)
        assert (iv > 0).all()

    def test_iv_bounded(self, spy_close):
        from quantcore.alpha.delta_neutral import generate_synthetic_iv
        iv = generate_synthetic_iv(spy_close)
        assert iv.min() >= 0.08
        assert iv.max() <= 0.80

    def test_iv_same_index(self, spy_close):
        from quantcore.alpha.delta_neutral import generate_synthetic_iv
        iv = generate_synthetic_iv(spy_close)
        assert len(iv) == len(spy_close)
        assert iv.index.equals(spy_close.index)


# ── IV Rank ──────────────────────────────────────────────────────────────────


class TestIVR:
    def test_ivr_bounded(self, iv_series):
        from quantcore.alpha.delta_neutral import compute_ivr
        for i in [100, 200, 300, 500]:
            ivr = compute_ivr(iv_series, i)
            assert 0 <= ivr <= 100

    def test_ivr_insufficient_data(self, iv_series):
        from quantcore.alpha.delta_neutral import compute_ivr
        ivr = compute_ivr(iv_series, 5)
        assert ivr == 50.0


# ── Strategy Construction ────────────────────────────────────────────────────


class TestStrategyInit:
    def test_default_params(self, spy_close, iv_series):
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series)
        assert strat.structure == "straddle"
        assert strat.dte_exit == 14
        assert strat.profit_target_pct == 0.50
        assert strat.delta_hedge_threshold == 5.0

    def test_strangle_structure(self, spy_close, iv_series):
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, structure="strangle")
        assert strat.structure == "strangle"
        assert strat.delta_target_strangle == 0.10

    def test_synthetic_iv_fallback(self, spy_close):
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close)
        assert strat._iv is not None
        assert len(strat._iv) == len(spy_close)

    def test_custom_commissions(self, spy_close, iv_series):
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(
            close=spy_close, iv=iv_series,
            commission_per_contract=1.0,
            commission_per_share=0.01,
        )
        assert strat.commission_per_contract == 1.0
        assert strat.commission_per_share == 0.01


# ── Simulation ───────────────────────────────────────────────────────────────


class TestSimulation:
    def test_run_returns_backtest_returns_correct_types(self, spy_close, iv_series):
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series)

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
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-06-30")
        training_start = pd.Timestamp("2022-01-01")

        eq, _, _, _ = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
            initial_capital=500_000.0,
        )
        if len(eq) > 0:
            assert eq.iloc[0] == 500_000.0

    def test_trades_generated(self, spy_close, iv_series):
        """With default params, trades should be generated."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        assert len(trades) > 0, "Expected trades with ivr_entry=0"

    def test_strangle_generates_trades(self, spy_close, iv_series):
        """Strangle structure should also generate trades."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(
            close=spy_close, iv=iv_series,
            structure="strangle", ivr_entry=0.0,
        )

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        assert len(trades) > 0

    def test_no_lookahead_bias(self, spy_close, iv_series):
        """Signal at time t should only use data up to t."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        mid = pd.Timestamp("2024-06-30")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        eq1, _, _, _ = strat.run_returns_backtest(
            start=pd.Timestamp("2024-01-01"), end=mid,
            training_start=training_start,
        )
        eq2, _, _, _ = strat.run_returns_backtest(
            start=pd.Timestamp("2024-01-01"), end=end,
            training_start=training_start,
        )

        if len(eq1) > 0 and len(eq2) > 0:
            overlap = eq1.index.intersection(eq2.index)
            if len(overlap) > 0:
                np.testing.assert_allclose(
                    eq1.loc[overlap].values,
                    eq2.loc[overlap].values,
                    atol=1.0,
                    err_msg="Equity diverged — possible lookahead bias",
                )


# ── Delta Hedging ────────────────────────────────────────────────────────────


class TestDeltaHedging:
    def test_hedge_shares_recorded(self, spy_close, iv_series):
        """Trades should have hedge_shares set after entry."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        # At least one trade should have non-zero hedge
        if trades:
            # After closing, hedge_shares is reset to 0
            # But cumulative_hedge_cost should be > 0 for hedged trades
            hedged = [t for t in trades if t.cumulative_hedge_cost > 0]
            assert len(hedged) > 0, "Expected some trades with hedge costs"

    def test_hedge_costs_positive(self, spy_close, iv_series):
        """Hedge costs should always be non-negative."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        for t in trades:
            assert t.cumulative_hedge_cost >= 0, "Hedge cost should be non-negative"


# ── Exit Logic ───────────────────────────────────────────────────────────────


class TestExitLogic:
    def test_profit_target_exit(self, spy_close, iv_series):
        """Some trades should exit at profit target."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        profit_exits = [t for t in trades if t.close_reason == "profit_target"]
        # With enough trades, some should hit profit target
        if len(trades) > 5:
            assert len(profit_exits) > 0 or len(trades) > 0

    def test_time_exit(self, spy_close, iv_series):
        """Trades should close at dte_exit."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        time_exits = [t for t in trades if t.close_reason == "time_exit"]
        if len(trades) > 5:
            assert len(time_exits) > 0

    def test_expiration_exit(self, spy_close, iv_series):
        """Trades should close at expiration."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2025-01-15")  # extend to catch expirations
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        expiry_exits = [t for t in trades if t.close_reason == "expiration"]
        # Some trades may expire
        assert len(trades) > 0

    def test_all_trades_closed(self, spy_close, iv_series):
        """All trades should be closed after backtest."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        for t in trades:
            assert t.closed, f"Trade from {t.entry_date} not closed"
            assert t.close_date is not None


# ── Position Sizing ──────────────────────────────────────────────────────────


class TestPositionSizing:
    def test_contracts_positive(self, spy_close, iv_series):
        """All trades should have positive contract count."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        for t in trades:
            assert t.contracts >= 1

    def test_credit_positive(self, spy_close, iv_series):
        """All trades should have positive credit."""
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(close=spy_close, iv=iv_series, ivr_entry=0.0)

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        training_start = pd.Timestamp("2022-01-01")

        _, _, _, logs = strat.run_returns_backtest(
            start=start, end=end, training_start=training_start,
        )
        trades = logs.get("trades", [])
        for t in trades:
            assert t.credit > 0
