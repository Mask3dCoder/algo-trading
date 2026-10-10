"""Test: look-ahead bias is eliminated from the backtest engine.

Creates scenarios where next-bar data, if visible during signal generation,
would change the signal.  Verifies signals are identical regardless of what
the next bar looks like after the architectural fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantcore.backtest.engine import BacktestEngine, SignalData


def _make_df(symbol, closes, opens=None, start="2020-01-02"):
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n)
    if opens is None:
        opens = [c * 0.99 for c in closes]
    rows = []
    for dt, cl, op in zip(dates, closes, opens):
        # Ensure OHLCV constraints: high >= max(open,close), low <= min(open,close)
        high = max(op, cl) * 1.01
        low = min(op, cl) * 0.99
        rows.append({
            "symbol": symbol, "timestamp": dt,
            "open": op, "high": high, "low": low,
            "close": cl, "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


def _make_engine(strategy, initial_cash=100_000, slippage_bps=5.0):
    """Construct BacktestEngine using the correct constructor signature."""
    return BacktestEngine(
        strategy=strategy,
        initial_cash=initial_cash,
        slippage_bps=slippage_bps,
    )


class _CanaryStrategy:
    """Records fill_sim._next_bar visibility during on_bar calls.

    The engine passes its own fill_sim reference to the strategy via a
    side-channel (set in the test) to probe whether next_bar leaks.
    """
    def __init__(self):
        self._engine_ref = None  # set by test after engine creation
        self.peeked_opens = []

    def on_bar(self, bar, portfolio):
        nb = self._engine_ref.fill_sim._next_bar if self._engine_ref else None
        self.peeked_opens.append(float(nb.open) if nb is not None else None)
        return []

    def on_fill(self, fill, portfolio):
        pass


class _PureSignalStrategy:
    """Emits BUY when close > threshold, based solely on current bar."""
    def __init__(self, symbol, threshold=100.0):
        self.symbol = symbol
        self.threshold = threshold
        self.signals = []

    def on_bar(self, bar, portfolio):
        if float(bar.close) > self.threshold:
            sig = SignalData(
                symbol=self.symbol, direction="long", strength=1.0,
                strategy_id="pure", notional_size=10_000.0,
            )
            self.signals.append(sig)
            return [sig]
        return []

    def on_fill(self, fill, portfolio):
        pass


class TestFillSimBarrierDuringSignalGeneration:
    """Core guarantee: fill_sim._next_bar is None when on_bar() is called."""

    def test_next_bar_hidden_at_signal_time(self):
        symbol = "TEST"
        df = _make_df(symbol, closes=[98.0, 105.0, 110.0, 90.0],
                      opens=[97.0, 104.0, 109.0, 89.0])
        strat = _CanaryStrategy()
        engine = _make_engine(strat, initial_cash=100_000)
        strat._engine_ref = engine  # wire after creation so strategy can probe fill_sim
        engine.run(df)
        for i, seen in enumerate(strat.peeked_opens):
            assert seen is None, (
                f"Bar {i}: fill_sim._next_bar.open={seen} was visible "
                "during on_bar() — LOOK-AHEAD BIAS DETECTED."
            )


class TestSignalIndependenceFromNextBar:
    """Signals at bar T must not depend on bar T+1 data."""

    def _run(self, symbol, bar1_close, bar2_close):
        df = _make_df(symbol, closes=[bar1_close, bar2_close])
        strat = _PureSignalStrategy(symbol, threshold=100.0)
        engine = _make_engine(strat)
        engine.run(df)
        return strat.signals

    def test_same_signal_with_spike_next_bar(self):
        # Bar 1 close = 105 (above threshold) -> BUY regardless of bar 2
        sigs_crash = self._run("SPY", bar1_close=105.0, bar2_close=50.0)
        sigs_spike = self._run("SPY", bar1_close=105.0, bar2_close=200.0)
        assert len(sigs_crash) > 0 and len(sigs_spike) > 0, "Expected BUY at bar 1"
        assert sigs_crash[0].direction == sigs_spike[0].direction, "Direction must be identical"
        assert sigs_crash[0].strength == sigs_spike[0].strength, "Strength must be identical"

    def test_no_spurious_signal_when_bar1_below_threshold(self):
        # Both bars below threshold (95 < 100) -> no signals at all.
        sigs_low_low = self._run("SPY", bar1_close=95.0, bar2_close=92.0)
        assert len(sigs_low_low) == 0, "No signal when close < threshold on all bars"

        # Bar 1 below, bar 2 above (200 > 100): exactly ONE signal expected (from bar 2).
        # This verifies bar 1 does NOT get a spurious signal due to look-ahead from bar 2.
        sigs_low_high = self._run("SPY", bar1_close=95.0, bar2_close=200.0)
        assert len(sigs_low_high) == 1, (
            f"Expected exactly 1 signal (bar 2), got {len(sigs_low_high)}"
        )
        assert sigs_low_high[0].direction == "long", "Bar 2 should emit a long signal"


class TestPurgedKfoldCVNoCrash:
    """Task 1.4: purged_kfold_cv must not crash with NameError(total)."""

    def test_basic_execution(self):
        from quantcore.backtest.walk_forward import purged_kfold_cv
        folds = purged_kfold_cv(n_bars=500, n_splits=3, test_size=60, embargo_pct=0.01)
        # First fold is skipped because train_end <= 0 (no prior data).
        # Only folds with training data chronologically before test data are kept.
        assert len(folds) == 2, f"Expected 2 folds, got {len(folds)}"

    def test_no_test_train_overlap(self):
        from quantcore.backtest.walk_forward import purged_kfold_cv
        folds = purged_kfold_cv(n_bars=1000, n_splits=5, test_size=100, embargo_pct=0.01)
        for i, fold in enumerate(folds):
            assert fold.test_end <= 1000, f"Fold {i}: test_end={fold.test_end} > n_bars=1000"
            # After the fix: training is always chronologically before test.
            assert fold.train_end <= fold.test_start, (
                f"Fold {i}: train [{fold.train_start},{fold.train_end}] "
                f"not before test [{fold.test_start},{fold.test_end}]"
            )

    def test_returns_correct_split_count(self):
        from quantcore.backtest.walk_forward import purged_kfold_cv
        # First fold always dropped (no prior training data), so expected
        # count is n_splits - 1 when n_bars is large enough for all folds.
        for n_splits in [2, 3, 5]:
            folds = purged_kfold_cv(n_bars=600, n_splits=n_splits)
            assert len(folds) == n_splits - 1, f"Expected {n_splits - 1} folds, got {len(folds)}"
