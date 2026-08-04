"""Regression: DeltaNeutralStrategy must run through BacktestEngine end-to-end.

The 2026-08-03 insurance check found ``DeltaNeutralStrategy`` crashed in the
engine on the FIRST snapshot with a twofold interface incompatibility:

1. ``AttributeError: 'DeltaNeutralStrategy' object has no attribute 'on_bar'``
   (``quantcore/backtest/engine.py`` ``_generate_signals``, legacy single-bar
   path). The strategy implemented the batch/live ``on_data`` interface but
   never implemented the engine's ``on_bar`` entry point.
2. Once ``on_bar`` existed, the inherited live-signature ``on_fill(symbol,
   fill_price, fill_qty, side)`` would ``TypeError`` at the first fill
   (``engine.py`` ``_execute_queued_orders``), because the engine's Strategy
   Protocol calls ``on_fill(fill, portfolio)``.

Both are pinned by these tests so the incompatibility cannot silently return.

**Scope / equity note (read before quoting any number from this test).** This
suite verifies *interface compatibility only*, not strategy performance. The
strategy trades options, but the engine is fed equity bars and cannot fill
option contracts, so every signal the adapter forwards is a stock-hedge leg
(see ``DeltaNeutralStrategy.on_bar`` docstring). On this fixture the engine
equity is *not* a meaningful P&L: the strategy's constant-fallback IV
(``implied_vol`` is absent from the bars, so ``on_data`` uses a flat 0.30) keeps
``_compute_ivr`` at its cold-start 55.0, so it churns entries, and its up-to-5
same-symbol signals per bar interact badly with the engine's per-symbol
position model, producing an unstable equity curve. Do not assert on equity
level here — a faithful options backtest requires the options data pipeline
(blocker #2 on the PIT ingestion line).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantcore.backtest.engine import BacktestEngine
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.options.delta_neutral import DeltaNeutralStrategy

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_BARS = _FIXTURES / "spy_raw_daily_2020_2024.csv"
_INITIAL_CASH = 1_000_000.0


def _load_spy() -> pd.DataFrame:
    bars = pd.read_csv(_BARS)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars["symbol"] = "SPY"
    cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    return bars[cols].sort_values("timestamp").reset_index(drop=True)


def _make_engine() -> tuple[DeltaNeutralStrategy, BacktestEngine]:
    config = StrategyConfig(
        name="options_delta_neutral",
        strategy_type=StrategyType.OPTIONS,
        max_position_notional=100_000.0,
    )
    strategy = DeltaNeutralStrategy(config)
    engine = BacktestEngine(
        strategy,
        initial_cash=_INITIAL_CASH,
        slippage_bps=0,
        commission_per_share=0,
    )
    return strategy, engine


def test_delta_neutral_runs_through_engine_without_crashing() -> None:
    """Engine completes end-to-end — would raise on the old on_bar AttributeError."""
    strategy, engine = _make_engine()
    bars = _load_spy()
    result = engine.run(bars)

    assert not result.empty
    # The engine must produce a finite equity column (not NaN/Inf from a blowup).
    equity = result["equity"]
    assert np.all(np.isfinite(equity)), "equity contains NaN/Inf"

    # The adapter must be invoked for every bar (it accumulates the rolling
    # close window that on_data needs), not no-op forever.
    assert len(strategy._hist_close) == len(bars), "on_bar adapter was not invoked for every bar"

    # The full on_data logic ran through the adapter: entries (long hedge) and
    # delta re-hedges (both long and short hedge legs) produce signals.
    assert len(engine.portfolio.signal_log) > 0, "strategy produced no signals through the adapter"
    directions = {s["direction"] for s in engine.portfolio.signal_log}
    assert {"long", "short"} <= directions, f"expected long and short hedge signals, got {directions}"

    # The on_fill signature tolerance must hold: the engine filled orders and
    # called strategy.on_fill(fill, portfolio) without a TypeError, starting
    # with the delta_neutral_entry stock hedge (BUY).
    assert len(engine.portfolio.trades) > 0, "no fills — signals were dropped, not executed"
    assert engine.portfolio.trades[0].side.value == "buy", (
        "first executed signal should be the delta_neutral_entry long-stock hedge"
    )


def test_delta_neutral_signals_flow_to_fills_not_silently_dropped() -> None:
    """The hedge-leg signals become real orders/fills.

    Guards against a future regression where the adapter silently no-ops (or the
    engine drops the signals) while the run still passes.
    """
    _, engine = _make_engine()
    engine.run(_load_spy())

    assert len(engine.portfolio.trades) > 0, "no fills — signals were dropped, not executed"
    # Fills carry both sides across the run (the strategy both buys and sells
    # hedge legs), confirming both the BUY and SELL paths execute.
    sides = {fill.side.value for fill in engine.portfolio.trades}
    assert sides == {"buy", "sell"}, f"expected both buy and sell fills, got {sides}"
