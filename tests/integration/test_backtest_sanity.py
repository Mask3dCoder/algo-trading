"""Backtest-engine honesty gates (audit Step 3).

``buy_and_hold(SPY)`` through ``BacktestEngine.run()`` with **zero** costs must
reproduce SPY's actual total return — price appreciation plus dividends —
computed independently from the same *unadjusted* data. With realistic costs the
drag must be exactly one round-trip of spread + commission, never mysterious
erosion. If either fails, the engine is fiction and no strategy validation built
on it means anything.

Fixtures are real SPY daily bars (unadjusted) + the dividend schedule for
2020-01-02 → 2024-12-31, committed under ``tests/fixtures/`` — no network at
test time. Deliberately NOT marked ``integration`` (no Docker required) so this
runs in the default suite on every CI run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantcore.backtest.corporate_actions import CorporateActionsConfig
from quantcore.backtest.engine import BacktestEngine
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.macro.simple_test_strategy import SimpleBuyHoldStrategy

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_BARS = _FIXTURES / "spy_raw_daily_2020_2024.csv"
_DIVS = _FIXTURES / "spy_dividends_2020_2024.csv"
_INITIAL_CASH = 1_000_000.0

# Cash-credited (not reinvested) dividends + fill-at-open mechanics vs Yahoo's
# close-to-close reinvested adj-close ratio. Tight enough to catch phantom alpha
# (a missing-dividend engine lands ~13% low; a double-count ~13% high) while
# leaving room for the reinvestment/timing model difference (~3-4%).
_BENCHMARK_TOL_REL = 0.08


def _load_spy() -> tuple[pd.DataFrame, dict[str, dict[str, float]], float]:
    """Load the fixture: (engine-ready bars, dividend schedule, Yahoo total return)."""
    bars = pd.read_csv(_BARS)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["symbol"] = "SPY"
    benchmark = float(bars["adj_close"].iloc[-1] / bars["adj_close"].iloc[0] - 1)
    divs = pd.read_csv(_DIVS)
    schedule = {"SPY": {row.ex_date: float(row.amount) for row in divs.itertuples()}}
    cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    return bars[cols], schedule, benchmark


def _shares_and_prices(data: pd.DataFrame) -> tuple[int, float, float]:
    """Replicate the engine's sizing + fill mechanics from the raw data.

    shares    = int(initial_cash / first_close)      (signal-snapshot close)
    exec_open = open of the SECOND snapshot          (queued order fills at next open)
    last_open = open of the LAST snapshot            (end-of-run forced liquidation)
    """
    first_close = float(data["close"].iloc[0])
    shares = int(_INITIAL_CASH / first_close)
    return shares, float(data["open"].iloc[1]), float(data["open"].iloc[-1])


def _div_total(data: pd.DataFrame, schedule: dict[str, dict[str, float]], exec_date) -> float:
    """Dividends credited while the position is held: ex-date in [exec_date, last_date]."""
    last_date = data["timestamp"].iloc[-1].date()
    return sum(
        amt for d, amt in schedule["SPY"].items()
        if exec_date <= pd.Timestamp(d).date() <= last_date
    )


def _buy_and_hold_return(data, schedule, *, slippage_bps, commission_per_share) -> float:
    config = StrategyConfig(
        name="simple_buy_hold",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=_INITIAL_CASH,
    )
    strategy = SimpleBuyHoldStrategy(config)
    engine = BacktestEngine(
        strategy,
        initial_cash=_INITIAL_CASH,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        corporate_actions_config=CorporateActionsConfig(dividends=schedule),
    )
    result = engine.run(data)
    return float(result["equity"].iloc[-1]) / _INITIAL_CASH - 1


def test_buy_and_hold_zero_cost_matches_spy_total_return() -> None:
    """With zero costs, buy-and-hold(SPY) == SPY total return (price + dividends)."""
    data, schedule, benchmark = _load_spy()
    engine_return = _buy_and_hold_return(data, schedule, slippage_bps=0, commission_per_share=0)

    shares, exec_open, last_open = _shares_and_prices(data)
    div_total = _div_total(data, schedule, data["timestamp"].iloc[1].date())
    expected = (shares * (last_open - exec_open) + shares * div_total) / _INITIAL_CASH

    assert shares > 0, "zero shares sized — the gate is testing nothing"
    # 1. Engine accounting matches an independent computation from raw prices + dividends.
    assert engine_return == pytest.approx(expected, rel=1e-6)
    # 2. That model is grounded in SPY's actual total return over the window.
    assert expected == pytest.approx(benchmark, rel=_BENCHMARK_TOL_REL)
    # 3. Dividends actually mattered: total return >> raw price return.
    price_return = float(data["close"].iloc[-1] / data["close"].iloc[0] - 1)
    assert expected > price_return + 0.05, "dividends appear to be missing from the return"


def test_buy_and_hold_cost_drag_is_exactly_one_round_trip() -> None:
    """Realistic costs must show exactly one round-trip of spread + commission."""
    data, schedule, _ = _load_spy()
    zero = _buy_and_hold_return(data, schedule, slippage_bps=0, commission_per_share=0)
    costed = _buy_and_hold_return(data, schedule, slippage_bps=5, commission_per_share=0.005)

    shares, exec_open, last_open = _shares_and_prices(data)
    div_total = _div_total(data, schedule, data["timestamp"].iloc[1].date())

    # Buy at open*(1+5bps) + commission; sell at open*(1-5bps) - commission.
    expected_costed = (
        shares * (last_open * (1 - 5e-4) - exec_open * (1 + 5e-4))
        + shares * div_total
        - 2 * shares * 0.005
    ) / _INITIAL_CASH
    assert costed == pytest.approx(expected_costed, rel=1e-6)

    # The drag is the explainable round-trip cost, nothing else.
    expected_drag = (shares * (exec_open + last_open) * 5e-4 + 2 * shares * 0.005) / _INITIAL_CASH
    assert zero - costed == pytest.approx(expected_drag, rel=1e-6)
    assert expected_drag < 0.01, "gate cost model changed — expected drag is trivial (<1%)"


def test_dividend_before_position_opens_is_not_credited() -> None:
    """A dividend on the first bar (before the queued buy fills next bar) is ignored."""
    data, schedule, _ = _load_spy()
    out = {"SPY": dict(schedule["SPY"])}
    earliest = data["timestamp"].iloc[0].date().isoformat()
    out["SPY"][earliest] = 1_000.0  # $1000/share — impossible to miss if wrongly credited

    baseline = _buy_and_hold_return(data, schedule, slippage_bps=0, commission_per_share=0)
    with_early_div = _buy_and_hold_return(data, out, slippage_bps=0, commission_per_share=0)
    assert with_early_div == pytest.approx(baseline, rel=1e-12)
