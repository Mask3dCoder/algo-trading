# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""Phase-4 routing guard tests (Audit Mandate 2026-07-12).

Asserts that an OPTIONS strategy is BLOCKED from routing an equity order, and
that a genuine option order is allowed.  No live broker required.
"""

from __future__ import annotations

import os

import pytest

from quantcore.execution.instrument_guard import (
    ASSET_EQUITY,
    ASSET_OPTION,
    build_occ_symbol,
    guard_route,
)


@pytest.fixture()
def kill_path(tmp_path):
    return str(tmp_path / "KILL_SWITCH")


def test_options_strategy_rejects_equity_symbol(kill_path):
    """An OPTIONS strategy trying to trade EQUITY must be blocked + kill switch."""
    assert not os.path.exists(kill_path)
    decision = guard_route(
        strategy_type="OPTIONS",
        asset_type=ASSET_EQUITY,
        strategy_name="options_delta_neutral",
        kill_switch_path=kill_path,
    )
    assert decision.allowed is False
    assert decision.killed is True
    assert "BLOCKING ORDER" in decision.reason
    assert os.path.exists(kill_path), "kill switch sentinel must be created"


def test_options_strategy_allows_option_contract(kill_path):
    """An OPTIONS strategy trading an OPTION_CONTRACT is allowed; no kill switch."""
    decision = guard_route(
        strategy_type="OPTIONS",
        asset_type=ASSET_OPTION,
        strategy_name="options_delta_neutral",
        kill_switch_path=kill_path,
    )
    assert decision.allowed is True
    assert decision.killed is False
    assert not os.path.exists(kill_path)


def test_equity_strategy_allows_equity(kill_path):
    """A genuine EQUITY strategy trading equity is unaffected by the guard."""
    decision = guard_route(
        strategy_type="EQUITY",
        asset_type=ASSET_EQUITY,
        strategy_name="equity_stat_arb",
        kill_switch_path=kill_path,
    )
    assert decision.allowed is True
    assert not os.path.exists(kill_path)


def test_missing_asset_type_blocked_for_options(kill_path):
    """Unknown/blank asset type on an OPTIONS strategy is treated as a violation."""
    decision = guard_route(
        strategy_type="OPTIONS",
        asset_type="",
        strategy_name="options_iron_condor",
        kill_switch_path=kill_path,
    )
    assert decision.allowed is False
    assert os.path.exists(kill_path)


def test_occ_symbol_construction():
    """OCC symbol must be 6-char root + YYMMDD + C/P + 8-digit strike*1000."""
    sym = build_occ_symbol("SPY", "240720", "C", 750.0)
    assert sym == "SPY   240720C00750000"
    assert len(sym) == 21
    put = build_occ_symbol("SPY", "240720", "P", 500.5)
    assert put == "SPY   240720P00500500"


def test_occ_symbol_rejects_bad_right():
    with pytest.raises(ValueError):
        build_occ_symbol("SPY", "240720", "X", 750.0)


def test_occ_symbol_rejects_bad_expiry():
    with pytest.raises(ValueError):
        build_occ_symbol("SPY", "2024-07-20", "C", 750.0)
