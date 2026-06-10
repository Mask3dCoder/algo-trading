from quantcore.risk.strategy_kill_switch import StrategyKillSwitch, StrategyKillSwitchConfig


def test_strategy_kill_switch_halts_at_drawdown_threshold() -> None:
    guard = StrategyKillSwitch(StrategyKillSwitchConfig(max_drawdown_pct=-0.20))

    first = guard.update("equity_stat_arb", 100.0)
    assert not first.is_halted
    assert first.drawdown_pct == 0.0

    second = guard.update("equity_stat_arb", 90.0)
    assert not second.is_halted
    assert second.drawdown_pct == -0.10

    third = guard.update("equity_stat_arb", 79.0)
    assert third.is_halted
    assert third.drawdown_pct <= -0.20
    assert guard.is_halted("equity_stat_arb")

    fourth = guard.update("equity_stat_arb", 85.0)
    assert fourth.is_halted


def test_strategy_kill_switch_reset_allows_strategy_again() -> None:
    guard = StrategyKillSwitch(StrategyKillSwitchConfig(max_drawdown_pct=-0.20))
    guard.update("macro_carry", 100.0)
    guard.update("macro_carry", 79.0)
    assert guard.is_halted("macro_carry")

    guard.reset("macro_carry")
    assert not guard.is_halted("macro_carry")
    assert guard.update("macro_carry", 100.0).drawdown_pct == 0.0
