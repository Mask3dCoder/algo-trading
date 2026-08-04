from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from quantcore.core.types import PortfolioSnapshot, Side
from quantcore.risk.manager import HardRiskManager, RiskConfig, RiskContext


class _Sig:
    def __init__(self, symbol: str, direction: str, strength: float, metadata: dict | None = None) -> None:
        self.symbol = symbol
        self.direction = direction
        self.strength = strength
        self.metadata = metadata or {}


def _portfolio(total_value: Decimal = Decimal("100000"), cash: Decimal = Decimal("100000")) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=datetime(2020, 1, 1),
        positions={},
        cash=cash,
        total_value=total_value,
    )


class _Market:
    def __init__(self, prices: dict[str, float]) -> None:
        class _Bar:
            def __init__(self, close: float) -> None:
                self.close = close

        self.bars = {symbol: _Bar(price) for symbol, price in prices.items()}


def _context(prices: dict[str, float], betas: dict[str, float] | None = None) -> RiskContext:
    return RiskContext(prices=prices, betas=betas or {})


def _market(prices: dict[str, float]) -> _Market:
    return _Market(prices)


def test_gross_cap_is_final_weight_normalization_step() -> None:
    manager = HardRiskManager(
        RiskConfig(max_gross_exposure=1.0, max_net_exposure=0.5, max_single_name_exposure=1.0)
    )
    signals = [
        _Sig("AAPL", "long", 2.0),
        _Sig("MSFT", "short", 1.0),
    ]
    decision = manager.pre_trade_check(
        signals,
        _portfolio(),
        _market({"AAPL": 100.0, "MSFT": 50.0}),
        _context({"AAPL": 100.0, "MSFT": 50.0}),
    )

    assert decision.gross_exposure <= 1.0
    assert abs(decision.net_exposure) <= 0.5
    assert sum(abs(t.target_weight) for t in decision.approved_trades) <= 1.0


def test_beta_neutralization_then_final_gross_cap() -> None:
    manager = HardRiskManager(
        RiskConfig(
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
            max_single_name_exposure=1.0,
            beta_neutral=True,
            beta_hedge_symbol="SPY",
        )
    )
    signals = [_Sig("AAPL", "long", 1.0, {"weight": 1.0})]
    decision = manager.pre_trade_check(
        signals,
        _portfolio(),
        _market({"AAPL": 100.0, "SPY": 400.0}),
        _context({"AAPL": 100.0, "SPY": 400.0}, betas={"AAPL": 1.2, "SPY": 1.0}),
    )

    weights = decision.normalized_weights
    beta_exposure = weights.get("AAPL", 0.0) * 1.2 + weights.get("SPY", 0.0) * 1.0
    assert abs(beta_exposure) < 1e-9
    assert sum(abs(w) for w in weights.values()) <= 1.0


def test_unauthorized_short_is_rejected() -> None:
    manager = HardRiskManager(RiskConfig(short_allowed=False, max_single_name_exposure=1.0))
    decision = manager.pre_trade_check(
        [_Sig("GME", "short", 1.0)],
        _portfolio(),
        _market({"GME": 100.0}),
        _context({"GME": 100.0}),
    )

    assert decision.approved_trades == []
    assert any(r.limit_name == "short_not_allowed" for r in decision.rejected)


def test_single_name_and_turnover_caps_clip_orders() -> None:
    manager = HardRiskManager(
        RiskConfig(
            max_gross_exposure=2.0,
            max_net_exposure=2.0,
            max_single_name_exposure=0.10,
            max_daily_turnover=5_000.0,
            min_order_notional=1.0,
        )
    )
    signals = [_Sig("NVDA", "long", 1.0)]

    decision = manager.pre_trade_check(
        signals,
        _portfolio(),
        _market({"NVDA": 100.0}),
        _context({"NVDA": 100.0}),
    )

    assert decision.approved_trades[0].target_notional <= 10_000.0
    assert abs(decision.approved_trades[0].delta_notional) <= 5_000.0


def test_short_margin_gate_downscales_short() -> None:
    manager = HardRiskManager(
        RiskConfig(
            short_allowed=True,
            short_margin_requirement=0.50,
            max_single_name_exposure=1.0,
            max_gross_exposure=1.0,
        )
    )
    decision = manager.pre_trade_check(
        [_Sig("TSLA", "short", 1.0)],
        _portfolio(total_value=Decimal("100000"), cash=Decimal("25000")),
        _market({"TSLA": 100.0}),
        _context({"TSLA": 100.0}),
    )

    assert decision.approved_trades
    assert decision.approved_trades[0].side == Side.SELL
    assert abs(decision.approved_trades[0].target_notional) <= 50_000.0
