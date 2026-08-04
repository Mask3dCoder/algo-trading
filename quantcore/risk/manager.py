"""Broker-neutral portfolio risk manager for pre-trade compliance.

The manager in this module intentionally does not know about Alpaca, IBKR,
or any execution API. It consumes broker-neutral strategy signals and portfolio
snapshots, then returns approved normalized trades or explicit rejections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

from quantcore.core.types import PortfolioSnapshot, Side


class MarketDataLike(Protocol):
    """Minimal market-data surface required by the risk manager."""

    bars: Mapping[str, Any]


@dataclass(frozen=True)
class RiskConfig:
    """Hard portfolio constraints applied before order construction."""

    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.25
    max_single_name_exposure: float = 0.10
    max_daily_turnover: float | None = None

    beta_neutral: bool = False
    beta_hedge_symbol: str | None = None

    short_allowed: bool = True
    long_margin_multiplier: float = 1.0
    short_margin_requirement: float = 0.50

    min_order_notional: float = 1.0


@dataclass(frozen=True)
class RiskContext:
    """Point-in-time risk inputs used to convert weights into quantities."""

    prices: Mapping[str, float]
    betas: Mapping[str, float] = field(default_factory=dict)
    adv_dollars: Mapping[str, float] = field(default_factory=dict)
    as_of_date: date | None = None

    @classmethod
    def from_bars(cls, bars: Mapping[str, Any], **kwargs: Any) -> RiskContext:
        prices = {
            symbol: float(bar.close)
            for symbol, bar in bars.items()
            if getattr(bar, "close", None) is not None
        }
        return cls(prices=prices, **kwargs)


@dataclass(frozen=True)
class ApprovedTrade:
    """Broker-neutral approved trade produced by RiskManager."""

    symbol: str
    side: Side
    target_weight: float
    normalized_weight: float
    target_notional: float
    delta_notional: float
    quantity: float
    price: float
    reason: str = "approved"


@dataclass(frozen=True)
class RiskRejection:
    """Rejected raw signal or clipped order reason."""

    symbol: str
    reason: str
    limit_name: str
    requested_notional: float = 0.0
    allowed_notional: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    """Result of pre-trade risk compliance."""

    approved_trades: list[ApprovedTrade]
    rejected: list[RiskRejection] = field(default_factory=list)
    normalized_weights: Mapping[str, float] = field(default_factory=dict)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0


class RiskManager(ABC):
    """Abstract broker-neutral pre-trade compliance gate."""

    @abstractmethod
    def pre_trade_check(
        self,
        raw_signals: Sequence[Any],
        portfolio: PortfolioSnapshot,
        market: MarketDataLike,
        context: RiskContext | None = None,
    ) -> RiskDecision:
        """Validate and normalize raw strategy signals before order creation."""


class HardRiskManager(RiskManager):
    """Strict portfolio-constraint manager for weights-to-orders gating.

    Weight pipeline:
      1. Extract raw signal weights.
      2. Beta-neutralize when configured.
      3. Apply net exposure cap.
      4. Apply gross exposure cap as the final weight-normalization step.
      5. Scale to NAV.
      6. Apply single-name, buying-power, short-selling, and turnover gates.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._turnover_by_date: dict[date, float] = defaultdict(float)

    def reset_daily_turnover(self, as_of: date | None = None) -> None:
        if as_of is None:
            self._turnover_by_date.clear()
        else:
            self._turnover_by_date.pop(as_of, None)

    def pre_trade_check(
        self,
        raw_signals: Sequence[Any],
        portfolio: PortfolioSnapshot,
        market: MarketDataLike,
        context: RiskContext | None = None,
    ) -> RiskDecision:
        context = context or RiskContext.from_bars(market.bars)
        nav = self._nav(portfolio)
        if nav <= 0:
            return RiskDecision(
                approved_trades=[],
                rejected=[RiskRejection("", "Portfolio NAV must be positive", "nav_required")],
            )

        raw_weights = self._raw_weights(raw_signals, nav)
        if not raw_weights:
            return RiskDecision(approved_trades=[], normalized_weights={})

        weights = self._beta_neutralize(raw_weights, context)
        weights = self._cap_net_exposure(weights)

        # Final weight-normalization boundary: after this point no function may
        # increase gross exposure. Later gates only reduce or reject trades.
        weights = self._final_gross_cap(weights)

        trades: list[ApprovedTrade] = []
        rejected: list[RiskRejection] = []

        for symbol, target_weight in weights.items():
            price = context.prices.get(symbol)
            if price is None or price <= 0:
                rejected.append(
                    RiskRejection(symbol, "Missing or invalid price", "price_required")
                )
                continue

            target_notional = target_weight * nav
            existing_notional = self._position_notional(portfolio, symbol)
            delta_notional = target_notional - existing_notional

            clipped_notional, rejection = self._apply_position_and_cash_gates(
                symbol=symbol,
                target_notional=target_notional,
                delta_notional=delta_notional,
                existing_notional=existing_notional,
                nav=nav,
                available_cash=self._available_cash(portfolio),
            )
            if rejection is not None:
                rejected.append(rejection)
                continue

            clipped_delta = clipped_notional - existing_notional
            clipped_delta, rejection = self._apply_turnover_gate(
                symbol, clipped_delta, context.as_of_date
            )
            if rejection is not None:
                rejected.append(rejection)
                continue

            quantity = abs(clipped_delta) / price
            if abs(clipped_delta) < self.config.min_order_notional or quantity <= 0:
                rejected.append(
                    RiskRejection(
                        symbol,
                        "Order notional below minimum",
                        "min_order_notional",
                        requested_notional=abs(clipped_delta),
                        allowed_notional=self.config.min_order_notional,
                    )
                )
                continue

            side = Side.BUY if clipped_delta > 0 else Side.SELL
            trades.append(
                ApprovedTrade(
                    symbol=symbol,
                    side=side,
                    target_weight=clipped_notional / nav,
                    normalized_weight=weights[symbol],
                    target_notional=clipped_notional,
                    delta_notional=clipped_delta,
                    quantity=quantity,
                    price=price,
                )
            )

        approved_weights = {
            trade.symbol: trade.target_notional / nav for trade in trades if nav > 0
        }
        return RiskDecision(
            approved_trades=trades,
            rejected=rejected,
            normalized_weights=approved_weights,
            gross_exposure=sum(abs(w) for w in approved_weights.values()),
            net_exposure=sum(approved_weights.values()),
        )

    def _raw_weights(self, signals: Sequence[Any], nav: float) -> dict[str, float]:
        weights: dict[str, float] = defaultdict(float)
        for sig in signals:
            symbol = str(getattr(sig, "symbol", "")).strip()
            if not symbol:
                continue
            weights[symbol] += self._weight_from_signal(sig, nav)
        return dict(weights)

    def _weight_from_signal(self, sig: Any, nav: float) -> float:
        metadata = getattr(sig, "metadata", {}) or {}

        for key in ("weight", "target_weight", "raw_weight"):
            if key in metadata:
                return float(metadata[key])

        notional = getattr(sig, "notional_size", None)
        if notional is None:
            notional = getattr(sig, "size", None)
        # If a signal carries absolute notional, convert it to a relative weight
        # here so later NAV scaling remains broker-neutral and deterministic.
        if notional is not None:
            try:
                return float(notional) / nav if nav > 0 else 0.0
            except (TypeError, ValueError):
                pass

        strength = float(getattr(sig, "strength", 0.0) or 0.0)
        direction = getattr(sig.direction, "value", sig.direction) if hasattr(sig, "direction") else None
        if direction in ("long", 1):
            sign = 1.0
        elif direction in ("short", -1):
            sign = -1.0
        elif direction in ("flat", 0):
            sign = 0.0
        else:
            sign = 1.0 if strength >= 0 else -1.0
        return sign * min(abs(strength), 1.0)

    def _beta_neutralize(self, weights: dict[str, float], context: RiskContext) -> dict[str, float]:
        if not self.config.beta_neutral or not context.betas:
            return dict(weights)

        result = dict(weights)
        beta_exposure = sum(result.get(symbol, 0.0) * context.betas.get(symbol, 0.0) for symbol in result)
        if abs(beta_exposure) < 1e-12:
            return result

        hedge_symbol = self.config.beta_hedge_symbol
        if hedge_symbol:
            hedge_beta = context.betas.get(hedge_symbol, 1.0)
            if hedge_beta != 0:
                result[hedge_symbol] = result.get(hedge_symbol, 0.0) - beta_exposure / hedge_beta
            return result

        denom = sum(beta * beta for beta in context.betas.values())
        if denom <= 0:
            return result
        for symbol, beta in context.betas.items():
            result[symbol] = result.get(symbol, 0.0) - beta_exposure * beta / denom
        return result

    def _cap_net_exposure(self, weights: dict[str, float]) -> dict[str, float]:
        if self.config.max_net_exposure <= 0:
            return dict(weights)
        net = sum(weights.values())
        if abs(net) <= self.config.max_net_exposure:
            return dict(weights)
        scale = self.config.max_net_exposure / abs(net)
        return {symbol: weight * scale for symbol, weight in weights.items()}

    def _final_gross_cap(self, weights: dict[str, float]) -> dict[str, float]:
        gross = sum(abs(weight) for weight in weights.values())
        if gross <= self.config.max_gross_exposure or gross <= 0:
            return dict(weights)
        scale = self.config.max_gross_exposure / gross
        return {symbol: weight * scale for symbol, weight in weights.items()}

    def _apply_position_and_cash_gates(
        self,
        symbol: str,
        target_notional: float,
        delta_notional: float,
        existing_notional: float,
        nav: float,
        available_cash: float,
    ) -> tuple[float, RiskRejection | None]:
        clipped_target = target_notional
        single_name_cap = nav * self.config.max_single_name_exposure
        if abs(target_notional) > single_name_cap:
            clipped_target = (1.0 if target_notional >= 0 else -1.0) * single_name_cap

        if clipped_target < 0 and not self.config.short_allowed:
            return 0.0, RiskRejection(
                symbol,
                "Short selling is not authorized",
                "short_not_allowed",
                requested_notional=abs(target_notional),
            )

        if clipped_target > 0:
            buying_power = available_cash * max(self.config.long_margin_multiplier, 1.0)
            if clipped_target > buying_power:
                clipped_target = buying_power
        elif clipped_target < 0:
            required_margin = abs(clipped_target) * self.config.short_margin_requirement
            if required_margin > available_cash:
                clipped_target = -available_cash / self.config.short_margin_requirement

        if abs(clipped_target) > single_name_cap:
            clipped_target = (1.0 if clipped_target >= 0 else -1.0) * single_name_cap

        if abs(clipped_target - existing_notional) < 1e-9:
            return existing_notional, None

        return clipped_target, None

    def _apply_turnover_gate(
        self,
        symbol: str,
        delta_notional: float,
        as_of: date | None,
    ) -> tuple[float, RiskRejection | None]:
        if self.config.max_daily_turnover is None or self.config.max_daily_turnover <= 0:
            return delta_notional, None

        trade_date = as_of or datetime.utcnow().date()
        used = self._turnover_by_date[trade_date]
        remaining = self.config.max_daily_turnover - used
        if remaining <= 0:
            return 0.0, RiskRejection(
                symbol,
                "Daily turnover budget exhausted",
                "max_daily_turnover",
                requested_notional=abs(delta_notional),
            )

        if abs(delta_notional) <= remaining:
            self._turnover_by_date[trade_date] += abs(delta_notional)
            return delta_notional, None

        clipped = (1.0 if delta_notional >= 0 else -1.0) * remaining
        self._turnover_by_date[trade_date] += remaining
        return clipped, None

    @staticmethod
    def _nav(portfolio: PortfolioSnapshot) -> float:
        return abs(float(portfolio.total_value))

    @staticmethod
    def _available_cash(portfolio: PortfolioSnapshot) -> float:
        if hasattr(portfolio, "available_cash"):
            return float(portfolio.available_cash())
        if hasattr(portfolio, "cash"):
            return float(portfolio.cash)
        return 0.0

    @staticmethod
    def _position_notional(portfolio: PortfolioSnapshot, symbol: str) -> float:
        pos = portfolio.positions.get(symbol)
        if pos is None:
            return 0.0
        return float(abs(pos.market_value))
