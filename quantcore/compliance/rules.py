"""Pre-trade compliance rules engine: restricted lists, wash sales, position limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

from quantcore.core.types import Order, PortfolioSnapshot, Position, Side


class RuleType(str, Enum):
    RESTRICTED_LIST = "restricted_list"
    POSITION_LIMIT = "position_limit"
    WASH_SALE = "wash_sale"
    SHORT_SALE = "short_sale"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    ORDER_SIZE = "order_size"
    PATTERN_DAY_TRADER = "pattern_day_trader"


class RuleSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


@dataclass
class ComplianceRule:
    rule_id: str = ""
    rule_type: RuleType = RuleType.RESTRICTED_LIST
    severity: RuleSeverity = RuleSeverity.BLOCK
    description: str = ""
    active: bool = True
    threshold: float = 0.0


@dataclass
class RuleViolation:
    rule: ComplianceRule
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: str = ""
    order_id: str | None = None
    symbol: str | None = None


class ComplianceRuleEngine:
    """Run all active compliance rules against orders and portfolios."""

    def __init__(self, rules: list[ComplianceRule] | None = None) -> None:
        self._rules: dict[str, ComplianceRule] = {}
        self._restricted: set[str] = set()
        self._violations: list[RuleViolation] = []
        self._trade_history: list[dict] = []
        if rules:
            for r in rules:
                self.add_rule(r)

    def add_rule(self, rule: ComplianceRule) -> None:
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        self._rules.pop(rule_id, None)

    def get_rules(self, rule_type: RuleType | None = None) -> list[ComplianceRule]:
        if rule_type:
            return [r for r in self._rules.values() if r.rule_type == rule_type]
        return list(self._rules.values())

    def check_order(
        self,
        order: Order,
        portfolio: PortfolioSnapshot,
        restricted_list: set[str] | None = None,
        daily_trades: dict[str, int] | None = None,
    ) -> list[RuleViolation]:
        """Run all active compliance rules against an order."""
        violations: list[RuleViolation] = []
        restricted = restricted_list or self._restricted
        daily = daily_trades or {}
        symbol = str(order.symbol)
        qty = float(order.quantity)

        for rule in self._rules.values():
            if not rule.active:
                continue

            if rule.rule_type == RuleType.RESTRICTED_LIST:
                if symbol in restricted:
                    violations.append(RuleViolation(
                        rule=rule, details=f"{symbol} is on the restricted list",
                        order_id=str(order.id), symbol=symbol,
                    ))

            elif rule.rule_type == RuleType.WASH_SALE:
                if order.side == Side.SELL:
                    if self._check_wash_sale(symbol, float(order.limit_price or 0)):
                        violations.append(RuleViolation(
                            rule=rule, details=f"Wash sale detected for {symbol}",
                            order_id=str(order.id), symbol=symbol,
                        ))

            elif rule.rule_type == RuleType.SHORT_SALE:
                if order.side == Side.SELL:
                    pos = portfolio.positions.get(symbol)
                    current_qty = float(pos.quantity) if pos else 0.0
                    if qty > current_qty:
                        violations.append(RuleViolation(
                            rule=rule, details=f"Short sale: sell {qty} > held {current_qty}",
                            order_id=str(order.id), symbol=symbol,
                        ))

            elif rule.rule_type == RuleType.POSITION_LIMIT:
                if rule.threshold > 0:
                    pos = portfolio.positions.get(symbol)
                    current_qty = float(pos.quantity) if pos else 0.0
                    new_qty = current_qty + qty if order.side == Side.BUY else current_qty - qty
                    if abs(new_qty) * float(pos.market_price if pos else 0) > rule.threshold:
                        violations.append(RuleViolation(
                            rule=rule, details=f"Position limit exceeded for {symbol}",
                            order_id=str(order.id), symbol=symbol,
                        ))

            elif rule.rule_type == RuleType.CONCENTRATION:
                if rule.threshold > 0 and float(portfolio.total_value) > 0:
                    pos = portfolio.positions.get(symbol)
                    current_val = float(pos.market_value) if pos else 0.0
                    order_val = qty * float(order.limit_price or 0)
                    new_val = current_val + order_val if order.side == Side.BUY else current_val - order_val
                    if new_val / float(portfolio.total_value) > rule.threshold:
                        violations.append(RuleViolation(
                            rule=rule, details=f"Concentration limit: {symbol} would be {new_val / float(portfolio.total_value):.1%}",
                            order_id=str(order.id), symbol=symbol,
                        ))

            elif rule.rule_type == RuleType.LEVERAGE:
                if rule.threshold > 0 and float(portfolio.total_value) > 0:
                    gross = sum(abs(float(p.market_value)) for p in portfolio.positions.values())
                    order_val = qty * float(order.limit_price or 0)
                    new_gross = gross + order_val
                    if new_gross / float(portfolio.total_value) > rule.threshold:
                        violations.append(RuleViolation(
                            rule=rule, details=f"Leverage limit: {new_gross / float(portfolio.total_value):.1f}x",
                            order_id=str(order.id), symbol=symbol,
                        ))

            elif rule.rule_type == RuleType.PATTERN_DAY_TRADER:
                day_trades_today = daily.get(symbol, 0)
                if day_trades_today >= 3:
                    portfolio_equity = float(portfolio.total_value)
                    if portfolio_equity < 25000.0:
                        violations.append(RuleViolation(
                            rule=rule, details=f"PDT: {day_trades_today} day trades, equity ${portfolio_equity:.0f} < $25k",
                            order_id=str(order.id), symbol=symbol,
                        ))

        self._violations.extend(violations)
        return violations

    def check_portfolio(self, portfolio: PortfolioSnapshot) -> list[RuleViolation]:
        """Run portfolio-level compliance checks."""
        violations: list[RuleViolation] = []
        for rule in self._rules.values():
            if not rule.active:
                continue
            if rule.rule_type == RuleType.CONCENTRATION and rule.threshold > 0:
                total = float(portfolio.total_value)
                if total <= 0:
                    continue
                for sym, pos in portfolio.positions.items():
                    val = float(pos.market_value)
                    if val / total > rule.threshold:
                        violations.append(RuleViolation(
                            rule=rule, details=f"Concentration: {sym} at {val / total:.1%}",
                            symbol=sym,
                        ))
            elif rule.rule_type == RuleType.LEVERAGE and rule.threshold > 0:
                total = float(portfolio.total_value)
                if total <= 0:
                    continue
                gross = sum(abs(float(p.market_value)) for p in portfolio.positions.values())
                if gross / total > rule.threshold:
                    violations.append(RuleViolation(
                        rule=rule, details=f"Leverage: {gross / total:.1f}x",
                    ))
        self._violations.extend(violations)
        return violations

    def is_restricted(self, symbol: str) -> bool:
        return symbol.upper() in self._restricted

    def add_restricted(self, symbol: str) -> None:
        self._restricted.add(symbol.upper())

    def remove_restricted(self, symbol: str) -> None:
        self._restricted.discard(symbol.upper())

    def _check_wash_sale(self, symbol: str, sell_price: float) -> bool:
        cutoff = datetime.now(UTC) - timedelta(days=30)
        for trade in self._trade_history:
            if trade.get("symbol") != symbol:
                continue
            trade_date = trade.get("date")
            if trade_date and trade_date < cutoff:
                continue
            if trade.get("side") == "buy" and sell_price > 0:
                buy_price = float(trade.get("price", 0))
                if sell_price < buy_price:
                    return True
        return False

    def record_trade(self, symbol: str, side: str, price: float, quantity: float) -> None:
        self._trade_history.append({
            "symbol": symbol, "side": side, "price": price,
            "quantity": quantity, "date": datetime.now(UTC),
        })

    @property
    def violations(self) -> list[RuleViolation]:
        return list(self._violations)

    def clear_violations(self) -> None:
        self._violations.clear()

    @property
    def restricted_symbols(self) -> set[str]:
        return set(self._restricted)


def build_default_rules() -> list[ComplianceRule]:
    """Build a standard set of compliance rules."""
    return [
        ComplianceRule("R001", RuleType.RESTRICTED_LIST, RuleSeverity.BLOCK, "Restricted list check"),
        ComplianceRule("R002", RuleType.POSITION_LIMIT, RuleSeverity.BLOCK, "Max position per symbol", threshold=1000000),
        ComplianceRule("R003", RuleType.WASH_SALE, RuleSeverity.BLOCK, "Wash sale detection (30-day window)"),
        ComplianceRule("R004", RuleType.SHORT_SALE, RuleSeverity.WARNING, "Reg SHO short sale check"),
        ComplianceRule("R005", RuleType.CONCENTRATION, RuleSeverity.WARNING, "Max single-name concentration", threshold=0.25),
        ComplianceRule("R006", RuleType.LEVERAGE, RuleSeverity.BLOCK, "Max portfolio leverage", threshold=2.0),
        ComplianceRule("R007", RuleType.ORDER_SIZE, RuleSeverity.WARNING, "Large order warning", threshold=500000),
        ComplianceRule("R008", RuleType.PATTERN_DAY_TRADER, RuleSeverity.WARNING, "Pattern day trader check"),
    ]


def detect_wash_sale(
    trades: list[dict], symbol: str, window_days: int = 30,
) -> bool:
    """Check if a symbol was bought within the window and is now being sold at a loss."""
    cutoff = datetime.now(UTC) - timedelta(days=window_days)
    buys = [
        t for t in trades
        if t.get("symbol") == symbol and t.get("side") == "buy"
        and t.get("date", datetime.now(UTC)) >= cutoff
    ]
    if not buys:
        return False
    sells = [
        t for t in trades
        if t.get("symbol") == symbol and t.get("side") == "sell"
        and t.get("date", datetime.now(UTC)) >= cutoff
    ]
    for sell in sells:
        sell_price = float(sell.get("price", 0))
        for buy in buys:
            buy_price = float(buy.get("price", 0))
            if sell_price < buy_price:
                return True
    return False


def check_reg_sho(
    order: Order, positions: dict[str, Position],
) -> RuleViolation | None:
    """Reg SHO short sale check: verify locate for short sales."""
    if order.side != Side.SELL:
        return None
    symbol = str(order.symbol)
    pos = positions.get(symbol)
    current_qty = float(pos.quantity) if pos else 0.0
    if float(order.quantity) <= current_qty:
        return None
    return RuleViolation(
        rule=ComplianceRule(
            rule_id="REG_SHO", rule_type=RuleType.SHORT_SALE,
            severity=RuleSeverity.WARNING, description="Reg SHO locate requirement",
        ),
        details=f"Short sale of {order.quantity} {symbol} requires locate (only {current_qty} held)",
        order_id=str(order.id), symbol=symbol,
    )
