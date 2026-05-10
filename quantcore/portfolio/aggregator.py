"""Cross-strategy portfolio aggregation and global exposure limits.

All strategies report positions into a shared exposure model that enforces:
  - Single-name concentration limits across ALL strategies combined.
  - Sector exposure limits at the system level.
  - Gross/net exposure limits aggregated across strategies.

Strategy isolation cannot bypass global exposure limits: a trade from
one strategy is rejected if it would cause a combined breach.

Design:
  - StrategyAggregator: central registry that tracks per-strategy positions.
  - GlobalExposureLimits: configurable limits for combined exposure.
  - AggregatorGate: pre-trade check that blocks orders violating global limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from quantcore.core.types import Order, Position, Side


class CheckVerdict(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class GlobalExposureLimits:
    """Global exposure limits aggregated across all strategies."""
    max_gross_exposure: float = 20_000_000.0   # $20M total gross
    max_net_exposure: float = 10_000_000.0     # $10M total net
    max_single_name: float = 2_000_000.0       # $2M per symbol (all strategies)
    max_single_name_pct: float = 0.25          # 25% of NAV per name
    max_sector_pct: float = 0.40               # 40% in any sector
    max_leverage: float = 2.0                  # 2x Reg T limit
    max_strategy_count: int = 20               # Max active strategies
    check_sector_on_order: bool = True         # Check sector limits pre-trade


@dataclass
class AggregationSnapshot:
    """Snapshot of aggregated portfolio state."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    n_strategies: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    single_name_exposures: dict[str, float] = field(default_factory=dict)
    sector_exposures: dict[str, float] = field(default_factory=dict)
    per_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0


@dataclass
class StrategyRegistration:
    """A strategy's registered position report."""
    strategy_id: str
    strategy_type: str = "unknown"
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


# Simple sector mapping (would be replaced by a proper data source)
DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
    "GOOG": "technology", "META": "technology", "NVDA": "technology",
    "AMD": "technology", "INTC": "technology", "CRM": "technology",
    "AMZN": "consumer_cyclical", "TSLA": "consumer_cyclical",
    "HD": "consumer_cyclical", "NFLX": "communication",
    "DIS": "communication", "JPM": "financial", "GS": "financial",
    "BAC": "financial", "V": "financial", "MA": "financial",
    "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "PG": "consumer_defensive", "KO": "consumer_defensive",
    "WMT": "consumer_defensive", "BA": "industrial", "CAT": "industrial",
    "GE": "industrial", "SPY": "etf", "QQQ": "etf", "IWM": "etf",
    "GLD": "commodity", "SLV": "commodity", "TLT": "fixed_income",
}


class StrategyAggregator:
    """Central registry for cross-strategy position aggregation.

    Usage:
        agg = StrategyAggregator(limits)
        agg.register_strategy("momentum_1", "equity")
        agg.update_positions("momentum_1", positions, cash)

        # Pre-trade check
        verdict = agg.check_order(order, symbol_sector_map)

        # Full aggregate report
        snapshot = agg.aggregate()
    """

    def __init__(
        self,
        limits: GlobalExposureLimits | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self._limits = limits or GlobalExposureLimits()
        self._sector_map = dict(sector_map or DEFAULT_SECTOR_MAP)
        self._strategies: dict[str, StrategyRegistration] = {}
        self._total_equity: float = 0.0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_strategy(
        self,
        strategy_id: str,
        strategy_type: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a strategy with the aggregator."""
        if len(self._strategies) >= self._limits.max_strategy_count:
            raise RuntimeError(
                f"Max strategy count ({self._limits.max_strategy_count}) exceeded"
            )
        self._strategies[strategy_id] = StrategyRegistration(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            metadata=metadata or {},
        )

    def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the aggregator."""
        self._strategies.pop(strategy_id, None)

    # ------------------------------------------------------------------
    # Position updates
    # ------------------------------------------------------------------

    def update_positions(
        self,
        strategy_id: str,
        positions: dict[str, Position],
        cash: float = 0.0,
    ) -> None:
        """Update positions for a registered strategy."""
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy not registered: {strategy_id}")
        self._strategies[strategy_id].positions = dict(positions)
        self._strategies[strategy_id].cash = cash
        self._strategies[strategy_id].last_updated = datetime.utcnow()

    def update_equity(self, total_equity: float) -> None:
        """Update total system equity (for percentage-based limits)."""
        self._total_equity = total_equity

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def check_order(
        self,
        order: Order,
        strategy_id: str = "",
        sector_map: dict[str, str] | None = None,
    ) -> CheckVerdict:
        """Check if an order would violate global exposure limits.

        This check runs BEFORE the per-strategy and per-broker checks.
        It ensures strategy isolation cannot bypass aggregate limits.
        """
        sym = order.symbol
        qty = float(order.quantity)
        price = float(order.limit_price) if order.limit_price else 0.0
        if price <= 0:
            return CheckVerdict.REJECTED

        order_value = qty * price

        # Aggregate current exposures across all strategies
        current_sym_exposure = 0.0
        for reg in self._strategies.values():
            for pos_sym, pos in reg.positions.items():
                if pos_sym == sym:
                    current_sym_exposure += abs(float(pos.market_value))

        # Factor in the new order
        if order.side == Side.BUY:
            new_exposure = current_sym_exposure + order_value
        else:
            new_exposure = max(0.0, current_sym_exposure - order_value)

        # Single-name dollar limit
        if new_exposure > self._limits.max_single_name:
            return CheckVerdict.REJECTED

        # Single-name percentage limit
        if self._total_equity > 0:
            pct = new_exposure / self._total_equity
            if pct > self._limits.max_single_name_pct:
                return CheckVerdict.REJECTED

        # Sector limit check
        if self._limits.check_sector_on_order:
            smap = sector_map or self._sector_map
            sym_sector = smap.get(sym)
            if sym_sector:
                current_sector = self._compute_sector_exposure()
                new_sector_val = current_sector.get(sym_sector, 0.0)
                if order.side == Side.BUY:
                    new_sector_val += order_value
                else:
                    new_sector_val = max(0.0, new_sector_val - order_value)
                if self._total_equity > 0:
                    sector_pct = new_sector_val / self._total_equity
                    if sector_pct > self._limits.max_sector_pct:
                        return CheckVerdict.REJECTED

        return CheckVerdict.APPROVED

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregationSnapshot:
        """Compute full aggregated portfolio snapshot."""
        gross_exposure = 0.0
        net_exposure = 0.0
        single_name: dict[str, float] = {}
        total_cash = 0.0

        per_strategy: dict[str, dict[str, Any]] = {}

        for sid, reg in self._strategies.items():
            strat_gross = 0.0
            strat_net = 0.0
            for sym, pos in reg.positions.items():
                mv = float(pos.quantity) * float(pos.market_price)
                single_name[sym] = single_name.get(sym, 0.0) + mv
                strat_gross += abs(mv)
                strat_net += mv
            gross_exposure += strat_gross
            net_exposure += strat_net
            total_cash += reg.cash

            per_strategy[sid] = {
                "gross_exposure": strat_gross,
                "net_exposure": strat_net,
                "n_positions": len(reg.positions),
                "cash": reg.cash,
                "type": reg.strategy_type,
                "last_updated": reg.last_updated.isoformat(),
            }

        equity = self._total_equity or gross_exposure + total_cash
        leverage = gross_exposure / equity if equity > 0 else 0.0

        # Sector exposures
        sector_exposures = self._compute_sector_exposure()

        snapshot = AggregationSnapshot(
            n_strategies=len(self._strategies),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
            cash=total_cash,
            equity=equity,
            single_name_exposures=single_name,
            sector_exposures=sector_exposures,
            per_strategy=per_strategy,
        )

        # Check limits and populate violations
        self._check_limits(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sector_exposure(self) -> dict[str, float]:
        """Compute total exposure per sector across all strategies."""
        sector_exp: dict[str, float] = {}
        for reg in self._strategies.values():
            for sym, pos in reg.positions.items():
                sector = self._sector_map.get(sym, "other")
                mv = abs(float(pos.quantity) * float(pos.market_price))
                sector_exp[sector] = sector_exp.get(sector, 0.0) + mv
        return sector_exp

    def _check_limits(self, snapshot: AggregationSnapshot) -> None:
        """Populate snapshot.violations with any limit breaches."""
        equity = snapshot.equity if snapshot.equity > 0 else 1.0
        violations: list[str] = []

        # Gross exposure
        if snapshot.gross_exposure > self._limits.max_gross_exposure:
            violations.append(
                f"Gross exposure ${snapshot.gross_exposure:,.0f} > "
                f"${self._limits.max_gross_exposure:,.0f}"
            )

        # Net exposure
        if abs(snapshot.net_exposure) > self._limits.max_net_exposure:
            violations.append(
                f"Net exposure ${snapshot.net_exposure:,.0f} > "
                f"${self._limits.max_net_exposure:,.0f}"
            )

        # Leverage
        if snapshot.leverage > self._limits.max_leverage:
            violations.append(
                f"Leverage {snapshot.leverage:.2f}x > {self._limits.max_leverage:.2f}x"
            )

        # Single-name concentration
        for sym, exp in snapshot.single_name_exposures.items():
            if exp > self._limits.max_single_name:
                violations.append(
                    f"{sym} exposure ${exp:,.0f} > ${self._limits.max_single_name:,.0f}"
                )
            pct = exp / equity
            if pct > self._limits.max_single_name_pct:
                violations.append(
                    f"{sym} at {pct:.1%} > {self._limits.max_single_name_pct:.1%} of NAV"
                )

        # Sector limits
        for sector, exp in snapshot.sector_exposures.items():
            pct = exp / equity
            if pct > self._limits.max_sector_pct:
                violations.append(
                    f"Sector {sector} at {pct:.1%} > {self._limits.max_sector_pct:.1%} of NAV"
                )

        snapshot.violations = violations

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def limits(self) -> GlobalExposureLimits:
        return self._limits

    @property
    def strategy_count(self) -> int:
        return len(self._strategies)

    def get_strategy(self, strategy_id: str) -> StrategyRegistration | None:
        return self._strategies.get(strategy_id)
