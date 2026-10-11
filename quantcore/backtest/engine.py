"""Event-driven backtest engine — MarketEvent → SignalEvent → OrderEvent → FillEvent.

Integrated with Phase 3 modules: risk limits, position sizing, regime filtering,
drawdown tracking, and risk metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quantcore.backtest.analytics import (
    PerformanceReport,
    compute_returns,
    generate_performance_report,
)
from quantcore.backtest.corporate_actions import CorporateActionsConfig, ex_date_key
from quantcore.backtest.financing import (
    FinancingConfig,
)
from quantcore.backtest.impact_config import (
    MarketImpactConfig,
    MarketImpactEngine,
    VolumeFilterConfig,
)
from quantcore.backtest.simulation import OptionFillSimulator
from quantcore.core import economic_kernel
from quantcore.core.cash import AssetClass, CashLedger
from quantcore.core.data_validation import clean_snapshot
from quantcore.core.types import OHLCV, Fill, Order, OrderType, Position, Side
from quantcore.risk.drawdown import DrawdownAction, DrawdownConfig, DrawdownTracker
from quantcore.risk.limits import LimitCheckResult, RiskLimits, check_pre_trade
from quantcore.risk.manager import ApprovedTrade, RiskContext, RiskDecision, RiskManager

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(int, Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass(order=True)
class Event:
    timestamp: float  # Unix timestamp for heap ordering
    event_type: EventType = field(compare=False)
    data: Any = field(compare=False, default=None)


@dataclass
class MarketData:
    bar: OHLCV


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Immutable cross-sectional market-data slice for one timestamp.

    The snapshot is the only market-data object exposed to snapshot-aware
    strategies. It contains every bar available for the timestamp and nothing
    from future timestamps. Omitted/missing symbols are absent from ``bars``;
    strategies should treat absence as missing data rather than as a zero price.
    """

    timestamp: datetime
    bars: Mapping[str, OHLCV] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.bars))

    def get_bar(self, symbol: str) -> OHLCV | None:
        return self.bars.get(symbol)

    def representative_bar(self) -> OHLCV | None:
        """Return a deterministic bar for legacy on_bar() compatibility.

        Snapshot-aware strategies should use ``on_snapshot``. Legacy single-bar
        strategies receive only this representative bar and therefore cannot see
        the full universe through the engine interface.
        """
        if not self.bars:
            return None
        symbol = self.symbols[0]
        return self.bars[symbol]


@dataclass
class SignalData:
    symbol: str
    direction: str  # "long", "short", "flat"
    strength: float
    strategy_id: str
    notional_size: float | None = None  # strategy-computed position size, overrides sizing_pct
    instrument_type: str = "equity"      # "equity" or "option"
    metadata: dict = field(default_factory=dict)


@dataclass
class OrderData:
    order: Order


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class Strategy(Protocol):
    """Protocol for backtest strategies.

    Preferred interface is ``on_snapshot`` so cross-sectional strategies receive
    the entire timestamp slice simultaneously. ``on_bar`` remains supported only
    as a legacy compatibility path.
    """

    def on_snapshot(self, snapshot: MarketDataSnapshot, portfolio: PortfolioState) -> list[SignalData]:
        """Generate signals from a complete market-data snapshot."""
        ...

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        """Legacy single-bar entry point."""
        ...

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        """React to a fill (optional)."""
        ...


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------


class PortfolioState:
    """Tracks positions, cash, and portfolio value during backtest.

    Integrated with Phase 3: drawdown tracking, regime state, trade logging.
    Optionally accepts a CashLedger for settlement-aware cash tracking
    (used by the live system; backtest defaults to a simple float).

    When a CashLedger is provided, all cash mutations route through it
    for proper settlement tracking.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        cash_ledger: CashLedger | None = None,
    ) -> None:
        self._cash: float = initial_cash
        self._cash_ledger: CashLedger | None = cash_ledger
        self.initial_cash: float = initial_cash
        self._positions: dict[str, Position] = {}
        self._prices: dict[str, float] = {}
        self.equity_curve: list[tuple[float, float]] = []  # (timestamp, value)
        self.trades: list[Fill] = []
        self.signal_log: list[dict[str, Any]] = []
        self.regime_state: str = "unknown"
        self.total_market_impact_cost: float = 0.0
        # Financing tracking (P2)
        self.total_borrow_cost: float = 0.0
        self.total_margin_interest: float = 0.0
        self._financing_config: FinancingConfig | None = None
        # Corporate actions (dividends / splits) — see CorporateActionsConfig
        self._corporate_actions: CorporateActionsConfig | None = None
        self.total_dividend_income: float = 0.0

    # ------------------------------------------------------------------
    # Property guards — economic state is READ-ONLY through the public API.
    # All mutation must go through EconomicEngineCore via apply_fill().
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Settled cash. Delegates to CashLedger if available."""
        if self._cash_ledger is not None:
            return self._cash_ledger.settled_cash
        return self._cash

    @cash.setter
    def cash(self, value: float) -> None:
        raise AttributeError(
            "Direct assignment to PortfolioState.cash is forbidden. "
            "All cash mutations must go through EconomicEngineCore via apply_fill()."
        )

    @property
    def cash_ledger(self) -> CashLedger | None:
        """Access the settlement-aware cash ledger, if configured."""
        return self._cash_ledger

    @property
    def unsettled_cash(self) -> float:
        """Cash pending settlement (T+1/T+2). 0 if no ledger."""
        if self._cash_ledger is not None:
            return self._cash_ledger.unsettled_cash
        return 0.0

    @property
    def committed_cash(self) -> float:
        """Cash reserved for open orders. 0 if no ledger."""
        if self._cash_ledger is not None:
            return self._cash_ledger.committed_cash
        return 0.0

    def available_cash(self) -> float:
        """Cash available for trading: settled - committed.

        This is the canonical method for order sizing in the live system.
        In backtest mode (no ledger), returns total cash.
        """
        if self._cash_ledger is not None:
            return self._cash_ledger.available()
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)  # return a copy to prevent external mutation

    @positions.setter
    def positions(self, value: dict[str, Position]) -> None:
        raise AttributeError(
            "Direct assignment to PortfolioState.positions is forbidden. "
            "All position mutations must go through EconomicEngineCore via apply_fill()."
        )

    @property
    def total_value(self) -> float:
        positions_value = sum(
            float(pos.quantity) * self._prices.get(pos.symbol, float(pos.market_price))
            for pos in self._positions.values()
            if float(pos.quantity) != 0
        )
        return self._cash + positions_value

    @property
    def returns(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return (self.total_value - self.initial_cash) / self.initial_cash

    def update_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price
        if symbol in self._positions:
            pos = self._positions[symbol]
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_price=Decimal(str(price)),
                unrealized_pnl=Decimal(str(float(pos.quantity) * (price - float(pos.avg_cost)))),
                realized_pnl=pos.realized_pnl,
            )

    def mark_to_market(self, snapshot: MarketDataSnapshot) -> None:
        """Update all held asset valuations from one immutable timestamp slice.

        This is the portfolio-side data-boundary enforcement: every symbol in
        the snapshot is marked before any strategy logic runs. The strategy never
        observes per-symbol price updates one row at a time.
        """
        for symbol, bar in snapshot.bars.items():
            self.update_price(symbol, float(bar.close))

    def apply_fill(self, fill: Fill, symbol: str) -> None:
        """Apply a fill to the portfolio.

        Delegates all position math to the canonical EconomicEngineCore.
        This method is a thin wrapper: it determines direction, calls the
        kernel, and applies the resulting state update to portfolio cash
        and positions.

        When a CashLedger is attached, cash mutations route through it
        for settlement tracking (unsettled entries created for fills).
        """
        qty = float(fill.quantity)
        price = float(fill.price)
        commission = float(fill.commission)
        is_buy = False

        # Track market impact cost as a separate line item (already reflected in fill price)
        self.total_market_impact_cost += float(fill.market_impact_cost)

        self.trades.append(fill)

        # Determine direction: prefer fill.side, fall back to order_id parsing
        if fill.side is not None:
            is_buy = fill.side == Side.BUY
        else:
            is_buy = fill.order_id.endswith("_buy")

        # Build current position for the kernel (None if no existing position)
        current = self._positions.get(symbol)
        current_kernel = None
        if current is not None and abs(float(current.quantity)) > 1e-10:
            current_kernel = {
                "symbol": symbol,
                "quantity": float(current.quantity),
                "avg_cost": float(current.avg_cost),
                "market_price": float(current.market_price),
                "realized_pnl": float(current.realized_pnl),
            }

        # Delegate to the canonical economic kernel
        fa = economic_kernel.apply_fill(
            current_kernel,
            fill_price=price,
            fill_quantity=qty,
            is_buy=is_buy,
            symbol=symbol,
        )

        # Apply cash delta from the kernel (after commission)
        total_cash_delta = fa.cash_delta - commission

        if self._cash_ledger is not None:
            # Settlement-aware path: create unsettled entries
            cash_amount = -total_cash_delta  # +cash_delta = money in
            asset_class: AssetClass = "option" if "/" in symbol else "equity"
            self._cash_ledger.add_unsettled(
                amount=cash_amount,
                asset_class=asset_class,
                description=f"{'Buy' if is_buy else 'Sell'} {qty} {symbol} @ {price}",
            )
        else:
            # Backtest path: immediate settlement
            self._cash += total_cash_delta

        # Apply position update from the kernel
        if fa.position is None:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(fa.position["quantity"])),
                avg_cost=Decimal(str(fa.position["avg_cost"])),
                market_price=Decimal(str(price)),
                realized_pnl=Decimal(str(fa.position["realized_pnl"])),
            )

    def record_equity(self, timestamp: float) -> None:
        self.equity_curve.append((timestamp, self.total_value))

    def _deduct_cash(self, amount: float) -> None:
        """Deduct cash, routing through CashLedger if available."""
        if self._cash_ledger is not None:
            self._cash_ledger.add_unsettled(-amount, "equity",
                                            description="financing cost")
        else:
            self._cash -= amount

    def _credit_cash(self, amount: float) -> None:
        """Credit cash (e.g. dividend income), routing through CashLedger if set."""
        if self._cash_ledger is not None:
            self._cash_ledger.add_unsettled(amount, "equity", description="dividend")
        else:
            self._cash += amount

    def set_corporate_actions(self, config: CorporateActionsConfig | None) -> None:
        """Set the dividend/split schedule applied during the backtest."""
        self._corporate_actions = config

    def daily_corporate_actions_update(self, timestamp: object) -> None:
        """Credit dividends and apply splits for the current bar.

        Called once per bar after mark-to-market and financing. A long position
        held on an ex-date receives ``amount`` cash per share (a short pays it);
        on a split date the position's quantity is rescaled by the ratio and its
        average cost divided by it, so value is preserved across the
        unadjusted-price jump. Ex-dates outside the run window are ignored.
        """
        config = self._corporate_actions
        if config is None or not (config.dividends or config.splits):
            return
        key = ex_date_key(timestamp)
        for symbol, pos in list(self._positions.items()):
            qty = float(pos.quantity)
            if abs(qty) < 1e-12:
                continue
            div_amt = config.dividends.get(symbol, {}).get(key)
            if div_amt:
                self._credit_cash(qty * div_amt)
                self.total_dividend_income += qty * div_amt
            ratio = config.splits.get(symbol, {}).get(key)
            if ratio and ratio > 0 and abs(ratio - 1.0) > 1e-12:
                new_qty = qty * ratio
                new_avg = float(pos.avg_cost) / ratio
                new_market = float(pos.market_price)
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal(str(new_qty)),
                    avg_cost=Decimal(str(new_avg)),
                    market_price=pos.market_price,
                    unrealized_pnl=Decimal(str(new_qty * (new_market - new_avg))),
                    realized_pnl=pos.realized_pnl,
                )

    def set_financing_config(self, config: FinancingConfig | None) -> None:
        """Set the financing config for borrow/margin cost calculations."""
        self._financing_config = config

    def daily_financing_update(self) -> None:
        """Accrue borrowing costs and margin interest for the current bar.

        Called once per bar in the engine loop after price updates.
        Deducts costs from cash and updates tracking totals.
        If margin limit is breached, sets a flag for forced liquidation.
        """
        config = self._financing_config
        if config is None:
            return

        registry = config.borrow_cost_registry
        total_short_value = 0.0
        total_long_value = 0.0

        for symbol, pos in self._positions.items():
            qty = float(pos.quantity)
            price = self._prices.get(symbol, float(pos.market_price))
            market_value = qty * price

            if qty < 0:  # Short position
                abs_value = abs(market_value)
                total_short_value += abs_value
                borrow_rate = registry.get_rate(symbol) if registry else 0.005
                daily_cost = abs_value * borrow_rate / 360.0
                if daily_cost > 0:
                    self._deduct_cash(daily_cost)
                    self.total_borrow_cost += daily_cost
            elif qty > 0:  # Long position
                total_long_value += market_value

        # Margin interest on leveraged long positions
        # Guard: floor cash at 0 — negative cash must not overstate margin
        effective_cash = max(0.0, self.cash)
        if config.margin_rate > 0 and total_long_value > effective_cash:
            margin_used = total_long_value - effective_cash
            daily_interest = margin_used * config.margin_rate / 360.0
            if daily_interest > 0:
                self._deduct_cash(daily_interest)
                self.total_margin_interest += daily_interest

    @property
    def is_margin_breach(self) -> bool:
        """Check if margin usage exceeds the configured limit."""
        config = self._financing_config
        if config is None:
            return False
        total_value = self.total_value
        if total_value <= 0:
            return False
        # Margin used = total position value / equity (1.0 = no leverage)
        long_value = sum(
            max(0, float(p.quantity)) * self._prices.get(s, float(p.market_price))
            for s, p in self._positions.items()
        )
        short_value = sum(
            abs(min(0, float(p.quantity))) * self._prices.get(s, float(p.market_price))
            for s, p in self._positions.items()
        )
        total_exposure = long_value + short_value
        leverage = total_exposure / total_value
        return leverage > config.margin_limit


# ---------------------------------------------------------------------------
# Data handler
# ---------------------------------------------------------------------------


class HistoricalDataHandler:
    """Feeds historical OHLCV bars as timestamp-aligned MarketDataSnapshots."""

    def __init__(self, data: pd.DataFrame) -> None:
        """data: DataFrame with columns [symbol, timestamp, open, high, low, close, volume]."""
        if data.empty:
            self._data = data.copy()
            self._snapshots: Sequence[MarketDataSnapshot] = ()
            self._index = 0
            return

        normalized = data.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
        normalized["symbol"] = normalized["symbol"].astype(str)
        normalized = normalized.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        self._data = normalized

        snapshots: list[MarketDataSnapshot] = []
        # Single pass over the (timestamp, symbol)-sorted frame, grouping
        # manually. One itertuples stream is ~10x faster than per-group
        # iteration (groupby's per-row boxing dominated handler cost).
        # Duplicate-symbol detection is unchanged: rows are visited in the
        # same order the old groupby produced.
        current_ts: pd.Timestamp | None = None
        current_dt: datetime | None = None
        bars: dict[str, OHLCV] = {}
        for row in normalized.itertuples(index=False):
            row_ts = row.timestamp
            if current_ts is None or row_ts != current_ts:
                if bars:
                    snapshots.append(
                        MarketDataSnapshot(
                            timestamp=current_dt,  # type: ignore[arg-type]
                            bars=MappingProxyType(bars),
                        )
                    )
                    bars = {}
                current_ts = row_ts
                current_dt = row_ts.to_pydatetime()
            symbol = str(row.symbol).strip()
            if symbol in bars:
                raise ValueError(
                    f"Duplicate bar for symbol={symbol!r} at timestamp={row_ts!r}; "
                    "cross-sectional snapshots require one row per symbol per timestamp."
                )
            bars[symbol] = OHLCV(
                symbol=symbol,
                timestamp=current_dt,
                open=Decimal(str(row.open)),
                high=Decimal(str(row.high)),
                low=Decimal(str(row.low)),
                close=Decimal(str(row.close)),
                volume=int(row.volume),
            )
        if bars:
            snapshots.append(
                MarketDataSnapshot(
                    timestamp=current_dt,  # type: ignore[arg-type]
                    bars=MappingProxyType(bars),
                )
            )
        self._snapshots = snapshots
        self._index = 0

    def has_next(self) -> bool:
        return self._index < len(self._snapshots)

    def next_snapshot(self) -> MarketDataSnapshot | None:
        """Return the next complete timestamp slice.

        Data-boundary enforcement: callers receive all symbols for exactly one
        timestamp. The handler never exposes a later timestamp while the current
        snapshot is being processed.
        """
        if not self.has_next():
            return None
        snapshot = self._snapshots[self._index]
        self._index += 1
        return snapshot

    def next_bar(self) -> OHLCV | None:
        """Legacy row compatibility helper.

        Prefer ``next_snapshot``. If legacy code calls ``next_bar``, only the
        deterministic representative bar from the next snapshot is returned.
        """
        snapshot = self.next_snapshot()
        if snapshot is None:
            return None
        return snapshot.representative_bar()

    def reset(self) -> None:
        self._index = 0


# ---------------------------------------------------------------------------
# Fill simulator
# ---------------------------------------------------------------------------


class FillSimulator:
    """Simulates order fills with slippage, commissions, and market impact."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        fill_at: str = "next_open",  # EXECUTION-BAR price level used for the fill.
                                  # The order is generated at the prior snapshot and queued, so it
                                  # always fills against the *execution* bar (>= the signal bar) --
                                  # never a future close. Both modes are therefore look-ahead-safe.
                                  #   "next_open" (default): fill at bar.open.
                                  #   "close": fill at bar.close.
        impact_config: MarketImpactConfig | None = None,
        volume_filter_config: VolumeFilterConfig | None = None,
    ) -> None:
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.fill_at = fill_at
        self._next_bar: OHLCV | None = None
        self._current_bar: OHLCV | None = None
        self._impact_engine: MarketImpactEngine | None = None
        self._permanent_impact: dict[str, float] = {}  # symbol → cumulative perm impact in price units

        if impact_config is not None:
            self._impact_engine = MarketImpactEngine(impact_config, volume_filter_config)

    def set_data(self, data: pd.DataFrame) -> None:
        """Provide historical data for ADV/sigma estimation."""
        if self._impact_engine is not None:
            self._impact_engine.set_data(data)

    def set_bars(self, current: OHLCV, next_bar: OHLCV | None = None) -> None:
        self._current_bar = current
        self._next_bar = next_bar

    def simulate_fill(self, order: Order) -> Fill | None:
        """Simulate a fill for the given order. Returns None if order cannot fill."""
        # Phase 3: route option orders to BSM-based pricing
        if order.metadata.get("instrument") == "OPTION":
            return self._simulate_option_fill(order)

        bar = self._current_bar
        if bar is None:
            return None

        # fill_at selects the execution bar's price level (see __init__ doc).
        fill_price = float(bar.open) if self.fill_at == "next_open" else float(bar.close)

        # Apply base slippage
        slip = fill_price * self.slippage_bps / 10_000.0
        if order.side == Side.BUY:
            fill_price += slip
        else:
            fill_price -= slip

        qty = float(order.quantity)
        bar_volume = int(bar.volume)
        market_impact_cost = 0.0
        impact_bps = 0.0

        # Volume filter and market impact
        if self._impact_engine is not None:
            # Check volume filter (may reduce fill quantity)
            adjusted_qty, is_clipped = self._impact_engine.check_volume_filter(
                qty, bar_volume
            )
            if adjusted_qty <= 0:
                return None  # Fill rejected by volume filter

            if is_clipped:
                qty = adjusted_qty

            # Compute market impact
            result = self._impact_engine.estimate_impact(
                shares=qty,
                price=float(bar.open),
                bar_volume=bar_volume,
            )
            impact_bps = result.total_cost_bps

            # Apply temporary impact to fill price
            temp_impact_price = float(bar.open) * result.temp_impact_bps / 10_000.0
            if order.side == Side.BUY:
                fill_price += temp_impact_price
            else:
                fill_price -= temp_impact_price

            # Apply accumulated permanent impact from prior trades.
            # Permanent impact is signed: buys push price up (+), sells push
            # price down (-).  The fill adjustment is always additive so that
            # a long accumulation makes subsequent buys more expensive AND
            # subsequent sells more profitable (unwind benefits).
            sym_perm = self._permanent_impact.get(order.symbol, 0.0)
            if sym_perm != 0.0:
                fill_price += sym_perm

            # Update cumulative permanent impact (signed by side)
            perm_impact_price = float(bar.open) * result.perm_impact_bps / 10_000.0
            sign = 1.0 if order.side == Side.BUY else -1.0
            self._permanent_impact[order.symbol] = (
                self._permanent_impact.get(order.symbol, 0.0) + sign * perm_impact_price
            )

            market_impact_cost = result.total_cost_dollars

        # Clip fill price to the bar's [low, high] range. A fill can never
        # land outside the observed bar, so this is a hard bound.
        # EDGE CASE (documented guardrail): on a zero-range bar
        # (open == high == low == close) the clip collapses any
        # slippage/impact to the single bar price -- i.e. slippage is
        # silently zeroed. This is intentional (you cannot pay slippage
        # on a bar with no range) but is called out here so it is
        # never mistaken for a missing-cost bug. See
        # tests/unit/backtest/test_impact.py::test_zero_range_bar_clamps_slippage.
        bar_high = float(bar.high)
        bar_low = float(bar.low)
        fill_price = max(bar_low, min(fill_price, bar_high))

        # Check limit orders
        if order.type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == Side.BUY and fill_price > float(order.limit_price):
                return None
            if order.side == Side.SELL and fill_price < float(order.limit_price):
                return None

        commission = qty * self.commission_per_share

        return Fill(
            order_id=f"{order.id}_{order.side.value}",
            price=Decimal(str(round(fill_price, 6))),
            quantity=Decimal(str(qty)),
            commission=Decimal(str(round(commission, 4))),
            timestamp=bar.timestamp,
            side=order.side,
            slippage_bps=Decimal(str(self.slippage_bps)),
            market_impact_cost=Decimal(str(round(market_impact_cost, 4))),
            impact_bps=Decimal(str(round(impact_bps, 4))),
        )

    def _simulate_option_fill(self, order: Order) -> Fill | None:
        """BSM-based fill for option orders using OptionFillSimulator."""
        meta = order.metadata
        strike = float(meta.get("strike", 0.0))
        iv = float(meta.get("iv", 0.0))
        dte = float(meta.get("dte", 0.0))
        contracts = int(meta.get("contracts", 0))
        if contracts <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
            return None

        bar = self._current_bar
        if bar is None:
            return None
        underlying_price = float(bar.close)

        is_call = meta.get("option_type", "PUT").upper() == "CALL"
        option_sim = OptionFillSimulator()
        result = option_sim.simulate_option_fill(
            underlying_price=underlying_price,
            strike=strike,
            iv=iv,
            dte=dte,
            r=float(meta.get("risk_free_rate", 0.045)),
            is_call=is_call,
            side=order.side,
            contracts=contracts,
            spread_pct=float(meta.get("bid_ask_spread_pct", 0.05)),
            commission_per_contract=float(meta.get("commission_per_contract", 0.65)),
        )

        return Fill(
            order_id=f"{order.id}_{order.side.value}",
            price=Decimal(str(round(result["fill_price"], 6))),
            quantity=order.quantity,
            commission=Decimal(str(round(result["commission"], 4))),
            timestamp=bar.timestamp,
            side=order.side,
            slippage_bps=Decimal("0"),
            market_impact_cost=Decimal("0"),
            impact_bps=Decimal("0"),
        )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven backtester with strict no-lookahead enforcement.

    Integrated with Phase 3 modules:
    - Risk limits: pre-trade checks prevent rule-breaking orders
    - Position sizing: configurable sizing (Kelly, vol-target, ATR, fixed fraction)
    - Drawdown tracking: monitors DD and triggers actions at thresholds
    - Regime filtering: optional regime model filters signals by market state
    - Risk metrics: summary() returns a full PerformanceReport
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 1_000_000.0,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        risk_limits: RiskLimits | None = None,
        sizing_pct: float = 0.10,
        sizing_fn: Callable[[float, float, dict[str, Any]], float] | None = None,
        drawdown_config: DrawdownConfig | None = None,
        regime_model: Any = None,
        impact_config: MarketImpactConfig | None = None,
        volume_filter_config: VolumeFilterConfig | None = None,
        financing_config: FinancingConfig | None = None,
        corporate_actions_config: CorporateActionsConfig | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.strategy = strategy
        self.portfolio = PortfolioState(initial_cash)
        self.portfolio.set_financing_config(financing_config)
        self._financing_config = financing_config
        self._corporate_actions_config = corporate_actions_config
        self.fill_sim = FillSimulator(
            slippage_bps,
            commission_per_share,
            impact_config=impact_config,
            volume_filter_config=volume_filter_config,
        )
        self.risk_limits = risk_limits
        self.sizing_pct = sizing_pct
        self.sizing_fn = sizing_fn  # (cash, price, context) -> shares
        self.drawdown_tracker = DrawdownTracker(drawdown_config) if drawdown_config else None
        self.regime_model = regime_model
        self.risk_manager = risk_manager
        self._risk_decisions: list[RiskDecision] = []
        self._rejected_orders: list[tuple[Order, LimitCheckResult]] = []
        # Phase 3: track option metadata by contract_id for EOP/liquidation pricing
        self._option_meta: dict[str, dict] = {}

    @property
    def rejected_orders(self) -> list[tuple[Order, LimitCheckResult]]:
        return self._rejected_orders

    @property
    def risk_decisions(self) -> list[RiskDecision]:
        return list(self._risk_decisions)

    def _force_fill(self, order: Order, bar: OHLCV) -> Fill:
        """Create a forced fill at worst-case price when simulate_fill fails.

        Uses bar.low for sells, bar.high for buys — worst-case execution.
        """
        if order.side == Side.SELL:
            price = Decimal(str(bar.low))
        else:
            price = Decimal(str(bar.high))
        return Fill(
            order_id=order.id,
            price=price,
            quantity=order.quantity,
            commission=Decimal("0"),
            exchange="FORCED",
            side=order.side,
        )

    def _reduce_positions(self, bar: OHLCV, fraction: float) -> None:
        """Reduce all positions by the given fraction (0.0 to 1.0).

        Args:
            bar: Current bar for fill simulation.
            fraction: Fraction of each position to close (e.g., 0.25 = close 25%).
        """
        for sym in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[sym]
            current_qty = float(pos.quantity)
            if abs(current_qty) < 1e-8:
                continue
            reduce_qty = int(abs(current_qty) * fraction)
            if reduce_qty <= 0:
                continue
            close_side = Side.SELL if current_qty > 0 else Side.BUY
            oid = f"ord_dd_reduce_{sym}_{close_side.value}"
            meta = getattr(self, '_option_meta', {}).get(sym, {})
            order = Order(
                id=oid, symbol=sym, side=close_side,
                type=OrderType.MARKET, quantity=Decimal(str(reduce_qty)),
                metadata=meta,
            )
            self.fill_sim.set_bars(bar, None)
            fill = self.fill_sim.simulate_fill(order)
            if fill is None:
                fill = self._force_fill(order, bar)
            self.portfolio.apply_fill(fill, sym)
            self.strategy.on_fill(fill, self.portfolio)

    def _flatten_all_positions(self, bar: OHLCV) -> None:
        """Close ALL open positions immediately at current bar price.

        Args:
            bar: Current bar for fill simulation.
        """
        for sym in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[sym]
            close_qty = int(abs(float(pos.quantity)))
            if close_qty == 0:
                continue
            close_side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
            oid = f"ord_dd_flatten_{sym}_{close_side.value}"
            meta = getattr(self, '_option_meta', {}).get(sym, {})
            order = Order(
                id=oid, symbol=sym, side=close_side,
                type=OrderType.MARKET, quantity=Decimal(str(close_qty)),
                metadata=meta,
            )
            self.fill_sim.set_bars(bar, None)
            fill = self.fill_sim.simulate_fill(order)
            if fill is None:
                fill = self._force_fill(order, bar)
            self.portfolio.apply_fill(fill, sym)
            self.strategy.on_fill(fill, self.portfolio)

    def _compute_sizing(
        self, side: Side, price: float, bar: OHLCV, sig: SignalData
    ) -> int:
        """Compute position size using the configured sizing strategy."""
        cash = self.portfolio.cash
        pv = self.portfolio.total_value

        # Strategy-computed notional_size (SignalData) or size (TradeSignal) takes priority
        # SignalData uses notional_size; TradeSignal uses size
        strat_size = getattr(sig, 'notional_size', None)
        if strat_size is None:
            strat_size = getattr(sig, 'size', 0)
        if strat_size is not None and strat_size > 0:
            qty = int(strat_size / price)
            if side == Side.BUY:
                return min(qty, int(cash / price)) if cash > 0 else 0
            else:
                # Short: limit to 2x portfolio value (Reg T margin)
                return min(qty, int(pv * 2.0 / price))

        if cash <= 0 and side == Side.BUY:
            return 0

        if self.sizing_fn is not None:
            context = {
                "cash": cash,
                "price": price,
                "signal_strength": sig.strength,
                "bar": bar,
                "portfolio_value": pv,
            }
            return int(self.sizing_fn(cash, price, context))

        qty = int(pv * self.sizing_pct / price)
        if side == Side.BUY:
            return min(qty, int(cash / price)) if cash > 0 else 0
        else:
            return qty

    def _current_regime(self, bar: OHLCV) -> str | None:
        """Detect current regime from the model if configured."""
        if self.regime_model is None:
            return None
        try:
            ret = float(bar.close) / float(bar.open) - 1.0
            state, prob = self.regime_model.predict_state(np.array([ret]))
            return self.regime_model.state_labels[state] if self.regime_model.state_labels else str(state)
        except Exception:
            return None

    def _execute_queued_orders(
        self,
        queued_orders: list[Order],
        snapshot: MarketDataSnapshot,
    ) -> list[Order]:
        """Fill orders queued from the previous timestamp using current open prices.

        This is the only place queued strategy/risk orders are executed. Orders
        generated from snapshot T are never filled with T prices; they are filled
        only when a later snapshot becomes available.
        """
        if not snapshot.bars:
            return list(queued_orders)

        unfilled: list[Order] = []
        for order in queued_orders:
            bar = snapshot.bars.get(order.symbol)
            if bar is None:
                # Missing/omitted symbol for this timestamp: keep the order queued
                # rather than injecting a synthetic price or breaking alignment.
                unfilled.append(order)
                continue

            # Fill at the current timestamp's open. The current snapshot is already
            # available, so this uses no future data.
            self.fill_sim.set_bars(bar, bar)
            fill = self.fill_sim.simulate_fill(order)
            if fill is None:
                fill = self._force_fill(order, bar)
            self.portfolio.apply_fill(fill, order.symbol)
            self.strategy.on_fill(fill, self.portfolio)

        return unfilled

    def _queue_position_adjustment(
        self,
        queued_orders: list[Order],
        snapshot: MarketDataSnapshot,
        fraction: float,
        reason: str,
    ) -> None:
        """Queue proportional position reductions for the next timestamp."""
        for sym, pos in self.portfolio.positions.items():
            if sym not in snapshot.bars:
                # Do not manufacture prices for missing symbols.
                continue
            current_qty = float(pos.quantity)
            if abs(current_qty) < 1e-8:
                continue
            reduce_qty = int(abs(current_qty) * fraction)
            if reduce_qty <= 0:
                continue
            close_side = Side.SELL if current_qty > 0 else Side.BUY
            meta = getattr(self, "_option_meta", {}).get(sym, {})
            queued_orders.append(
                Order(
                    id=f"ord_{snapshot.timestamp.isoformat()}_{sym}_{reason}_{close_side.value}",
                    symbol=sym,
                    side=close_side,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(reduce_qty)),
                    metadata=meta,
                )
            )

    def _queue_flatten_all_positions(
        self,
        queued_orders: list[Order],
        snapshot: MarketDataSnapshot,
        reason: str,
    ) -> None:
        """Queue full liquidation for the next timestamp."""
        for sym, pos in self.portfolio.positions.items():
            if sym not in snapshot.bars:
                continue
            close_qty = int(abs(float(pos.quantity)))
            if close_qty == 0:
                continue
            close_side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
            meta = getattr(self, "_option_meta", {}).get(sym, {})
            queued_orders.append(
                Order(
                    id=f"ord_{snapshot.timestamp.isoformat()}_{sym}_{reason}_{close_side.value}",
                    symbol=sym,
                    side=close_side,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(close_qty)),
                    metadata=meta,
                )
            )

    def _generate_signals(self, snapshot: MarketDataSnapshot) -> list[SignalData]:
        """Generate cross-sectional signals from an immutable snapshot.

        Data boundary: fill_sim._next_bar is explicitly None while strategy code
        runs. The strategy receives either a full snapshot via on_snapshot(), or
        a single representative bar via the legacy on_bar() compatibility path.
        """
        clean_snapshot(snapshot)
        representative = snapshot.representative_bar()
        if representative is None:
            return []

        # Strict look-ahead barrier: no next-timestamp bar is visible to strategy.
        self.fill_sim.set_bars(representative, None)

        regime = self._current_regime(representative)
        if regime:
            self.portfolio.regime_state = regime

        if hasattr(self.strategy, "on_snapshot"):
            return list(self.strategy.on_snapshot(snapshot, self.portfolio) or [])

        # Legacy compatibility only. Cross-sectional strategies should implement
        # on_snapshot() to avoid single-row leakage.
        return list(self.strategy.on_bar(representative, self.portfolio) or [])

    @staticmethod
    def _direction_value(sig: Any) -> Any:
        return getattr(sig.direction, "value", sig.direction)

    @staticmethod
    def _is_flat_signal(sig: Any) -> bool:
        direction = BacktestEngine._direction_value(sig)
        return direction == "flat" or direction == 0

    @staticmethod
    def _is_long_signal(sig: Any) -> bool:
        direction = BacktestEngine._direction_value(sig)
        return direction == "long" or direction == 1

    @staticmethod
    def _is_short_signal(sig: Any) -> bool:
        direction = BacktestEngine._direction_value(sig)
        return direction == "short" or direction == -1

    def _signal_reference_price(self, sig: Any, snapshot: MarketDataSnapshot) -> float | None:
        symbol = str(getattr(sig, "symbol", "")).strip()
        bar = snapshot.get_bar(symbol)
        if bar is None:
            return None
        return float(bar.close)

    def _orders_from_risk_decision(
        self,
        decision_trades: Sequence[ApprovedTrade],
        timestamp: datetime,
    ) -> list[Order]:
        """Convert broker-neutral approved trades into internal orders."""
        orders: list[Order] = []
        for trade in decision_trades:
            quantity = max(1, int(trade.quantity))
            orders.append(
                Order(
                    id=f"ord_{timestamp.isoformat()}_{trade.symbol}_{trade.side.value}_risk",
                    symbol=trade.symbol,
                    side=trade.side,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(quantity)),
                    metadata={
                        "risk_managed": True,
                        "target_weight": trade.target_weight,
                        "normalized_weight": trade.normalized_weight,
                        "target_notional": trade.target_notional,
                        "delta_notional": trade.delta_notional,
                    },
                )
            )
        return orders

    def _queue_orders_from_signals(
        self,
        queued_orders: list[Order],
        snapshot: MarketDataSnapshot,
        signals: Sequence[SignalData],
    ) -> None:
        """Convert signals into orders queued for the next timestamp window.

        If a broker-neutral RiskManager is configured, raw strategy outputs are
        normalized and compliance-gated before this engine constructs Order
        objects. This keeps alpha logic isolated from portfolio constraints.
        """
        if self.risk_manager is not None:
            context = RiskContext.from_bars(snapshot.bars)
            decision = self.risk_manager.pre_trade_check(
                raw_signals=signals,
                portfolio=self.portfolio,
                market=snapshot,
                context=context,
            )
            self._risk_decisions.append(decision)
            queued_orders.extend(
                self._orders_from_risk_decision(decision.approved_trades, snapshot.timestamp)
            )
            return

        for sig in signals:
            symbol = str(getattr(sig, "symbol", "")).strip()
            if not symbol:
                continue

            price = self._signal_reference_price(sig, snapshot)
            if price is None or price <= 0:
                # Missing/omitted symbol: skip signal instead of inventing data.
                continue

            metadata = getattr(sig, "metadata", {}) or {}
            is_option = getattr(sig, "instrument_type", "") == "option" or metadata.get("instrument") == "OPTION"
            if is_option:
                order_symbol = str(metadata.get("contract_id", symbol))
                multiplier = int(metadata.get("multiplier", 100))
                contracts = int(metadata.get("contracts", 0))
                option_qty = contracts * multiplier if contracts > 0 else 0
            else:
                order_symbol = symbol
                option_qty = 0

            pos = self.portfolio.positions.get(order_symbol)
            current_qty = float(pos.quantity) if pos else 0.0

            if self._is_flat_signal(sig):
                if abs(current_qty) > 1e-8:
                    close_side = Side.SELL if current_qty > 0 else Side.BUY
                    queued_orders.append(
                        Order(
                            id=f"ord_{snapshot.timestamp.isoformat()}_{order_symbol}_exit",
                            symbol=order_symbol,
                            side=close_side,
                            type=OrderType.MARKET,
                            quantity=Decimal(str(int(abs(current_qty)))),
                            metadata=metadata,
                        )
                    )
                continue

            if self._is_long_signal(sig):
                side = Side.BUY
            elif self._is_short_signal(sig):
                side = Side.SELL
            else:
                continue

            # Queue any required opposing-position close before the new target order.
            # Both orders execute only on the next timestamp window.
            if not is_option:
                if side == Side.BUY and current_qty < 0:
                    cover_qty = int(abs(current_qty))
                    queued_orders.append(
                        Order(
                            id=f"ord_{snapshot.timestamp.isoformat()}_{order_symbol}_buy_cover",
                            symbol=order_symbol,
                            side=Side.BUY,
                            type=OrderType.MARKET,
                            quantity=Decimal(str(cover_qty)),
                            metadata=metadata,
                        )
                    )
                elif side == Side.SELL and current_qty > 0:
                    close_qty = int(abs(current_qty))
                    queued_orders.append(
                        Order(
                            id=f"ord_{snapshot.timestamp.isoformat()}_{order_symbol}_sell_close",
                            symbol=order_symbol,
                            side=Side.SELL,
                            type=OrderType.MARKET,
                            quantity=Decimal(str(close_qty)),
                            metadata=metadata,
                        )
                    )

            representative = snapshot.representative_bar()
            qty = option_qty if is_option else self._compute_sizing(side, price, representative, sig)
            if qty <= 0:
                continue

            order = Order(
                id=f"ord_{snapshot.timestamp.isoformat()}_{order_symbol}_{side.value}",
                symbol=order_symbol,
                side=side,
                type=OrderType.MARKET,
                quantity=Decimal(str(qty)),
                metadata=metadata,
            )

            if self.risk_limits is not None:
                from quantcore.core.types import PortfolioSnapshot

                snap = PortfolioSnapshot(
                    timestamp=snapshot.timestamp,
                    positions=self.portfolio.positions,
                    cash=Decimal(str(max(0.0, self.portfolio.cash))),
                    total_value=Decimal(str(self.portfolio.total_value)),
                    returns=self.portfolio.returns,
                )
                result = check_pre_trade(
                    order,
                    snap,
                    self.risk_limits,
                    market_price=price,
                )
                if result.status.value == "rejected":
                    self._rejected_orders.append((order, result))
                    continue

            queued_orders.append(order)

            if is_option:
                self._option_meta[order_symbol] = metadata

            self.portfolio.signal_log.append({
                "timestamp": snapshot.timestamp,
                "symbol": order_symbol,
                "direction": self._direction_value(sig),
                "strength": getattr(sig, "strength", 0.0),
                "regime": self.portfolio.regime_state,
            })

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the backtest on historical data.

        data: DataFrame with columns [symbol, timestamp, open, high, low, close, volume]
        Returns DataFrame with equity curve and risk metrics accessible via summary().

        Core event loop: each timestamp is processed as one immutable
        MarketDataSnapshot. The loop never peeks at a later timestamp during
        signal generation.
        """
        handler = HistoricalDataHandler(data)
        self.portfolio = PortfolioState(self.portfolio.initial_cash)
        self.portfolio.set_financing_config(self._financing_config)
        self.portfolio.set_corporate_actions(self._corporate_actions_config)
        self._risk_decisions = []
        self._rejected_orders = []
        self._option_meta = {}

        self.fill_sim.set_data(data)

        if self.drawdown_tracker:
            self.drawdown_tracker = DrawdownTracker(
                self.drawdown_tracker.config if hasattr(self.drawdown_tracker, 'config') else DrawdownConfig()
            )

        # Orders generated from snapshot T are queued here and executed only when
        # snapshot T+1 arrives. This is the execution queue boundary.
        queued_orders: list[Order] = []
        daily_returns: list[float] = []
        halted = False
        last_snapshot: MarketDataSnapshot | None = None

        while handler.has_next():
            # 1. Grouping by Timestamp: HistoricalDataHandler yields one complete
            # timestamp slice containing all available symbols for that timestamp.
            snapshot = handler.next_snapshot()
            if snapshot is None:
                break
            clean_snapshot(snapshot)
            last_snapshot = snapshot

            # Execution of prior snapshot's queued orders at the current open.
            # These orders were created after the previous snapshot was fully
            # processed, so filling them now cannot leak future data backward.
            queued_orders = self._execute_queued_orders(queued_orders, snapshot)

            # 3. Mark-to-Market Valuation: update the entire portfolio from the
            # current snapshot before any strategy logic can observe prices.
            self.portfolio.mark_to_market(snapshot)

            # 4. Financing & Fees: borrow/margin costs now use newly marked prices.
            self.portfolio.daily_financing_update()

            # 4b. Corporate actions: credit dividends / apply splits for today.
            self.portfolio.daily_corporate_actions_update(snapshot.timestamp)

            # Risk circuit breakers queue liquidation for the next timestamp
            # instead of filling intra-bar from the signal-generation snapshot.
            if self.portfolio.is_margin_breach:
                queued_orders.clear()
                self._queue_flatten_all_positions(queued_orders, snapshot, "margin_liquidation")
                ts = snapshot.timestamp.timestamp() if hasattr(snapshot.timestamp, "timestamp") else 0.0
                self.portfolio.record_equity(ts)
                continue

            # Drawdown circuit breakers.
            if self.drawdown_tracker is not None:
                dd_state = self.drawdown_tracker.update(self.portfolio.total_value)
                self.portfolio.regime_state = dd_state.action.value

                if dd_state.action == DrawdownAction.REDUCE_25:
                    self._queue_position_adjustment(queued_orders, snapshot, 0.25, "dd_reduce_25")
                elif dd_state.action == DrawdownAction.REDUCE_50:
                    self._queue_position_adjustment(queued_orders, snapshot, 0.50, "dd_reduce_50")
                elif dd_state.action == DrawdownAction.REDUCE_75:
                    self._queue_position_adjustment(queued_orders, snapshot, 0.75, "dd_reduce_75")
                elif dd_state.action == DrawdownAction.FLATTEN:
                    queued_orders.clear()
                    self._queue_flatten_all_positions(queued_orders, snapshot, "dd_flatten")
                elif dd_state.action == DrawdownAction.HALT:
                    queued_orders.clear()
                    self._queue_flatten_all_positions(queued_orders, snapshot, "dd_halt")
                    halted = True

            if halted:
                # No new alpha signals after halt; only queued liquidation/fills remain.
                ts = snapshot.timestamp.timestamp() if hasattr(snapshot.timestamp, "timestamp") else 0.0
                self.portfolio.record_equity(ts)
                continue

            # 5. Cross-Sectional Signal Generation: strategy sees the complete
            # immutable snapshot. No future timestamp is visible here.
            signals = self._generate_signals(snapshot)

            # 6. Execution Queueing: convert signals into internal orders for the
            # next timestamp window. No immediate fill occurs in this step.
            self._queue_orders_from_signals(queued_orders, snapshot, signals)

            # Record daily return after MTM, financing, fills, and queued orders.
            prev_equity = self.portfolio.equity_curve[-1][1] if self.portfolio.equity_curve else self.portfolio.initial_cash
            current_equity = self.portfolio.total_value
            daily_returns.append((current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0)

            ts = snapshot.timestamp.timestamp() if hasattr(snapshot.timestamp, "timestamp") else 0.0
            self.portfolio.record_equity(ts)

        # End-of-run cleanup: process remaining queued orders and liquidate open
        # positions using the last available snapshot. This is a forced terminal
        # liquidation, not part of the normal intra-run event loop.
        if last_snapshot is not None:
            for order in queued_orders:
                bar = last_snapshot.bars.get(order.symbol)
                if bar is None:
                    representative = last_snapshot.representative_bar()
                    if representative is None:
                        continue
                    bar = representative
                self.fill_sim.set_bars(bar, None)
                fill = self.fill_sim.simulate_fill(order)
                if fill is None:
                    fill = self._force_fill(order, bar)
                self.portfolio.apply_fill(fill, order.symbol)
                self.strategy.on_fill(fill, self.portfolio)

            for sym, pos in list(self.portfolio.positions.items()):
                close_qty = int(abs(float(pos.quantity)))
                if close_qty == 0:
                    continue
                bar = last_snapshot.bars.get(sym)
                if bar is None:
                    representative = last_snapshot.representative_bar()
                    if representative is None:
                        continue
                    bar = representative
                close_side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
                meta = getattr(self, "_option_meta", {}).get(sym, {})
                order = Order(
                    id=f"ord_eop_{sym}_{'sell' if close_side == Side.SELL else 'buy_cover'}",
                    symbol=sym,
                    side=close_side,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(close_qty)),
                    metadata=meta,
                )
                self.fill_sim.set_bars(bar, None)
                fill = self.fill_sim.simulate_fill(order)
                if fill is None:
                    fill = self._force_fill(order, bar)
                self.portfolio.apply_fill(fill, sym)
                self.strategy.on_fill(fill, self.portfolio)
                self.portfolio.signal_log.append({
                    "timestamp": last_snapshot.timestamp,
                    "symbol": sym,
                    "direction": "eop_close",
                    "strength": 1.0,
                    "regime": self.portfolio.regime_state,
                })

            ts = last_snapshot.timestamp.timestamp() if hasattr(last_snapshot.timestamp, "timestamp") else 0.0
            self.portfolio.record_equity(ts)

        if not self.portfolio.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])

        timestamps, values = zip(*self.portfolio.equity_curve)
        self._last_returns = np.array(daily_returns)
        return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, unit="s"), "equity": values})

    def summary(self) -> PerformanceReport:
        """Generate a full performance report from the backtest results.

        Uses the risk metrics module for all ratio computations.
        """
        if not self.portfolio.equity_curve:
            return PerformanceReport()

        equity = np.array([v for _, v in self.portfolio.equity_curve])
        returns = compute_returns(equity)
        trade_pnls = self._compute_trade_pnls(self.portfolio.trades)

        # Detect multi-symbol: count unique symbols across all trades
        # Symbol is embedded in order_id (not a direct Fill field)
        unique_symbols = set()
        for t in self.portfolio.trades:
            unique_symbols.add(self._parse_symbol_from_fill(t.order_id))
        n_symbols = len(unique_symbols)

        return generate_performance_report(
            equity,
            returns,
            np.array(trade_pnls) if trade_pnls else None,
            total_market_impact_cost=self.portfolio.total_market_impact_cost,
            total_borrow_cost=self.portfolio.total_borrow_cost,
            total_margin_interest=self.portfolio.total_margin_interest,
            is_multi_symbol=(n_symbols > 1),
            constituent_symbol_count=n_symbols,
        )

    @staticmethod
    def _parse_symbol_from_fill(order_id: str) -> str:
        """Extract symbol from a fill's order_id.

        order_id format: ord_{iso_timestamp}_{symbol}_{side}_{side}
        e.g. 'ord_2020-01-02T14:30:00_AAPL_buy_buy'
        """
        base = order_id.rsplit("_", 2)[0]
        return base.split("_", 2)[2]

    @staticmethod
    def _compute_trade_pnls(trades: list) -> list[float]:
        """Compute realized P&Ls via FIFO pairing of buys and sells per symbol."""
        from collections import defaultdict, deque

        by_symbol: dict[str, list] = defaultdict(list)
        for fill in trades:
            symbol = BacktestEngine._parse_symbol_from_fill(fill.order_id)
            by_symbol[symbol].append(fill)

        realized: list[float] = []
        for fills in by_symbol.values():
            buy_queue: deque = deque()
            for fill in fills:
                qty = float(fill.quantity)
                price = float(fill.price)
                comm = float(fill.commission)

                if fill.side == Side.BUY:
                    buy_queue.append((price, qty, comm))
                elif fill.side == Side.SELL:
                    sell_qty = qty
                    sell_price = price
                    total_sell_qty = sell_qty

                    while sell_qty > 0 and buy_queue:
                        buy_price, buy_qty, buy_comm = buy_queue[0]
                        matched = min(sell_qty, buy_qty)

                        buy_comm_alloc = buy_comm * (matched / buy_qty)
                        sell_comm_alloc = comm * (matched / total_sell_qty)

                        pnl = (sell_price - buy_price) * matched - buy_comm_alloc - sell_comm_alloc
                        realized.append(pnl)

                        sell_qty -= matched
                        if matched >= buy_qty:
                            buy_queue.popleft()
                        else:
                            buy_queue[0] = (buy_price, buy_qty - matched, buy_comm - buy_comm_alloc)

        return realized

    def risk_metrics(self) -> dict[str, float]:
        """Compute risk-adjusted metrics for the backtest using the risk module."""
        from quantcore.risk.metrics import compute_all_metrics
        if not self.portfolio.equity_curve or len(self.portfolio.equity_curve) < 2:
            return {}
        equity = np.array([v for _, v in self.portfolio.equity_curve])
        returns = compute_returns(equity)
        return compute_all_metrics(returns)
