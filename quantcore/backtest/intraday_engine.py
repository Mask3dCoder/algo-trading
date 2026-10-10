"""IntradayBacktestEngine — event-driven backtester for 5-minute bars.

Extends the base BacktestEngine to support:
1. 5-minute OHLCV bar ingestion (from Alpaca historical API)
2. TickContext session tracking (pre-market volume, first-5min VWAP, overnight gap)
3. Realistic slippage: 1 bp SPY/QQQ, 2 bp + IV-rank adjustment for options,
   2× for first-5-min of RTH (opening auction volatility)
4. Session-aware trading: entry only on first RTH bar, hard exit by EXIT_TIME (10:00 AM ET)
5. IntradayDrawdownCalculator for per-bar MTM tracking at 3 horizons

This engine is the BACKTEST engine for opening_gap_fade (Task 2.2).

Slippage model reference:
    ── Instrument ─────── Base ── First-5-min ── IV-Rank adj ─
    SPY/QQQ (equity)      1 bp      2 bp           N/A
    All options            2 bp      4 bp     +0.5 bp per 10% IV Rank

Slippage is added to fill price for BUY orders and subtracted for SELL orders,
consistent with the base engine's behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from quantcore.backtest.simulation import OptionFillSimulator
from quantcore.backtest.tick_context import (
    EXIT_TIME,
    FIRST_5_MIN_CLOSE,
    FIRST_5_MIN_OPEN,
    PRE_MARKET_OPEN,
    RTH_CLOSE,
    RTH_OPEN,
    TickContext,
    TickContextHistory,
    is_exit_time,
    is_premarket_bar,
    is_rth_bar,
)
from quantcore.core.types import OHLCV, Fill, Order, OrderType, Position, Side
from quantcore.risk.intraday_drawdown import IntradayDrawdownCalculator
from quantcore.risk.limits import RiskLimits, check_pre_trade

# Export tick_context constants for strategy use
__all__ = [
    "EXIT_TIME",
    "FIRST_5_MIN_CLOSE",
    "FIRST_5_MIN_OPEN",
    "PRE_MARKET_OPEN",
    "RTH_CLOSE",
    "RTH_OPEN",
    "TickContext",
    "TickContextHistory",
    "IntradayBacktestEngine",
]


# ── Slippage configuration ──────────────────────────────────────────────────────

@dataclass
class IntradaySlippageConfig:
    """Realistic slippage parameters for intraday execution."""

    # Base slippage in bps
    spy_equity_bps: float = 1.0    # 1 bp for SPY/QQQ
    option_bps: float = 2.0        # 2 bp for equity options

    # First-5-min multiplier (opening auction volatility)
    first_5_min_multiplier: float = 2.0

    # IV-rank bonus (added to option spread, per 10% IV Rank)
    iv_rank_bonus_per_10: float = 0.5

    # Slippery tickers (> 5% ADV → 50% more slippage)
    adv_threshold_pct: float = 5.0
    liquidity_penalty_multiplier: float = 1.5

    def get_slippage_bps(
        self,
        is_option: bool,
        is_first_5_min: bool,
        iv_rank: float = 50.0,
        ticker: str = "SPY",
        participation_rate: float = 0.01,
    ) -> float:
        """Compute the actual slippage for a fill.

        Args:
            is_option: True if instrument is an option
            is_first_5_min: True if this is the first 5 min of RTH
            iv_rank: IV rank 0–100 (for options)
            ticker: Symbol (SPY/QQQ/etc.)
            participation_rate: Fraction of ADV (for liquidity penalty)

        Returns:
            Slippage in basis points
        """
        slip = float(self.option_bps if is_option else self.spy_equity_bps)

        # Opening auction volatility multiplier
        if is_first_5_min:
            slip *= self.first_5_min_multiplier

        # IV-rank bonus for options
        if is_option:
            slip += self.iv_rank_bonus_per_10 * (iv_rank / 10.0)

        # Liquidity penalty for illiquid participation rates
        if participation_rate > self.adv_threshold_pct:
            slip *= self.liquidity_penalty_multiplier

        return slip


# ── Intraday bar data handler ──────────────────────────────────────────────────

@dataclass
class IntradayDataHandler:
    """Feeds 5-minute OHLCV bars as MarketEvents, building TickContext.

    Supports:
    - Alpaca 5-min historical bar DataFrame input
    - Pre-market volume accumulation before RTH open
    - Session context wrapping each bar
    """

    def __init__(
        self,
        data: pd.DataFrame,
        prev_close_map: dict[datetime, float] | None = None,
        vix_map: dict[datetime, float] | None = None,
        timeframe: str = "5min",
    ) -> None:
        """Initialize the intraday handler.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume, symbol]
                 OR raw tick data that will be resampled via build_5min_bar_df
            prev_close_map: {date: close} for gap computation
            vix_map: {timestamp: vix_level} for vol regime detection
            timeframe: Resampling timeframe if raw ticks provided
        """
        if not data.empty:
            # Validate and build OHLCV structure
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(set(data.columns)):
                raise ValueError(
                    f"IntradayDataHandler requires columns {required}, got {set(data.columns)}"
                )
            self._data = data.sort_values("timestamp").reset_index(drop=True)
        else:
            self._data = pd.DataFrame()

        self._index = 0
        self._timeframe = timeframe
        self._ctx_history = TickContextHistory(prev_close_map, vix_map)

    def has_next(self) -> bool:
        return self._index < len(self._data)

    def next_bar(self) -> tuple[OHLCV, TickContext] | None:
        """Return the next bar wrapped in its TickContext.

        Returns:
            (bar, context) tuple, or None if exhausted
        """
        if not self.has_next():
            return None

        row = self._data.iloc[self._index]
        self._index += 1

        ts_raw = row["timestamp"]
        ts: datetime
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw.to_pydatetime()
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = pd.Timestamp(ts_raw).to_pydatetime()

        bar = OHLCV(
            symbol=str(row.get("symbol", "SPY")),
            timestamp=ts,
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=int(row["volume"]),
        )

        # Get current VIX for vol regime
        vix = self._ctx_history._vix_map.get(ts, np.nan)

        ctx = self._ctx_history.update(bar, vix=vix)

        return bar, ctx

    def reset(self) -> None:
        self._index = 0


# ── Intraday fill simulator ────────────────────────────────────────────────────

class IntradayFillSimulator:
    """Simulates fills for intraday bars with realistic slippage.

    Key differences from base FillSimulator:
    1. IntradaySlippageConfig: 1 bp base for SPY, 2× for first 5 min, IV bonus
    2. Bid-ask midpoint model for options using half-spread
    3. VIX-aware spread adjustment
    """

    def __init__(
        self,
        slippage_config: IntradaySlippageConfig | None = None,
        commission_per_share: float = 0.005,
        commission_per_contract: float = 0.65,
        fill_at: str = "midpoint",  # Intraday: midpoint is more realistic than open
    ) -> None:
        self.config = slippage_config or IntradaySlippageConfig()
        self.commission_per_share = float(commission_per_share)
        self.commission_per_contract = float(commission_per_contract)
        self.fill_at = fill_at  # "midpoint" | "vwap" | "close"
        self._ctx: TickContext | None = None

    def set_context(self, ctx: TickContext) -> None:
        """Inject the current TickContext for slippage calculation."""
        self._ctx = ctx

    def simulate_fill(
        self,
        order: Order,
        bar: OHLCV,
        iv_rank: float = 50.0,
    ) -> Fill | None:
        """Simulate a fill on an intraday bar.

        Fill price computation:
        1. Use bar VWAP (or midpoint) instead of open — more realistic for intraday
        2. Apply slippage from IntradaySlippageConfig
        3. Clip to bar range

        Args:
            order: The order to fill
            bar: The current 5-min OHLCV bar
            iv_rank: Current IV rank for option spread adjustment

        Returns:
            Fill or None (if rejected)
        """
        if order.side not in (Side.BUY, Side.SELL):
            return None

        # Determine fill price
        O, H, L, C = float(bar.open), float(bar.high), float(bar.low), float(bar.close)

        if self.fill_at == "vwap":
            fill_price = (H + L + C) / 3.0  # Typical intraday VWAP approximation
        elif self.fill_at == "midpoint":
            fill_price = (H + L) / 2.0  # Midpoint of bar range
        else:
            fill_price = C  # Close

        # Slippage
        is_option = order.metadata.get("instrument") == "OPTION" or "/" in order.symbol
        is_first_5 = self._ctx is not None and self._ctx.bars_processed == 1

        slip_bps = self.config.get_slippage_bps(
            is_option=is_option,
            is_first_5_min=is_first_5,
            iv_rank=iv_rank,
            ticker=order.symbol,
        )

        slip = fill_price * slip_bps / 10_000.0
        if order.side == Side.BUY:
            fill_price += slip
        else:
            fill_price -= slip

        # Clip to observed bar range
        fill_price = max(L, min(fill_price, H))

        # Commission
        if is_option:
            contracts = int(order.metadata.get("contracts", abs(float(order.quantity)) / 100))
            commission = contracts * self.commission_per_contract
        else:
            qty = int(abs(float(order.quantity)))
            commission = qty * self.commission_per_share

        return Fill(
            order_id=f"{order.id}_{order.side.value}_intraday",
            price=Decimal(str(round(fill_price, 6))),
            quantity=order.quantity,
            commission=Decimal(str(round(commission, 4))),
            timestamp=bar.timestamp,
            side=order.side,
            slippage_bps=Decimal(str(slip_bps)),
            market_impact_cost=Decimal("0"),
            impact_bps=Decimal("0"),
        )

    def simulate_option_fill(
        self,
        order: Order,
        bar: OHLCV,
        underlying_price: float,
        strike: float,
        iv: float,
        dte: float,
        is_call: bool,
        contracts: int,
        iv_rank: float = 50.0,
        r: float = 0.045,
    ) -> Fill | None:
        """BSM-based option fill for intraday bars.

        Uses BSM with IV rank adjustment for realistic bid-ask spread.
        """
        if contracts <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
            return None

        # Half-spread based on IV rank + base
        base_spread_bps = self.config.get_slippage_bps(
            is_option=True,
            is_first_5_min=(self._ctx is not None and self._ctx.bars_processed == 1),
            iv_rank=iv_rank,
        )
        total_spread_bps = base_spread_bps * 2  # Full spread = 2 × half-spread

        # Option price via BSM midpoint
        from quantcore.math_models.options import black_scholes
        try:
            opt_price = black_scholes(
                S=underlying_price,
                K=strike,
                T=dte / 252.0,
                r=r,
                sigma=iv,
                is_call=is_call,
            )
        except Exception:
            opt_price = float(bar.open)  # Fallback

        # Apply half-spread: buys get ask (mid + spread/2), sells get bid (mid - spread/2)
        half_spread_dollars = opt_price * (total_spread_bps / 10_000.0) / 2.0
        if order.side == Side.BUY:
            fill_price = opt_price + half_spread_dollars
        else:
            fill_price = opt_price - half_spread_dollars

        # Clip to bar range
        fill_price = max(float(bar.low), min(fill_price, float(bar.high)))

        commission = contracts * self.commission_per_contract

        return Fill(
            order_id=f"{order.id}_{order.side.value}_intraday_opt",
            price=Decimal(str(round(fill_price, 6))),
            quantity=Decimal(str(contracts * 100)),
            commission=Decimal(str(round(commission, 4))),
            timestamp=bar.timestamp,
            side=order.side,
            slippage_bps=Decimal(str(base_spread_bps)),
            market_impact_cost=Decimal("0"),
            impact_bps=Decimal("0"),
        )


# ── Intraday backtest engine ──────────────────────────────────────────────────

class IntradayBacktestEngine:
    """Event-driven backtester for 5-minute intraday bars.

    Key features beyond the base BacktestEngine:
    - Processes 5-min OHLCV bars with TickContext wrapping
    - Realistic intraday slippage model
    - IntradayDrawdownCalculator tracking MTM at 3 horizons
    - Session-aware exit (hard exit by EXIT_TIME = 10:00 AM ET)
    - Pre-market volume tracking
    - VIX regime detection

    Designed for: opening_gap_fade, any other intraday equity strategy.

    NOT designed for: options strategies that need Greeks-level precision
    (for those, use IntradayOptionsBacktestEngine — TODO: future work).

    Usage:
        from quantcore.backtest.tick_context import TickContextHistory

        engine = IntradayBacktestEngine(
            strategy=OpeningGapFadeStrategy(config),
            initial_cash=1_000_000,
            slippage_config=IntradaySlippageConfig(),
            prev_close_map=prev_closes,    # date → prev_close
            vix_map=vix_times,            # timestamp → VIX level
        )

        result = engine.run(intraday_5min_df)
        report = engine.summary()
        dd_report = engine.intraday_dd.report()
    """

    def __init__(
        self,
        strategy,                 # Strategy with on_bar_intraday()
        initial_cash: float = 1_000_000,
        slippage_config: IntradaySlippageConfig | None = None,
        prev_close_map: dict[datetime, float] | None = None,
        vix_map: dict[datetime, float] | None = None,
        risk_limits: RiskLimits | None = None,
        sizing_pct: float = 0.10,
        sizing_fn: Callable | None = None,
    ) -> None:
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.slippage_config = slippage_config or IntradaySlippageConfig()
        self.prev_close_map = prev_close_map or {}
        self.vix_map = vix_map or {}
        self.risk_limits = risk_limits
        self.sizing_pct = sizing_pct
        self.sizing_fn = sizing_fn

        # Internal state
        self._cash = float(initial_cash)
        self._positions: dict[str, Position] = {}
        self._prices: dict[str, float] = {}
        self._equity_curve: list[float] = []
        self._timestamps: list[datetime] = []
        self._trades: list[Fill] = []
        self._signal_log: list[dict] = []

        # Intraday-specific: MTM tracker
        self._dd_calc = IntradayDrawdownCalculator(
            initial_nav=float(initial_cash),
            vix_base=20.0,
            shock_magnitude=5.0,
        )

        # Context
        self._current_ctx: TickContext | None = None
        self._pending_orders: list[Order] = []

        # Options support
        self._option_meta: dict[str, dict] = {}
        self._option_fill_sim = OptionFillSimulator()

    @property
    def rejected_orders(self) -> list:
        return []  # Stub — filled by pre-trade checks

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the intraday backtest on 5-minute bar data.

        Args:
            data: DataFrame with [timestamp, symbol, open, high, low, close, volume]

        Returns:
            DataFrame with equity_curve (timestamp, equity)
        """
        if data.empty:
            return pd.DataFrame(columns=["timestamp", "equity"])

        # Build TickContext history
        ctx_history = TickContextHistory(self.prev_close_map, self.vix_map)

        # Initialize fill simulator with context injection
        fill_sim = IntradayFillSimulator(self.slippage_config)

        self._cash = float(self.initial_cash)
        self._positions = {}
        self._equity_curve = []
        self._timestamps = []
        self._trades = []
        self._signal_log = []
        self._rejected_orders: list = []

        data = data.sort_values("timestamp").reset_index(drop=True)

        for i in range(len(data)):
            row = data.iloc[i]
            ts_raw = row["timestamp"]
            ts: datetime
            if isinstance(ts_raw, pd.Timestamp):
                ts = ts_raw.to_pydatetime()
            elif isinstance(ts_raw, datetime):
                ts = ts_raw
            else:
                ts = pd.Timestamp(ts_raw).to_pydatetime()

            bar = OHLCV(
                symbol=str(row.get("symbol", "SPY")),
                timestamp=ts,
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=int(row["volume"]),
            )

            # Update prices for mark-to-market
            self._prices[bar.symbol] = float(bar.close)

            # Build/Update TickContext
            vix = self.vix_map.get(ts, np.nan)
            ctx = ctx_history.update(bar, vix=vix)
            self._current_ctx = ctx

            # Inject context for slippage calculation
            fill_sim.set_context(ctx)

            # ── Process pending orders ────────────────────────────────────────
            for order in self._pending_orders:
                fill = self._simulate_fill(order, bar, fill_sim)
                if fill is not None:
                    self._apply_fill(fill, order.symbol)
                    self.strategy.on_fill(fill, self._create_portfolio())
            self._pending_orders.clear()

            # ── HARD EXIT CHECK ─────────────────────────────────────────────
            if is_exit_time(ts):
                self._flatten_all(bar, fill_sim)
                self._record_equity(ts)
                continue

            # ── Skip pre-market bars for signal generation ───────────────────
            if is_premarket_bar(ts) or not is_rth_bar(ts):
                # Still update MTM for drawdown tracking
                self._update_mtm(ts)
                self._record_equity(ts)
                continue

            # ── Generate signals ─────────────────────────────────────────────
            iv_rank = self._estimate_iv_rank(ts)

            signals = self._call_strategy_on_bar(bar, ctx)

            # ── Convert signals to orders ────────────────────────────────────
            for sig in signals:
                order = self._signal_to_order(sig, bar, ctx)
                if order is None:
                    continue

                # Pre-trade risk check
                if self.risk_limits is not None:
                    from quantcore.core.types import PortfolioSnapshot
                    snap = PortfolioSnapshot(
                        timestamp=ts,
                        positions=self._positions,
                        total_value=Decimal(str(self._total_value())),
                    )
                    result = check_pre_trade(order, snap, self.risk_limits, market_price=float(bar.close))
                    if result.status.value == "rejected":
                        self._rejected_orders.append((order, result))
                        continue

                self._pending_orders.append(order)

                # Log signal
                self._signal_log.append({
                    "timestamp": ts,
                    "symbol": order.symbol,
                    "direction": sig.direction,
                    "strength": sig.strength,
                    "session_open": ctx.session_open,
                    "overnight_gap_pct": ctx.overnight_gap_pct,
                    "pre_market_volume": ctx.pre_market_volume,
                    "vol_regime": ctx.vol_regime,
                })

            # ── Update MTM ───────────────────────────────────────────────────
            self._update_mtm(ts)

            # ── End-of-session equity record ─────────────────────────────────
            self._record_equity(ts)

        # End-of-session: close all positions at last close
        if len(data) > 0:
            last_row = data.iloc[-1]
            ts_raw = last_row["timestamp"]
            ts: datetime
            if isinstance(ts_raw, pd.Timestamp):
                ts = ts_raw.to_pydatetime()
            else:
                ts = pd.Timestamp(ts_raw).to_pydatetime()
            last_bar = OHLCV(
                symbol=str(last_row.get("symbol", "SPY")),
                timestamp=ts,
                open=Decimal(str(last_row["open"])),
                high=Decimal(str(last_row["high"])),
                low=Decimal(str(last_row["low"])),
                close=Decimal(str(last_row["close"])),
                volume=int(last_row["volume"]),
            )
            self._flatten_all(last_bar, fill_sim)
            self._record_equity(ts)
            self._dd_calc.finalize_day(ts, self._total_value())

        return pd.DataFrame({
            "timestamp": pd.to_datetime(self._timestamps),
            "equity": self._equity_curve,
        })

    def _call_strategy_on_bar(self, bar: OHLCV, ctx: TickContext) -> list:
        """Call the strategy's on_bar_intraday if available, else on_bar."""
        sig_method = getattr(self.strategy, 'on_bar_intraday', None)
        if sig_method is not None:
            return sig_method(bar, ctx)
        # Fallback to standard on_bar
        data = {
            "close": np.array([float(bar.close)]),
            "open": np.array([float(bar.open)]),
            "high": np.array([float(bar.high)]),
            "low": np.array([float(bar.low)]),
            "volume": np.array([float(bar.volume)]),
            "symbol": bar.symbol,
            "ctx": ctx,
        }
        return self.strategy.on_data(data)

    def _estimate_iv_rank(self, ts: datetime) -> float:
        """Estimate current IV rank for slippage calculation.

        This is a rough estimate using the VIX level.
        VIX > 30 → IV rank ~ 85
        VIX ~ 20 → IV rank ~ 50
        VIX < 15 → IV rank ~ 25
        """
        vix = self.vix_map.get(ts, np.nan)
        if not np.isfinite(vix):
            return 50.0
        # Map VIX to percentile
        iv_rank = min(100.0, max(0.0, (vix - 10.0) / 30.0 * 100.0))
        return float(iv_rank)

    def _simulate_fill(
        self,
        order: Order,
        bar: OHLCV,
        fill_sim: IntradayFillSimulator,
    ) -> Fill | None:
        is_option = order.metadata.get("instrument") == "OPTION" or "/" in order.symbol
        iv_rank = self._estimate_iv_rank(bar.timestamp)

        if is_option:
            meta = order.metadata
            return fill_sim.simulate_option_fill(
                order=order,
                bar=bar,
                underlying_price=float(bar.close),
                strike=float(meta.get("strike", float(bar.close))),
                iv=float(meta.get("iv", 0.25)),
                dte=float(meta.get("dte", 30)),
                is_call=meta.get("option_type", "PUT").upper() == "CALL",
                contracts=int(meta.get("contracts", 0)),
                iv_rank=iv_rank,
            )
        else:
            return fill_sim.simulate_fill(order, bar, iv_rank=iv_rank)

    def _apply_fill(self, fill: Fill, symbol: str) -> None:
        """Apply a fill to the portfolio (simplified from BacktestEngine)."""
        from quantcore.core import economic_kernel
        qty = float(fill.quantity)
        price = float(fill.price)
        is_buy = fill.side == Side.BUY

        fa = economic_kernel.apply_fill(
            current_kernel=None,
            fill_price=price,
            fill_quantity=qty,
            is_buy=is_buy,
            symbol=symbol,
        )

        self._cash += fa.cash_delta - float(fill.commission)

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
        self._trades.append(fill)

    def _flatten_all(self, bar: OHLCV, fill_sim: IntradayFillSimulator) -> None:
        for sym in list(self._positions.keys()):
            pos = self._positions[sym]
            qty = int(abs(float(pos.quantity)))
            if qty == 0:
                continue
            side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
            oid = f"ord_flatten_{sym}_{side.value}"
            order = Order(
                id=oid, symbol=sym, side=side, type=OrderType.MARKET,
                quantity=Decimal(str(qty)),
                metadata=self._option_meta.get(sym, {}),
            )
            fill = self._simulate_fill(order, bar, fill_sim)
            if fill is not None:
                self._apply_fill(fill, sym)
                self.strategy.on_fill(fill, self._create_portfolio())

    def _total_value(self) -> float:
        pv = sum(
            float(p.quantity) * self._prices.get(p.symbol, float(p.market_price))
            for p in self._positions.values()
            if float(p.quantity) != 0
        )
        return self._cash + pv

    def _update_mtm(self, ts: datetime) -> None:
        nav = self._total_value()
        # Gamma estimate: rough from net delta * 0.01 per % move (placeholder)
        gamma_estimate = self._estimate_portfolio_gamma()
        self._dd_calc.update_mtm(ts, nav, gamma=gamma_estimate)

    def _estimate_portfolio_gamma(self) -> float:
        """Estimate portfolio gamma for stressed scenario.

        This is a CRUDE estimate: assumes 1% SPY move → 0.5% portfolio loss.
        Real implementation requires option Greeks from BSM.
        """
        return 0.05  # 5% of NAV per 1% SPY move (placeholder)

    def _record_equity(self, ts: datetime) -> None:
        nav = self._total_value()
        self._equity_curve.append(nav)
        self._timestamps.append(ts)

    def _create_portfolio(self):
        """Create a mock portfolio for strategy callbacks."""
        class MockPortfolio:
            def __init__(p):
                p.total_value = self._total_value()
                p.positions = dict(self._positions)
                p.cash = self._cash
        return MockPortfolio()

    def _signal_to_order(self, sig, bar: OHLCV, ctx: TickContext) -> Order | None:
        """Convert a TradeSignal to an Order."""
        from quantcore.core.types import TradeSignal

        if isinstance(sig, TradeSignal):
            direction = sig.direction
            symbol = sig.symbol
            metadata = dict(sig.metadata)
            notional_size = sig.size
        else:
            direction = sig.direction
            symbol = sig.symbol
            metadata = {}
            notional_size = 0.0

        price = float(bar.close)

        if direction == 0:  # FLAT — close position
            pos = self._positions.get(symbol)
            if pos and abs(float(pos.quantity)) > 1e-8:
                close_side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
                close_qty = int(abs(float(pos.quantity)))
                return Order(
                    id=f"ord_flat_{symbol}_{close_side.value}",
                    symbol=symbol, side=close_side, type=OrderType.MARKET,
                    quantity=Decimal(str(close_qty)),
                    metadata=metadata,
                )
            return None

        side = Side.BUY if direction == 1 else Side.SELL

        # Sizing
        if notional_size > 0:
            qty = max(1, int(notional_size / price))
        elif self.sizing_fn is not None:
            qty = self.sizing_fn(self._cash, price, {"signal": sig, "bar": bar, "ctx": ctx})
            qty = max(1, int(qty))
        else:
            qty = max(1, int(self._total_value() * self.sizing_pct / price))

        return Order(
            id=f"ord_{bar.timestamp.isoformat()}_{symbol}_{side.value}",
            symbol=symbol, side=side, type=OrderType.MARKET,
            quantity=Decimal(str(qty)),
            metadata=metadata,
        )

    def summary(self):
        """Generate performance report using the analytics module."""
        from quantcore.backtest.analytics import compute_returns, generate_performance_report
        equity = np.array(self._equity_curve)
        returns = compute_returns(equity) if len(equity) > 1 else np.array([])
        return generate_performance_report(equity, returns)

    @property
    def intraday_dd(self) -> IntradayDrawdownCalculator:
        """Access the intraday drawdown calculator for detailed reporting."""
        return self._dd_calc
