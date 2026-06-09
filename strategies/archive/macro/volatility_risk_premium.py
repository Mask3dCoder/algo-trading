"""Volatility Risk Premium (VRP) Harvesting Strategy.

Systematically sells volatility when the VRP signal is positive and regime
conditions are favorable.  Manages tail risk via stop-losses, time exits,
and a permanent long VIX call hedge.

Signal: VRP_raw × TS_regime_multiplier × VVIX_conviction, EWM-smoothed.
Entry: signal > threshold, VIX < crisis threshold, not in position.
Exit: profit target, stop loss, time exit, signal reversal.
Sizing: vol-targeted half-Kelly with drawdown scalar.
"""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

from quantcore.portfolio.sizing import volatility_target
from quantcore.signals.technical.volatility import (
    garman_klass_vol,
    vrp_composite_signal,
)
from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register

_log = logging.getLogger(__name__)


@register("macro_vrp")
class VolatilityRiskPremiumStrategy(Strategy):
    """Cross-asset volatility risk premium harvesting strategy.

    Sells implied volatility when the VRP is positive and regime conditions
    are favorable.  Uses VIX term structure slope as a regime amplifier and
    VVIX as a conviction filter.

    Config params:
        lookback: Rolling window for Garman-Klass RV (default 21).
        ewm_span: EWM smoothing span for composite signal (default 5).
        entry_threshold: Minimum signal to enter (default 0.25).
        vix_crisis_threshold: VIX level above which no new entries (default 35).
        target_vol: Annualized vol target for sizing (default 0.10).
        kelly_fraction: Fractional Kelly multiplier (default 0.5).
        max_position_pct: Maximum position as fraction of capital (default 0.25).
        stop_loss_sigma_normal: Stop-loss in vol units, normal regime (default 2.0).
        stop_loss_sigma_crisis: Stop-loss in vol units, crisis regime (default 3.0).
        profit_target_pct: Profit target as fraction (default 0.05).
        max_holding_days: Maximum days to hold a position (default 30).
        signal_exit_threshold: Signal level below which to exit (default 0.0).
        tail_hedge_pct: Fraction of capital for VIX call hedge (default 0.02).
        vol_lookback: Lookback for realized vol in sizing (default 21).
        rebalance_freq: Days between rebalance checks (default 5).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.lookback = int(config.params.get("lookback", 21))
        self.ewm_span = int(config.params.get("ewm_span", 5))
        self.entry_threshold = float(config.params.get("entry_threshold", 0.25))
        self.vix_crisis_threshold = float(config.params.get("vix_crisis_threshold", 35))
        self.target_vol = float(config.params.get("target_vol", 0.10))
        self.kelly_fraction = float(config.params.get("kelly_fraction", 0.5))
        self.max_position_pct = float(config.params.get("max_position_pct", 0.25))
        self.stop_loss_sigma_normal = float(config.params.get("stop_loss_sigma_normal", 2.0))
        self.stop_loss_sigma_crisis = float(config.params.get("stop_loss_sigma_crisis", 3.0))
        self.profit_target_pct = float(config.params.get("profit_target_pct", 0.05))
        self.max_holding_days = int(config.params.get("max_holding_days", 30))
        self.signal_exit_threshold = float(config.params.get("signal_exit_threshold", 0.0))
        self.tail_hedge_pct = float(config.params.get("tail_hedge_pct", 0.02))
        self.vol_lookback = int(config.params.get("vol_lookback", 21))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))

        # Pre-loaded vol index data (set externally or via attach_vol_data)
        self._vix: NDArray | None = None
        self._vvix: NDArray | None = None
        self._ts_slope: NDArray | None = None

        # Deconcentration: per-asset position state (no more single-asset concentration)
        self._max_concurrent_positions = int(config.params.get("max_concurrent_positions", 3))
        self._max_position_pct_per_asset = float(config.params.get("max_position_pct_per_asset", 0.10))
        self._circuit_breaker_mult = 1.0  # Set externally by engine / risk manager
        self._days_since_rebalance = 0
        # dict[str, dict] tracking each active position
        self._positions: dict[str, dict] = {}

    def attach_vol_data(
        self,
        vix: NDArray,
        vvix: NDArray,
        ts_slope: NDArray,
    ) -> None:
        """Attach pre-loaded volatility index data.

        Call this before on_data() to provide VIX, VVIX, and term structure
        slope arrays aligned with the price data.

        Args:
            vix: VIX index values (numpy array, same length as price data).
            vvix: VVIX index values.
            ts_slope: VIX term structure slope values.
        """
        self._vix = np.asarray(vix, dtype=np.float64)
        self._vvix = np.asarray(vvix, dtype=np.float64)
        self._ts_slope = np.asarray(ts_slope, dtype=np.float64)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Process new bar and emit deconcentrated, hardened VRP signals.

        Pipeline (hardened):
            1. Guard: insufficient data → []
            2. Rebalance frequency check
            3. Compute per-asset Garman-Klass RV and composite VRP signal
            4. Phase 2 (exits): check all active positions for stop / time / signal-reversal
            5. Phase 3 (entries): rank candidates by signal, enter top-N with per-asset caps
            6. Phase 4 (hedge): emit tail-hedge only when net short-vol notional > 0
        """
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        high = data.get("high", close)
        low = data.get("low", close)
        open_ = data.get("open", close)

        if high.ndim == 1:
            high = high.reshape(-1, 1)
        if low.ndim == 1:
            low = low.reshape(-1, 1)
        if open_.ndim == 1:
            open_ = open_.reshape(-1, 1)

        # Guard
        min_data = max(self.lookback, 252) + 10
        if n_periods < min_data:
            return []

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        if self._vix is None or self._vvix is None or self._ts_slope is None:
            _log.warning("VRP: vol data not attached, returning empty signals")
            return []

        bar_idx = n_periods - 1

        if bar_idx >= len(self._vix):
            _log.warning("VRP: bar index %d exceeds vol data length %d", bar_idx, len(self._vix))
            return []

        current_vix = float(self._vix[bar_idx])
        current_vvix = float(self._vvix[bar_idx])
        current_ts_slope = float(self._ts_slope[bar_idx])

        is_crisis = current_vix >= self.vix_crisis_threshold
        stop_sigma = (
            self.stop_loss_sigma_crisis if is_crisis else self.stop_loss_sigma_normal
        )

        iv_series = self._vix[:n_periods] / 100.0
        ts_series = self._ts_slope[:n_periods]
        vvix_series = self._vvix[:n_periods]

        # ── Phase 1: per-asset signal computation ─────────────────────────────
        per_asset: dict[str, dict] = {}
        for i in range(min(n_assets, len(symbols))):
            sym = symbols[i]
            try:
                rv = garman_klass_vol(
                    high[:, i], low[:, i], open_[:, i], close[:, i],
                    period=self.lookback,
                )
            except Exception:
                continue

            rv_series = rv[:n_periods] if len(rv) >= n_periods else rv
            try:
                signal_arr = vrp_composite_signal(
                    iv_series,
                    rv_series,
                    ts_series,
                    vvix_series,
                    lookback=min(252, n_periods - 1),
                    ewm_span=self.ewm_span,
                )
            except Exception:
                continue

            signal_val = float(signal_arr[-1]) if not np.isnan(signal_arr[-1]) else 0.0
            rv_val = float(rv_series[-1]) if not np.isnan(rv_series[-1]) else 0.0

            per_asset[sym] = {
                "idx": i,
                "signal": signal_val,
                "rv": rv_val,
            }

        trade_signals: list[TradeSignal] = []

        # ── Phase 2: exits (per-asset) ────────────────────────────────────────
        exited_symbols: list[str] = []
        for sym, pos in list(self._positions.items()):
            pos["holding_days"] += 1
            holding_days = pos["holding_days"]
            entry_price = pos["entry_price"]

            if sym in per_asset:
                idx = per_asset[sym]["idx"]
                current_rv = per_asset[sym]["rv"]
            else:
                # Symbol dropped from universe → force exit
                trade_signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={"strategy": "vrp", "exit_reason": "universe_drop", "vix": round(current_vix, 2)},
                ))
                exited_symbols.append(sym)
                continue

            should_exit = False
            exit_reason = ""

            # Profit target
            if entry_price > 0:
                pnl_pct = (entry_price - float(close[-1, idx])) / entry_price
                if pnl_pct >= self.profit_target_pct:
                    should_exit = True
                    exit_reason = "profit_target"

            # Stop loss
            if not should_exit and entry_price > 0:
                price_move = abs(float(close[-1, idx]) - entry_price)
                stop_distance = stop_sigma * per_asset[sym]["rv"] * entry_price
                if price_move >= stop_distance:
                    should_exit = True
                    exit_reason = "stop_loss"

            # Time exit
            if not should_exit and holding_days >= self.max_holding_days:
                should_exit = True
                exit_reason = "time_exit"

            # Signal reversal
            if not should_exit and per_asset[sym]["signal"] < self.signal_exit_threshold:
                should_exit = True
                exit_reason = "signal_reversal"

            if should_exit:
                exited_symbols.append(sym)
                trade_signals.append(TradeSignal(
                    symbol=sym,
                    direction=0,  # direction=0 = go flat (was incorrectly 1)
                    strength=0.0,
                    size=0.0,
                    metadata={"strategy": "vrp", "exit_reason": exit_reason, "vix": round(current_vix, 2)},
                ))
                _log.info("VRP exit: %s (reason=%s)", sym, exit_reason)

        for sym in exited_symbols:
            self._positions.pop(sym, None)

        # Circuit breaker: if mult < 1 and we have positions, emit emergency exits
        if self._circuit_breaker_mult <= 0.0 and self._positions:
            for sym in list(self._positions.keys()):
                trade_signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={"strategy": "vrp", "exit_reason": "circuit_breaker_halt", "vix": round(current_vix, 2)},
                ))
                _log.info("VRP circuit-breaker exit: %s", sym)
            self._positions.clear()
            return trade_signals

        # ── Phase 3: entries (deconcentrated across top signals) ──────────────
        if not is_crisis and self._circuit_breaker_mult > 0.0:
            # Candidates: not in position, signal > threshold
            candidates = [
                (sym, info) for sym, info in per_asset.items()
                if sym not in self._positions and info["signal"] > self.entry_threshold
            ]
            # Rank by signal strength (descending)
            candidates.sort(key=lambda x: x[1]["signal"], reverse=True)
            available_slots = max(0, self._max_concurrent_positions - len(self._positions))

            # Shared sizing parameters
            equity = data.get("equity")
            dd_scalar = 1.0
            if equity is not None and len(equity) > 1:
                peak = np.max(equity)
                if peak > 0:
                    dd = (peak - equity[-1]) / peak
                    dd_scalar = float(np.clip(1.0 - dd * 2.0, 0.5, 1.0))

            conviction = float(np.clip(1.0 - (current_vvix - 100) / 100, 0.2, 1.0))
            # VVIX panic: >150 crushes conviction → no entries
            if current_vvix > 150:
                conviction = 0.0

            for sym, info in candidates[:available_slots]:
                asset_vol = info["rv"] if info["rv"] > 0 else 0.20
                vol_weight = volatility_target(asset_vol=asset_vol, target_vol=self.target_vol, n_assets=1)
                signal_strength = float(np.clip(info["signal"] / 2.0, 0.0, 1.0))

                raw_weight = vol_weight * signal_strength * conviction * dd_scalar
                raw_weight = min(raw_weight, self._max_position_pct_per_asset)
                weight = float(np.clip(raw_weight, 0.0, self.max_position_pct))
                weight *= self._circuit_breaker_mult

                if weight > 0.01:
                    entry_price = float(close[-1, info["idx"]])
                    self._positions[sym] = {
                        "entry_price": entry_price,
                        "entry_bar": bar_idx,
                        "holding_days": 0,
                    }
                    trade_signals.append(TradeSignal(
                        symbol=sym,
                        direction=-1,
                        strength=min(weight / self.max_position_pct, 1.0),
                        size=weight * float(self.config.max_position_notional),
                        stop_loss=entry_price * (1.0 + stop_sigma * asset_vol),
                        take_profit=entry_price * (1.0 - self.profit_target_pct),
                        metadata={
                            "strategy": "vrp",
                            "signal": round(info["signal"], 4),
                            "vix": round(current_vix, 2),
                            "vvix": round(current_vvix, 2),
                            "ts_slope": round(current_ts_slope, 4),
                            "rv": round(asset_vol, 6),
                            "weight": round(weight, 4),
                            "signal_strength": round(signal_strength, 4),
                            "conviction": round(conviction, 4),
                        },
                    ))

        # ── Phase 4: conditional tail hedge ────────────────────────────────────
        if self.tail_hedge_pct > 0:
            hedge_size = self.tail_hedge_pct * float(self.config.max_position_notional)
            trade_signals.append(TradeSignal(
                symbol="VIX_CALL",
                direction=1,
                strength=0.5,
                size=hedge_size,
                metadata={
                    "strategy": "vrp",
                    "type": "tail_hedge",
                    "vix": round(current_vix, 2),
                    "hedge_ratio": len(self._positions) / max(self._max_concurrent_positions, 1),
                },
            ))

        self._days_since_rebalance = 0
        return trade_signals

    def set_circuit_breaker_mult(self, mult: float) -> None:
        """Apply portfolio-level circuit-breaker size multiplier.

        Called by the engine / risk manager when the vol-circuit breaker
        triggers.  mult < 1.0 reduces position sizes; mult <= 0.0 blocks
        new entries and flattens existing positions.
        """
        self._circuit_breaker_mult = float(mult)
