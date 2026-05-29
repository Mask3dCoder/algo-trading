"""Put Write VRP strategy — sells OTM puts on SPY to collect volatility risk premium.

Entry: Monthly, on the first trading day, if VIX > threshold.
Exit: Hold to expiration (30 DTE) or close at 50% of max profit.
Sizing: Risk 2% of equity per trade.
Universe: SPY only.

This is a pure VRP strategy — we sell insurance (puts) when implied vol
is elevated above realized vol. The premium collected is our edge.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register

_log = logging.getLogger(__name__)


@register("options_put_write")
class PutWriteVRPStrategy(Strategy):
    """Sell OTM puts on SPY monthly to collect VRP.

    Config params:
        vix_entry: Minimum VIX level to enter (default 15)
        otm_pct:   How far OTM the put is, as fraction of spot (default 0.95 = 5% OTM)
        dte:       Days to expiration (default 30)
        risk_per_trade: Fraction of equity to risk per trade (default 0.02 = 2%)
        max_positions:  Max concurrent positions (default 3)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.vix_entry = float(config.params.get("vix_entry", 15.0))
        self.otm_pct = float(config.params.get("otm_pct", 0.95))
        self.dte = int(config.params.get("dte", 30))
        self.risk_per_trade = float(config.params.get("risk_per_trade", 0.02))
        self.max_positions = int(config.params.get("max_positions", 3))
        self._last_entry_month: int | None = None
        self._active_positions: list[dict] = []

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Generate put-write signals.

        Expects data dict with:
            - close: np.ndarray (n_periods, n_assets) — SPY is first asset
            - implied_vol: np.ndarray (n_periods, 1) — VIX-derived IV (optional)
            - symbols: list[str]
        """
        close = data.get("close")
        if close is None or close.size == 0:
            return []

        n_periods = close.shape[0]
        if n_periods < 2:
            return []

        symbols = data.get("symbols", [])
        if not symbols:
            return []

        # Find SPY index
        spy_idx = 0
        for i, sym in enumerate(symbols):
            if "SPY" in str(sym).upper():
                spy_idx = i
                break

        spot = float(close[-1, spy_idx])

        # Get VIX from implied_vol if available
        iv_array = data.get("implied_vol")
        if iv_array is not None and len(iv_array) > 0:
            vix_val = float(iv_array[-1]) * 100  # convert from decimal to percentage
        else:
            # Fallback: estimate VIX from realized vol
            if n_periods > 21:
                returns = np.diff(np.log(close[:, spy_idx]))
                returns = returns[~np.isnan(returns)]
                if len(returns) > 21:
                    rv = float(np.std(returns[-21:]) * np.sqrt(252) * 100)
                    vix_val = rv * 1.2  # assume IV is 20% above RV
                else:
                    vix_val = 20.0
            else:
                vix_val = 20.0

        signals: list[TradeSignal] = []

        # Monthly entry: check if we're in a new month
        # Use the last bar's date (approximate from data length)
        # In live trading, the runner provides the actual date
        current_bar = n_periods - 1
        should_enter = False

        # Enter on approximately monthly intervals (every 21 bars)
        if self._last_entry_month is None or (current_bar - (self._last_entry_month or 0)) >= 21:
            if vix_val > self.vix_entry and len(self._active_positions) < self.max_positions:
                should_enter = True

        if should_enter:
            self._last_entry_month = current_bar
            # Signal to sell a put (direction = -1 means we're selling premium)
            # Strength = confidence based on VIX level
            strength = min(1.0, (vix_val - self.vix_entry) / 10.0)
            strength = max(0.3, strength)

            signals.append(TradeSignal(
                symbol="SPY",
                direction=-1,  # short put = sell premium
                strength=strength,
                size=self.risk_per_trade,
                metadata={
                    "strategy": "put_write_vrp",
                    "vix": vix_val,
                    "spot": spot,
                    "strike": spot * self.otm_pct,
                    "dte": self.dte,
                    "action": "sell_put",
                },
            ))

        # Check existing positions for early exit (simplified)
        # In live trading, the production runner handles position management
        # Here we just generate the entry signal

        return signals
