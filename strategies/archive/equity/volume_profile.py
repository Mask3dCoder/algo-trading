"""Volume profile (smart money) strategy: institutional flow detection.

Combines multiple volume-based indicators to detect accumulation and
distribution by institutional players:

 - Unusual volume: z-score > 2 signals institutional activity
 - OBV divergence: price/obv disagreement signals reversal
 - Chaikin Money Flow: sustained positive/negative CMF = accumulation/distribution
 - MFI: volume-weighted RSI for overbought/oversold
 - VWAP deviation: institutional benchmark for execution quality

Uses signals from quantcore.signals.technical.volume for indicator computation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("equity_volume_profile")
class VolumeProfileStrategy(Strategy):
    """Smart money / institutional flow detection via volume signals.

    Config params:
        volume_z_threshold: unusual volume z-score threshold (default 2.0)
        cmf_period: Chaikin Money Flow period (default 21)
        cmf_threshold: CMF threshold for accumulation/distribution (default 0.1)
        mfi_period: Money Flow Index period (default 14)
        mfi_oversold: MFI oversold threshold (default 20)
        mfi_overbought: MFI overbought threshold (default 80)
        vwap_period: VWAP deviation period (default 20)
        obv_lookback: OBV divergence lookback (default 20)
        min_indicators: minimum indicators agreeing for signal (default 2)
        hold_days: max hold period (default 5)
        stop_loss_pct: stop loss percentage (default 3.0)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.volume_z_threshold = float(config.params.get("volume_z_threshold", 2.0))
        self.cmf_period = int(config.params.get("cmf_period", 21))
        self.cmf_threshold = float(config.params.get("cmf_threshold", 0.1))
        self.mfi_period = int(config.params.get("mfi_period", 14))
        self.mfi_oversold = float(config.params.get("mfi_oversold", 20))
        self.mfi_overbought = float(config.params.get("mfi_overbought", 80))
        self.vwap_period = int(config.params.get("vwap_period", 20))
        self.obv_lookback = int(config.params.get("obv_lookback", 20))
        self.min_indicators = int(config.params.get("min_indicators", 2))
        self.hold_days = int(config.params.get("hold_days", 5))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 3.0))
        self._entry_days: dict[str, int] = {}

    def _sma(self, x: NDArray, period: int) -> NDArray:
        n = len(x)
        result = np.full(n, np.nan, dtype=float)
        if n < period:
            return result
        cumsum = np.cumsum(x)
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
        return result

    def _compute_obv(self, close: NDArray, volume: NDArray) -> NDArray:
        n = len(close)
        obv = np.zeros(n, dtype=float)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        return obv

    def _compute_cmf(self, high: NDArray, low: NDArray, close: NDArray, volume: NDArray) -> float:
        n = len(close)
        if n < self.cmf_period:
            return 0.0
        clv_sum = 0.0
        vol_sum = 0.0
        for i in range(n - self.cmf_period, n):
            hl_range = high[i] - low[i]
            if hl_range > 0:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            else:
                clv = 0.0
            clv_sum += clv * volume[i]
            vol_sum += volume[i]
        return clv_sum / vol_sum if vol_sum > 0 else 0.0

    def _compute_mfi(self, high: NDArray, low: NDArray, close: NDArray, volume: NDArray) -> float:
        n = len(close)
        if n < self.mfi_period + 1:
            return 50.0
        tp = (high + low + close) / 3.0
        pos_mf = 0.0
        neg_mf = 0.0
        for i in range(n - self.mfi_period, n):
            if tp[i] > tp[i - 1]:
                pos_mf += tp[i] * volume[i]
            elif tp[i] < tp[i - 1]:
                neg_mf += tp[i] * volume[i]
            else:
                pos_mf += tp[i] * volume[i] * 0.5
                neg_mf += tp[i] * volume[i] * 0.5
        if neg_mf < 1e-10:
            return 100.0
        return 100.0 - 100.0 / (1.0 + pos_mf / neg_mf)

    def _obv_divergence_signal(self, close: NDArray, volume: NDArray) -> float:
        if len(close) < self.obv_lookback:
            return 0.0
        obv = self._compute_obv(close, volume)
        start = len(close) - self.obv_lookback

        price_up = close[-1] > close[start]
        obv_up = obv[-1] > obv[start]
        price_down = close[-1] < close[start]
        obv_down = obv[-1] < obv[start]

        if price_up and obv_down:
            return -1.0  # Bearish divergence
        elif price_down and obv_up:
            return 1.0   # Bullish divergence
        return 0.0

    def _unusual_volume_signal(self, volume: NDArray) -> float:
        if len(volume) < 20:
            return 0.0
        window = volume[-20:]
        avg = np.mean(window)
        std = np.std(window, ddof=1)
        if std < 1e-10:
            return 0.0
        return (volume[-1] - avg) / std

    def _vwap_deviation_signal(self, close: NDArray, volume: NDArray) -> float:
        if len(close) < self.vwap_period:
            return 0.0
        window = close[-self.vwap_period:]
        vol_window = volume[-self.vwap_period:]
        total_vol = np.sum(vol_window)
        if total_vol < 1e-10:
            return 0.0
        vwap = np.sum(window * vol_window) / total_vol
        # Z-score of deviation
        deviations = np.zeros(self.vwap_period)
        cum_v = np.cumsum(vol_window)
        cum_pv = np.cumsum(window * vol_window)
        for i in range(self.vwap_period):
            v_s = 0 if i == 0 else cum_v[i - 1]
            pv_s = 0 if i == 0 else cum_pv[i - 1]
            tv = cum_v[i] - v_s
            if tv > 0:
                deviations[i] = window[i] - (cum_pv[i] - pv_s) / tv
        std = np.std(deviations, ddof=1)
        if std > 1e-10:
            return -(close[-1] - vwap) / std  # Negative: below VWAP = buy
        return 0.0

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 2:
            close = close[:, 0]

        volume = data.get("volume")
        if volume is None:
            # Synthetic volume from price movement
            volume = np.abs(np.diff(close, prepend=close[0])) * 1_000_000 + 100_000
        if volume.ndim == 2:
            volume = volume[:, 0]

        high = data.get("high", close * 1.005)
        low = data.get("low", close * 0.995)
        if high.ndim == 2:
            high = high[:, 0]
        if low.ndim == 2:
            low = low[:, 0]

        n = len(close)
        min_period = max(self.cmf_period, self.mfi_period, self.obv_lookback, self.vwap_period) + 1
        if n < min_period:
            return []

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols

        # Compute all volume indicators
        vol_z = self._unusual_volume_signal(volume)
        cmf = self._compute_cmf(high, low, close, volume)
        mfi = self._compute_mfi(high, low, close, volume)
        obv_div = self._obv_divergence_signal(close, volume)
        vwap_dev = self._vwap_deviation_signal(close, volume)

        # Composite score: each indicator contributes direction
        composite = 0.0
        n_agreeing = 0

        # Unusual volume amplifies other signals
        volume_spike = abs(vol_z) > self.volume_z_threshold

        # CMF: positive = accumulation (buy), negative = distribution (sell)
        if abs(cmf) > self.cmf_threshold:
            composite += np.sign(cmf) * min(abs(cmf) * 3, 1.0)
            n_agreeing += 1

        # MFI: oversold = buy, overbought = sell
        if mfi < self.mfi_oversold:
            composite += 1.0
            n_agreeing += 1
        elif mfi > self.mfi_overbought:
            composite -= 1.0
            n_agreeing += 1

        # OBV divergence
        if abs(obv_div) > 0.5:
            composite += obv_div
            n_agreeing += 1

        # VWAP deviation
        if abs(vwap_dev) > 1.0:
            composite += np.sign(vwap_dev) * min(abs(vwap_dev), 1.0)
            n_agreeing += 1

        # Volume spike boost
        if volume_spike and n_agreeing >= self.min_indicators - 1:
            composite *= 1.5

        if n_agreeing < self.min_indicators:
            composite = 0.0

        composite = np.clip(composite / 3.0, -1.0, 1.0)

        # Exits
        if symbol in self._entry_days:
            self._entry_days[symbol] += 1
            if self._entry_days[symbol] >= self.hold_days or (abs(composite) < 0.2 and n_agreeing < 1):
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "volume_exit"},
                ))
                self._entry_days.pop(symbol, None)
                return signals

        # Entry
        if abs(composite) > 0.3:
            direction = 1 if composite > 0 else -1
            strength = abs(composite)
            size = strength * self.config.max_position_notional * self.config.risk_budget_pct
            stop = close[-1] * (1 - direction * self.stop_loss_pct / 100.0)

            signals.append(TradeSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                size=size,
                stop_loss=stop,
                metadata={
                    "cmf": float(cmf),
                    "mfi": float(mfi),
                    "vol_z": float(vol_z),
                    "obv_div": float(obv_div),
                    "vwap_dev": float(vwap_dev),
                    "n_agreeing": n_agreeing,
                    "volume_spike": volume_spike,
                },
            ))
            self._entry_days[symbol] = 0

        return signals
