"""Sentiment momentum strategy.

Ranks assets by trailing 5-day average news sentiment score.
Long top tercile, short bottom tercile. Equal weight, dollar-neutral.

Requires pre-loaded sentiment data passed via config params or set_sentiment_data().

Usage:
    from quantcore.alpha.sentiment_momentum import SentimentMomentumStrategy
    strat = SentimentMomentumStrategy(config)
    strat.set_sentiment_data(sentiment_df)  # date, ticker, sentiment_score
    strat.initialize()
    signals = strat.on_data(bar_data)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from pandas import Timestamp

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("sentiment_momentum")
class SentimentMomentumStrategy(Strategy):
    """Cross-sectional sentiment momentum.

    Signal logic:
    1. Compute trailing N-day average sentiment per ticker.
    2. Rank tickers by average sentiment.
    3. Go long top tercile, short bottom tercile.
    4. Equal weight within each tercile, dollar-neutral overall.
    5. Rebalance every `rebalance_freq` bars.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        p = config.params
        self._lookback: int = int(p.get("sentiment_lookback", 5))
        self._rebalance_freq: int = int(p.get("rebalance_freq", 1))
        self._tercile_pct: float = float(p.get("tercile_pct", 0.33))
        self._max_strength: float = float(p.get("max_strength", 1.0))

        # Sentiment data: date -> {ticker: score}
        self._sentiment_by_date: dict[str, dict[str, float]] = {}
        self._sentiment_dates: list[str] = []

        # State
        self._bar_count: int = 0
        self._current_ts: Timestamp | None = None

    def set_sentiment_data(self, df) -> None:
        """Load pre-computed daily sentiment scores.

        Args:
            df: DataFrame with columns: date, ticker, sentiment_score.
                date should be convertible to Timestamp.
        """
        self._sentiment_by_date = defaultdict(dict)
        for _, row in df.iterrows():
            date_key = str(Timestamp(row["date"]).date())
            self._sentiment_by_date[date_key][row["ticker"]] = float(row["sentiment_score"])
        self._sentiment_dates = sorted(self._sentiment_by_date.keys())

    def set_timestamp(self, ts) -> None:
        """Set the current bar timestamp (called by backtest runner)."""
        self._current_ts = Timestamp(ts) if ts is not None else None

    def _get_trailing_sentiment(self, symbols: list[str]) -> dict[str, float]:
        """Compute trailing N-day average sentiment for each symbol.

        Looks back `self._lookback` calendar days from the current date.
        """
        if self._current_ts is None:
            return {}

        current_date = self._current_ts.date()
        avg_scores: dict[str, float] = {}

        for sym in symbols:
            scores: list[float] = []
            for days_back in range(self._lookback):
                d = current_date - __import__("datetime").timedelta(days=days_back)
                d_key = str(d)
                if d_key in self._sentiment_by_date and sym in self._sentiment_by_date[d_key]:
                    scores.append(self._sentiment_by_date[d_key][sym])

            if scores:
                avg_scores[sym] = float(np.mean(scores))
            else:
                avg_scores[sym] = 0.0

        return avg_scores

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Process bar data and emit sentiment-ranked signals.

        Args:
            data: dict with 'close', 'symbols', 'timestamp' (and other OHLCV fields).

        Returns:
            List of TradeSignal for long/short positions.
        """
        self._bar_count += 1
        symbols: list[str] = list(data.get("symbols", []))
        if not symbols:
            return []

        # Update timestamp from data dict if available
        if "timestamp" in data:
            self.set_timestamp(data["timestamp"])

        # Only rebalance on schedule
        if self._bar_count % self._rebalance_freq != 0:
            return []

        # Get trailing sentiment scores
        avg_scores = self._get_trailing_sentiment(symbols)

        if not avg_scores or all(v == 0.0 for v in avg_scores.values()):
            return []

        # Rank by sentiment
        ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        n_long = max(1, int(n * self._tercile_pct))
        n_short = max(1, int(n * self._tercile_pct))

        longs = ranked[:n_long]
        shorts = ranked[-n_short:]

        # Avoid overlap
        long_syms = {s for s, _ in longs}
        shorts = [(s, sc) for s, sc in shorts if s not in long_syms]

        signals: list[TradeSignal] = []

        # Long signals
        if longs:
            weight = 1.0 / len(longs)
            for sym, score in longs:
                # Strength proportional to sentiment score (normalized)
                strength = min(abs(score), self._max_strength)
                signals.append(TradeSignal(
                    symbol=sym,
                    direction=1,
                    strength=weight * strength,
                    size=weight * self.config.max_position_notional,
                    metadata={"source": "sentiment", "sentiment_score": score},
                ))

        # Short signals
        if shorts:
            weight = 1.0 / len(shorts)
            for sym, score in shorts:
                strength = min(abs(score), self._max_strength)
                signals.append(TradeSignal(
                    symbol=sym,
                    direction=-1,
                    strength=weight * strength,
                    size=weight * self.config.max_position_notional,
                    metadata={"source": "sentiment", "sentiment_score": score},
                ))

        return signals
