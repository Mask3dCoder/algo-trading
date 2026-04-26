"""Sentiment-based trading signals: composite scoring from news, social, and filings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SentimentSignal:
    """Composite sentiment signal for a ticker."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    ticker: str = ""
    composite_score: float = 0.0
    news_score: float = 0.0
    social_score: float = 0.0
    filing_score: float = 0.0
    confidence: float = 0.0
    components: list[str] = field(default_factory=list)


@dataclass
class SentimentConfig:
    """Weights and thresholds for sentiment signal generation."""
    news_weight: float = 0.35
    social_weight: float = 0.35
    filing_weight: float = 0.30
    lookback_days: int = 5
    min_confidence: float = 0.3
    min_articles_for_signal: int = 3


class SentimentSignalGenerator:
    """Generate composite sentiment signals from multiple sources."""

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or SentimentConfig()

    def compute_news_score(self, articles: list[Any], ticker: str) -> float:
        """Average sentiment of recent articles for a ticker."""
        relevant = [a for a in articles if ticker in getattr(a, "tickers", [])]
        if len(relevant) < self.config.min_articles_for_signal:
            return 0.0
        scores = [getattr(a, "sentiment_score", 0.0) for a in relevant]
        return float(np.mean(scores)) if scores else 0.0

    def compute_social_score(self, posts: list[Any], ticker: str) -> float:
        """Volume-adjusted social sentiment score."""
        relevant = [p for p in posts if getattr(p, "ticker", "") == ticker]
        if not relevant:
            return 0.0
        scores = [getattr(p, "score", 0) for p in relevant]
        total_engagement = sum(max(s, 1) for s in scores)
        if total_engagement == 0:
            return 0.0
        weighted = sum(
            max(getattr(p, "score", 0), 1) * max(len(getattr(p, "body", "") or getattr(p, "title", "")), 1)
            for p in relevant
        )
        norm = float(weighted) / total_engagement if total_engagement > 0 else 0.0
        return min(max(norm / 100.0, -1.0), 1.0)

    def compute_filing_score(self, filing_sentiment: dict[str, float] | None) -> float:
        """Convert LM filing sentiment to a signal score."""
        if not filing_sentiment:
            return 0.0
        return float(np.mean(list(filing_sentiment.values())))

    def generate(
        self,
        ticker: str,
        articles: list[Any],
        posts: list[Any],
        filing_sentiment: dict[str, float] | None = None,
    ) -> SentimentSignal:
        """Generate a composite sentiment signal for a ticker."""
        news_score = self.compute_news_score(articles, ticker)
        social_score = self.compute_social_score(posts, ticker)
        filing_score = self.compute_filing_score(filing_sentiment)

        composite = (
            news_score * self.config.news_weight
            + social_score * self.config.social_weight
            + filing_score * self.config.filing_weight
        )

        components = []
        active_count = 0
        if abs(news_score) > 0.01:
            components.append("news")
            active_count += 1
        if abs(social_score) > 0.01:
            components.append("social")
            active_count += 1
        if filing_sentiment:
            components.append("filing")
            active_count += 1

        confidence = min(active_count / 3.0, 1.0) * (1.0 - 0.5 * abs(composite))

        return SentimentSignal(
            ticker=ticker, composite_score=float(composite),
            news_score=float(news_score), social_score=float(social_score),
            filing_score=float(filing_score), confidence=float(confidence),
            components=components,
        )

    def is_signal_strong(self, signal: SentimentSignal) -> bool:
        """Check if the signal confidence exceeds the threshold."""
        return signal.confidence >= self.config.min_confidence


def sentiment_momentum(
    signals: list[SentimentSignal], ticker: str, window: int = 5,
) -> float:
    """Rate of change of sentiment score over the last N signals."""
    ticker_signals = [s for s in signals if s.ticker == ticker]
    if len(ticker_signals) < 2:
        return 0.0
    recent = ticker_signals[-window:]
    scores = np.array([s.composite_score for s in recent])
    if len(scores) < 2:
        return 0.0
    return float(scores[-1] - scores[0]) / max(len(scores) - 1, 1)


def sentiment_divergence(
    price_returns: NDArray[np.floating], sentiment_scores: NDArray[np.floating],
) -> float:
    """Correlation between price returns and sentiment scores.

    Negative values indicate divergence (price up, sentiment down or vice versa).
    """
    if len(price_returns) < 5 or len(sentiment_scores) < 5:
        return 0.0
    min_len = min(len(price_returns), len(sentiment_scores))
    r = price_returns[-min_len:]
    s = sentiment_scores[-min_len:]
    if np.std(r) == 0 or np.std(s) == 0:
        return 0.0
    return float(np.corrcoef(r, s)[0, 1])


def aggregate_cross_sectional(signals: list[SentimentSignal]) -> dict[str, float]:
    """Rank tickers by composite sentiment score."""
    ranked: dict[str, float] = {}
    for s in sorted(signals, key=lambda x: x.composite_score, reverse=True):
        ranked[s.ticker] = s.composite_score
    return ranked
