"""Sentiment analysis: news, FinBERT, SEC filings, social media, composite signals."""

from quantcore.alt_data.sentiment.filings import (
    FilingDocument,
    FilingIngestor,
    FilingType,
    LMDictionary,
    compute_filing_sentiment_trend,
)
from quantcore.alt_data.sentiment.finbert import (
    FinBERTAnalyzer,
    FinBERTConfig,
    FinBERTSentiment,
    scored_articles,
)
from quantcore.alt_data.sentiment.news import NewsArticle, NewsIngestor, NewsSourceConfig
from quantcore.alt_data.sentiment.signals import (
    SentimentConfig,
    SentimentSignal,
    SentimentSignalGenerator,
    aggregate_cross_sectional,
    sentiment_divergence,
    sentiment_momentum,
)
from quantcore.alt_data.sentiment.social import (
    SocialConfig,
    SocialIngestor,
    SocialPost,
    SocialSource,
    bot_filter,
)

__all__ = [
    "FilingDocument",
    "FilingIngestor",
    "FilingType",
    "FinBERTAnalyzer",
    "FinBERTConfig",
    "FinBERTSentiment",
    "LMDictionary",
    "NewsArticle",
    "NewsIngestor",
    "NewsSourceConfig",
    "SentimentConfig",
    "SentimentSignal",
    "SentimentSignalGenerator",
    "SocialConfig",
    "SocialIngestor",
    "SocialPost",
    "SocialSource",
    "aggregate_cross_sectional",
    "bot_filter",
    "compute_filing_sentiment_trend",
    "scored_articles",
    "sentiment_divergence",
    "sentiment_momentum",
]
