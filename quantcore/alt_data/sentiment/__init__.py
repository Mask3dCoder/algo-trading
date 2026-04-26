"""Sentiment analysis: news, FinBERT, SEC filings, social media, composite signals."""

from quantcore.alt_data.sentiment.news import NewsSourceConfig, NewsArticle, NewsIngestor
from quantcore.alt_data.sentiment.finbert import (
    FinBERTSentiment, FinBERTConfig, FinBERTAnalyzer, scored_articles,
)
from quantcore.alt_data.sentiment.filings import (
    FilingType, FilingDocument, LMDictionary, FilingIngestor,
    compute_filing_sentiment_trend,
)
from quantcore.alt_data.sentiment.social import (
    SocialSource, SocialPost, SocialConfig, SocialIngestor, bot_filter,
)
from quantcore.alt_data.sentiment.signals import (
    SentimentSignal, SentimentConfig, SentimentSignalGenerator,
    sentiment_momentum, sentiment_divergence, aggregate_cross_sectional,
)

__all__ = [
    "NewsSourceConfig", "NewsArticle", "NewsIngestor",
    "FinBERTSentiment", "FinBERTConfig", "FinBERTAnalyzer", "scored_articles",
    "FilingType", "FilingDocument", "LMDictionary", "FilingIngestor",
    "compute_filing_sentiment_trend",
    "SocialSource", "SocialPost", "SocialConfig", "SocialIngestor", "bot_filter",
    "SentimentSignal", "SentimentConfig", "SentimentSignalGenerator",
    "sentiment_momentum", "sentiment_divergence", "aggregate_cross_sectional",
]
