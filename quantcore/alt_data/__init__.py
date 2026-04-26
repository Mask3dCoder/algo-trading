"""Alternative data: sentiment analysis and options flow intelligence."""

from quantcore.alt_data.sentiment import (
    NewsSourceConfig, NewsArticle, NewsIngestor,
    FinBERTSentiment, FinBERTConfig, FinBERTAnalyzer, scored_articles,
    FilingType, FilingDocument, LMDictionary, FilingIngestor, compute_filing_sentiment_trend,
    SocialSource, SocialPost, SocialConfig, SocialIngestor, bot_filter,
    SentimentSignal, SentimentConfig, SentimentSignalGenerator,
    sentiment_momentum, sentiment_divergence, aggregate_cross_sectional,
)
from quantcore.alt_data.options_flow import (
    UOAType, UOAConfig, UOATrade, UOADetector,
    detect_sweep_cascade, compute_premium_sentiment,
    GEXConfig, GEXResult, GEXCalculator, gex_profile, detect_gamma_squeeze,
    MaxPainConfig, MaxPainResult, MaxPainCalculator,
    put_call_ratio, put_call_oi_ratio, pcr_signal,
    PCRConfig, PCRSnapshot, PutCallAnalyzer,
    compute_equity_only_pcr, pcr_term_structure,
)

__all__ = [
    "NewsSourceConfig", "NewsArticle", "NewsIngestor",
    "FinBERTSentiment", "FinBERTConfig", "FinBERTAnalyzer", "scored_articles",
    "FilingType", "FilingDocument", "LMDictionary", "FilingIngestor",
    "compute_filing_sentiment_trend",
    "SocialSource", "SocialPost", "SocialConfig", "SocialIngestor", "bot_filter",
    "SentimentSignal", "SentimentConfig", "SentimentSignalGenerator",
    "sentiment_momentum", "sentiment_divergence", "aggregate_cross_sectional",
    "UOAType", "UOAConfig", "UOATrade", "UOADetector",
    "detect_sweep_cascade", "compute_premium_sentiment",
    "GEXConfig", "GEXResult", "GEXCalculator", "gex_profile", "detect_gamma_squeeze",
    "MaxPainConfig", "MaxPainResult", "MaxPainCalculator",
    "put_call_ratio", "put_call_oi_ratio", "pcr_signal",
    "PCRConfig", "PCRSnapshot", "PutCallAnalyzer",
    "compute_equity_only_pcr", "pcr_term_structure",
]
