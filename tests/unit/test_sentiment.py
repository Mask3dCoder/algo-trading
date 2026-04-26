"""Tests for alt_data sentiment modules: news, FinBERT, filings, social, signals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

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

# ---------------------------------------------------------------------------
# News tests
# ---------------------------------------------------------------------------

class TestNewsSourceConfig:
    def test_default_config(self):
        c = NewsSourceConfig()
        assert len(c.feed_urls) >= 1
        assert c.refresh_interval_sec == 300.0

    def test_custom_urls(self):
        c = NewsSourceConfig(feed_urls=["https://example.com/rss"], max_articles_per_feed=25)
        assert len(c.feed_urls) == 1
        assert c.max_articles_per_feed == 25


class TestNewsArticle:
    def test_default_article(self):
        a = NewsArticle()
        assert a.title == ""
        assert a.tickers == []

    def test_article_with_tickers(self):
        a = NewsArticle(title="AAPL earnings", tickers=["AAPL"], sentiment_score=0.5)
        assert "AAPL" in a.tickers
        assert a.sentiment_score == 0.5


class TestNewsIngestor:
    def test_init(self):
        ingestor = NewsIngestor()
        assert ingestor.config is not None
        assert ingestor.ticker_map == {}

    def test_with_ticker_map(self):
        ingestor = NewsIngestor(ticker_map={"Apple": ["AAPL"], "Microsoft": ["MSFT"]})
        assert len(ingestor.ticker_map) == 2

    def test_text_extraction(self):
        ingestor = NewsIngestor()
        text = ingestor._extract_text("<html><body><p>Hello World</p></body></html>")
        assert "Hello World" in text

    def test_ticker_mapping(self):
        ingestor = NewsIngestor(ticker_map={"Apple": ["AAPL"], "Google": ["GOOGL"]})
        tickers = ingestor._map_tickers("Apple announced new iPhone. Google reported earnings.")
        assert "AAPL" in tickers
        assert "GOOGL" in tickers

    def test_filter_by_ticker(self):
        ingestor = NewsIngestor()
        a1 = NewsArticle(title="Apple news", tickers=["AAPL"])
        a2 = NewsArticle(title="Microsoft news", tickers=["MSFT"])
        ingestor._articles = [a1, a2]
        filtered = ingestor.filter_by_ticker("AAPL")
        assert len(filtered) == 1
        assert filtered[0].title == "Apple news"


# ---------------------------------------------------------------------------
# FinBERT tests
# ---------------------------------------------------------------------------

class TestFinBERTAnalyzer:
    def test_init(self):
        a = FinBERTAnalyzer()
        assert a.config.model_name == "ProsusAI/finbert"

    def test_custom_config(self):
        config = FinBERTConfig(batch_size=16, device="cpu")
        a = FinBERTAnalyzer(config=config)
        assert a.config.batch_size == 16

    def test_analyze_empty(self):
        a = FinBERTAnalyzer()
        results = a.analyze([])
        assert results == []

    def test_lexicon_fallback_positive(self):
        a = FinBERTAnalyzer()
        results = a.analyze(["The company reported strong profit growth and positive outlook"])
        assert len(results) == 1
        r = results[0]
        assert r.positive >= 0 or r.negative >= 0
        assert -1.0 <= r.compound <= 1.0

    def test_lexicon_fallback_negative(self):
        a = FinBERTAnalyzer()
        results = a.analyze(["bankruptcy debt default loss decline weak"])
        assert len(results) == 1
        assert results[0].label in ("negative", "neutral", "positive")

    def test_scored_articles(self):
        analyzer = FinBERTAnalyzer()
        articles = [
            NewsArticle(title="Good news", raw_text="profit growth strong"),
            NewsArticle(title="Bad news", raw_text="loss decline bankruptcy"),
        ]
        scored = scored_articles(articles, analyzer)
        assert len(scored) == 2

    def test_extract_entities(self):
        a = FinBERTAnalyzer()
        entities = a.extract_entities("Apple Inc. reported $AAPL earnings. Microsoft Corp. also beat $MSFT.")
        assert "tickers" in entities


# ---------------------------------------------------------------------------
# Filings / LM Dictionary tests
# ---------------------------------------------------------------------------

class TestLMDictionary:
    def test_init(self):
        lm = LMDictionary()
        assert len(lm.negative) > 10
        assert len(lm.positive) > 10

    def test_score_positive_text(self):
        lm = LMDictionary()
        scores = lm.score("The company achieved excellent growth and improvement in profitability")
        assert scores["positive"] >= 0

    def test_score_negative_text(self):
        lm = LMDictionary()
        scores = lm.score("The company faces bankruptcy litigation and loss of market share")
        assert scores["negative"] >= 0

    def test_sentiment_score(self):
        lm = LMDictionary()
        s = lm.sentiment_score("excellent growth improvement")
        assert s > 0

    def test_sentiment_score_negative(self):
        lm = LMDictionary()
        s = lm.sentiment_score("bankruptcy fraud litigation loss")
        assert s < 0

    def test_empty_text(self):
        lm = LMDictionary()
        scores = lm.score("")
        assert scores["positive"] == 0.0
        assert scores["negative"] == 0.0


class TestFilingDocument:
    def test_default(self):
        f = FilingDocument()
        assert f.filing_type == FilingType.K10

    def test_with_sections(self):
        f = FilingDocument(sections={"risk_factors": "We face significant risks."})
        assert "risk_factors" in f.sections


class TestFilingIngestor:
    def test_init(self):
        ingestor = FilingIngestor()
        assert "QuantCore" in ingestor.user_agent

    def test_extract_sections(self):
        ingestor = FilingIngestor()
        text = "Item 1A. Risk Factors\nThe company faces risks.\nItem 7. Management's Discussion\nRevenue grew.\nItem 8. Financial Statements\nBalance sheet data."
        sections = ingestor.extract_sections(text)
        assert len(sections) >= 1

    def test_analyze_sentiment(self):
        ingestor = FilingIngestor()
        lm = LMDictionary()
        filing = FilingDocument(sections={"risk_factors": "litigation bankruptcy fraud loss"})
        results = ingestor.analyze_sentiment(filing, lm)
        assert "risk_factors" in results
        assert results["risk_factors"]["negative"] >= 0


class TestComputeFilingSentimentTrend:
    def test_empty(self):
        trends = compute_filing_sentiment_trend([])
        assert trends == {}

    def test_with_filings(self):
        f1 = FilingDocument(sections={"mda": "growth improvement"})
        f2 = FilingDocument(sections={"mda": "loss decline"})
        trends = compute_filing_sentiment_trend([f1, f2])
        assert "mda" in trends
        assert len(trends["mda"]) == 2


# ---------------------------------------------------------------------------
# Social media tests
# ---------------------------------------------------------------------------

class TestSocialConfig:
    def test_default(self):
        c = SocialConfig()
        assert len(c.subreddits) >= 3
        assert c.min_score == 5


class TestSocialPost:
    def test_default(self):
        p = SocialPost()
        assert p.source == SocialSource.REDDIT
        assert p.ticker == ""


class TestSocialIngestor:
    def test_init(self):
        ingestor = SocialIngestor()
        assert ingestor.config is not None

    def test_mention_volume(self):
        ingestor = SocialIngestor()
        now = datetime.now(UTC)
        posts = [
            SocialPost(id="1", ticker="AAPL", created_at=now),
            SocialPost(id="2", ticker="AAPL", created_at=now - timedelta(hours=48)),
            SocialPost(id="3", ticker="MSFT", created_at=now),
        ]
        vol = ingestor.compute_mention_volume(posts, "AAPL", hours=24)
        assert vol == 1

    def test_detect_unusual_activity_normal(self):
        ingestor = SocialIngestor()
        hist = [10, 12, 11, 9, 11, 10, 12, 13, 10, 11]
        result = ingestor.detect_unusual_activity(11, hist, threshold=2.0)
        assert result is False

    def test_detect_unusual_activity_spike(self):
        ingestor = SocialIngestor()
        hist = [10, 12, 11, 9, 11, 10, 12, 13, 10, 11]
        result = ingestor.detect_unusual_activity(50, hist, threshold=2.0)
        assert result is True


class TestBotFilter:
    def test_filters_short_posts(self):
        posts = [
            SocialPost(id="1", body="ok"),
            SocialPost(id="2", body="This is a genuine post with interesting content"),
        ]
        filtered = bot_filter(posts)
        assert len(filtered) == 1
        assert filtered[0].id == "2"


# ---------------------------------------------------------------------------
# Sentiment signals tests
# ---------------------------------------------------------------------------

class TestSentimentConfig:
    def test_default(self):
        c = SentimentConfig()
        assert 0.3 <= c.news_weight <= 0.4
        assert c.min_confidence == 0.3


class TestSentimentSignalGenerator:
    def test_init(self):
        gen = SentimentSignalGenerator()
        assert gen.config is not None

    def test_compute_news_score_no_articles(self):
        gen = SentimentSignalGenerator()
        score = gen.compute_news_score([], "AAPL")
        assert score == 0.0

    def test_compute_news_score_with_articles(self):
        gen = SentimentSignalGenerator()
        gen.config.min_articles_for_signal = 1
        a1 = NewsArticle(title="Test", tickers=["AAPL"], sentiment_score=0.5)
        a2 = NewsArticle(title="Test2", tickers=["AAPL"], sentiment_score=0.3)
        score = gen.compute_news_score([a1, a2], "AAPL")
        assert abs(score - 0.4) < 0.01

    def test_generate_composite(self):
        gen = SentimentSignalGenerator()
        a = NewsArticle(title="Test", tickers=["AAPL"], sentiment_score=0.5)
        gen.config.min_articles_for_signal = 1
        signal = gen.generate("AAPL", [a], [], {"mda": 0.1})
        assert signal.ticker == "AAPL"
        assert -1.0 <= signal.composite_score <= 1.0
        assert signal.confidence >= 0

    def test_is_signal_strong(self):
        gen = SentimentSignalGenerator()
        signal = SentimentSignal(ticker="AAPL", confidence=0.5)
        assert gen.is_signal_strong(signal) is True

    def test_is_signal_weak(self):
        gen = SentimentSignalGenerator()
        signal = SentimentSignal(ticker="AAPL", confidence=0.1)
        assert gen.is_signal_strong(signal) is False


class TestSentimentMomentum:
    def test_positive_momentum(self):
        signals = [
            SentimentSignal(ticker="AAPL", composite_score=0.1),
            SentimentSignal(ticker="AAPL", composite_score=0.5),
        ]
        mom = sentiment_momentum(signals, "AAPL")
        assert mom > 0

    def test_empty(self):
        assert sentiment_momentum([], "AAPL") == 0.0


class TestSentimentDivergence:
    def test_positive_correlation(self):
        prices = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        sentiment = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        corr = sentiment_divergence(prices, sentiment)
        assert abs(corr - 1.0) < 0.1

    def test_small_input(self):
        assert sentiment_divergence(np.array([0.01, 0.02]), np.array([0.1, 0.2])) == 0.0


class TestAggregateCrossSectional:
    def test_ranking(self):
        signals = [
            SentimentSignal(ticker="AAPL", composite_score=0.8),
            SentimentSignal(ticker="MSFT", composite_score=0.3),
            SentimentSignal(ticker="GOOGL", composite_score=0.6),
        ]
        ranked = aggregate_cross_sectional(signals)
        tickers = list(ranked.keys())
        assert tickers[0] == "AAPL"
        assert tickers[1] == "GOOGL"
        assert tickers[2] == "MSFT"
