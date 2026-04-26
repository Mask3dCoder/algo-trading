"""FinBERT sentiment scoring with lazy model loading and entity extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FinBERTSentiment:
    """Sentiment scores for a text span."""
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 1.0
    compound: float = 0.0
    label: str = "neutral"


@dataclass
class FinBERTConfig:
    """Configuration for the FinBERT sentiment analyzer."""
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 8
    max_length: int = 512
    device: str = "cpu"


class FinBERTAnalyzer:
    """Lazy-loading FinBERT sentiment analyzer.

    Uses the transformers pipeline when available, falling back to
    a lexicon-based heuristic for offline/lightweight use.
    """

    def __init__(self, config: FinBERTConfig | None = None) -> None:
        self.config = config or FinBERTConfig()
        self._pipeline: Any = None
        self._loaded = False

    def _ensure_model(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.model_name,
                device=self.config.device,
                max_length=self.config.max_length,
                truncation=True,
            )
        except ImportError:
            self._pipeline = None
        self._loaded = True

    def analyze(self, texts: list[str]) -> list[FinBERTSentiment]:
        """Run sentiment analysis on a batch of texts."""
        self._ensure_model()
        if not texts:
            return []
        if self._pipeline is not None:
            try:
                raw = self._pipeline(texts, batch_size=self.config.batch_size)
                return [_finbert_result_to_sentiment(r) for r in raw]
            except Exception:
                pass
        return [_lexicon_sentiment(t) for t in texts]

    def analyze_article(self, article: Any) -> FinBERTSentiment:
        """Score a single NewsArticle."""
        text = getattr(article, "raw_text", "") or getattr(article, "summary", "") or ""
        results = self.analyze([text])
        return results[0] if results else FinBERTSentiment()

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract organizations and tickers from text using regex heuristics."""
        org_patterns = [
            r'\b([A-Z][a-z]+ (?:Inc\.?|Corp\.?|Ltd\.?|LLC|Group|Holdings|Technologies|Pharmaceuticals?|Bancorp))\b',
        ]
        orgs: list[str] = []
        for pat in org_patterns:
            orgs.extend(re.findall(pat, text))
        tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        return {
            "organizations": list(dict.fromkeys(orgs)),
            "tickers": list(dict.fromkeys(tickers)),
        }


def _finbert_result_to_sentiment(result: dict) -> FinBERTSentiment:
    label = result.get("label", "neutral").lower()
    score = float(result.get("score", 0.5))
    pos = score if label == "positive" else (1.0 - score) * 0.5 if label == "neutral" else 0.0
    neg = score if label == "negative" else (1.0 - score) * 0.5 if label == "neutral" else 0.0
    neu = 1.0 - pos - neg
    compound = pos - neg
    return FinBERTSentiment(positive=pos, negative=neg, neutral=neu, compound=compound, label=label)


def _lexicon_sentiment(text: str) -> FinBERTSentiment:
    positive_words = {
        "profit", "growth", "gain", "positive", "strong", "beat", "upgrade",
        "bullish", "rally", "surge", "opportunity", "outperform", "dividend",
        "buyback", "expansion", "recovery", "momentum", "record", "boost",
        "optimistic", "outlook", "raised", "guidance", "exceed",
    }
    negative_words = {
        "loss", "decline", "drop", "negative", "weak", "miss", "downgrade",
        "bearish", "crash", "plunge", "risk", "underperform", "layoff",
        "lawsuit", "bankruptcy", "debt", "default", "cut", "warned",
        "volatility", "concern", "caution", "headwind", "downturn",
    }
    words = set(text.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    total = pos + neg
    if total == 0:
        return FinBERTSentiment(neutral=1.0, label="neutral")
    pos_score = pos / total
    neg_score = neg / total
    compound = pos_score - neg_score
    label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
    return FinBERTSentiment(
        positive=pos_score, negative=neg_score,
        neutral=1.0 - abs(compound), compound=compound, label=label,
    )


def scored_articles(
    articles: list[Any], analyzer: FinBERTAnalyzer,
) -> list[tuple[Any, FinBERTSentiment]]:
    """Score a list of articles and return (article, sentiment) pairs."""
    texts = [getattr(a, "raw_text", "") or getattr(a, "summary", "") or "" for a in articles]
    sentiments = analyzer.analyze(texts)
    return list(zip(articles, sentiments))
