"""RSS news ingestion: feed parsing, text extraction, ticker mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
import re
import html as html_mod

from quantcore.core.types import Position


@dataclass
class NewsSourceConfig:
    """Configuration for RSS news ingestion."""
    feed_urls: list[str] = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://seekingalpha.com/market_currents.xml",
    ])
    refresh_interval_sec: float = 300.0
    max_articles_per_feed: int = 50
    user_agent: str = "QuantCore/0.1"


@dataclass
class NewsArticle:
    """Parsed and enriched news article."""
    title: str = ""
    url: str = ""
    source: str = ""
    published_at: datetime | None = None
    raw_text: str = ""
    summary: str = ""
    tickers: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0


class NewsIngestor:
    """Ingest news from RSS feeds, extract text, and map tickers."""

    def __init__(
        self,
        config: NewsSourceConfig | None = None,
        ticker_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.config = config or NewsSourceConfig()
        self.ticker_map = ticker_map or {}
        self._articles: list[NewsArticle] = []

    def fetch_feeds(self) -> list[NewsArticle]:
        """Parse all configured RSS feeds and return new articles."""
        try:
            import feedparser
        except ImportError:
            return []
        new_articles: list[NewsArticle] = []
        for url in self.config.feed_urls:
            try:
                feed = feedparser.parse(url, agent=self.config.user_agent)
                for entry in feed.entries[:self.config.max_articles_per_feed]:
                    article = self._parse_entry(entry, url)
                    if article and self._is_new(article):
                        new_articles.append(article)
                        self._articles.append(article)
            except Exception:
                pass
        return new_articles

    def _parse_entry(self, entry: Any, feed_url: str) -> NewsArticle | None:
        try:
            title = entry.get("title", "")
            link = entry.get("link", "")
            if not title or not link:
                return None
            raw_html = entry.get("summary", entry.get("description", entry.get("content", "")))
            if hasattr(raw_html, "value"):
                raw_html = raw_html.value
            raw_text = self._extract_text(str(raw_html))
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
            tickers = self._map_tickers(title + " " + raw_text)
            return NewsArticle(
                title=title, url=link, source=feed_url,
                published_at=published, raw_text=raw_text,
                summary=raw_text[:300], tickers=tickers,
            )
        except Exception:
            return None

    def _extract_text(self, html_text: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except ImportError:
            clean = re.sub(r"<[^>]+>", " ", html_text)
            return re.sub(r"\s+", " ", clean).strip()

    def _map_tickers(self, text: str) -> list[str]:
        """Find company names in text using ticker_map and return tickers."""
        found: list[str] = []
        text_lower = text.lower()
        for name, tickers in self.ticker_map.items():
            if name.lower() in text_lower:
                found.extend(tickers)
        # Also extract $TICKER patterns
        dollar_tickers = re.findall(r"\$([A-Z]{1,5})", text)
        found.extend(dollar_tickers)
        return list(dict.fromkeys(found))  # dedup, preserve order

    def _is_new(self, article: NewsArticle) -> bool:
        existing_urls = {a.url for a in self._articles}
        return article.url not in existing_urls

    def filter_by_ticker(self, ticker: str) -> list[NewsArticle]:
        """Get all articles mentioning a specific ticker."""
        return [a for a in self._articles if ticker in a.tickers]

    def recent_articles(self, hours: float = 24.0) -> list[NewsArticle]:
        """Get articles published within the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self._articles if a.published_at and a.published_at >= cutoff]

    @property
    def article_count(self) -> int:
        return len(self._articles)
