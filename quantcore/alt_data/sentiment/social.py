"""Social media sentiment: Reddit, StockTwits ingestion and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np


class SocialSource(str, Enum):
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    TWITTER = "twitter"


@dataclass
class SocialPost:
    """A single social media post about a ticker."""
    id: str = ""
    source: SocialSource = SocialSource.REDDIT
    ticker: str = ""
    title: str = ""
    body: str = ""
    score: int = 0
    num_comments: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    url: str = ""


@dataclass
class SocialConfig:
    """Configuration for social media ingestion."""
    subreddits: list[str] = field(default_factory=lambda: [
        "wallstreetbets", "stocks", "investing", "trading", "options",
        "stockmarket", "securityanalysis",
    ])
    min_score: int = 5
    max_posts: int = 200
    refresh_interval_sec: float = 120.0


class SocialIngestor:
    """Ingest social media posts and compute engagement metrics."""

    def __init__(self, config: SocialConfig | None = None) -> None:
        self.config = config or SocialConfig()
        self._posts: list[SocialPost] = []
        self._session: Any = None

    def _get_session(self) -> Any:
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "QuantCore/0.1"})
        return self._session

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def fetch_reddit_posts(self, ticker: str | None = None) -> list[SocialPost]:
        """Fetch posts from configured subreddits."""
        new_posts: list[SocialPost] = []
        session = self._get_session()
        for sub in self.config.subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json?q={ticker or ''}&sort=new&limit={self.config.max_posts}"
                resp = session.get(url, timeout=10)
                data = resp.json()
                children = data.get("data", {}).get("children", [])
                for child in children:
                    post_data = child.get("data", {})
                    post = self._parse_reddit_post(post_data, sub, ticker)
                    if post and post.score >= self.config.min_score:
                        if self._is_new(post.id):
                            new_posts.append(post)
                            self._posts.append(post)
            except Exception:
                continue
        return new_posts

    def _parse_reddit_post(
        self, data: dict, subreddit: str, ticker: str | None,
    ) -> SocialPost | None:
        post_id = data.get("id", "")
        title = data.get("title", "")
        body = data.get("selftext", "")
        if not title:
            return None
        detected_ticker = ticker or self._detect_ticker(title + " " + body)
        if not detected_ticker:
            return None
        return SocialPost(
            id=post_id, source=SocialSource.REDDIT, ticker=detected_ticker,
            title=title, body=body, score=int(data.get("score", 0)),
            num_comments=int(data.get("num_comments", 0)),
            created_at=datetime.fromtimestamp(data.get("created_utc", 0), tz=UTC),
            url=f"https://reddit.com{data.get('permalink', '')}",
        )

    def _detect_ticker(self, text: str) -> str | None:
        import re
        tickers = re.findall(r'\$?([A-Z]{1,5})\b', text.upper())
        common_words = {"THE", "A", "I", "AM", "PM", "DD", "YOLO", "USA", "US", "CEO", "CFO", "IPO", "ATH", "IT", "IS", "BE", "FOR", "ARE", "HAS", "AND", "BUT", "OR", "NOT", "NEW", "ONE", "ALL", "CAN", "WILL", "JUST", "NOW", "MORE", "OVER", "ALSO", "VERY", "MUCH", "WELL", "EVEN", "THAN", "THEN", "THAT", "THIS", "WITH", "FROM", "THEY", "SAYS", "YOUR", "SOME", "WHAT", "WHEN", "MAKE", "LIKE", "ABOUT", "WOULD", "THERE", "THEIR"}
        for t in tickers:
            if t not in common_words and len(t) >= 2:
                return t
        return None

    def fetch_stocktwits(self, ticker: str) -> list[SocialPost]:
        """Fetch StockTwits messages for a ticker."""
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            resp = self._get_session().get(url, timeout=10)
            data = resp.json()
            messages = data.get("messages", [])
            new_posts: list[SocialPost] = []
            for msg in messages:
                post = SocialPost(
                    id=str(msg.get("id", "")),
                    source=SocialSource.STOCKTWITS,
                    ticker=ticker,
                    body=msg.get("body", ""),
                    created_at=datetime.fromisoformat(
                        msg.get("created_at", datetime.now(UTC).isoformat()),
                    ),
                )
                if self._is_new(post.id):
                    new_posts.append(post)
                    self._posts.append(post)
            return new_posts
        except Exception:
            return []

    def _is_new(self, post_id: str) -> bool:
        return post_id not in {p.id for p in self._posts}

    def compute_mention_volume(
        self, posts: list[SocialPost], ticker: str, hours: float = 24.0,
    ) -> int:
        """Count mentions of a ticker within a time window."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return sum(1 for p in posts if p.ticker == ticker and p.created_at >= cutoff)

    def compute_sentiment(self, posts: list[SocialPost], analyzer: Any) -> float:
        """Average FinBERT sentiment across posts for a ticker."""
        texts = [p.title + " " + p.body for p in posts]
        if not texts:
            return 0.0
        sentiments = analyzer.analyze(texts)
        return float(np.mean([s.compound for s in sentiments]))

    def detect_unusual_activity(
        self, current_volume: int, historical_volumes: list[int], threshold: float = 2.0,
    ) -> bool:
        """Detect spikes in mention volume using z-score."""
        if len(historical_volumes) < 10:
            return False
        hist = np.array(historical_volumes, dtype=float)
        mean = float(np.mean(hist))
        std = float(np.std(hist, ddof=1))
        if std == 0:
            return False
        return (current_volume - mean) / std > threshold

    @property
    def post_count(self) -> int:
        return len(self._posts)

    def recent_posts(self, ticker: str, hours: float = 24.0) -> list[SocialPost]:
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return [
            p for p in self._posts
            if p.ticker == ticker and p.created_at >= cutoff
        ]


def bot_filter(posts: list[SocialPost]) -> list[SocialPost]:
    """Remove likely bot posts based on simple heuristics."""
    filtered: list[SocialPost] = []
    for p in posts:
        if p.body and len(p.body) < 10:
            continue
        if p.title and len(set(p.title.split())) <= 2:
            continue
        filtered.append(p)
    return filtered
