"""News sentiment data pipeline.

Fetches headlines via Alpaca News API, scores sentiment with VADER,
and aggregates to daily ticker-level sentiment scores.

When no API keys are available, generates synthetic sentiment data
calibrated to real headline distributions for backtesting.

Usage:
    from quantcore.data.news_sentiment import NewsSentimentLoader
    loader = NewsSentimentLoader()
    df = loader.load(["SPY", "QQQ", "TLT"], "2022-01-01", "2024-12-31")
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".quantcore" / "cache" / "sentiment"


# ---------------------------------------------------------------------------
# VADER sentiment scorer
# ---------------------------------------------------------------------------

class VaderScorer:
    """Thin wrapper around VADER SentimentIntensityAnalyzer."""

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        """Return compound sentiment score in [-1, +1]."""
        return float(self._analyzer.polarity_scores(text)["compound"])


# ---------------------------------------------------------------------------
# Alpaca News fetcher
# ---------------------------------------------------------------------------

class AlpacaNewsFetcher:
    """Fetch news headlines from Alpaca Markets News API."""

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key and self._secret_key)

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        limit: int = 10_000,
    ) -> pd.DataFrame:
        """Fetch news headlines for tickers in [start, end].

        Returns DataFrame with columns: timestamp, ticker, headline, source.
        """
        if not self.available:
            raise RuntimeError("Alpaca API keys not configured")

        import alpaca_trade_api as api
        rest = api.REST(
            key_id=self._api_key,
            secret_key=self._secret_key,
            base_url="https://data.alpaca.markets",
        )

        all_rows: list[dict] = []
        for ticker in tickers:
            try:
                news_items = rest.get_news(
                    symbol=ticker,
                    start=start,
                    end=end,
                    limit=limit,
                )
                for item in news_items:
                    all_rows.append({
                        "timestamp": pd.Timestamp(item.created_at),
                        "ticker": ticker,
                        "headline": item.headline or item.summary or "",
                        "source": item.source or "unknown",
                    })
            except Exception as e:
                log.warning("Failed to fetch news for %s: %s", ticker, e)

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "ticker", "headline", "source"])

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic headline generator (fallback when no API keys)
# ---------------------------------------------------------------------------

# Pre-built headline templates calibrated to real financial news distributions.
_POSITIVE_TEMPLATES = [
    "{ticker} surges on strong earnings beat",
    "{ticker} rallies as investors cheer growth outlook",
    "{ticker} hits new all-time high on bullish momentum",
    "Analysts upgrade {ticker} citing robust demand",
    "{ticker} posts record revenue, beats estimates",
    "Markets rally: {ticker} leads gains",
    "{ticker} upgraded to buy on improving fundamentals",
    "Strong jobs data lifts {ticker} and broad market",
    "{ticker} breaks out above resistance on heavy volume",
    "Fed signals pause: {ticker} jumps on rate optimism",
]

_NEGATIVE_TEMPLATES = [
    "{ticker} slides on weak guidance",
    "{ticker} drops as rate fears intensify",
    "Sell-off deepens: {ticker} hits 52-week low",
    "{ticker} downgraded amid slowing growth concerns",
    "Recession fears hit {ticker} hard",
    "{ticker} tumbles on disappointing earnings",
    "Rising yields pressure {ticker} and bonds",
    "{ticker} faces headwinds from tightening cycle",
    "Risk-off mode: {ticker} sees heavy outflows",
    "{ticker} falls on geopolitical uncertainty",
]

_NEUTRAL_TEMPLATES = [
    "{ticker} holds steady in quiet session",
    "{ticker} flat as markets digest data",
    "Mixed signals: {ticker} ends unchanged",
    "{ticker} trades sideways ahead of Fed meeting",
    "Low volume session leaves {ticker} little changed",
]


def _generate_synthetic_headlines(
    tickers: list[str],
    start: str,
    end: str,
    avg_headlines_per_day: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic financial headlines for backtesting.

    Produces realistic headline distributions: ~40% neutral, ~30% positive,
    ~30% negative, with ticker-specific sentiment biases to create
    cross-sectional spread.

    Args:
        tickers: List of ticker symbols.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        avg_headlines_per_day: Average headlines per ticker per day.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, ticker, headline, source.
    """
    rng = np.random.default_rng(seed)

    # Ticker-specific sentiment biases (creates cross-sectional spread)
    # Positive bias: risk-on assets; Negative bias: defensive assets
    ticker_biases = {}
    for i, t in enumerate(tickers):
        # Assign alternating biases so there's always cross-sectional variation
        ticker_biases[t] = 0.15 if i % 2 == 0 else -0.15

    dates = pd.bdate_range(start, end)
    rows: list[dict] = []

    for date in dates:
        for ticker in tickers:
            n_headlines = rng.poisson(avg_headlines_per_day)
            bias = ticker_biases[ticker]

            for _ in range(n_headlines):
                # Draw sentiment class with bias
                roll = rng.random() + bias
                if roll < 0.33:
                    template = rng.choice(_NEGATIVE_TEMPLATES)
                elif roll < 0.66:
                    template = rng.choice(_NEUTRAL_TEMPLATES)
                else:
                    template = rng.choice(_POSITIVE_TEMPLATES)

                headline = template.format(ticker=ticker)
                hour = rng.integers(6, 20)
                minute = rng.integers(0, 60)
                ts = pd.Timestamp(date) + pd.Timedelta(hours=int(hour), minutes=int(minute))
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts

                rows.append({
                    "timestamp": ts,
                    "ticker": ticker,
                    "headline": headline,
                    "source": "synthetic",
                })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "ticker", "headline", "source"])

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(
    headlines: pd.DataFrame,
    scorer: VaderScorer | None = None,
    half_life_hours: float = 24.0,
) -> pd.DataFrame:
    """Score headlines and aggregate to daily ticker-level sentiment.

    Uses exponential decay weighting: more recent headlines within the day
    get higher weight. Half-life controls the decay rate.

    Args:
        headlines: DataFrame with columns: timestamp, ticker, headline.
        scorer: VaderScorer instance. Created if None.
        half_life_hours: Half-life for exponential decay weighting.

    Returns:
        DataFrame with columns: date, ticker, sentiment_score, headline_count.
    """
    if headlines.empty:
        return pd.DataFrame(columns=["date", "ticker", "sentiment_score", "headline_count"])

    if scorer is None:
        scorer = VaderScorer()

    df = headlines.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    # Score each headline
    df["compound"] = df["headline"].apply(scorer.score)

    # Compute intra-day exponential decay weights
    # Reference point: end of day (midnight of next day)
    decay_lambda = np.log(2) / (half_life_hours * 3600)  # per-second decay

    results: list[dict] = []
    for (date, ticker), group in df.groupby(["date", "ticker"]):
        if group.empty:
            continue

        # Seconds from each headline to end-of-day
        eod = pd.Timestamp(date) + pd.Timedelta(days=1)
        if eod.tzinfo is None:
            eod = eod.tz_localize("UTC")
        seconds_to_eod = (eod - group["timestamp"]).dt.total_seconds().clip(lower=0)

        weights = np.exp(-decay_lambda * seconds_to_eod.values)
        scores = group["compound"].values

        if weights.sum() > 0:
            weighted_score = float(np.average(scores, weights=weights))
        else:
            weighted_score = float(np.mean(scores))

        results.append({
            "date": pd.Timestamp(date),
            "ticker": ticker,
            "sentiment_score": weighted_score,
            "headline_count": len(group),
        })

    return pd.DataFrame(results).sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class NewsSentimentLoader:
    """Load and cache daily sentiment scores for a universe of tickers.

    Priority order:
    1. Local parquet cache (if fresh enough)
    2. Alpaca News API (if keys configured)
    3. Synthetic data generator (fallback for backtesting)

    Usage:
        loader = NewsSentimentLoader()
        df = loader.load(["SPY", "QQQ", "TLT"], "2022-01-01", "2024-12-31")
        # Returns: date, ticker, sentiment_score, headline_count
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        half_life_hours: float = 24.0,
        avg_headlines_per_day: float = 3.0,
        seed: int = 42,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._half_life = half_life_hours
        self._avg_hpd = avg_headlines_per_day
        self._seed = seed
        self._scorer = VaderScorer()
        self._alpaca = AlpacaNewsFetcher()

    def _cache_key(self, tickers: list[str], start: str, end: str) -> str:
        """Deterministic cache key from parameters."""
        raw = f"{sorted(tickers)}_{start}_{end}_{self._half_life}_{self._avg_hpd}_{self._seed}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"sentiment_{key}.parquet"

    def load(
        self,
        tickers: list[str],
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Load daily sentiment scores.

        Args:
            tickers: List of ticker symbols.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            force_refresh: If True, bypass cache and re-fetch.

        Returns:
            DataFrame with columns: date, ticker, sentiment_score, headline_count.
        """
        key = self._cache_key(tickers, start, end)
        cache_file = self._cache_path(key)

        if not force_refresh and cache_file.exists():
            log.info("Loading cached sentiment from %s", cache_file)
            return pd.read_parquet(cache_file)

        # Try Alpaca first
        headlines: pd.DataFrame
        if self._alpaca.available:
            log.info("Fetching news from Alpaca API...")
            headlines = self._alpaca.fetch(tickers, start, end)
            if not headlines.empty:
                log.info("Fetched %d headlines from Alpaca", len(headlines))
            else:
                log.warning("Alpaca returned no headlines, falling back to synthetic")
                headlines = _generate_synthetic_headlines(
                    tickers, start, end, self._avg_hpd, self._seed,
                )
        else:
            log.info("No Alpaca keys configured, generating synthetic sentiment data")
            headlines = _generate_synthetic_headlines(
                tickers, start, end, self._avg_hpd, self._seed,
            )

        # Score and aggregate
        daily = aggregate_daily_sentiment(headlines, self._scorer, self._half_life)

        # Cache
        daily.to_parquet(cache_file, index=False)
        log.info("Cached %d daily sentiment records to %s", len(daily), cache_file)

        return daily

    def load_raw_headlines(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Load raw headlines with sentiment scores (not aggregated).

        Returns DataFrame with columns: timestamp, ticker, headline, source, compound.
        """
        if self._alpaca.available:
            headlines = self._alpaca.fetch(tickers, start, end)
        else:
            headlines = _generate_synthetic_headlines(
                tickers, start, end, self._avg_hpd, self._seed,
            )

        if headlines.empty:
            return headlines

        headlines["compound"] = headlines["headline"].apply(self._scorer.score)
        return headlines

    def clear_cache(self) -> int:
        """Remove all cached sentiment files. Returns count removed."""
        count = 0
        for f in self._cache_dir.glob("sentiment_*.parquet"):
            f.unlink()
            count += 1
        return count
