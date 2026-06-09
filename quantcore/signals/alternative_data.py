"""Alternative data signals for statistical arbitrage.

Implements:
  - News sentiment analysis (FinBERT)
  - Social media sentiment
  - Options flow analysis
  - Insider trading signals
  - Alternative data combination

References:
    - Tetlock (2007): Giving Content to Investor Sentiment
    - Baker & Wurgler (2006): Investor Sentiment and the Cross-Section of Stock Returns
    - Bollen, Mao, Zeng (2011): Twitter mood predicts the stock market
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    score: float  # Sentiment score [-1, 1]
    confidence: float  # Confidence in the score [0, 1]
    volume: int  # Number of mentions/articles
    momentum: float  # Sentiment momentum (change over time)
    source: str  # Data source


@dataclass
class OptionsFlowSignal:
    """Options flow signal."""
    put_call_ratio: float  # Put/Call volume ratio
    unusual_activity: bool  # Detected unusual activity
    iv_skew: float  # Implied volatility skew
    net_gamma: float  # Net gamma exposure
    signal: float  # Combined signal [-1, 1]


@dataclass
class InsiderSignal:
    """Insider trading signal."""
    cluster_score: float  # Cluster insider activity
    momentum: float  # Insider trading momentum
    smart_money: float  # Smart money indicator
    signal: float  # Combined signal [-1, 1]


# ---------------------------------------------------------------------------
# News Sentiment Analysis
# ---------------------------------------------------------------------------


class NewsSentimentAnalyzer:
    """NLP-based news sentiment analysis.
    
    Uses FinBERT or similar models to analyze financial news
    and generate sentiment signals.
    
    Parameters
    ----------
    model_name : str
        NLP model to use (default 'finbert').
    lookback_days : int
        Days to look back for sentiment (default 7).
    min_mentions : int
        Minimum mentions for valid signal (default 5).
    """
    
    def __init__(
        self,
        model_name: str = "finbert",
        lookback_days: int = 7,
        min_mentions: int = 5,
    ) -> None:
        self.model_name = model_name
        self.lookback_days = lookback_days
        self.min_mentions = min_mentions
        self._model = None
        
    def analyze_headlines(
        self,
        headlines: list[str],
        timestamps: list[datetime],
    ) -> SentimentResult:
        """Analyze news headlines for sentiment.
        
        Parameters
        ----------
        headlines : list[str]
            News headlines.
        timestamps : list[datetime]
            Publication timestamps.
            
        Returns
        -------
        SentimentResult
            Sentiment analysis result.
        """
        if not headlines:
            return SentimentResult(0.0, 0.0, 0, 0.0, "news")
        
        # Simplified sentiment analysis (in production, use FinBERT)
        # This is a placeholder - real implementation would use transformers
        
        positive_words = {
            "upgrade", "beat", "exceed", "strong", "growth", "bullish",
            "outperform", "buy", "positive", "surge", "rally", "record",
        }
        negative_words = {
            "downgrade", "miss", "weak", "decline", "bearish", "underperform",
            "sell", "negative", "drop", "crash", "recession", "loss",
        }
        
        scores = []
        for headline in headlines:
            words = set(headline.lower().split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            
            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                score = 0.0
            
            scores.append(score)
        
        # Compute weighted average (more recent = higher weight)
        if timestamps:
            now = max(timestamps)
            weights = []
            for ts in timestamps:
                days_ago = (now - ts).days
                weight = np.exp(-days_ago / self.lookback_days)
                weights.append(weight)
            
            weights = np.array(weights)
            weights /= weights.sum()
            
            avg_score = np.average(scores, weights=weights)
        else:
            avg_score = np.mean(scores)
        
        # Compute momentum (recent vs older sentiment)
        if len(scores) >= 10:
            recent = np.mean(scores[-5:])
            older = np.mean(scores[-10:-5])
            momentum = recent - older
        else:
            momentum = 0.0
        
        # Confidence based on volume
        confidence = min(len(headlines) / self.min_mentions, 1.0)
        
        return SentimentResult(
            score=float(np.clip(avg_score, -1, 1)),
            confidence=confidence,
            volume=len(headlines),
            momentum=momentum,
            source="news",
        )
    
    def compute_sentiment_signal(
        self,
        sentiment_history: list[SentimentResult],
        lookback: int = 20,
    ) -> float:
        """Compute trading signal from sentiment history.
        
        Parameters
        ----------
        sentiment_history : list[SentimentResult]
            Historical sentiment results.
        lookback : int
            Lookback window for signal computation.
            
        Returns
        -------
        float
            Trading signal [-1, 1].
        """
        if len(sentiment_history) < lookback:
            return 0.0
        
        recent = sentiment_history[-lookback:]
        
        # Weighted average by confidence and recency
        scores = [s.score * s.confidence for s in recent]
        weights = np.array([np.exp(-i / lookback) for i in range(lookback)])[::-1]
        weights /= weights.sum()
        
        signal = np.average(scores, weights=weights)
        
        # Add momentum component
        momentum = recent[-1].momentum
        signal = 0.7 * signal + 0.3 * momentum
        
        return float(np.clip(signal, -1, 1))


# ---------------------------------------------------------------------------
# Social Media Sentiment
# ---------------------------------------------------------------------------


class SocialMediaSentiment:
    """Social media sentiment analysis.
    
    Aggregates sentiment from Twitter, Reddit, StockTwits, etc.
    to capture retail investor sentiment.
    
    Parameters
    ----------
    platforms : list[str]
        Platforms to monitor (default ['twitter', 'reddit', 'stocktwits']).
    lookback_hours : int
        Hours to look back (default 24).
    min_volume : int
        Minimum mention volume (default 100).
    """
    
    def __init__(
        self,
        platforms: Optional[list[str]] = None,
        lookback_hours: int = 24,
        min_volume: int = 100,
    ) -> None:
        self.platforms = platforms or ["twitter", "reddit", "stocktwits"]
        self.lookback_hours = lookback_hours
        self.min_volume = min_volume
        
    def compute_retail_sentiment(
        self,
        mentions: dict[str, list[dict]],
    ) -> SentimentResult:
        """Compute retail investor sentiment.
        
        Parameters
        ----------
        mentions : dict[str, list[dict]]
            Mentions per platform. Each dict has:
            - 'timestamp': datetime
            - 'sentiment': float [-1, 1]
            - 'engagement': int (likes, retweets, etc.)
            
        Returns
        -------
        SentimentResult
            Aggregated sentiment.
        """
        all_sentiments = []
        total_volume = 0
        
        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)
        
        for platform, posts in mentions.items():
            if platform not in self.platforms:
                continue
            
            platform_sentiments = []
            for post in posts:
                if post["timestamp"] >= cutoff:
                    # Weight by engagement
                    weight = np.log1p(post.get("engagement", 1))
                    platform_sentiments.append(post["sentiment"] * weight)
            
            if platform_sentiments:
                # Platform average
                platform_avg = np.mean(platform_sentiments)
                all_sentiments.append(platform_avg)
                total_volume += len(platform_sentiments)
        
        if not all_sentiments:
            return SentimentResult(0.0, 0.0, 0, 0.0, "social")
        
        # Cross-platform average
        avg_sentiment = np.mean(all_sentiments)
        
        # Confidence based on volume
        confidence = min(total_volume / self.min_volume, 1.0)
        
        return SentimentResult(
            score=float(np.clip(avg_sentiment, -1, 1)),
            confidence=confidence,
            volume=total_volume,
            momentum=0.0,  # Would need historical data
            source="social",
        )
    
    def detect_sentiment_spikes(
        self,
        sentiment_history: list[SentimentResult],
        z_threshold: float = 2.0,
    ) -> list[int]:
        """Detect unusual sentiment spikes.
        
        Parameters
        ----------
        sentiment_history : list[SentimentResult]
            Historical sentiment.
        z_threshold : float
            Z-score threshold for spike detection.
            
        Returns
        -------
        list[int]
            Indices of detected spikes.
        """
        if len(sentiment_history) < 20:
            return []
        
        scores = [s.score * s.volume for s in sentiment_history]
        
        # Compute rolling statistics
        window = 20
        spikes = []
        
        for i in range(window, len(scores)):
            window_data = scores[i-window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                z = (scores[i] - mean) / std
                if abs(z) > z_threshold:
                    spikes.append(i)
        
        return spikes


# ---------------------------------------------------------------------------
# Options Flow Analysis
# ---------------------------------------------------------------------------


class OptionsFlowAnalyzer:
    """Options market signals analysis.
    
    Analyzes options flow to detect:
      - Put/Call ratios (volume and open interest)
      - Unusual options activity
      - Implied volatility signals
      - Market maker positioning
    
    Parameters
    ----------
    unusual_volume_threshold : float
        Multiple of average volume to flag as unusual (default 3.0).
    iv_percentile_lookback : int
        Days for IV percentile calculation (default 252).
    """
    
    def __init__(
        self,
        unusual_volume_threshold: float = 3.0,
        iv_percentile_lookback: int = 252,
    ) -> None:
        self.unusual_volume_threshold = unusual_volume_threshold
        self.iv_percentile_lookback = iv_percentile_lookback
        
    def compute_put_call_signal(
        self,
        put_volume: NDArray,
        call_volume: NDArray,
        put_oi: NDArray,
        call_oi: NDArray,
        lookback: int = 20,
    ) -> float:
        """Compute Put/Call ratio signal.
        
        Parameters
        ----------
        put_volume : NDArray
            Daily put volume.
        call_volume : NDArray
            Daily call volume.
        put_oi : NDArray
            Put open interest.
        call_oi : NDArray
            Call open interest.
        lookback : int
            Lookback for z-scoring.
            
        Returns
        -------
        float
            Signal [-1, 1]. High P/C ratio = bearish (-1), low = bullish (+1).
        """
        # Volume P/C ratio
        total_vol = put_volume + call_volume
        pc_ratio_vol = np.where(total_vol > 0, put_volume / total_vol, 0.5)
        
        # OI P/C ratio
        total_oi = put_oi + call_oi
        pc_ratio_oi = np.where(total_oi > 0, put_oi / total_oi, 0.5)
        
        # Combined ratio
        pc_ratio = 0.6 * pc_ratio_vol + 0.4 * pc_ratio_oi
        
        # Z-score
        if len(pc_ratio) >= lookback:
            recent = pc_ratio[-lookback:]
            mean = np.mean(recent)
            std = np.std(recent)
            
            if std > 0:
                z = (pc_ratio[-1] - mean) / std
                # Invert: high P/C = bearish (negative signal)
                signal = -np.clip(z / 3, -1, 1)
            else:
                signal = 0.0
        else:
            signal = 0.0
        
        return float(signal)
    
    def detect_unusual_activity(
        self,
        volume: NDArray,
        avg_volume: NDArray,
    ) -> NDArray:
        """Detect unusual options activity.
        
        Parameters
        ----------
        volume : NDArray
            Current volume.
        avg_volume : NDArray
            Average volume.
            
        Returns
        -------
        NDArray
            Boolean array indicating unusual activity.
        """
        # Avoid division by zero
        safe_avg = np.maximum(avg_volume, 1)
        ratio = volume / safe_avg
        
        return ratio > self.unusual_volume_threshold
    
    def compute_iv_signal(
        self,
        current_iv: float,
        iv_history: NDArray,
    ) -> float:
        """Compute implied volatility signal.
        
        Parameters
        ----------
        current_iv : float
            Current IV level.
        iv_history : NDArray
            Historical IV values.
            
        Returns
        -------
        float
            Signal [-1, 1]. High IV percentile = sell vol (-1), low = buy vol (+1).
        """
        if len(iv_history) < 20:
            return 0.0
        
        # IV percentile
        percentile = np.mean(iv_history < current_iv) * 100
        
        # Convert to signal
        # High IV (>80th percentile) = sell vol (negative signal)
        # Low IV (<20th percentile) = buy vol (positive signal)
        if percentile > 80:
            signal = -(percentile - 80) / 20
        elif percentile < 20:
            signal = (20 - percentile) / 20
        else:
            signal = 0.0
        
        return float(np.clip(signal, -1, 1))
    
    def compute_net_gamma(
        self,
        call_gamma: NDArray,
        put_gamma: NDArray,
        call_oi: NDArray,
        put_oi: NDArray,
    ) -> float:
        """Compute net gamma exposure.
        
        Positive net gamma = dealers are long gamma (mean reversion).
        Negative net gamma = dealers are short gamma (trend following).
        
        Parameters
        ----------
        call_gamma : NDArray
            Call gamma per contract.
        put_gamma : NDArray
            Put gamma per contract.
        call_oi : NDArray
            Call open interest.
        put_oi : NDArray
            Put open interest.
            
        Returns
        -------
        float
            Net gamma exposure (normalized).
        """
        # Net gamma = call gamma * call OI - put gamma * put OI
        net = np.sum(call_gamma * call_oi) - np.sum(put_gamma * put_oi)
        
        # Normalize
        total_oi = np.sum(call_oi) + np.sum(put_oi)
        if total_oi > 0:
            return float(net / total_oi)
        return 0.0
    
    def compute_flow_signal(
        self,
        put_call_signal: float,
        unusual_activity: bool,
        iv_signal: float,
        net_gamma: float,
    ) -> OptionsFlowSignal:
        """Combine options flow signals.
        
        Parameters
        ----------
        put_call_signal : float
            Put/Call ratio signal.
        unusual_activity : bool
            Unusual activity detected.
        iv_signal : float
            IV signal.
        net_gamma : float
            Net gamma exposure.
            
        Returns
        -------
        OptionsFlowSignal
            Combined options flow signal.
        """
        # Weighted combination
        signal = (
            0.3 * put_call_signal +
            0.2 * (1.0 if unusual_activity else 0.0) +
            0.3 * iv_signal +
            0.2 * np.sign(net_gamma) * min(abs(net_gamma) * 10, 1.0)
        )
        
        return OptionsFlowSignal(
            put_call_ratio=put_call_signal,
            unusual_activity=unusual_activity,
            iv_skew=iv_signal,
            net_gamma=net_gamma,
            signal=float(np.clip(signal, -1, 1)),
        )


# ---------------------------------------------------------------------------
# Insider Trading Signals
# ---------------------------------------------------------------------------


class InsiderTradingSignal:
    """Insider trading signal generation.
    
    Analyzes insider buying/selling patterns to detect:
      - Cluster insider activity
      - Insider trading momentum
      - Smart money indicators
    
    Parameters
    ----------
    lookback_days : int
        Days to look back for insider activity (default 90).
    min_trades : int
        Minimum trades for valid signal (default 5).
    """
    
    def __init__(
        self,
        lookback_days: int = 90,
        min_trades: int = 5,
    ) -> None:
        self.lookback_days = lookback_days
        self.min_trades = min_trades
        
    def compute_insider_signal(
        self,
        insider_trades: list[dict],
        current_price: float,
    ) -> InsiderSignal:
        """Compute insider trading signal.
        
        Parameters
        ----------
        insider_trades : list[dict]
            Insider trade records with keys:
            - 'date': datetime
            - 'type': 'buy' or 'sell'
            - 'shares': int
            - 'price': float
            - 'insider_type': 'officer', 'director', '10%owner', etc.
        current_price : float
            Current stock price.
            
        Returns
        -------
        InsiderSignal
            Insider trading signal.
        """
        if len(insider_trades) < self.min_trades:
            return InsiderSignal(0.0, 0.0, 0.0, 0.0)
        
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent_trades = [t for t in insider_trades if t["date"] >= cutoff]
        
        if len(recent_trades) < self.min_trades:
            return InsiderSignal(0.0, 0.0, 0.0, 0.0)
        
        # Separate buys and sells
        buys = [t for t in recent_trades if t["type"] == "buy"]
        sells = [t for t in recent_trades if t["type"] == "sell"]
        
        # Cluster score: more buys = positive, more sells = negative
        total = len(buys) + len(sells)
        if total > 0:
            cluster_score = (len(buys) - len(sells)) / total
        else:
            cluster_score = 0.0
        
        # Momentum: recent vs older activity
        mid_date = cutoff + timedelta(days=self.lookback_days // 2)
        recent_buys = len([t for t in buys if t["date"] >= mid_date])
        older_buys = len([t for t in buys if t["date"] < mid_date])
        recent_sells = len([t for t in sells if t["date"] >= mid_date])
        older_sells = len([t for t in sells if t["date"] < mid_date])
        
        recent_net = recent_buys - recent_sells
        older_net = older_buys - older_sells
        momentum = recent_net - older_net
        
        # Smart money: officers and directors weighted higher
        smart_weights = {"officer": 2.0, "director": 1.5, "10%owner": 1.0}
        smart_score = 0.0
        for trade in recent_trades:
            weight = smart_weights.get(trade.get("insider_type", ""), 1.0)
            if trade["type"] == "buy":
                smart_score += weight
            else:
                smart_score -= weight
        
        smart_money = smart_score / (total * 2)  # Normalize to [-1, 1]
        
        # Combined signal
        signal = 0.4 * cluster_score + 0.3 * np.sign(momentum) * min(abs(momentum) / 5, 1.0) + 0.3 * smart_money
        
        return InsiderSignal(
            cluster_score=float(np.clip(cluster_score, -1, 1)),
            momentum=float(np.clip(momentum / 10, -1, 1)),
            smart_money=float(np.clip(smart_money, -1, 1)),
            signal=float(np.clip(signal, -1, 1)),
        )


# ---------------------------------------------------------------------------
# Alternative Data Combiner
# ---------------------------------------------------------------------------


class AlternativeDataCombiner:
    """Combines multiple alternative data sources.
    
    Weights signals by data quality, freshness, and historical
    predictive power.
    
    Parameters
    ----------
    signal_weights : dict[str, float] | None
        Weights for each signal source. Default: equal weights.
    freshness_decay : float
        Exponential decay for signal freshness (default 0.94).
    min_sources : int
        Minimum sources for valid combined signal (default 2).
    """
    
    def __init__(
        self,
        signal_weights: Optional[dict[str, float]] = None,
        freshness_decay: float = 0.94,
        min_sources: int = 2,
    ) -> None:
        self.signal_weights = signal_weights or {
            "news": 0.25,
            "social": 0.20,
            "options": 0.30,
            "insider": 0.25,
        }
        self.freshness_decay = freshness_decay
        self.min_sources = min_sources
        self._signal_history: dict[str, list[tuple[datetime, float]]] = {}
        
    def update_signal(
        self,
        source: str,
        timestamp: datetime,
        signal: float,
    ) -> None:
        """Update signal history for a source.
        
        Parameters
        ----------
        source : str
            Signal source name.
        timestamp : datetime
            Signal timestamp.
        signal : float
            Signal value.
        """
        if source not in self._signal_history:
            self._signal_history[source] = []
        
        self._signal_history[source].append((timestamp, signal))
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=30)
        self._signal_history[source] = [
            (ts, sig) for ts, sig in self._signal_history[source]
            if ts >= cutoff
        ]
    
    def compute_combined_signal(
        self,
        current_signals: dict[str, float],
        current_time: datetime,
    ) -> float:
        """Compute combined alternative data signal.
        
        Parameters
        ----------
        current_signals : dict[str, float]
            Current signal values per source.
        current_time : datetime
            Current timestamp.
            
        Returns
        -------
        float
            Combined signal [-1, 1].
        """
        if len(current_signals) < self.min_sources:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for source, signal in current_signals.items():
            if source not in self.signal_weights:
                continue
            
            weight = self.signal_weights[source]
            
            # Apply freshness decay (assume signals are recent)
            weight *= self.freshness_decay
            
            weighted_sum += signal * weight
            weight_sum += weight
        
        if weight_sum > 0:
            combined = weighted_sum / weight_sum
        else:
            combined = 0.0
        
        return float(np.clip(combined, -1, 1))
    
    def compute_alpha_decay(
        self,
        source: str,
        holding_period: int = 5,
    ) -> float:
        """Estimate alpha decay for a signal source.
        
        Parameters
        ----------
        source : str
            Signal source.
        holding_period : int
            Holding period in days.
            
        Returns
        -------
        float
            Estimated alpha decay factor.
        """
        if source not in self._signal_history:
            return 1.0
        
        history = self._signal_history[source]
        if len(history) < 10:
            return 1.0
        
        # Compute autocorrelation of signals
        signals = [sig for _, sig in history]
        if len(signals) < holding_period + 1:
            return 1.0
        
        # Simple autocorrelation
        autocorr = np.corrcoef(signals[:-holding_period], signals[holding_period:])[0, 1]
        
        # Decay factor (higher autocorr = slower decay)
        decay = max(autocorr, 0.5)
        
        return float(decay)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_alternative_data_pipeline(
    sources: list[str],
    **kwargs,
) -> AlternativeDataCombiner:
    """Create alternative data pipeline.
    
    Parameters
    ----------
    sources : list[str]
        Data sources to include.
    **kwargs
        Additional arguments for combiner.
        
    Returns
    -------
    AlternativeDataCombiner
        Configured combiner.
    """
    weights = {source: 1.0 / len(sources) for source in sources}
    return AlternativeDataCombiner(signal_weights=weights, **kwargs)


def compute_sentiment_momentum(
    sentiment_series: NDArray,
    lookback: int = 5,
) -> NDArray:
    """Compute sentiment momentum.
    
    Parameters
    ----------
    sentiment_series : NDArray
        Sentiment scores over time.
    lookback : int
        Lookback for momentum calculation.
        
    Returns
    -------
    NDArray
        Sentiment momentum.
    """
    momentum = np.zeros_like(sentiment_series)
    for i in range(lookback, len(sentiment_series)):
        momentum[i] = sentiment_series[i] - sentiment_series[i - lookback]
    
    return momentum
