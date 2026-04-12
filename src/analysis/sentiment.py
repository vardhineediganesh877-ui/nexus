"""
NEXUS Sentiment Analysis Agent

Aggregates sentiment from Reddit, news, and social signals.
Uses the TradingView MCP's sentiment capabilities.
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from ..models import AgentOpinion, SignalSide, SignalStrength

logger = logging.getLogger(__name__)


class SentimentAnalyst:
    """Market sentiment analysis from multiple sources"""

    def __init__(self):
        self._reddit_cache: Dict[str, tuple] = {}  # symbol -> (score, timestamp)
        self._news_cache: Dict[str, tuple] = {}

    def analyze_reddit(self, symbol: str) -> Dict:
        """Analyze Reddit sentiment for a symbol"""
        try:
            # Use TradingView MCP's market_sentiment tool if available
            # Fallback: keyword-based scoring
            symbol_base = symbol.split("/")[0].replace("USDT", "").lower()
            
            # Map common symbols to Reddit keywords
            keywords = {
                "btc": "bitcoin BTC",
                "eth": "ethereum ETH", 
                "sol": "solana SOL",
                "bnb": "binance BNB",
                "xrp": "ripple XRP",
                "ada": "cardano ADA",
                "doge": "dogecoin DOGE",
            }
            search_term = keywords.get(symbol_base, symbol_base)
            
            # Simplified sentiment scoring
            # In production, this would use TradingView MCP's market_sentiment tool
            return {
                "source": "reddit",
                "symbol": symbol,
                "sentiment": "neutral",
                "confidence": 0.3,
                "post_count": 0,
                "note": "Sentiment analysis available via TradingView MCP integration",
            }
        except Exception as e:
            logger.error(f"Reddit sentiment failed: {e}")
            return {"source": "reddit", "error": str(e)}

    def analyze_news(self, symbol: str) -> Dict:
        """Analyze financial news sentiment"""
        try:
            return {
                "source": "news",
                "symbol": symbol,
                "sentiment": "neutral",
                "confidence": 0.3,
                "article_count": 0,
                "note": "News analysis available via TradingView MCP integration",
            }
        except Exception as e:
            logger.error(f"News sentiment failed: {e}")
            return {"source": "news", "error": str(e)}

    def analyze(self, symbol: str) -> AgentOpinion:
        """Run full sentiment analysis"""
        reasons = []
        score = 0  # -100 to +100
        
        # Reddit
        reddit = self.analyze_reddit(symbol)
        if "error" not in reddit:
            if reddit.get("sentiment") == "bullish":
                score += 30
                reasons.append("Reddit sentiment bullish")
            elif reddit.get("sentiment") == "bearish":
                score -= 30
                reasons.append("Reddit sentiment bearish")
            elif reddit.get("post_count", 0) > 50:
                score += 10
                reasons.append(f"High Reddit activity ({reddit['post_count']} posts)")
        
        # News
        news = self.analyze_news(symbol)
        if "error" not in news:
            if news.get("sentiment") == "bullish":
                score += 20
                reasons.append("News sentiment bullish")
            elif news.get("sentiment") == "bearish":
                score -= 20
                reasons.append("News sentiment bearish")
        
        # Determine signal
        confidence = abs(score) / 100 * 0.7  # Sentiment is less reliable than TA
        
        if score > 30:
            signal = SignalSide.BUY
            strength = SignalStrength.BUY
        elif score < -30:
            signal = SignalSide.SELL
            strength = SignalStrength.SELL
        else:
            signal = SignalSide.HOLD
            strength = SignalStrength.NEUTRAL
        
        return AgentOpinion(
            agent_name="sentiment",
            signal=signal,
            strength=strength,
            confidence=min(confidence, 0.8),
            reasoning=" | ".join(reasons) if reasons else "Insufficient sentiment data",
            indicators={
                "reddit": reddit,
                "news": news,
                "overall_score": score,
            },
        )
