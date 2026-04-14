"""
NEXUS Sentiment Analysis Agent

Uses real market data (price momentum, volume anomalies, ticker breadth)
as sentiment proxy. No external API keys needed — pure on-chain/market data.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

import ccxt

from ..models import AgentOpinion, SignalSide, SignalStrength
from .ccxt_helpers import retry_exchange, safe_fetch_tickers, safe_fetch_ohlcv

logger = logging.getLogger(__name__)


class SentimentAnalyst:
    """Market sentiment analysis from price action + volume signals"""

    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        self._exchange = exchange
        self._ticker_cache: Dict[str, tuple] = {}  # symbol -> (tickers_dict, timestamp)
        self._ohlcv_cache: Dict[str, tuple] = {}    # symbol -> (ohlcv_data, timestamp)

    def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is not None:
            return self._exchange
        # Fallback: create a default binance instance
        self._exchange = ccxt.binance({"rateLimit": 1000})
        return self._exchange

    @retry_exchange(max_retries=3, base_delay=1.5)
    def _fetch_tickers(self) -> Dict:
        """Fetch all tickers with caching (5 min TTL)"""
        now = datetime.utcnow()
        if self._ticker_cache:
            ts = self._ticker_cache.get("_timestamp")
            if ts and (now - ts).seconds < 300:
                return self._ticker_cache.get("_data", {})

        exchange = self._get_exchange()
        data = exchange.fetch_tickers()
        self._ticker_cache = {"_data": data, "_timestamp": now}
        return data

    @retry_exchange(max_retries=3, base_delay=1.0)
    def _fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 24) -> List:
        """Fetch OHLCV data with caching (5 min TTL)"""
        cache_key = f"{symbol}_{timeframe}"
        now = datetime.utcnow()

        if cache_key in self._ohlcv_cache:
            ts = self._ohlcv_cache[cache_key][1]
            if (now - ts).seconds < 300:
                return self._ohlcv_cache[cache_key][0]

        exchange = self._get_exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        self._ohlcv_cache[cache_key] = (ohlcv, now)
        return ohlcv

    def _analyze_momentum(self, symbol: str) -> Dict:
        """Analyze price momentum from OHLCV data"""
        ohlcv = self._fetch_ohlcv(symbol, "1h", 24)
        if not ohlcv or len(ohlcv) < 12:
            return {"sentiment": "neutral", "confidence": 0.2, "reason": "insufficient_data"}

        closes = [c[4] for c in ohlcv]  # Close prices
        volumes = [c[5] for c in ohlcv]  # Volumes

        # Price change over windows
        current = closes[-1]
        change_4h = (current - closes[-4]) / closes[-4] if len(closes) >= 5 else 0
        change_12h = (current - closes[-12]) / closes[-12] if len(closes) >= 13 else 0
        change_24h = (current - closes[0]) / closes[0] if len(closes) >= 2 else 0

        # Volume analysis: compare recent vs average
        avg_vol = sum(volumes[:-6]) / max(len(volumes[:-6]), 1)
        recent_vol = sum(volumes[-6:]) / max(len(volumes[-6:]), 1)
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # RSI-like momentum (simplified)
        gains = []
        losses = []
        for i in range(1, min(len(closes), 14)):
            diff = closes[-i] - closes[-i - 1]
            if diff > 0:
                gains.append(diff)
            else:
                losses.append(abs(diff))

        avg_gain = sum(gains) / max(len(gains), 1)
        avg_loss = sum(losses) / max(len(losses), 1)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # Scoring
        score = 0  # -100 to +100

        # Momentum signals
        if change_4h > 0.02:
            score += 20
        elif change_4h < -0.02:
            score -= 20
        elif change_4h > 0.005:
            score += 8
        elif change_4h < -0.005:
            score -= 8

        if change_12h > 0.05:
            score += 15
        elif change_12h < -0.05:
            score -= 15

        if change_24h > 0.10:
            score += 10
        elif change_24h < -0.10:
            score -= 10

        # Volume confirmation (surge = conviction)
        if vol_surge > 2.0:
            # Volume surge amplifies the direction
            score += int(10 * (1 if change_4h > 0 else -1))
        elif vol_surge > 1.5:
            score += int(5 * (1 if change_4h > 0 else -1))

        # RSI extremes
        if rsi > 70:
            score -= 15  # Overbought = bearish sentiment
        elif rsi < 30:
            score += 15  # Oversold = bullish sentiment (contrarian)
        elif rsi > 55:
            score += 5
        elif rsi < 45:
            score -= 5

        # Determine sentiment
        if score > 25:
            sentiment = "bullish"
        elif score < -25:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        confidence = min(abs(score) / 80, 0.85)  # Cap at 0.85

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "score": score,
            "change_4h": round(change_4h, 4),
            "change_12h": round(change_12h, 4),
            "change_24h": round(change_24h, 4),
            "volume_surge": round(vol_surge, 2),
            "rsi": round(rsi, 1),
            "reason": f"momentum_{sentiment}_rsi{rsi:.0f}",
        }

    def _analyze_breadth(self, symbol: str) -> Dict:
        """Analyze market breadth — how the symbol compares to overall market"""
        try:
            tickers = self._fetch_tickers()
            if not tickers:
                return {"sentiment": "neutral", "confidence": 0.1, "reason": "no_ticker_data"}

            # Get symbol ticker
            sym_ticker = tickers.get(symbol)
            if not sym_ticker:
                return {"sentiment": "neutral", "confidence": 0.1, "reason": "symbol_not_found"}

            sym_change = sym_ticker.get("percentage", 0) or 0

            # Market breadth: what % of USDT pairs are up?
            up_count = 0
            total_count = 0
            for sym, t in tickers.items():
                if "USDT" in sym and sym != symbol:
                    pct = t.get("percentage", 0) or 0
                    total_count += 1
                    if pct > 0:
                        up_count += 1

            breadth = up_count / total_count if total_count > 0 else 0.5

            score = 0
            # Above-average performer = bullish sentiment
            if sym_change > 0 and breadth > 0.5:
                score += 15  # Rising in a rising market
            elif sym_change > 0 and breadth < 0.5:
                score += 25  # Rising while market falls = strong relative strength
            elif sym_change < 0 and breadth > 0.5:
                score -= 25  # Falling while market rises = weakness
            elif sym_change < 0 and breadth < 0.5:
                score -= 10  # Falling with market

            # Overall market sentiment
            if breadth > 0.65:
                score += 10  # Risk-on environment
            elif breadth < 0.35:
                score -= 10  # Risk-off environment

            if score > 15:
                sentiment = "bullish"
            elif score < -15:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            confidence = min(abs(score) / 60, 0.6)

            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "score": score,
                "symbol_change_pct": round(sym_change, 2),
                "market_breadth": round(breadth, 3),
                "reason": f"breadth_{sentiment}_pct{sym_change:.1f}%",
            }
        except Exception as e:
            logger.debug(f"Breadth analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.1, "reason": str(e)}

    def analyze(self, symbol: str) -> AgentOpinion:
        """Run full sentiment analysis using price momentum + breadth"""
        reasons = []
        total_score = 0

        # 1. Price Momentum (weighted 60%)
        momentum = self._analyze_momentum(symbol)
        if momentum.get("confidence", 0) > 0.1:
            total_score += momentum.get("score", 0) * 0.6
            reasons.append(
                f"Momentum: {momentum.get('sentiment')} "
                f"(4h: {momentum.get('change_4h', 0):+.1%}, "
                f"RSI: {momentum.get('rsi', 0):.0f}, "
                f"Vol surge: {momentum.get('volume_surge', 0):.1f}x)"
            )

        # 2. Market Breadth (weighted 40%)
        breadth = self._analyze_breadth(symbol)
        if breadth.get("confidence", 0) > 0.1:
            total_score += breadth.get("score", 0) * 0.4
            reasons.append(
                f"Breadth: {breadth.get('sentiment')} "
                f"(change: {breadth.get('symbol_change_pct', 0):+.1f}%, "
                f"market: {breadth.get('market_breadth', 0):.0%} up)"
            )

        # Final signal
        if total_score > 20:
            signal = SignalSide.BUY
            strength = SignalStrength.STRONG_BUY if total_score > 40 else SignalStrength.BUY
        elif total_score < -20:
            signal = SignalSide.SELL
            strength = SignalStrength.STRONG_SELL if total_score < -40 else SignalStrength.SELL
        else:
            signal = SignalSide.HOLD
            strength = SignalStrength.NEUTRAL

        # Confidence based on score magnitude, capped
        confidence = min(abs(total_score) / 80, 0.80)

        return AgentOpinion(
            agent_name="sentiment",
            signal=signal,
            strength=strength,
            confidence=round(confidence, 3),
            reasoning=" | ".join(reasons) if reasons else "Neutral sentiment — no strong signals",
            indicators={
                "momentum": momentum,
                "breadth": breadth,
                "total_score": round(total_score, 1),
            },
        )
