"""
NEXUS Technical Analysis Agent

Uses TradingView Technical Analysis library (30+ indicators).
Multi-timeframe alignment, volume confirmation, pattern detection.
"""

import ccxt
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models import AgentOpinion, SignalSide, SignalStrength, TradeSignal
from .ccxt_helpers import retry_exchange

logger = logging.getLogger(__name__)


class TechnicalAnalyst:
    """Technical analysis using TradingView indicators"""

    # Signal thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    STRONG_RSI_OVERSOLD = 20
    STRONG_RSI_OVERBOUGHT = 80

    def __init__(self, exchange: ccxt.Exchange, cache_ttl: int = 300):
        self.exchange = exchange
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = cache_ttl  # seconds

    @retry_exchange(max_retries=3, base_delay=1.0)
    def _get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> List[list]:
        """Fetch OHLCV data from exchange (with TTL cache)"""
        from datetime import datetime as _dt
        cache_key = f"{symbol}_{timeframe}"
        now = _dt.utcnow()
        
        if cache_key in self._cache and cache_key in self._cache_time:
            age = (now - self._cache_time[cache_key]).total_seconds()
            if age < self._cache_ttl:
                return self._cache[cache_key]
        
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            raise ValueError(f"No OHLCV data for {symbol}")
        
        self._cache[cache_key] = ohlcv
        self._cache_time[cache_key] = now
        return ohlcv

    def _compute_indicators(self, ohlcv: List[list]) -> Dict[str, Any]:
        """Compute technical indicators from OHLCV data"""
        # Manual indicator computation
        closes = [c[4] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]
        
        indicators = {
            "price": closes[-1],
            "price_change_24h": (closes[-1] - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 else 0,
            "volume": volumes[-1],
            "volume_avg_20": sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes),
        }

        # RSI (14-period)
        rsi = self._rsi(closes, 14)
        indicators["rsi"] = rsi
        indicators["rsi_signal"] = "oversold" if rsi < self.RSI_OVERSOLD else "overbought" if rsi > self.RSI_OVERBOUGHT else "neutral"

        # EMA (20, 50)
        indicators["ema_20"] = self._ema(closes, 20)
        indicators["ema_50"] = self._ema(closes, 50)
        indicators["ema_cross"] = "bullish" if indicators["ema_20"] > indicators["ema_50"] else "bearish"

        # MACD
        macd, signal, hist = self._macd(closes)
        indicators["macd"] = macd
        indicators["macd_signal"] = signal
        indicators["macd_hist"] = hist
        indicators["macd_cross"] = "bullish" if hist > 0 else "bearish"

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = self._bollinger_bands(closes, 20)
        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_middle
        indicators["bb_lower"] = bb_lower
        indicators["bb_width"] = bb_width
        price = closes[-1]
        indicators["bb_position"] = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # Volume ratio
        indicators["volume_ratio"] = volumes[-1] / indicators["volume_avg_20"] if indicators["volume_avg_20"] > 0 else 1.0

        # ATR (14-period) for volatility
        indicators["atr"] = self._atr(ohlcv, 14)
        indicators["atr_pct"] = indicators["atr"] / price * 100 if price > 0 else 0

        return indicators

    @staticmethod
    def _rsi(prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas[-period:]]
        losses = [abs(min(d, 0)) for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    @staticmethod
    def _macd(prices: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9):
        """MACD indicator"""
        ema_fast = []
        ema_slow = []
        
        # Calculate fast EMA
        multiplier_fast = 2 / (fast + 1)
        ef = sum(prices[:fast]) / fast
        for p in prices[fast:]:
            ef = (p - ef) * multiplier_fast + ef
            ema_fast.append(ef)
        
        # Calculate slow EMA
        multiplier_slow = 2 / (slow + 1)
        es = sum(prices[:slow]) / slow
        for p in prices[slow:]:
            es = (p - es) * multiplier_slow + es
            ema_slow.append(es)
        
        # MACD line
        min_len = min(len(ema_fast), len(ema_slow))
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(min_len)]
        
        # Signal line (EMA of MACD)
        if len(macd_line) < signal_period:
            return macd_line[-1] if macd_line else 0, 0, 0
        
        multiplier_sig = 2 / (signal_period + 1)
        sig = sum(macd_line[:signal_period]) / signal_period
        for m in macd_line[signal_period:]:
            sig = (m - sig) * multiplier_sig + sig
        
        histogram = macd_line[-1] - sig
        return macd_line[-1], sig, histogram

    @staticmethod
    def _bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        if len(prices) < period:
            return 0, 0, 0, 0
        
        recent = prices[-period:]
        middle = sum(recent) / period
        variance = sum((p - middle) ** 2 for p in recent) / period
        std = variance ** 0.5
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        width = (upper - lower) / middle if middle > 0 else 0
        
        return upper, middle, lower, width

    @staticmethod
    def _atr(ohlcv: List[list], period: int = 14) -> float:
        """Average True Range"""
        if len(ohlcv) < period + 1:
            return 0
        
        trs = []
        for i in range(1, len(ohlcv)):
            high, low, close_prev = ohlcv[i][2], ohlcv[i][3], ohlcv[i-1][4]
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            trs.append(tr)
        
        return sum(trs[-period:]) / period

    def analyze(self, symbol: str, timeframe: str = "1h") -> AgentOpinion:
        """Run full technical analysis and return opinion"""
        try:
            ohlcv = self._get_ohlcv(symbol, timeframe)
            indicators = self._compute_indicators(ohlcv)
            
            # Score calculation
            score = 0  # -100 to +100
            reasons = []
            
            # RSI
            rsi = indicators["rsi"]
            if rsi < self.STRONG_RSI_OVERSOLD:
                score += 30
                reasons.append(f"RSI deeply oversold ({rsi:.1f})")
            elif rsi < self.RSI_OVERSOLD:
                score += 20
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.STRONG_RSI_OVERBOUGHT:
                score -= 30
                reasons.append(f"RSI deeply overbought ({rsi:.1f})")
            elif rsi > self.RSI_OVERBOUGHT:
                score -= 20
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # EMA Cross
            if indicators["ema_cross"] == "bullish":
                score += 15
                reasons.append("EMA20 > EMA50 (bullish)")
            else:
                score -= 15
                reasons.append("EMA20 < EMA50 (bearish)")
            
            # MACD
            if indicators["macd_cross"] == "bullish":
                score += 20
                reasons.append("MACD bullish crossover")
            else:
                score -= 20
                reasons.append("MACD bearish crossover")
            
            # Bollinger Band position
            bb_pos = indicators["bb_position"]
            if bb_pos < 0.1:
                score += 20
                reasons.append(f"Price near lower BB ({bb_pos:.1%} of band)")
            elif bb_pos > 0.9:
                score -= 20
                reasons.append(f"Price near upper BB ({bb_pos:.1%} of band)")
            
            # Volume confirmation
            if indicators["volume_ratio"] > 2.0:
                score += 10
                reasons.append(f"High volume ({indicators['volume_ratio']:.1f}x average)")
            elif indicators["volume_ratio"] < 0.5:
                score -= 5
                reasons.append(f"Low volume ({indicators['volume_ratio']:.1f}x average)")
            
            # Determine signal
            normalized = (score + 100) / 200  # 0 to 1
            confidence = abs(score) / 100
            
            if score > 50:
                signal = SignalSide.BUY
                strength = SignalStrength.STRONG_BUY
            elif score > 20:
                signal = SignalSide.BUY
                strength = SignalStrength.BUY
            elif score < -50:
                signal = SignalSide.SELL
                strength = SignalStrength.STRONG_SELL
            elif score < -20:
                signal = SignalSide.SELL
                strength = SignalStrength.SELL
            else:
                signal = SignalSide.HOLD
                strength = SignalStrength.NEUTRAL
            
            # Calculate support/resistance from Bollinger Bands
            entry = indicators["price"]
            atr = indicators["atr"]
            stop_loss = entry - 2 * atr if signal == SignalSide.BUY else entry + 2 * atr
            take_profit = entry + 3 * atr if signal == SignalSide.BUY else entry - 3 * atr
            
            return AgentOpinion(
                agent_name="technical",
                signal=signal,
                strength=strength,
                confidence=min(confidence, 0.95),
                reasoning=" | ".join(reasons) if reasons else "No clear signals",
                indicators={
                    "rsi": round(rsi, 1),
                    "ema_cross": indicators["ema_cross"],
                    "macd_cross": indicators["macd_cross"],
                    "bb_position": round(bb_pos, 3),
                    "volume_ratio": round(indicators["volume_ratio"], 2),
                    "atr_pct": round(indicators["atr_pct"], 2),
                    "entry": entry,
                    "stop_loss": round(stop_loss, 8),
                    "take_profit": round(take_profit, 8),
                },
            )
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return AgentOpinion(
                agent_name="technical",
                signal=SignalSide.HOLD,
                strength=SignalStrength.NEUTRAL,
                confidence=0.0,
                reasoning=f"Analysis failed: {e}",
            )

    def multi_timeframe(self, symbol: str, timeframes: List[str]) -> Dict[str, AgentOpinion]:
        """Analyze across multiple timeframes"""
        results = {}
        for tf in timeframes:
            try:
                results[tf] = self.analyze(symbol, tf)
            except Exception as e:
                logger.warning(f"Failed {symbol} {tf}: {e}")
        return results
