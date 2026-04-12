"""
NEXUS Signal Engine — Orchestrates all analysis agents into a consensus signal.

This is the brain. It runs Technical, Sentiment, and Risk agents,
weights their opinions, and produces a final trade signal.
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime

import ccxt

from ..models import TradeSignal, SignalSide, SignalStrength, AgentOpinion
from ..config import NexusConfig, ExchangeConfig
from .technical import TechnicalAnalyst
from .sentiment import SentimentAnalyst
from .risk import RiskManager

logger = logging.getLogger(__name__)


class SignalEngine:
    """Multi-agent signal consensus engine"""

    # Agent weights for consensus calculation
    AGENT_WEIGHTS = {
        "technical": 0.40,   # Technical analysis is most reliable
        "sentiment": 0.20,   # Sentiment is noisy but directional
        "fundamental": 0.15, # Fundamentals for conviction
        "risk": 0.25,        # Risk has veto power
    }

    def __init__(self, config: NexusConfig):
        self.config = config
        self._exchanges: Dict[str, ccxt.Exchange] = {}

    def _get_exchange(self, exchange_id: str = "binance") -> ccxt.Exchange:
        """Get or create exchange instance"""
        if exchange_id in self._exchanges:
            return self._exchanges[exchange_id]
        
        exchange_class = getattr(ccxt, exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Exchange {exchange_id} not supported by CCXT")
        
        ex_config = self.config.exchanges.get(exchange_id, ExchangeConfig(id=exchange_id))
        
        kwargs = {"rateLimit": 1000}
        if ex_config.is_configured:
            kwargs["apiKey"] = ex_config.api_key
            kwargs["secret"] = ex_config.api_secret
            if ex_config.passphrase:
                kwargs["password"] = ex_config.passphrase
            if ex_config.testnet:
                kwargs["options"] = {"defaultType": "spot"}
                if exchange_id == "binance":
                    kwargs["urls"] = {"api": {"public": "https://testnet.binance.vision", "private": "https://testnet.binance.vision"}}
        
        exchange = exchange_class(kwargs)
        self._exchanges[exchange_id] = exchange
        return exchange

    def analyze(self, symbol: str, exchange_id: str = "binance",
                timeframe: str = "1h") -> TradeSignal:
        """Run full multi-agent analysis on a symbol"""
        
        signal = TradeSignal(
            symbol=symbol,
            exchange=exchange_id,
            timeframe=timeframe,
            timestamp=datetime.utcnow(),
        )

        try:
            exchange = self._get_exchange(exchange_id)
            
            # 1. Technical Analysis (heaviest weight)
            technical = TechnicalAnalyst(exchange)
            tech_opinion = technical.analyze(symbol, timeframe)
            signal.opinions.append(tech_opinion)
            
            # Extract entry/SL/TP from technical analysis
            if tech_opinion.indicators:
                signal.entry_price = tech_opinion.indicators.get("entry")
                signal.stop_loss = tech_opinion.indicators.get("stop_loss")
                signal.take_profit = tech_opinion.indicators.get("take_profit")

            # 2. Sentiment Analysis
            sentiment = SentimentAnalyst()
            sent_opinion = sentiment.analyze(symbol)
            signal.opinions.append(sent_opinion)

            # 3. Calculate weighted consensus
            signal = self._compute_consensus(signal)

            # 4. Risk Management (veto power)
            risk_mgr = RiskManager(self.config.risk)
            risk_opinion = risk_mgr.analyze(signal, portfolio_value=10000)
            signal.opinions.append(risk_opinion)

            # Risk can veto the signal
            if not risk_opinion.indicators.get("approved", True):
                signal.side = SignalSide.HOLD
                signal.strength = SignalStrength.NEUTRAL
                signal.confidence *= 0.3  # Heavily reduce confidence
            else:
                signal.position_size_pct = risk_opinion.indicators.get("position_size_pct", 0)

            # Calculate risk/reward
            if signal.entry_price and signal.stop_loss and signal.take_profit:
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.entry_price)
                signal.risk_reward_ratio = reward / risk if risk > 0 else 0

            logger.info(
                f"Signal {signal.id}: {signal.symbol} {signal.side.value} "
                f"conf={signal.confidence:.0%} strength={signal.strength.value}"
            )

        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            signal.side = SignalSide.HOLD
            signal.confidence = 0

        return signal

    def _compute_consensus(self, signal: TradeSignal) -> TradeSignal:
        """Weight agent opinions into consensus signal"""
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for opinion in signal.opinions:
            weight = self.AGENT_WEIGHTS.get(opinion.agent_name, 0.1)
            total_weight += weight

            if opinion.signal == SignalSide.BUY:
                buy_score += weight * opinion.confidence
            elif opinion.signal == SignalSide.SELL:
                sell_score += weight * opinion.confidence
            # HOLD contributes nothing

        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        # Determine final signal
        if buy_score > sell_score and buy_score > 0.3:
            signal.side = SignalSide.BUY
            signal.confidence = buy_score
            signal.strength = (
                SignalStrength.STRONG_BUY if buy_score > 0.7
                else SignalStrength.BUY
            )
        elif sell_score > buy_score and sell_score > 0.3:
            signal.side = SignalSide.SELL
            signal.confidence = sell_score
            signal.strength = (
                SignalStrength.STRONG_SELL if sell_score > 0.7
                else SignalStrength.SELL
            )
        else:
            signal.side = SignalSide.HOLD
            signal.confidence = max(buy_score, sell_score)
            signal.strength = SignalStrength.NEUTRAL

        return signal

    def scan(self, exchange_id: str = "binance", base: str = "USDT",
             top_n: int = 10) -> List[TradeSignal]:
        """Scan top coins on an exchange and rank by signal strength"""
        try:
            exchange = self._get_exchange(exchange_id)
            
            # Get top coins by volume
            tickers = exchange.fetch_tickers()
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if base in symbol and ticker.get("quoteVolume", 0) > 1_000_000:
                    usdt_pairs.append((symbol, ticker["quoteVolume"]))
            
            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Analyze top N
            signals = []
            for symbol, _ in usdt_pairs[:top_n]:
                try:
                    sig = self.analyze(symbol, exchange_id)
                    if sig.side != SignalSide.HOLD:
                        signals.append(sig)
                except Exception as e:
                    logger.warning(f"Scan failed for {symbol}: {e}")
            
            # Sort by confidence
            signals.sort(key=lambda s: s.confidence, reverse=True)
            return signals

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []
