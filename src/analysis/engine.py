"""
NEXUS Signal Engine — Orchestrates all analysis agents into a consensus signal.

This is the brain. It runs Technical, Sentiment, and Risk agents,
weights their opinions, and produces a final trade signal.
"""

import asyncio
import concurrent.futures
import logging
from typing import List, Optional, Dict
from datetime import datetime, timezone

import ccxt

from ..models import TradeSignal, SignalSide, SignalStrength, AgentOpinion
from ..config import NexusConfig, ExchangeConfig
from .technical import TechnicalAnalyst
from .sentiment import SentimentAnalyst
from .risk import RiskManager
from .correlation import CorrelationMatrix
from .rate_limited import RateLimitedExchange, RateLimitConfig
from .ccxt_helpers import retry_exchange, safe_fetch_tickers

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
        self._exchanges: Dict[str, RateLimitedExchange] = {}
        self._correlation_matrix = CorrelationMatrix(config)
        self._risk_mgr: Optional[RiskManager] = None

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
        # Wrap with rate limiting (transparent proxy — analysts call fetch_ohlcv etc. as before)
        safe_exchange = RateLimitedExchange(exchange)
        self._exchanges[exchange_id] = safe_exchange
        return safe_exchange

    async def analyze_async(self, symbol: str, exchange_id: str = "binance",
                            timeframe: str = "1h") -> TradeSignal:
        """Async version of analyze with TTL timeout support"""
        ttl = self.config.ttl.get_timeout()
        
        try:
            return await asyncio.wait_for(
                self._analyze_full(symbol, exchange_id, timeframe),
                timeout=ttl,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Analysis TTL exceeded ({ttl}s) for {symbol}, degrading to technical-only")
            return await self._analyze_technical_only(symbol, exchange_id, timeframe)

    async def _analyze_full(self, symbol: str, exchange_id: str,
                            timeframe: str) -> TradeSignal:
        """Full multi-agent analysis (runs within TTL)"""
        signal = TradeSignal(
            symbol=symbol,
            exchange=exchange_id,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            exchange = self._get_exchange(exchange_id)
            
            # 1. Technical Analysis
            technical = TechnicalAnalyst(exchange)
            tech_opinion = technical.analyze(symbol, timeframe)
            signal.opinions.append(tech_opinion)
            
            if tech_opinion.indicators:
                signal.entry_price = tech_opinion.indicators.get("entry")
                signal.stop_loss = tech_opinion.indicators.get("stop_loss")
                signal.take_profit = tech_opinion.indicators.get("take_profit")

            # 2. Sentiment Analysis
            sentiment = SentimentAnalyst(exchange)
            sent_opinion = sentiment.analyze(symbol)
            signal.opinions.append(sent_opinion)

            # 3. Calculate weighted consensus
            signal = self._compute_consensus(signal)

            # 4. Risk Management (cached instance with correlation matrix)
            if self._risk_mgr is None:
                self._risk_mgr = RiskManager(self.config.risk)
                self._risk_mgr.set_correlation_matrix(self._correlation_matrix)
            risk_opinion = self._risk_mgr.analyze(signal, portfolio_value=10000)
            signal.opinions.append(risk_opinion)

            if not risk_opinion.indicators.get("approved", True):
                signal.side = SignalSide.HOLD
                signal.strength = SignalStrength.NEUTRAL
                signal.confidence *= 0.3
            else:
                signal.position_size_pct = risk_opinion.indicators.get("position_size_pct", 0)

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

    async def _analyze_technical_only(self, symbol: str, exchange_id: str,
                                      timeframe: str) -> TradeSignal:
        """Degraded analysis — technical only, used when TTL expires"""
        signal = TradeSignal(
            symbol=symbol,
            exchange=exchange_id,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
        )
        signal.metadata["degraded"] = True
        signal.metadata["degradation_reason"] = "ttl_expired"

        try:
            exchange = self._get_exchange(exchange_id)
            technical = TechnicalAnalyst(exchange)
            tech_opinion = technical.analyze(symbol, timeframe)
            signal.opinions.append(tech_opinion)
            
            if tech_opinion.indicators:
                signal.entry_price = tech_opinion.indicators.get("entry")
                signal.stop_loss = tech_opinion.indicators.get("stop_loss")
                signal.take_profit = tech_opinion.indicators.get("take_profit")

            # Technical-only consensus (reduced weight)
            signal = self._compute_consensus(signal)
            
            # Reduce confidence for degraded signal
            signal.confidence *= 0.7
            signal.metadata["confidence_penalty"] = "0.7x (degraded)"

        except Exception as e:
            logger.error(f"Technical-only analysis also failed for {symbol}: {e}")
            signal.side = SignalSide.HOLD
            signal.confidence = 0

        return signal

    def analyze(self, symbol: str, exchange_id: str = "binance",
                timeframe: str = "1h") -> TradeSignal:
        """Synchronous analyze — uses TTL if configured"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Already in async context — run in thread
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.analyze_async(symbol, exchange_id, timeframe)
                )
                return future.result(timeout=self.config.ttl.get_timeout() + 1)
        else:
            return asyncio.run(self.analyze_async(symbol, exchange_id, timeframe))

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
        """Scan top coins on an exchange and rank by signal strength (sync wrapper)"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — run in thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._scan_sync, exchange_id, base, top_n)
                return future.result(timeout=120)
        else:
            return self._scan_sync(exchange_id, base, top_n)

    async def scan_async(self, exchange_id: str = "binance", base: str = "USDT",
                         top_n: int = 10) -> List[TradeSignal]:
        """Async scan — analyzes top N coins in parallel using asyncio.gather."""
        try:
            exchange = self._get_exchange(exchange_id)

            # Get top coins by volume (single API call with retry)
            tickers = await asyncio.to_thread(safe_fetch_tickers, exchange)

            # Filter pairs by volume
            usdt_pairs = [
                (symbol, ticker["quoteVolume"])
                for symbol, ticker in tickers.items()
                if base in symbol and ticker.get("quoteVolume", 0) > 1_000_000
            ]
            usdt_pairs.sort(key=lambda x: x[1], reverse=True)

            # Analyze top N in parallel via thread pool
            loop = asyncio.get_running_loop()
            tasks = []
            for symbol, _ in usdt_pairs[:top_n]:
                tasks.append(loop.run_in_executor(None, self.analyze, symbol, exchange_id))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            signals = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Scan task failed: {result}")
                    continue
                if result.side != SignalSide.HOLD:
                    signals.append(result)

            signals.sort(key=lambda s: s.confidence, reverse=True)
            return signals

        except Exception as e:
            logger.error(f"Async scan failed: {e}")
            return []

    def _scan_sync(self, exchange_id: str, base: str, top_n: int) -> List[TradeSignal]:
        """Internal synchronous scan implementation."""
        try:
            exchange = self._get_exchange(exchange_id)

            # Get top coins by volume
            tickers = safe_fetch_tickers(exchange)

            # Filter pairs by volume
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
