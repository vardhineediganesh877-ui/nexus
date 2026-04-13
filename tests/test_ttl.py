"""Tests for Decision TTL feature"""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import StrategyType, TTLConfig, TradeSignal, SignalSide, SignalStrength
from src.config import NexusConfig, RiskConfig
from src.analysis.engine import SignalEngine


class TestTTLConfig:
    def test_default_swing(self):
        config = TTLConfig()
        assert config.get_timeout() == 5.0

    def test_scalp_ttl(self):
        config = TTLConfig(strategy_type=StrategyType.SCALP)
        assert config.get_timeout() == 0.5

    def test_long_ttl(self):
        config = TTLConfig(strategy_type=StrategyType.LONG)
        assert config.get_timeout() == 30.0

    def test_custom_strategy(self):
        config = TTLConfig(strategy_type=StrategyType.CUSTOM, custom_timeout=2.0)
        assert config.get_timeout() == 2.0

    def test_custom_strategy_no_timeout_falls_back(self):
        config = TTLConfig(strategy_type=StrategyType.CUSTOM, custom_timeout=None)
        assert config.get_timeout() == 5.0  # Falls back to SWING

    def test_ttl_zero_raises_error(self):
        config = TTLConfig(strategy_type=StrategyType.CUSTOM, custom_timeout=0.0)
        with pytest.raises(ValueError, match="TTL timeout must be >= 0.1s"):
            config.get_timeout()

    def test_ttl_negative_raises_error(self):
        config = TTLConfig(strategy_type=StrategyType.CUSTOM, custom_timeout=-1.0)
        with pytest.raises(ValueError, match="TTL timeout must be >= 0.1s"):
            config.get_timeout()


class TestSignalEngineTTL:
    def _make_config(self, strategy_type: StrategyType = StrategyType.SWING) -> NexusConfig:
        config = NexusConfig()
        config.ttl.strategy_type = strategy_type
        return config

    def test_analyze_sync_wrapper(self):
        """Verify sync analyze() works outside async context"""
        config = self._make_config()
        engine = SignalEngine(config)

        with patch.object(engine, '_get_exchange') as mock_ex:
            mock_exchange = MagicMock()
            mock_ex.return_value = mock_exchange

            with patch('src.analysis.engine.TechnicalAnalyst') as MockTech:
                mock_tech = MagicMock()
                from src.models import AgentOpinion
                mock_tech.analyze.return_value = AgentOpinion(
                    agent_name="technical",
                    signal=SignalSide.BUY,
                    strength=SignalStrength.BUY,
                    confidence=0.8,
                    reasoning="Test",
                    indicators={"entry": 50000.0, "stop_loss": 49000.0, "take_profit": 52000.0},
                )
                MockTech.return_value = mock_tech

                with patch('src.analysis.engine.SentimentAnalyst') as MockSent:
                    mock_sent = MagicMock()
                    mock_sent.analyze.return_value = AgentOpinion(
                        agent_name="sentiment",
                        signal=SignalSide.BUY,
                        strength=SignalStrength.BUY,
                        confidence=0.7,
                        reasoning="Bullish",
                    )
                    MockSent.return_value = mock_sent

                    signal = engine.analyze("BTC/USDT", "binance", "1h")
                    assert isinstance(signal, TradeSignal)

    def test_ttl_scalp_completes_fast(self):
        """Scalp strategy should use short TTL"""
        config = self._make_config(StrategyType.SCALP)
        assert config.ttl.get_timeout() == 0.5

    def test_degraded_signal_has_lower_confidence(self):
        """When TTL expires, degraded signal gets 0.7x confidence penalty"""
        engine = SignalEngine(self._make_config())

        async def run():
            with patch.object(engine, '_get_exchange') as mock_ex:
                mock_exchange = MagicMock()
                mock_ex.return_value = mock_exchange

                with patch('src.analysis.engine.TechnicalAnalyst') as MockTech:
                    mock_tech = MagicMock()
                    from src.models import AgentOpinion
                    mock_tech.analyze.return_value = AgentOpinion(
                        agent_name="technical",
                        signal=SignalSide.BUY,
                        strength=SignalStrength.BUY,
                        confidence=0.8,
                        reasoning="Test",
                    )
                    MockTech.return_value = mock_tech

                    signal = await engine._analyze_technical_only("BTC/USDT", "binance", "1h")
                    assert signal.metadata.get("degraded") is True
                    assert signal.metadata.get("degradation_reason") == "ttl_expired"
                    assert signal.metadata.get("confidence_penalty") == "0.7x (degraded)"

        asyncio.run(run())

    def test_ttl_timeout_degrades_to_technical_only(self):
        """When full analysis times out, degrade to technical-only"""
        config = self._make_config(StrategyType.SWING)
        config.ttl.timeout_seconds = 0.1
        engine = SignalEngine(config)

        async def slow_analysis(*args, **kwargs):
            await asyncio.sleep(10)  # Way longer than TTL
            return TradeSignal(symbol="BTC/USDT")

        async def run():
            with patch.object(engine, '_analyze_full', side_effect=slow_analysis):
                with patch.object(engine, '_analyze_technical_only') as mock_degraded:
                    from src.models import AgentOpinion
                    mock_degraded.return_value = TradeSignal(
                        symbol="BTC/USDT",
                        metadata={"degraded": True},
                    )
                    signal = await engine.analyze_async("BTC/USDT", "binance", "1h")
                    mock_degraded.assert_called_once()

        asyncio.run(run())
