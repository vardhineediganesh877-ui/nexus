"""Tests for NEXUS data models."""
import pytest
from datetime import datetime
from src.models import (
    SignalSide, SignalStrength, AgentOpinion, TradeSignal, Trade, BacktestResult,
)


class TestSignalSide:
    def test_values(self):
        assert SignalSide.BUY.value == "buy"
        assert SignalSide.SELL.value == "sell"
        assert SignalSide.HOLD.value == "hold"


class TestSignalStrength:
    def test_values(self):
        assert SignalStrength.STRONG_BUY.value == "strong_buy"
        assert SignalStrength.SELL.value == "sell"
        assert SignalStrength.NEUTRAL.value == "neutral"


class TestAgentOpinion:
    def test_creation(self):
        op = AgentOpinion(
            agent_name="technical", signal=SignalSide.BUY,
            strength=SignalStrength.STRONG_BUY, confidence=0.85,
            reasoning="RSI oversold",
        )
        assert op.agent_name == "technical"
        assert op.confidence == 0.85

    def test_to_dict(self):
        op = AgentOpinion(
            agent_name="sentiment", signal=SignalSide.SELL,
            strength=SignalStrength.SELL, confidence=0.6,
            reasoning="bearish", indicators={"reddit": 0.3},
        )
        d = op.to_dict()
        assert d["agent"] == "sentiment"
        assert d["signal"] == "sell"
        assert d["confidence"] == 0.6
        assert d["indicators"]["reddit"] == 0.3
        assert "timestamp" in d


class TestTradeSignal:
    def _make_signal(self):
        op1 = AgentOpinion("technical", SignalSide.BUY, SignalStrength.BUY, 0.8, "RSI oversold")
        op2 = AgentOpinion("sentiment", SignalSide.BUY, SignalStrength.BUY, 0.7, "Positive sentiment")
        return TradeSignal(
            symbol="BTC/USDT", exchange="binance", timeframe="1h",
            side=SignalSide.BUY, strength=SignalStrength.BUY,
            confidence=0.75, opinions=[op1, op2],
            entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0,
            position_size_pct=5.0, risk_reward_ratio=1.5,
        )

    def test_creation(self):
        sig = self._make_signal()
        assert sig.symbol == "BTC/USDT"
        assert len(sig.opinions) == 2
        assert len(sig.id) == 8

    def test_agents_agree(self):
        sig = self._make_signal()
        assert sig.agents_agree is True

    def test_agents_disagree(self):
        op1 = AgentOpinion("technical", SignalSide.BUY, SignalStrength.BUY, 0.8, "")
        op2 = AgentOpinion("sentiment", SignalSide.SELL, SignalStrength.SELL, 0.7, "")
        sig = TradeSignal(opinions=[op1, op2])
        assert sig.agents_agree is False

    def test_to_dict(self):
        sig = self._make_signal()
        d = sig.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "buy"
        assert d["confidence"] == 0.75
        assert len(d["opinions"]) == 2
        assert d["agents_agree"] is True

    def test_to_telegram(self):
        sig = self._make_signal()
        text = sig.to_telegram()
        assert "BTC/USDT" in text
        assert "BUY" in text
        assert "$50,000.0000" in text

    def test_technical_score(self):
        sig = self._make_signal()
        assert sig.technical_score == pytest.approx(0.8)

    def test_sentiment_score(self):
        sig = self._make_signal()
        assert sig.sentiment_score == pytest.approx(0.7)


class TestTrade:
    def test_creation(self):
        t = Trade(symbol="ETH/USDT", exchange="binance", side=SignalSide.BUY,
                  entry_price=3000.0, quantity=1.5)
        assert t.symbol == "ETH/USDT"
        assert t.status == "open"
        assert t.paper is True

    def test_to_dict(self):
        t = Trade(symbol="ETH/USDT", exchange="binance", side=SignalSide.SELL,
                  entry_price=3000.0, quantity=2.0, pnl=100.0, status="closed")
        d = t.to_dict()
        assert d["symbol"] == "ETH/USDT"
        assert d["side"] == "sell"
        assert d["pnl"] == 100.0
        assert d["status"] == "closed"
        assert d["timestamp_closed"] is None


class TestBacktestResult:
    def test_grade_a_plus(self):
        r = BacktestResult("BTC", "rsi", "1y", sharpe_ratio=2.5, max_drawdown_pct=5)
        assert r.grade == "A+"

    def test_grade_a(self):
        r = BacktestResult("BTC", "rsi", "1y", sharpe_ratio=1.6, max_drawdown_pct=12)
        assert r.grade == "A"

    def test_grade_b(self):
        r = BacktestResult("BTC", "rsi", "1y", sharpe_ratio=1.2, win_rate=60)
        assert r.grade == "B"

    def test_grade_c(self):
        r = BacktestResult("BTC", "rsi", "1y", sharpe_ratio=0.6)
        assert r.grade == "C"

    def test_grade_d(self):
        r = BacktestResult("BTC", "rsi", "1y", sharpe_ratio=0.1, total_return_pct=5)
        assert r.grade == "D"

    def test_grade_f(self):
        r = BacktestResult("BTC", "rsi", "1y", total_return_pct=-10, sharpe_ratio=-0.5)
        assert r.grade == "F"

    def test_to_dict(self):
        r = BacktestResult("BTC", "rsi", "1y", total_return_pct=42.5, sharpe_ratio=1.3)
        d = r.to_dict()
        assert d["symbol"] == "BTC"
        assert d["total_return_pct"] == 42.5
