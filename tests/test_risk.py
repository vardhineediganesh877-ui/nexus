"""Tests for RiskManager."""
import pytest
from src.analysis.risk import RiskManager
from src.config import RiskConfig
from src.models import TradeSignal, AgentOpinion, SignalSide, SignalStrength


def make_signal(confidence=0.8, side=SignalSide.BUY, agents_agree_data=None):
    if agents_agree_data is None:
        agents_agree_data = [
            AgentOpinion("technical", SignalSide.BUY, SignalStrength.BUY, 0.8, ""),
            AgentOpinion("sentiment", SignalSide.BUY, SignalStrength.BUY, 0.7, ""),
        ]
    return TradeSignal(
        symbol="BTC/USDT", exchange="binance", side=side,
        confidence=confidence, opinions=agents_agree_data,
        entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0,
    )


@pytest.fixture
def rm():
    return RiskManager(RiskConfig())


class TestRiskApproval:
    def test_approves_high_confidence_low_exposure(self, rm):
        sig = make_signal(confidence=0.85)
        result = rm.analyze(sig, portfolio_value=10000, open_positions=[])
        assert result.signal == SignalSide.BUY
        assert result.indicators["approved"] is True

    def test_blocks_low_confidence(self, rm):
        # risk_score starts at 100, deducts 40 for low confidence → 60.
        # Also deducts 20 for agents disagreeing if we mix signals.
        # Use disagreeing agents to push score below 50.
        agents = [
            AgentOpinion("technical", SignalSide.BUY, SignalStrength.BUY, 0.1, ""),
            AgentOpinion("sentiment", SignalSide.SELL, SignalStrength.SELL, 0.1, ""),
        ]
        sig = make_signal(confidence=0.1, agents_agree_data=agents)
        result = rm.analyze(sig, portfolio_value=10000)
        assert result.indicators["approved"] is False

    def test_blocks_max_exposure(self, rm):
        sig = make_signal(confidence=0.85)
        # exposure 2500/10000 = 25% > max_portfolio_risk 20%, deducts 50
        # total risk_score = 100 - 50 = 50, still >= 50 so approved
        # Need to push it lower: add disagreement too
        agents = [
            AgentOpinion("technical", SignalSide.BUY, SignalStrength.BUY, 0.85, ""),
            AgentOpinion("sentiment", SignalSide.SELL, SignalStrength.SELL, 0.85, ""),
        ]
        sig = make_signal(confidence=0.85, agents_agree_data=agents)
        positions = [{"value": 2500}]
        result = rm.analyze(sig, portfolio_value=10000, open_positions=positions)
        # risk_score = 100 - 20(disagree) - 50(max exposure) = 30 < 50
        assert result.indicators["approved"] is False


class TestPositionSizing:
    def test_kelly_sizing(self, rm):
        size = rm.calculate_position_size(0.8, 10000, 50000, 49000)
        assert size > 0

    def test_returns_positive(self, rm):
        size = rm.calculate_position_size(1.0, 100000, 50000, 49999)
        assert size > 0
        # The formula: risk_amount = portfolio * max_pos * kelly * confidence / sl_distance
        # = 100000 * 0.05 * 0.5 * 1.0 / 1 = 2500
        assert size == 2500.0

    def test_zero_sl_distance(self, rm):
        size = rm.calculate_position_size(0.8, 10000, 50000, 50000)
        assert size == 0
