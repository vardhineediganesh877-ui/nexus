"""Tests for KnowledgeEngine — GBrain integration, page writing, pattern queries."""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.knowledge.engine import KnowledgeEngine
from src.models import (
    AgentOpinion,
    SignalSide,
    SignalStrength,
    Trade,
    TradeSignal,
)


@pytest.fixture
def ke(tmp_path):
    """Create KnowledgeEngine with a temp brain directory."""
    return KnowledgeEngine(brain_dir=str(tmp_path / "brain"))


@pytest.fixture
def sample_trade():
    return Trade(
        id="t1",
        signal_id="s1",
        symbol="BTC/USDT",
        exchange="binance",
        side=SignalSide.BUY,
        entry_price=50000.0,
        quantity=0.01,
        pnl=50.0,
        pnl_pct=10.0,
        status="closed",
        paper=True,
    )


@pytest.fixture
def sample_signal():
    return TradeSignal(
        id="s1",
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        side=SignalSide.BUY,
        strength=SignalStrength.BUY,
        confidence=0.85,
        opinions=[
            AgentOpinion(
                agent_name="technical",
                signal=SignalSide.BUY,
                strength=SignalStrength.BUY,
                confidence=0.9,
                reasoning="RSI oversold bounce",
            )
        ],
        entry_price=50000.0,
    )


# ── Initialization ───────────────────────────────────────────────────

class TestInit:
    def test_creates_directories(self, tmp_path):
        ke = KnowledgeEngine(brain_dir=str(tmp_path / "mybrain"))
        assert (tmp_path / "mybrain" / "concepts").is_dir()
        assert (tmp_path / "mybrain" / "companies").is_dir()


# ── _slug ────────────────────────────────────────────────────────────

class TestSlug:
    def test_btc_usdt(self, ke):
        assert ke._slug("BTC/USDT") == "btc-usdt"

    def test_spaces(self, ke):
        assert ke._slug("BTC USDT") == "btc-usdt"


# ── log_trade ────────────────────────────────────────────────────────

class TestLogTrade:
    @patch.object(KnowledgeEngine, "_gbrain_call", return_value=None)
    def test_creates_page(self, mock_gbrain, ke, sample_trade, sample_signal):
        ke.log_trade(sample_trade, sample_signal)
        page = ke._page_path("BTC/USDT")
        assert page.exists()
        content = page.read_text()
        assert "BTC/USDT" in content
        assert "BUY" in content
        assert "$50,000.0000" in content

    @patch.object(KnowledgeEngine, "_gbrain_call", return_value=None)
    def test_appends_to_existing_page(self, mock_gbrain, ke, sample_trade, sample_signal):
        ke.log_trade(sample_trade, sample_signal)
        ke.log_trade(sample_trade, sample_signal)
        page = ke._page_path("BTC/USDT")
        content = page.read_text()
        # Should have two timeline entries
        assert content.count("BUY BTC/USDT") == 2

    @patch.object(KnowledgeEngine, "_gbrain_call", return_value=None)
    def test_includes_pnl(self, mock_gbrain, ke, sample_trade, sample_signal):
        ke.log_trade(sample_trade, sample_signal)
        page = ke._page_path("BTC/USDT")
        content = page.read_text()
        assert "P&L" in content
        assert "10.0%" in content


# ── query_pattern ────────────────────────────────────────────────────

class TestQueryPattern:
    @patch.object(KnowledgeEngine, "_gbrain_call", return_value=None)
    def test_unknown_symbol(self, mock_gbrain, ke):
        result = ke.query_pattern("ETH/USDT")
        assert result["known"] is False
        assert result["trades"] == 0

    @patch.object(KnowledgeEngine, "_gbrain_call", return_value=None)
    def test_known_symbol_after_trade(self, mock_gbrain, ke, sample_trade, sample_signal):
        ke.log_trade(sample_trade, sample_signal)
        result = ke.query_pattern("BTC/USDT")
        assert result["known"] is True


# ── log_analysis ─────────────────────────────────────────────────────

class TestLogAnalysis:
    def test_creates_page_with_analysis(self, ke, sample_signal):
        ke.log_analysis(sample_signal)
        page = ke._page_path("BTC/USDT")
        assert page.exists()
        content = page.read_text()
        assert "Analysis" in content
        assert "technical" in content


# ── market_intelligence ──────────────────────────────────────────────

class TestMarketIntelligence:
    def test_empty_when_no_pages(self, ke):
        result = ke.market_intelligence()
        assert result == []

    @patch.object(KnowledgeEngine, "_gbrain_call", return_value=None)
    def test_returns_tracked_symbols(self, mock_gbrain, ke, sample_trade, sample_signal):
        ke.log_trade(sample_trade, sample_signal)
        result = ke.market_intelligence()
        assert len(result) >= 1
        assert result[0]["symbol"] == "BTC/USDT"
