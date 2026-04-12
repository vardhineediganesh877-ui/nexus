"""Tests for ExecutionEngine (SQLite-based, temp dir, no network)."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.execution.engine import ExecutionEngine
from src.config import NexusConfig
from src.models import TradeSignal, Trade, SignalSide, SignalStrength, AgentOpinion


@pytest.fixture
def tmp_config(tmp_path):
    return NexusConfig(data_dir=tmp_path, paper_mode=True)


@pytest.fixture
def engine(tmp_config):
    return ExecutionEngine(tmp_config)


def make_signal(side=SignalSide.BUY, symbol="BTC/USDT", exchange="binance"):
    return TradeSignal(
        symbol=symbol, exchange=exchange, side=side,
        strength=SignalStrength.BUY, confidence=0.85,
        entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0,
        opinions=[AgentOpinion("technical", side, SignalStrength.BUY, 0.85, "test")],
    )


class TestDBInit:
    def test_db_created(self, tmp_config):
        engine = ExecutionEngine(tmp_config)
        assert (tmp_config.data_dir / "trades.db").exists()


class TestPaperTrade:
    def test_execute_creates_record(self, engine):
        sig = make_signal()
        mock_ticker = {"last": 50000.0}
        with patch.object(engine, '_get_exchange') as mock_ex:
            ex = MagicMock()
            ex.fetch_ticker.return_value = mock_ticker
            mock_ex.return_value = ex
            trade = engine.execute(sig)

        assert trade.status == "open"
        assert trade.entry_price == 50000.0
        assert trade.paper is True

        # Verify in DB
        positions = engine.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"

    def test_hold_signal_returns_cancelled(self, engine):
        sig = make_signal(side=SignalSide.HOLD)
        trade = engine.execute(sig)
        assert trade.status == "cancelled"


class TestPortfolioSummary:
    def test_empty_summary(self, engine):
        summary = engine.get_portfolio_summary()
        assert summary["total_trades"] == 0
        assert summary["open_positions"] == 0

    def test_after_trade(self, engine):
        sig = make_signal()
        with patch.object(engine, '_get_exchange') as mock_ex:
            ex = MagicMock()
            ex.fetch_ticker.return_value = {"last": 50000.0}
            mock_ex.return_value = ex
            engine.execute(sig)

        summary = engine.get_portfolio_summary()
        assert summary["total_trades"] == 1
        assert summary["open_positions"] == 1


class TestTradeHistory:
    def test_returns_closed_trades(self, engine):
        # Execute and close a trade
        sig = make_signal()
        with patch.object(engine, '_get_exchange') as mock_ex:
            ex = MagicMock()
            ex.fetch_ticker.return_value = {"last": 50000.0}
            mock_ex.return_value = ex
            trade = engine.execute(sig)

        # Close it
        with patch.object(engine, '_get_exchange') as mock_ex:
            ex = MagicMock()
            ex.fetch_ticker.return_value = {"last": 52000.0}
            mock_ex.return_value = ex
            engine.close_position(trade, exit_price=52000.0)

        history = engine.get_trade_history()
        assert len(history) == 1
        assert history[0].status == "closed"
