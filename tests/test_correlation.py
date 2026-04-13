"""Tests for Portfolio Correlation Matrix feature"""
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import NexusConfig, RiskConfig
from src.models import TradeSignal, SignalSide, SignalStrength, AgentOpinion
from src.analysis.correlation import CorrelationMatrix
from src.analysis.risk import RiskManager
from src.execution.engine import ExecutionEngine


@pytest.fixture
def tmp_config(tmp_path):
    config = NexusConfig()
    config.data_dir = tmp_path
    config.paper_mode = True
    return config


@pytest.fixture
def corr_matrix(tmp_config):
    matrix = CorrelationMatrix(tmp_config)
    matrix._init_db()
    return matrix


class TestCorrelationMatrix:
    def test_compute_and_persist(self, corr_matrix, tmp_config):
        """Compute correlations and verify DB rows exist"""
        price_data = {
            "BTC/USDT": np.random.randn(30).tolist(),
            "ETH/USDT": np.random.randn(30).tolist(),
        }
        corr_matrix.compute_correlations(price_data)
        
        # Verify DB has the row
        conn = sqlite3.connect(str(tmp_config.data_dir / "trades.db"))
        rows = conn.execute("SELECT * FROM correlation_matrix").fetchall()
        conn.close()
        assert len(rows) == 1  # One pair: BTC/USDT ↔ ETH/USDT

    def test_get_correlation(self, corr_matrix):
        """Get correlation from cache"""
        # Manually set cache
        corr_matrix._cache[("BTC/USDT", "ETH/USDT")] = 0.85
        corr_matrix._cache_timestamp = datetime.now(timezone.utc)
        
        result = corr_matrix.get_correlation("BTC/USDT", "ETH/USDT")
        assert result == 0.85

    def test_get_correlation_none_when_no_data(self, corr_matrix):
        """Returns None when no data for pair"""
        result = corr_matrix.get_correlation("DOGE/USDT", "XRP/USDT")
        assert result is None

    def test_check_correlation_high_blocks(self, corr_matrix):
        """High correlation should block trade"""
        corr_matrix._cache[("BTC/USDT", "ETH/USDT")] = 0.85
        corr_matrix._cache_timestamp = datetime.now(timezone.utc)
        
        approved, correlated_with, corr = corr_matrix.check_correlation(
            "ETH/USDT", ["BTC/USDT"]
        )
        assert approved is False
        assert correlated_with == "BTC/USDT"
        assert corr == 0.85

    def test_check_correlation_low_allows(self, corr_matrix):
        """Low correlation should allow trade"""
        corr_matrix._cache[("BTC/USDT", "DOGE/USDT")] = 0.3
        corr_matrix._cache_timestamp = datetime.now(timezone.utc)
        
        approved, correlated_with, corr = corr_matrix.check_correlation(
            "DOGE/USDT", ["BTC/USDT"]
        )
        assert approved is True

    def test_no_correlation_data_allows_trade(self, corr_matrix):
        """Empty DB should allow trade (no data = no block)"""
        approved, correlated_with, corr = corr_matrix.check_correlation(
            "ETH/USDT", ["BTC/USDT"]
        )
        assert approved is True
        assert correlated_with is None

    def test_empty_positions_allows_trade(self, corr_matrix):
        """No existing positions = trivially approved"""
        approved, correlated_with, corr = corr_matrix.check_correlation(
            "ETH/USDT", []
        )
        assert approved is True

    def test_negative_correlation_allows_trade(self, corr_matrix):
        """Negative correlation is good for diversification - should NOT block"""
        corr_matrix._cache[("BTC/USDT", "XRP/USDT")] = -0.8
        corr_matrix._cache_timestamp = datetime.now(timezone.utc)
        
        approved, _, _ = corr_matrix.check_correlation(
            "XRP/USDT", ["BTC/USDT"]
        )
        assert approved is True  # Negative correlation should pass

    def test_cache_refresh(self, corr_matrix, tmp_config):
        """Verify cache reloads from DB after TTL"""
        # Insert data directly into DB
        conn = sqlite3.connect(str(tmp_config.data_dir / "trades.db"))
        conn.execute(
            "INSERT OR REPLACE INTO correlation_matrix VALUES (?, ?, ?, ?)",
            ("BTC/USDT", "ETH/USDT", 0.9, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
        
        # Set cache as stale
        corr_matrix._cache_timestamp = datetime.now(timezone.utc) - timedelta(hours=7)
        
        heatmap = corr_matrix.get_portfolio_heatmap()
        assert ("BTC/USDT", "ETH/USDT") in heatmap
        assert heatmap[("BTC/USDT", "ETH/USDT")] == 0.9


class TestRiskManagerCorrelation:
    def test_correlation_check_in_risk(self):
        """RiskManager should use correlation matrix when injected"""
        risk_config = RiskConfig(max_correlation=0.7, correlation_override=False)
        mgr = RiskManager(risk_config)
        
        mock_matrix = MagicMock()
        mock_matrix.check_correlation.return_value = (False, "BTC/USDT", 0.85)
        mgr.set_correlation_matrix(mock_matrix)
        
        signal = TradeSignal(symbol="ETH/USDT", side=SignalSide.BUY, confidence=0.8)
        opinion = mgr.check_signal(signal, 10000, [{"symbol": "BTC/USDT", "value": 500}])
        
        assert opinion.confidence < 1.0  # Risk score should be reduced

    def test_correlation_override_proceeds(self):
        """With override enabled, high correlation proceeds with warning"""
        risk_config = RiskConfig(max_correlation=0.7, correlation_override=True)
        mgr = RiskManager(risk_config)
        
        mock_matrix = MagicMock()
        mock_matrix.check_correlation.return_value = (False, "BTC/USDT", 0.85)
        mgr.set_correlation_matrix(mock_matrix)
        
        signal = TradeSignal(symbol="ETH/USDT", side=SignalSide.BUY, confidence=0.8)
        opinion = mgr.check_signal(signal, 10000, [{"symbol": "BTC/USDT", "value": 500}])
        
        # Should still proceed (override) but with reduced score
        assert opinion.indicators["risk_score"] < 100


class TestExecutionCorrelation:
    def test_correlation_blocks_trade(self, tmp_config):
        """Execution engine should block highly correlated trades"""
        engine = ExecutionEngine(tmp_config)
        
        # Set up correlation data: BTC ↔ ETH = 0.85
        engine._correlation._cache[("BTC/USDT", "ETH/USDT")] = 0.85
        engine._correlation._cache_timestamp = datetime.now(timezone.utc)
        
        # Insert an open BTC position
        conn = sqlite3.connect(str(tmp_config.data_dir / "trades.db"))
        conn.execute(
            "INSERT INTO trades (id, symbol, exchange, side, entry_price, quantity, status, paper, timestamp_opened, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test1", "BTC/USDT", "binance", "buy", 50000.0, 0.01, "open", 1,
             datetime.now(timezone.utc).isoformat(), "{}")
        )
        conn.commit()
        conn.close()
        
        signal = TradeSignal(
            symbol="ETH/USDT",
            exchange="binance",
            side=SignalSide.BUY,
            confidence=0.8,
        )
        
        trade = engine.execute(signal)
        assert trade.status == "cancelled"
        assert trade.metadata.get("blocked_reason") == "high_correlation"
        assert trade.metadata.get("correlated_with") == "BTC/USDT"

    def test_no_correlation_data_allows_trade(self, tmp_config):
        """No correlation data should allow trade"""
        engine = ExecutionEngine(tmp_config)
        
        # No correlation data, no open positions
        signal = TradeSignal(
            symbol="ETH/USDT",
            exchange="binance",
            side=SignalSide.BUY,
            confidence=0.8,
        )
        
        # Mock exchange to avoid real API call
        with patch.object(engine, '_get_exchange') as mock_ex:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ticker.return_value = {"last": 3000.0}
            mock_ex.return_value = mock_exchange
            
            trade = engine.execute(signal)
            assert trade.status != "cancelled" or trade.metadata.get("blocked_reason") != "high_correlation"
