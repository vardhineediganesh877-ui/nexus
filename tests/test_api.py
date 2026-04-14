"""Tests for NEXUS FastAPI endpoints — uses TestClient with mocked engines."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models import (
    BacktestResult,
    SignalSide,
    SignalStrength,
    Trade,
    TradeSignal,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.paper_mode = True
    cfg.exchanges = {
        "binance": MagicMock(is_configured=True, testnet=False),
        "mexc": MagicMock(is_configured=True, testnet=True),
    }
    return cfg


@pytest.fixture
def mock_signal_engine():
    engine = MagicMock()
    signal = TradeSignal(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        side=SignalSide.BUY,
        strength=SignalStrength.BUY,
        confidence=0.85,
    )
    engine.analyze.return_value = signal
    engine.scan_async = MagicMock()  # async, will be awaited
    return engine


@pytest.fixture
def mock_execution_engine():
    engine = MagicMock()
    engine.get_portfolio_summary.return_value = {"total_value": 10000, "positions": 0}
    engine.get_open_positions.return_value = []
    engine.get_trade_history.return_value = []
    trade = Trade(
        id="t1", symbol="BTC/USDT", exchange="binance",
        side=SignalSide.BUY, entry_price=50000, quantity=0.01,
    )
    engine.execute.return_value = trade
    return engine


@pytest.fixture
def client(mock_config, mock_signal_engine, mock_execution_engine):
    """Create TestClient with mocked globals to avoid real exchange connections."""
    with patch("src.api.app.NexusConfig") as MockConfig, \
         patch("src.api.app.SignalEngine", return_value=mock_signal_engine), \
         patch("src.api.app.ExecutionEngine", return_value=mock_execution_engine):

        MockConfig.from_env.return_value = mock_config

        # Import AFTER patching so lifespan uses mocks
        from src.api.app import app
        with TestClient(app) as c:
            yield c


# ── Health ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["paper_mode"] is True
        assert "binance" in data["exchanges_configured"]


# ── Analyze ──────────────────────────────────────────────────────────

class TestAnalyze:
    def test_analyze_returns_signal(self, client, mock_signal_engine):
        r = client.get("/api/v1/analyze/BTC-USDT", params={"exchange": "binance"})
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "BTC/USDT"
        assert data["side"] == "buy"

    def test_analyze_error_returns_500(self, client, mock_signal_engine):
        mock_signal_engine.analyze.side_effect = Exception("exchange down")
        r = client.get("/api/v1/analyze/BTC-USDT")
        assert r.status_code == 500


# ── Portfolio ────────────────────────────────────────────────────────

class TestPortfolio:
    def test_portfolio_returns_summary(self, client):
        r = client.get("/api/v1/portfolio")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert "open_positions" in data

    def test_portfolio_history(self, client):
        r = client.get("/api/v1/portfolio/history")
        assert r.status_code == 200
        data = r.json()
        assert "trades" in data


# ── Trade ────────────────────────────────────────────────────────────

class TestTrade:
    def test_trade_executes_buy(self, client):
        r = client.post("/api/v1/trade", json={
            "symbol": "BTC/USDT",
            "side": "buy",
            "exchange": "binance",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "BTC/USDT"

    def test_trade_rejects_hold(self, client):
        r = client.post("/api/v1/trade", json={
            "symbol": "BTC/USDT",
            "side": "hold",
        })
        assert r.status_code == 400

    def test_trade_invalid_side(self, client):
        r = client.post("/api/v1/trade", json={
            "symbol": "BTC/USDT",
            "side": "invalid_side",
        })
        assert r.status_code == 400 or r.status_code == 422


# ── Exchanges ────────────────────────────────────────────────────────

class TestExchanges:
    def test_exchanges_list(self, client):
        r = client.get("/api/v1/exchanges")
        assert r.status_code == 200
        data = r.json()
        assert len(data["exchanges"]) == 2
        ids = [e["id"] for e in data["exchanges"]]
        assert "binance" in ids


# ── Close Trade ──────────────────────────────────────────────────────

class TestCloseTrade:
    def test_close_not_found(self, client, mock_execution_engine):
        mock_execution_engine.get_open_positions.return_value = []
        r = client.post("/api/v1/trade/nonexistent/close")
        assert r.status_code == 404
