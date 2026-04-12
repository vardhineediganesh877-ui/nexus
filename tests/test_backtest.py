"""Tests for BacktestEngine (uses synthetic data, no network)."""
import pytest
from unittest.mock import patch
from src.analysis.backtest import BacktestEngine
from src.models import BacktestResult


def synthetic_ohlcv(n, base=100, pattern="flat"):
    data = []
    for i in range(n):
        if pattern == "flat":
            c = base
        elif pattern == "up":
            c = base + i * 0.5
        else:
            c = base
        data.append([i * 86400000, c, c + 1, c - 1, c, 1000])
    return data


@pytest.fixture
def engine():
    return BacktestEngine(exchange_id="mexc")


class TestStrategies:
    def test_rsi_generates_trades(self, engine):
        closes = [100] * 15
        for i in range(30):
            closes.append(100 - i * 2)
        for i in range(30):
            closes.append(closes[-1] + i * 3)
        trades = engine._strategy_rsi(closes)
        assert len(trades) >= 1

    def test_bollinger_generates_buy(self, engine):
        # Build data where price drops below lower band then recovers
        closes = [100.0] * 25
        closes.extend([80, 75, 70])  # drops below lower band
        closes.extend([85, 95, 100, 105])  # recovers past middle
        trades = engine._strategy_bollinger(closes)
        assert len(trades) >= 1
        # Entry is at one of the low prices
        assert trades[0]["entry"] <= 100.0

    def test_compare_strategies_sorted(self, engine):
        ohlcv = synthetic_ohlcv(200, pattern="up")
        with patch.object(engine, '_get_ohlcv', return_value=ohlcv):
            results = engine.compare_strategies("BTC/USDT", period_days=200)
            assert len(results) > 0
            sharpes = [r.sharpe_ratio for r in results]
            assert sharpes == sorted(sharpes, reverse=True)

    def test_walk_forward(self, engine):
        ohlcv = synthetic_ohlcv(300, pattern="up")
        with patch.object(engine, '_get_ohlcv', return_value=ohlcv):
            wf = engine.walk_forward("BTC/USDT", strategy="rsi", total_days=300)
            assert "robustness_score" in wf
            assert "verdict" in wf
            assert len(wf["folds"]) > 0


class TestBacktestResultGrade:
    def test_grade_from_result(self):
        r = BacktestResult("BTC", "rsi", "1y", sharpe_ratio=2.1, max_drawdown_pct=5)
        assert r.grade == "A+"
