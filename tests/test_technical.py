"""Tests for TechnicalAnalyst indicator calculations (no network)."""
import pytest
from src.analysis.technical import TechnicalAnalyst


class TestRSI:
    def test_flat_prices(self):
        # All same price → no losses → avg_loss=0 → RSI=100
        prices = [100.0] * 20
        rsi = TechnicalAnalyst._rsi(prices, 14)
        assert rsi == 100.0

    def test_all_increasing(self):
        prices = list(range(1, 30))
        rsi = TechnicalAnalyst._rsi(prices, 14)
        assert rsi == 100.0

    def test_all_decreasing(self):
        prices = list(range(30, 1, -1))
        rsi = TechnicalAnalyst._rsi(prices, 14)
        assert rsi == pytest.approx(0.0)

    def test_mixed(self):
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        rsi = TechnicalAnalyst._rsi(prices, 14)
        assert 30 < rsi < 80

    def test_too_few_prices(self):
        assert TechnicalAnalyst._rsi([1, 2], 14) == 50.0


class TestEMA:
    def test_basic(self):
        prices = [10.0] * 10 + [20.0] * 10
        ema = TechnicalAnalyst._ema(prices, 10)
        assert 10 < ema < 20

    def test_fewer_than_period(self):
        prices = [5.0, 10.0, 15.0]
        ema = TechnicalAnalyst._ema(prices, 10)
        assert ema == pytest.approx(10.0)


class TestBollingerBands:
    def test_ordering(self):
        prices = list(range(1, 31))
        upper, middle, lower, width = TechnicalAnalyst._bollinger_bands(prices, 20)
        assert upper > middle > lower

    def test_width_positive(self):
        prices = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50,
                  10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10]
        _, _, _, width = TechnicalAnalyst._bollinger_bands(prices, 20)
        assert width > 0

    def test_too_few(self):
        assert TechnicalAnalyst._bollinger_bands([1, 2], 20) == (0, 0, 0, 0)


class TestATR:
    def test_positive(self):
        ohlcv = [[0, 100, 105, 95, 100, 1000]]
        for i in range(1, 20):
            ohlcv.append([i, 100, 102, 98, 100 + i * 0.5, 1000])
        atr = TechnicalAnalyst._atr(ohlcv, 14)
        assert atr > 0

    def test_too_few(self):
        assert TechnicalAnalyst._atr([[0, 1, 2, 1, 1, 1]], 14) == 0


class TestMACD:
    def test_returns_three_values(self):
        prices = list(range(1, 60))
        macd, signal, hist = TechnicalAnalyst._macd(prices)
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)

    def test_too_few(self):
        macd, signal, hist = TechnicalAnalyst._macd([1, 2, 3])
        assert macd == 0
