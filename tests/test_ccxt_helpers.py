"""Tests for CCXT rate-limiting helpers — retry logic, backoff, safe wrappers."""
import time
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from src.analysis.ccxt_helpers import (
    ExchangeRateLimitError,
    _calc_delay,
    _detect_exchange_id,
    get_exchange_semaphore,
    retry_exchange,
    safe_fetch_ohlcv,
    safe_fetch_ticker,
    safe_fetch_tickers,
)


# ── _calc_delay ──────────────────────────────────────────────────────

class TestCalcDelay:
    def test_exponential_backoff(self):
        d0 = _calc_delay(0, 1.0, 30.0, 0.0)
        d1 = _calc_delay(1, 1.0, 30.0, 0.0)
        d2 = _calc_delay(2, 1.0, 30.0, 0.0)
        assert d0 == pytest.approx(1.0, abs=0.05)
        assert d1 == pytest.approx(2.0, abs=0.05)
        assert d2 == pytest.approx(4.0, abs=0.05)

    def test_max_delay_cap(self):
        d = _calc_delay(10, 1.0, 8.0, 0.0)
        assert d <= 8.0 + 0.01  # jitter=0 so no randomness

    def test_jitter_adds_randomness(self):
        delays = {_calc_delay(1, 1.0, 30.0, 0.5) for _ in range(20)}
        assert len(delays) > 1  # jitter should produce varied values

    def test_minimum_delay(self):
        d = _calc_delay(0, 0.01, 30.0, 0.0)
        assert d >= 0.1


# ── _detect_exchange_id ─────────────────────────────────────────────

class TestDetectExchangeId:
    def test_from_exchange_attr(self):
        obj = MagicMock()
        obj.exchange = ccxt.binance()
        obj.exchange.id = "binance"
        assert _detect_exchange_id((obj,), {}) == "binance"

    def test_from_ccxt_arg(self):
        ex = ccxt.kraken()
        ex.id = "kraken"
        assert _detect_exchange_id((ex,), {}) == "kraken"

    def test_unknown_when_no_exchange(self):
        assert _detect_exchange_id((), {}) == "unknown"


# ── retry_exchange decorator ────────────────────────────────────────

class TestRetryExchange:
    def test_success_on_first_try(self):
        call_count = 0

        @retry_exchange(max_retries=3, throttle=False, enforce_semaphore=False)
        def ok():
            nonlocal call_count
            call_count += 1
            return "done"

        assert ok() == "done"
        assert call_count == 1

    @patch("src.analysis.ccxt_helpers.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        call_count = 0

        @retry_exchange(max_retries=2, base_delay=0.01, throttle=False, enforce_semaphore=False)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ccxt.RateLimitExceeded("429")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.analysis.ccxt_helpers.time.sleep")
    def test_raises_after_exhausting_retries(self, mock_sleep):
        @retry_exchange(max_retries=1, base_delay=0.01, throttle=False, enforce_semaphore=False)
        def always_rate_limited():
            raise ccxt.RateLimitExceeded("429")

        with pytest.raises(ExchangeRateLimitError):
            always_rate_limited()
        assert mock_sleep.call_count == 1

    def test_no_retry_on_bad_symbol(self):
        call_count = 0

        @retry_exchange(max_retries=3, throttle=False, enforce_semaphore=False)
        def bad():
            nonlocal call_count
            call_count += 1
            raise ccxt.BadSymbol("nope")

        with pytest.raises(ccxt.BadSymbol):
            bad()
        assert call_count == 1  # no retry

    @patch("src.analysis.ccxt_helpers.time.sleep")
    def test_retries_on_network_error(self, mock_sleep):
        call_count = 0

        @retry_exchange(max_retries=2, base_delay=0.01, throttle=False, enforce_semaphore=False)
        def net_fail():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ccxt.NetworkError("timeout")
            return "recovered"

        assert net_fail() == "recovered"
        assert call_count == 2


# ── Semaphore ────────────────────────────────────────────────────────

class TestSemaphore:
    def test_get_exchange_semaphore_returns_same(self):
        s1 = get_exchange_semaphore("binance_test")
        s2 = get_exchange_semaphore("binance_test")
        assert s1 is s2


# ── safe_fetch_* wrappers ───────────────────────────────────────────

class TestSafeFetchWrappers:
    @patch("src.analysis.ccxt_helpers.time.sleep")
    def test_safe_fetch_ticker(self, mock_sleep):
        mock_ex = MagicMock()
        mock_ex.id = "binance"
        mock_ex.fetch_ticker.return_value = {"symbol": "BTC/USDT", "last": 50000}
        result = safe_fetch_ticker(mock_ex, "BTC/USDT")
        assert result["last"] == 50000

    @patch("src.analysis.ccxt_helpers.time.sleep")
    def test_safe_fetch_ohlcv(self, mock_sleep):
        mock_ex = MagicMock()
        mock_ex.id = "binance"
        mock_ex.fetch_ohlcv.return_value = [[1, 100, 101, 99, 100, 500]]
        result = safe_fetch_ohlcv(mock_ex, "BTC/USDT", "1h", 200)
        assert len(result) == 1

    @patch("src.analysis.ccxt_helpers.time.sleep")
    def test_safe_fetch_tickers(self, mock_sleep):
        mock_ex = MagicMock()
        mock_ex.id = "binance"
        mock_ex.fetch_tickers.return_value = {"BTC/USDT": {"last": 50000}}
        result = safe_fetch_tickers(mock_ex, None)
        assert "BTC/USDT" in result
