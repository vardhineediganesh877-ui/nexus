"""
NEXUS Rate-Limited Exchange Wrapper

Wraps CCXT exchange calls with:
- Exponential backoff with jitter on 429/5xx errors
- Configurable max retries
- Per-endpoint rate limit tracking
- Automatic throttle when approaching limits
"""

import logging
import random
import time
import threading
from typing import Dict, List, Optional, Any
from functools import wraps

import ccxt

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Rate limit configuration"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.5,
        backoff_factor: float = 2.0,
        requests_per_second: float = 5.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.backoff_factor = backoff_factor
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second


class RateLimiter:
    """Token-bucket style rate limiter for API calls"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._request_count: int = 0
        self._error_count: int = 0
        self._backoff_until: float = 0.0

    def acquire(self) -> None:
        """Block until a request slot is available"""
        with self._lock:
            now = time.monotonic()

            # If we're in a backoff period, wait
            if now < self._backoff_until:
                wait = self._backoff_until - now
                logger.debug(f"Rate limiter: waiting {wait:.1f}s (backoff)")
                time.sleep(wait)
                now = time.monotonic()

            # Enforce minimum interval between requests
            elapsed = now - self._last_request_time
            if elapsed < self.config.min_interval:
                wait = self.config.min_interval - elapsed
                logger.debug(f"Rate limiter: throttling {wait:.3f}s")
                time.sleep(wait)

            self._last_request_time = time.monotonic()
            self._request_count += 1

    def report_error(self, retry_attempt: int) -> float:
        """Report an error and get the backoff delay"""
        with self._lock:
            self._error_count += 1
            delay = min(
                self.config.base_delay * (self.config.backoff_factor ** retry_attempt),
                self.config.max_delay,
            )
            # Add jitter to avoid thundering herd
            jitter = random.uniform(0, self.config.jitter * delay)
            total_delay = delay + jitter
            self._backoff_until = time.monotonic() + total_delay
            return total_delay

    def report_success(self) -> None:
        """Report a successful request"""
        with self._lock:
            self._error_count = max(0, self._error_count - 1)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "backoff_active": time.monotonic() < self._backoff_until,
        }


class RateLimitedExchange:
    """
    Wraps a CCXT exchange with automatic retry, backoff, and rate limiting.

    Usage:
        exchange = ccxt.binance({...})
        safe = RateLimitedExchange(exchange)
        ohlcv = safe.fetch_ohlcv("BTC/USDT", "1h", limit=200)
        tickers = safe.fetch_tickers()
    """

    # Methods that need rate limiting
    PROTECTED_METHODS = {
        "fetch_ohlcv", "fetch_tickers", "fetch_ticker", "fetch_order_book",
        "fetch_trades", "fetch_balance", "fetch_orders", "fetch_open_orders",
        "fetch_closed_orders", "create_order", "cancel_order", "fetch_positions",
    }

    def __init__(
        self,
        exchange: ccxt.Exchange,
        config: Optional[RateLimitConfig] = None,
    ):
        self._exchange = exchange
        self._config = config or RateLimitConfig()
        self._limiter = RateLimiter(self._config)
        self._method_cache: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access, wrapping protected methods with retry logic"""
        if name.startswith("_"):
            raise AttributeError(name)

        attr = getattr(self._exchange, name)

        if name in self.PROTECTED_METHODS and callable(attr):
            if name not in self._method_cache:
                self._method_cache[name] = self._make_retry_wrapper(name, attr)
            return self._method_cache[name]

        return attr

    def _make_retry_wrapper(self, method_name: str, original_method):
        """Create a retry wrapper for an exchange method"""

        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(self._config.max_retries + 1):
                try:
                    self._limiter.acquire()
                    result = original_method(*args, **kwargs)
                    self._limiter.report_success()
                    return result

                except ccxt.RateLimitExceeded as e:
                    last_error = e
                    delay = self._limiter.report_error(attempt)
                    logger.warning(
                        f"[{method_name}] Rate limited (429), attempt {attempt + 1}/{self._config.max_retries + 1}, "
                        f"backing off {delay:.1f}s"
                    )
                    time.sleep(delay)

                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                    last_error = e
                    delay = self._limiter.report_error(attempt)
                    logger.warning(
                        f"[{method_name}] Network error: {type(e).__name__}, attempt {attempt + 1}, "
                        f"retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)

                except ccxt.ExchangeError as e:
                    # Don't retry on auth errors, invalid symbols, etc.
                    if isinstance(e, (ccxt.AuthenticationError, ccxt.BadSymbol, ccxt.InvalidOrder)):
                        logger.error(f"[{method_name}] Non-retryable error: {e}")
                        raise
                    last_error = e
                    delay = self._limiter.report_error(attempt)
                    logger.warning(
                        f"[{method_name}] Exchange error, attempt {attempt + 1}, "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

            # All retries exhausted
            logger.error(
                f"[{method_name}] All {self._config.max_retries + 1} attempts failed: {last_error}"
            )
            raise last_error

        wrapper.__name__ = method_name
        wrapper.__doc__ = original_method.__doc__
        return wrapper

    @property
    def id(self) -> str:
        return self._exchange.id

    @property
    def rate_limit_stats(self) -> Dict[str, Any]:
        return self._limiter.stats

    @property
    def underlying(self) -> ccxt.Exchange:
        """Access the raw exchange if needed"""
        return self._exchange


def create_rate_limited_exchange(
    exchange_id: str = "binance",
    config: Optional[RateLimitConfig] = None,
    **exchange_kwargs,
) -> RateLimitedExchange:
    """Factory: create a rate-limited exchange from scratch"""
    exchange_class = getattr(ccxt, exchange_id, None)
    if not exchange_class:
        raise ValueError(f"Exchange '{exchange_id}' not supported by CCXT")

    # Set sane defaults if not specified
    if "rateLimit" not in exchange_kwargs:
        exchange_kwargs["rateLimit"] = 200  # CCXT internal rate limiter

    exchange = exchange_class(exchange_kwargs)
    return RateLimitedExchange(exchange, config)
