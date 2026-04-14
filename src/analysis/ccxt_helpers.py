"""
NEXUS CCXT Helpers — Rate limiting, retry, and backoff for all exchange calls.

Provides a single decorator `retry_exchange` that wraps any CCXT API call with:
- Exponential backoff (1s → 2s → 4s)
- Configurable max retries (default 3)
- RateLimitError / NetworkError specific handling
- Jitter to avoid thundering herd on parallel scans
- Per-exchange rate limiter semaphore
"""

import logging
import random
import threading
import time
import functools
from typing import Dict, Optional

import ccxt

logger = logging.getLogger(__name__)

# ── Global rate-limiting semaphore per exchange ──────────────────────────────
# Binance: 1200 req/min = 20 req/sec → allow bursts of 10 with 0.5s refill
# Default conservative: 5 concurrent requests per exchange
_EXCHANGE_SEMAPHORES: Dict[str, threading.Semaphore] = {}
_SEMAPHORE_LOCK = threading.Lock()
MAX_CONCURRENT_PER_EXCHANGE = 5

# Minimum delay between requests to same exchange (seconds)
_MIN_INTERVAL: Dict[str, float] = {}
_MIN_INTERVAL_LOCK = threading.Lock()
_last_request_time: Dict[str, float] = {}


def get_exchange_semaphore(exchange_id: str) -> threading.Semaphore:
    """Get or create a concurrency limiter for an exchange."""
    with _SEMAPHORE_LOCK:
        if exchange_id not in _EXCHANGE_SEMAPHORES:
            _EXCHANGE_SEMAPHORES[exchange_id] = threading.Semaphore(MAX_CONCURRENT_PER_EXCHANGE)
        return _EXCHANGE_SEMAPHORES[exchange_id]


def set_min_interval(exchange_id: str, seconds: float):
    """Set minimum interval between requests for a specific exchange."""
    with _MIN_INTERVAL_LOCK:
        _MIN_INTERVAL[exchange_id] = seconds


def _throttle(exchange_id: str):
    """Enforce minimum interval between requests to the same exchange."""
    min_interval = _MIN_INTERVAL.get(exchange_id, 0.2)  # Default 200ms
    with _MIN_INTERVAL_LOCK:
        last = _last_request_time.get(exchange_id, 0)
        now = time.monotonic()
        elapsed = now - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_time[exchange_id] = time.monotonic()


class ExchangeRateLimitError(Exception):
    """Raised when all retries are exhausted."""
    pass


def retry_exchange(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.25,
    throttle: bool = True,
    enforce_semaphore: bool = True,
):
    """
    Decorator that adds retry + exponential backoff to CCXT exchange calls.

    Handles:
    - ccxt.RateLimitExceeded → always retry with backoff
    - ccxt.NetworkError → retry (transient)
    - ccxt.ExchangeNotAvailable → retry (maintenance)
    - ccxt.BadSymbol → DO NOT retry (bad request)
    - Other exceptions → retry once (unknown may be transient)

    Args:
        max_retries: Maximum retry attempts (default 3 = 4 total tries)
        base_delay: Initial backoff delay in seconds
        max_delay: Maximum backoff delay cap
        jitter: Random jitter factor (0.25 = ±25% randomness)
        throttle: Whether to enforce per-exchange rate limiting
        enforce_semaphore: Whether to use concurrency semaphore
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract exchange_id from self.exchange or first ccxt.Exchange arg
            exchange_id = _detect_exchange_id(args, kwargs)
            sem = get_exchange_semaphore(exchange_id) if enforce_semaphore else None

            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    # Throttle: enforce minimum interval
                    if throttle and exchange_id:
                        _throttle(exchange_id)

                    # Acquire semaphore for concurrency control
                    if sem:
                        sem.acquire()

                    try:
                        return func(*args, **kwargs)
                    finally:
                        if sem:
                            sem.release()

                except ccxt.RateLimitExceeded as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = _calc_delay(attempt, base_delay, max_delay, jitter)
                        logger.warning(
                            f"[{exchange_id}] Rate limited on attempt {attempt + 1}/{max_retries + 1}, "
                            f"retrying in {delay:.1f}s — {func.__name__}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{exchange_id}] Rate limit exhausted after {max_retries + 1} attempts — {func.__name__}"
                        )

                except ccxt.NetworkError as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = _calc_delay(attempt, base_delay, max_delay, jitter)
                        logger.warning(
                            f"[{exchange_id}] Network error on attempt {attempt + 1}: {e}, "
                            f"retrying in {delay:.1f}s — {func.__name__}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{exchange_id}] Network error exhausted after {max_retries + 1} attempts — {func.__name__}"
                        )

                except ccxt.ExchangeNotAvailable as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = _calc_delay(attempt, base_delay, max_delay, jitter)
                        logger.warning(
                            f"[{exchange_id}] Exchange unavailable on attempt {attempt + 1}: {e}, "
                            f"retrying in {delay:.1f}s — {func.__name__}"
                        )
                        time.sleep(delay)

                except (ccxt.BadSymbol, ccxt.AuthenticationError, ccxt.PermissionDenied, ValueError) as e:
                    # These are not transient — don't retry
                    logger.debug(f"[{exchange_id}] Non-retryable error in {func.__name__}: {type(e).__name__}: {e}")
                    raise

                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = _calc_delay(attempt, base_delay, max_delay, jitter)
                        logger.warning(
                            f"[{exchange_id}] Unexpected error on attempt {attempt + 1}: {e}, "
                            f"retrying in {delay:.1f}s — {func.__name__}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{exchange_id}] All retries exhausted for {func.__name__}: {e}"
                        )

            raise ExchangeRateLimitError(
                f"All {max_retries + 1} attempts failed for {func.__name__}: {last_exception}"
            ) from last_exception

        return wrapper
    return decorator


def _detect_exchange_id(args, kwargs) -> str:
    """Try to extract exchange ID from method args."""
    import ccxt as _ccxt

    # Check self.exchange (common pattern in our classes)
    if args and hasattr(args[0], 'exchange'):
        ex = args[0].exchange
        if isinstance(ex, _ccxt.Exchange):
            return getattr(ex, 'id', 'unknown')
    if args and hasattr(args[0], '_exchange'):
        ex = args[0]._exchange
        if isinstance(ex, _ccxt.Exchange):
            return getattr(ex, 'id', 'unknown')
    if args and hasattr(args[0], '_exchanges'):
        # Pick first exchange as proxy
        exs = args[0]._exchanges
        if exs:
            try:
                first = next(iter(exs.values()))
                return getattr(first, 'id', 'unknown')
            except StopIteration:
                pass

    # Check if first arg is an Exchange instance
    for arg in args:
        if isinstance(arg, _ccxt.Exchange):
            return getattr(arg, 'id', 'unknown')

    return 'unknown'


def _calc_delay(attempt: int, base: float, max_delay: float, jitter: float) -> float:
    """Calculate exponential backoff with jitter."""
    delay = min(base * (2 ** attempt), max_delay)
    jitter_amount = delay * jitter * (random.random() * 2 - 1)  # ±jitter%
    return max(0.1, delay + jitter_amount)


def safe_fetch_tickers(exchange: ccxt.Exchange, params: dict = None) -> dict:
    """Standalone helper: fetch all tickers with retry."""
    @retry_exchange(max_retries=3, base_delay=2.0)
    def _fetch(ex: ccxt.Exchange, p):
        return ex.fetch_tickers(p or {})
    return _fetch(exchange, params)


def safe_fetch_ohlcv(exchange: ccxt.Exchange, symbol: str,
                     timeframe: str = "1h", limit: int = 200) -> list:
    """Standalone helper: fetch OHLCV with retry."""
    @retry_exchange(max_retries=3, base_delay=1.0)
    def _fetch(ex: ccxt.Exchange, s, tf, l):
        return ex.fetch_ohlcv(s, timeframe=tf, limit=l)
    return _fetch(exchange, symbol, timeframe, limit)


def safe_fetch_ticker(exchange: ccxt.Exchange, symbol: str) -> dict:
    """Standalone helper: fetch single ticker with retry."""
    @retry_exchange(max_retries=3, base_delay=1.0)
    def _fetch(ex: ccxt.Exchange, s):
        return ex.fetch_ticker(s)
    return _fetch(exchange, symbol)
