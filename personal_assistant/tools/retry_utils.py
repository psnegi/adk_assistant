"""Retry utilities with exponential backoff for handling API rate limits."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)

# Default retry settings
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_INITIAL_DELAY = 1.0   # seconds
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_MAX_DELAY = 60.0      # seconds


def retry_with_backoff(
    max_retries: int = _DEFAULT_MAX_RETRIES,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    max_delay: float = _DEFAULT_MAX_DELAY,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> Callable:
    """Decorator that retries a function with exponential backoff on failure.

    Handles HTTP 429 (Too Many Requests / resource limit) and transient server
    errors automatically, logging each retry attempt.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        initial_delay: Initial delay in seconds before the first retry (default 1).
        backoff_factor: Multiplier applied to the delay after each retry (default 2).
        max_delay: Maximum delay cap in seconds (default 60).
        retryable_exceptions: Exception types that should trigger a retry.
        retryable_status_codes: HTTP status codes that should trigger a retry.

    Returns:
        Decorated function with retry logic applied.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    exc_str = str(exc)

                    # Check if this is an HTTP error with a retryable status code
                    is_retryable = False
                    for code in retryable_status_codes:
                        if str(code) in exc_str:
                            is_retryable = True
                            break

                    # Also check for common rate-limit keywords
                    rate_limit_keywords = [
                        "quota", "rate limit", "resource exhausted",
                        "too many requests", "429", "limit exceeded",
                    ]
                    if any(kw in exc_str.lower() for kw in rate_limit_keywords):
                        is_retryable = True

                    if not is_retryable or attempt == max_retries:
                        raise

                    actual_delay = min(delay, max_delay)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs…",
                        attempt + 1,
                        max_retries + 1,
                        func.__name__,
                        exc_str[:120],
                        actual_delay,
                    )
                    time.sleep(actual_delay)
                    delay *= backoff_factor

        return wrapper

    return decorator
