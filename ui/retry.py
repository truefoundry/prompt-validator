"""Retry utilities for transient LLM and network failures."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BASE_WAIT = 1.0  # seconds; doubles each retry: 1s → 2s → 4s


def call_with_retry(
    fn: Callable[[], _T],
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_wait: float = _DEFAULT_BASE_WAIT,
    is_retryable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, int, BaseException], None] | None = None,
) -> _T:
    """Call *fn* up to *max_attempts* times with exponential back-off between retries.

    Args:
        fn: Zero-argument callable to invoke.
        max_attempts: Total number of attempts allowed (default 3).
        base_wait: Base wait in seconds; actual wait = base_wait * 2^(attempt-1),
                   giving 1 s, 2 s, 4 s for base_wait=1.
        is_retryable: Predicate that returns True when *exc* should trigger a retry.
                      When None, all exceptions are retried.
        on_retry: Optional hook called just before sleeping:
                  ``on_retry(attempt_number, max_attempts, exc)``.
                  Useful for surfacing warnings in the UI.

    Returns:
        The return value of *fn* on the first successful call.

    Raises:
        The last exception raised by *fn* after all attempts are exhausted,
        or immediately if *is_retryable* returns False for the exception.
    """
    last_exc: BaseException
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            if is_retryable is not None and not is_retryable(exc):
                raise
            if attempt == max_attempts:
                break
            wait = base_wait * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                wait,
            )
            if on_retry is not None:
                on_retry(attempt, max_attempts, exc)
            time.sleep(wait)

    raise last_exc  # type: ignore[misc]  # always assigned after first iteration
