import logging
from typing import Any

import requests
import streamlit as st

from .config import DEFAULT_GRID, REQUEST_TIMEOUT_SECONDS
from .retry import call_with_retry

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BASE_WAIT_SECONDS = 1.0


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for transient network/server errors that warrant a retry."""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def _post_once(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    response = requests.post(url=url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected API response format; expected a JSON object.")
    return data


def post_traces_fetch(
    tfy_host: str,
    tfy_api_key: str,
    hours: int = 24,
    limit: int = 200,
    fqn_filter: str | None = None,
    email_filter: str | None = None,
    data_routing_destination: str = "default",
) -> dict[str, Any]:
    """POST /traces/fetch — fetch live spans from a TrueFoundry tenant via the backend."""
    base_url = st.session_state.base_url.strip().rstrip("/")
    url = f"{base_url}/traces/fetch"
    payload: dict[str, Any] = {
        "tfy_host": tfy_host,
        "tfy_api_key": tfy_api_key,
        "hours": hours,
        "limit": limit,
        "data_routing_destination": data_routing_destination,
    }
    if fqn_filter:
        payload["fqn_filter"] = fqn_filter
    if email_filter:
        payload["email_filter"] = email_filter

    def _on_retry(attempt: int, max_attempts: int, exc: BaseException) -> None:
        logger.warning("post_traces_fetch retry %d/%d: %s", attempt, max_attempts, exc)
        st.toast(f"Trace fetch failed — retrying ({attempt}/{max_attempts - 1})…", icon="⚠️")

    return call_with_retry(
        fn=lambda: _post_once(url, payload, {"Content-Type": "application/json"}),
        max_attempts=_MAX_ATTEMPTS,
        base_wait=_BASE_WAIT_SECONDS,
        is_retryable=_is_retryable_http_error,
        on_retry=_on_retry,
    )


def post_chat(payload: dict[str, Any], include_grid_header: bool) -> dict[str, Any]:
    """POST /chat with automatic retry on transient failures.

    Retries up to 3 times on connection errors, timeouts, HTTP 429, and HTTP 5xx,
    using exponential back-off (1 s, 2 s). A toast notification is shown to the
    user on each retry attempt.
    """
    base_url = st.session_state.base_url.strip().rstrip("/")
    url = f"{base_url}/chat"
    headers = {"Content-Type": "application/json"}
    if include_grid_header:
        headers["x-grid"] = DEFAULT_GRID

    def _on_retry(attempt: int, max_attempts: int, exc: BaseException) -> None:
        logger.warning("post_chat retry %d/%d: %s", attempt, max_attempts, exc)
        st.toast(f"Request failed — retrying ({attempt}/{max_attempts - 1})…", icon="⚠️")

    return call_with_retry(
        fn=lambda: _post_once(url, payload, headers),
        max_attempts=_MAX_ATTEMPTS,
        base_wait=_BASE_WAIT_SECONDS,
        is_retryable=_is_retryable_http_error,
        on_retry=_on_retry,
    )
