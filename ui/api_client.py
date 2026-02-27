from typing import Any

import requests
import streamlit as st

from .config import DEFAULT_GRID, REQUEST_TIMEOUT_SECONDS

def post_chat(payload: dict[str, Any], include_grid_header: bool) -> dict[str, Any]:
    base_url = st.session_state.base_url.strip().rstrip("/")
    url = f"{base_url}/chat"
    headers = {"Content-Type": "application/json"}
    if include_grid_header:
        headers["x-grid"] = DEFAULT_GRID

    response = requests.post(
        url=url,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected API response format; expected a JSON object.")
    return data
