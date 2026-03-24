import os
import json

import streamlit as st

from ..config import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME, DEFAULT_GRID, REQUEST_TIMEOUT_SECONDS
from ..state import init_session_state


def _load_available_models() -> list[str]:
    """Load model names from api_response.json."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(project_root, "api_response.json")
        with open(json_path) as f:
            data = json.load(f)
        for entry in data.get("data", []):
            if entry.get("apiKeyValue") == "tfy.request.model_name":
                return sorted(k for k in entry.get("keys", []) if k.startswith("tfy-"))
    except Exception:
        pass
    return []


_AVAILABLE_MODELS = _load_available_models()


EXAMPLE_PROMPT_FQNS = [
    "chat_prompt:truefoundry/new-repo/test-ocr:1",
    "chat_prompt:truefoundry/new-repo/test-fetchtraces:1",
    "chat_prompt:truefoundry/new-repo/test-nl-to-filters:1",
    "chat_prompt:truefoundry/new-repo/cvs_guardrails_prompt:1",
    "chat_prompt:truefoundry/new-repo/cvs_intent_classifier_prompt:1",
]


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Configuration")
        st.session_state.base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            help="Backend service base URL. Example: http://localhost:21120",
        )
        st.session_state.session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
        )
        current_model = st.session_state.model_name or ""
        model_options = [""] + _AVAILABLE_MODELS
        if current_model and current_model not in model_options:
            model_options = [current_model] + model_options
        current_idx = model_options.index(current_model) if current_model in model_options else 0
        selected_model = st.selectbox(
            "Model Name (Optional)",
            options=model_options,
            index=current_idx,
            placeholder="Type to search…",
            help="Type any substring to filter models.",
        )
        st.session_state.model_name = selected_model or ""

        st.session_state.max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=100000,
            value=st.session_state.max_tokens,
            step=1000,
            help="Maximum number of tokens in the model response.",
        )

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controls randomness. Lower values are more deterministic.",
        )

        model_lower = (st.session_state.model_name or "").lower()
        _GEMINI_PATTERNS = ("gemini-2.5", "gemini-3")
        _OPENAI_PATTERNS = (
            "o4-mini", "o4-preview", "o3", "o1",
            "gpt-5-mini", "gpt-5-nano", "gpt-5",
        )
        is_gemini = any(p in model_lower for p in _GEMINI_PATTERNS)
        is_openai_reasoning = any(p in model_lower for p in _OPENAI_PATTERNS)

        if is_gemini or is_openai_reasoning:
            options = ["none", "low", "medium", "high"]
            if is_gemini:
                options.insert(1, "no")
            current = st.session_state.reasoning_effort
            idx = options.index(current) if current in options else 0
            st.session_state.reasoning_effort = st.selectbox(
                "Reasoning Effort",
                options=options,
                index=idx,
                help=(
                    "Controls how much reasoning the model performs. "
                    "Supported by Gemini 2.5+, o1/o3/o4, and GPT-5 families."
                ),
            )
        else:
            st.session_state.reasoning_effort = "none"

        st.divider()
        st.caption("Example Prompt FQNs")
        selected_example_fqn = st.selectbox(
            "Choose example Prompt FQN",
            options=EXAMPLE_PROMPT_FQNS,
            index=None,
            placeholder="Select an example",
            key="sidebar_example_prompt_fqn",
        )
        if selected_example_fqn:
            st.session_state.prompt_input_mode = "TFY Prompt FQN"
            st.session_state.prompt_fqn = selected_example_fqn
            st.session_state.deepeval_input_mode = "TFY Prompt FQN"
            st.session_state.deepeval_prompt_fqn = selected_example_fqn
            st.session_state.deepeval_fqn_input = selected_example_fqn
            st.session_state.exact_match_input_mode = "TFY Prompt FQN"
            st.session_state.exact_match_prompt_fqn = selected_example_fqn
            st.session_state.exact_match_fqn_input = selected_example_fqn
