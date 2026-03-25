import os
import json

import streamlit as st

from ..config import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME, DEFAULT_GRID, REQUEST_TIMEOUT_SECONDS
from ..state import init_session_state


def _load_available_models() -> list[str]:
    """Load model names from models.json."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(project_root, "models.json")
        with open(json_path) as f:
            data = json.load(f)
        return sorted(data.get("models", []))
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


def _get_tfy_tenant() -> str:
    """Extract tenant name from TFY_HOST env var (e.g. https://acme.truefoundry.tech/ → acme)."""
    host = os.environ.get("TFY_HOST", "")
    if host:
        # strip scheme and trailing slash
        host = host.replace("https://", "").replace("http://", "").rstrip("/")
        # take the first subdomain segment
        return host.split(".")[0]
    return ""


def render_sidebar() -> None:
    with st.sidebar:
        tenant = _get_tfy_tenant()
        st.markdown(
            f"""<div style="padding:16px 4px 8px 4px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:{'12' if tenant else '0'}px;">
    <span style="font-size:1.1rem;">⚙️</span>
    <span style="font-size:0.95rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.01em;">Configuration</span>
  </div>
  {'<div style="background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.35);border-radius:6px;padding:4px 10px;display:inline-block;">'
   f'<span style="font-size:0.72rem;font-weight:600;color:#a5b4fc;letter-spacing:0.04em;">TENANT: {tenant.upper()}</span>'
   '</div>' if tenant else ''}
</div>""",
            unsafe_allow_html=True,
        )
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

        # Models that support reasoning_effort (per provider docs)
        _GEMINI_PATTERNS = ("gemini-2.5", "gemini-3")
        _ANTHROPIC_PATTERNS = ("claude-opus-4", "claude-sonnet-4", "claude-3-7", "claude-sonnet-3.7")
        _OPENAI_PATTERNS = (
            "o4-mini", "o4-preview", "o3-pro", "o3-mini", "/o3", "/o1", "o1-mini", "o1-preview",
            "gpt-5-mini", "gpt-5-nano", "gpt-5", "codex-mini",
        )
        _GROQ_PATTERNS = ("deepseek-r1", "qwen3", "qwen-3", "gpt-oss")
        _XAI_PATTERNS = ("grok-3-mini",)

        is_gemini = any(p in model_lower for p in _GEMINI_PATTERNS)
        is_anthropic = any(p in model_lower for p in _ANTHROPIC_PATTERNS)
        is_openai_reasoning = any(p in model_lower for p in _OPENAI_PATTERNS)
        is_groq = any(p in model_lower for p in _GROQ_PATTERNS)
        is_xai = any(p in model_lower for p in _XAI_PATTERNS)

        supports_reasoning = is_gemini or is_anthropic or is_openai_reasoning or is_groq or is_xai

        if supports_reasoning:
            if is_xai:
                options = ["low", "high"]
            elif is_gemini:
                options = ["none", "minimal", "low", "medium", "high"]
            elif is_anthropic:
                options = ["none", "low", "medium", "high"]
            else:
                options = ["none", "low", "medium", "high"]
            current = st.session_state.reasoning_effort
            idx = options.index(current) if current in options else 0
            st.session_state.reasoning_effort = st.selectbox(
                "Reasoning Effort",
                options=options,
                index=idx,
                help=(
                    "Controls how much reasoning the model performs. "
                    "Supported by Gemini 2.5+, Claude Sonnet 3.7/4/Opus 4, "
                    "o1/o3/o4/GPT-5, Groq DeepSeek R1/Qwen3, and xAI Grok-3-mini."
                ),
            )
        else:
            st.session_state.reasoning_effort = "none"

        st.divider()
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.06em;color:#64748b;margin:0 0 6px 0;">📎 Example Prompt FQNs</p>',
            unsafe_allow_html=True,
        )
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
