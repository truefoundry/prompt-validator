import streamlit as st

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME

def init_session_state() -> None:
    defaults = {
        "base_url": DEFAULT_BASE_URL,
        "session_id": "123",
        "model_name": DEFAULT_MODEL_NAME,
        "max_tokens": 15000,
        "temperature": 0.1,
        "reasoning_effort": "none",
        "prompt_fqn": "",
        "original_prompt": "",
        "recommendations": [],
        "selected_recommendations": [],
        "editable_recommendations": [],
        "enhanced_prompt": "",
        "total_score": None,
        "criteria_scores": {},
        "explanations": {},
        "last_selected_signature": "",
        "api_response_debug": {},
        # DeepEval tab
        "deepeval_prompt_fqn": "",
        "deepeval_is_rag": False,
        "deepeval_result": None,
        "deepeval_api_debug": {},
        "deepeval_uploaded_tests": None,
        # Exact Match tab
        "exact_match_prompt_fqn": "",
        "exact_match_result": None,
        "exact_match_api_debug": {},
        "exact_match_uploaded_tests": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
