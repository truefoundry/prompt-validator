import requests
import streamlit as st

from ..api_client import post_chat
from ..extractors import (
    extract_criteria_scores,
    extract_enhanced_prompt,
    extract_explanations,
    extract_original_prompt,
    extract_recommendations,
    extract_test_evaluation_result,
    extract_total_score,
)
from ..tfy_prompt_client import get_prompt_text_by_fqn
from .helpers import _build_prompt_payload, _is_paste_mode, _validate_prompt_input


def fetch_recommendations() -> None:
    if not _validate_prompt_input("prompt_input_mode", "prompt_fqn", "system_prompt_text"):
        return

    paste_mode = _is_paste_mode("prompt_input_mode")
    prompt_fields = _build_prompt_payload(
        "prompt_input_mode", "prompt_fqn", "system_prompt_text", "user_prompt_template_text",
    )

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        **prompt_fields,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "validation",
        "recommendations": None,
    }

    with st.spinner("Fetching prompt and recommendations..."):
        try:
            tfy_prompt_text = ""
            if not paste_mode:
                try:
                    tfy_prompt_text = get_prompt_text_by_fqn(prompt_fields["promptFQN"])
                except Exception as tfy_error:
                    st.warning(f"Unable to fetch prompt directly from TrueFoundry: {tfy_error}")

            data = post_chat(payload, include_grid_header=True)
            extracted_original_prompt = extract_original_prompt(data)

            if paste_mode:
                sys_text = st.session_state.get("system_prompt_text", "").strip()
                user_tpl = st.session_state.get("user_prompt_template_text", "").strip()
                pasted_text = f"system: {sys_text}"
                if user_tpl:
                    pasted_text += f"\n\nuser: {user_tpl}"
                st.session_state.original_prompt = pasted_text
            else:
                st.session_state.original_prompt = tfy_prompt_text or extracted_original_prompt

            st.session_state.recommendations = extract_recommendations(data)
            st.session_state.total_score = extract_total_score(data)
            st.session_state.criteria_scores = extract_criteria_scores(data)
            st.session_state.explanations = extract_explanations(data)
            st.session_state.selected_recommendations = []
            st.session_state.editable_recommendations = []
            st.session_state.last_selected_signature = ""
            st.session_state.enhanced_prompt = ""
            st.session_state.api_response_debug = data

            if st.session_state.original_prompt:
                st.success("Recommendations fetched successfully.")
            else:
                st.success(
                    "Recommendations fetched successfully. "
                    "Note: prompt content could not be loaded for display."
                )
        except requests.RequestException as exc:
            st.error(f"Failed to fetch recommendations: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error while fetching recommendations: {exc}")


def apply_recommendations() -> None:
    if not _validate_prompt_input("prompt_input_mode", "prompt_fqn", "system_prompt_text"):
        return

    selected = st.session_state.editable_recommendations or st.session_state.selected_recommendations
    if not selected:
        st.error("Select at least one recommendation before applying.")
        return

    prompt_fields = _build_prompt_payload(
        "prompt_input_mode", "prompt_fqn", "system_prompt_text", "user_prompt_template_text",
    )

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        **prompt_fields,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "validation",
        "recommendations": selected,
    }

    with st.spinner("Applying recommendations and enhancing prompt..."):
        try:
            data = post_chat(payload, include_grid_header=False)
            enhanced_prompt = extract_enhanced_prompt(data)
            if enhanced_prompt:
                st.session_state.enhanced_prompt = enhanced_prompt

            if not st.session_state.original_prompt and enhanced_prompt:
                st.session_state.original_prompt = enhanced_prompt

            st.session_state.api_response_debug = data
            st.success("Prompt enhanced successfully.")
        except requests.RequestException as exc:
            st.error(f"Failed to apply recommendations: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error while applying recommendations: {exc}")


def fetch_behavioral_recommendations() -> None:
    """Fetch behavioral recommendations from the backend using trace examples."""
    trace_examples = st.session_state.get("rec_trace_examples")
    if not trace_examples:
        st.error("No trace examples selected. Please select trace rows with bad outputs first.")
        return

    # Resolve system prompt
    if st.session_state.get("prompt_input_mode") == "Paste Prompt Text":
        system_prompt = st.session_state.get("system_prompt_text", "").strip()
        if not system_prompt:
            st.error("Please enter a system prompt before analyzing failures.")
            return
        prompt_fields = {"systemPrompt": system_prompt}
    else:
        fqn = st.session_state.get("prompt_fqn", "").strip()
        if not fqn:
            st.error("Please enter a Prompt FQN before analyzing failures.")
            return
        # Try to resolve to text for the behavioral call
        try:
            system_prompt = get_prompt_text_by_fqn(fqn)
        except Exception:
            system_prompt = ""
        if system_prompt:
            prompt_fields = {"systemPrompt": system_prompt}
        else:
            prompt_fields = {"promptFQN": fqn}

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        **prompt_fields,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "get_behavioral_recommendations",
        "recommendations": None,
        "traceExamples": trace_examples,
    }

    with st.spinner("Analyzing behavioral failures..."):
        try:
            data = post_chat(payload, include_grid_header=False)
            result = extract_test_evaluation_result(data)
            recs = []
            if isinstance(result, dict):
                recs = result.get("behavioral_recommendations", [])
            if not isinstance(recs, list):
                recs = []
            st.session_state.behavioral_recommendations = recs
            if recs:
                st.success(f"Found {len(recs)} behavioral recommendation(s).")
            else:
                st.warning("No behavioral recommendations returned.")
        except requests.RequestException as exc:
            st.error(f"Failed to fetch behavioral recommendations: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
