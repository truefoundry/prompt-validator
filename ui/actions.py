import requests
import streamlit as st

from .api_client import post_chat
from .extractors import (
    extract_criteria_scores,
    extract_enhanced_prompt,
    extract_explanations,
    extract_original_prompt,
    extract_recommendations,
    extract_test_evaluation_result,
    extract_total_score,
)
from .tfy_prompt_client import get_prompt_text_by_fqn


def fetch_recommendations() -> None:
    prompt_fqn = st.session_state.prompt_fqn.strip()
    if not prompt_fqn:
        st.error("Please enter Prompt FQN before fetching recommendations.")
        return

    payload = {
        "sessionId": st.session_state.session_id,
        "promptFQN": prompt_fqn,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "type": "validation",
        "recommendations": None,
    }

    with st.spinner("Fetching prompt and recommendations..."):
        try:
            tfy_prompt_text = ""
            try:
                tfy_prompt_text = get_prompt_text_by_fqn(prompt_fqn)
            except Exception as tfy_error:
                st.warning(f"Unable to fetch prompt directly from TrueFoundry: {tfy_error}")

            data = post_chat(payload, include_grid_header=True)
            extracted_original_prompt = extract_original_prompt(data)
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
                    "Note: prompt content could not be loaded from TrueFoundry for this FQN."
                )
        except requests.RequestException as exc:
            st.error(f"Failed to fetch recommendations: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error while fetching recommendations: {exc}")


def apply_recommendations() -> None:
    prompt_fqn = st.session_state.prompt_fqn.strip()
    if not prompt_fqn:
        st.error("Prompt FQN is required. Fetch recommendations first.")
        return

    selected = st.session_state.editable_recommendations or st.session_state.selected_recommendations
    if not selected:
        st.error("Select at least one recommendation before applying.")
        return

    payload = {
        "sessionId": st.session_state.session_id,
        "promptFQN": prompt_fqn,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
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


def run_tests_with_file(tab: str) -> None:
    """Run tests using uploaded test cases."""
    fqn_key = f"{tab}_prompt_fqn"
    uploaded_key = f"{tab}_uploaded_tests"
    result_key = f"{tab}_result"
    debug_key = f"{tab}_api_debug"
    req_type = "verify_tests" if tab == "deepeval" else "verify_tests_exact"
    is_rag = st.session_state.get("deepeval_is_rag", False) if tab == "deepeval" else False

    prompt_fqn = st.session_state[fqn_key].strip()
    if not prompt_fqn:
        st.error("Please enter a Prompt FQN.")
        return

    uploaded_tests = st.session_state.get(uploaded_key)
    if not uploaded_tests:
        st.error("Please upload a test cases JSON file before running tests.")
        return

    payload = {
        "sessionId": st.session_state.session_id,
        "promptFQN": prompt_fqn,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "type": req_type,
        "isRAG": is_rag,
        "testCases": uploaded_tests,
        "recommendations": None,
    }

    spinner_msg = (
        "Running DeepEval tests — this may take several minutes..."
        if tab == "deepeval"
        else "Running exact match tests..."
    )
    with st.spinner(spinner_msg):
        try:
            data = post_chat(payload, include_grid_header=True)
            result = extract_test_evaluation_result(data)
            st.session_state[result_key] = result
            st.session_state[debug_key] = data
            if result:
                st.success("Tests completed successfully.")
            else:
                st.warning("Tests ran but no evaluation result was returned.")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error while running tests: {exc}")
