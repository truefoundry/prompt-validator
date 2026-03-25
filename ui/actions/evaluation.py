import requests
import streamlit as st

from ..api_client import post_chat
from ..extractors import extract_enhanced_prompt, extract_test_evaluation_result
from .helpers import _build_prompt_payload, _validate_prompt_input


def run_enhance_evaluation() -> None:
    """Run LLM-as-judge evaluation comparing original vs enhanced prompt."""
    original_prompt = st.session_state.get("enhance_eval_orig_prompt", "").strip()
    enhanced_prompt = st.session_state.get("enhance_eval_enh_prompt", "").strip()

    if not original_prompt:
        st.error("Please enter the original system prompt.")
        return
    if not enhanced_prompt:
        st.error("Please enter the enhanced system prompt.")
        return

    test_cases = st.session_state.get("enhance_eval_uploaded_tests")
    if not test_cases:
        st.error("Please enter a test input before running evaluation.")
        return

    reasoning = st.session_state.reasoning_effort
    enh_model = (st.session_state.get("enhance_eval_enh_model_name") or "").strip() or None
    enh_reasoning = st.session_state.get("enhance_eval_enh_reasoning_effort", "none")

    payload = {
        "sessionId": st.session_state.session_id,
        "systemPrompt": original_prompt,
        "enhancedSystemPrompt": enhanced_prompt,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "llm_judge",
        "isRAG": False,
        "testCases": test_cases,
        "recommendations": None,
        "enhancedModelName": enh_model,
        "enhancedTemperature": st.session_state.get("enhance_eval_enh_temperature") if enh_model else None,
        "enhancedMaxTokens": st.session_state.get("enhance_eval_enh_max_tokens") if enh_model else None,
        "enhancedReasoningEffort": (enh_reasoning if enh_reasoning != "none" else None) if enh_model else None,
        "judgeMetrics": st.session_state.get("enhance_eval_selected_metrics"),
    }

    with st.spinner("Running LLM-as-judge evaluation..."):
        try:
            data = post_chat(payload, include_grid_header=True)
            result = extract_test_evaluation_result(data)
            st.session_state.enhance_eval_judge_result = result
            st.session_state.enhance_eval_api_debug_original = data
            if result:
                st.success("Evaluation complete.")
            else:
                st.warning("Evaluation ran but no result was returned.")
        except requests.RequestException as exc:
            st.error(f"Evaluation failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def _generate_suggestions(
    *,
    judge_key: str,
    original_prompt_key: str,
    enhanced_prompt_key: str,
    result_key: str,
    spinner_msg: str,
    no_judge_error: str,
) -> None:
    """Generate holistic, priority-ranked suggestions from a judge result."""
    judge_result = st.session_state.get(judge_key)
    if not judge_result:
        st.error(no_judge_error)
        return

    original_prompt = st.session_state.get(original_prompt_key, "").strip()
    enhanced_prompt = st.session_state.get(enhanced_prompt_key, "").strip()
    reasoning = st.session_state.reasoning_effort

    payload = {
        "sessionId": st.session_state.session_id,
        "systemPrompt": original_prompt or "placeholder",
        "enhancedSystemPrompt": enhanced_prompt,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": 0.2,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "generate_suggestions",
        "recommendations": None,
        "judgeResult": judge_result,
    }

    with st.spinner(spinner_msg):
        try:
            data = post_chat(payload, include_grid_header=False)
            result = extract_test_evaluation_result(data)
            if result:
                st.session_state[result_key] = result
            else:
                st.warning("No suggestions returned.")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except Exception as exc:
            st.error(f"Suggestion generation failed: {exc}")


def generate_overall_suggestions() -> None:
    _generate_suggestions(
        judge_key="trace_llm_judge_result",
        original_prompt_key="trace_original_system_prompt",
        enhanced_prompt_key="trace_enhanced_system_prompt",
        result_key="trace_suggestions_result",
        spinner_msg="Generating overall suggestions...",
        no_judge_error="Run LLM Judge on traces first before generating suggestions.",
    )


def generate_enhance_suggestions() -> None:
    _generate_suggestions(
        judge_key="enhance_eval_judge_result",
        original_prompt_key="enhance_eval_orig_prompt",
        enhanced_prompt_key="enhance_eval_enh_prompt",
        result_key="enhance_eval_suggestions_result",
        spinner_msg="Generating suggestions...",
        no_judge_error="Run LLM Judge evaluation first before generating suggestions.",
    )


def run_deepeval_prompt_metrics() -> None:
    """Run DeepEval reference-free metrics by delegating to the backend."""
    judge_result = st.session_state.get("trace_llm_judge_result")
    if not judge_result:
        st.error("Run LLM Judge on traces first — DeepEval metrics use those outputs.")
        return

    test_results = judge_result.get("test_results", [])
    test_cases = [
        {
            "input": tr.get("input", ""),
            "original_output": tr.get("original_output", ""),
            "enhanced_output": tr.get("enhanced_output", ""),
        }
        for tr in test_results
        if tr.get("input") and tr.get("original_output") and tr.get("enhanced_output")
    ]
    if not test_cases:
        st.error("No valid test outputs found — run evaluation first.")
        return

    geval_criteria = st.session_state.get("trace_deepeval_metrics_geval_criteria", "").strip()
    raw_instructions = st.session_state.get("trace_deepeval_metrics_prompt_instructions", "").strip()
    prompt_instructions = [l.strip() for l in raw_instructions.splitlines() if l.strip()] if raw_instructions else []
    reasoning = st.session_state.reasoning_effort

    payload = {
        "sessionId": st.session_state.session_id,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": 0.0,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "deepeval_prompt_metrics",
        "recommendations": None,
        "testCases": test_cases,
        "gevalCriteria": geval_criteria,
        "promptInstructions": prompt_instructions,
    }

    with st.spinner("Running DeepEval metrics (this may take a moment)..."):
        try:
            data = post_chat(payload, include_grid_header=False)
            result = extract_test_evaluation_result(data)
            if result:
                st.session_state.trace_deepeval_metrics_result = result
            else:
                st.warning("DeepEval metrics returned no result.")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except Exception as exc:
            st.error(f"DeepEval metrics failed: {exc}")


def apply_judge_recommendations() -> None:
    """Apply judge-derived recommendations to the original prompt and store as enhanced."""
    original_prompt = st.session_state.get("enhance_eval_orig_prompt", "").strip()
    if not original_prompt:
        st.error("Please enter the original system prompt before applying recommendations.")
        return

    selected = st.session_state.get("enhance_eval_judge_recs_selected", [])
    if not selected:
        st.error("Select at least one recommendation to apply.")
        return

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        "systemPrompt": original_prompt,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "validation",
        "recommendations": selected,
    }

    with st.spinner("Generating enhanced prompt from judge recommendations..."):
        try:
            data = post_chat(payload, include_grid_header=False)
            enhanced = extract_enhanced_prompt(data)
            if enhanced:
                st.session_state.enhance_eval_enh_prompt = enhanced
                st.success("Enhanced prompt generated — check the Enhanced System Prompt field above.")
            else:
                st.warning("API returned no enhanced prompt.")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def run_tests_with_file(tab: str) -> None:
    """Run tests using uploaded test cases."""
    mode_key = f"{tab}_input_mode"
    fqn_key = f"{tab}_prompt_fqn"
    sys_key = f"{tab}_system_prompt"
    user_tpl_key = f"{tab}_user_prompt_template"
    uploaded_key = f"{tab}_uploaded_tests"
    result_key = f"{tab}_result"
    debug_key = f"{tab}_api_debug"
    req_type = "verify_tests" if tab == "deepeval" else "verify_tests_exact"
    is_rag = st.session_state.get("deepeval_is_rag", False) if tab == "deepeval" else False

    if not _validate_prompt_input(mode_key, fqn_key, sys_key):
        return

    uploaded_tests = st.session_state.get(uploaded_key)
    if not uploaded_tests:
        st.error("Please upload a test cases JSON file before running tests.")
        return

    prompt_fields = _build_prompt_payload(mode_key, fqn_key, sys_key, user_tpl_key)

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        **prompt_fields,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
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
