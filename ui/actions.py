import json

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


def _is_paste_mode(mode_key: str) -> bool:
    return st.session_state.get(mode_key) == "Paste Prompt Text"


def _build_prompt_payload(mode_key: str, fqn_key: str, sys_key: str, user_tpl_key: str) -> dict:
    """Return the prompt-identification fields for the API payload.

    In FQN mode the dict contains ``promptFQN``; in paste mode it contains
    ``systemPrompt`` and optionally ``userPromptTemplate``.
    """
    if _is_paste_mode(mode_key):
        result: dict = {"systemPrompt": st.session_state.get(sys_key, "").strip()}
        user_tpl = st.session_state.get(user_tpl_key, "").strip()
        if user_tpl:
            result["userPromptTemplate"] = user_tpl
        return result
    return {"promptFQN": st.session_state.get(fqn_key, "").strip()}


def _validate_prompt_input(mode_key: str, fqn_key: str, sys_key: str) -> bool:
    """Validate that at least one prompt source has been provided. Returns True when valid."""
    if _is_paste_mode(mode_key):
        if not st.session_state.get(sys_key, "").strip():
            st.error("Please enter a system prompt before proceeding.")
            return False
    else:
        if not st.session_state.get(fqn_key, "").strip():
            st.error("Please enter a Prompt FQN before proceeding.")
            return False
    return True


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


def generate_overall_suggestions() -> None:
    """Generate holistic, priority-ranked suggestions from trace evaluation results via a single LLM call."""
    judge_result = st.session_state.get("trace_llm_judge_result")
    if not judge_result:
        st.error("Run LLM Judge on traces first before generating suggestions.")
        return

    original_prompt = st.session_state.get("trace_original_system_prompt", "").strip()
    enhanced_prompt = st.session_state.get("trace_enhanced_system_prompt", "").strip()

    summary = judge_result.get("summary", {})
    test_results = judge_result.get("test_results", [])
    metrics = ["clarity", "completeness", "accuracy", "conciseness", "professional_tone", "overall"]

    # Build metrics table string
    avg_orig = summary.get("avg_original", {})
    avg_enh = summary.get("avg_enhanced", {})
    avg_delta = summary.get("avg_delta", {})
    metrics_lines = ["Metric | Original | Enhanced | Delta"]
    metrics_lines.append("--- | --- | --- | ---")
    for m in metrics:
        o = avg_orig.get(m)
        e = avg_enh.get(m)
        d = avg_delta.get(m)
        metrics_lines.append(
            f"{m.replace('_', ' ').title()} | "
            f"{f'{o:.2f}' if o is not None else '—'} | "
            f"{f'{e:.2f}' if e is not None else '—'} | "
            f"{f'{d:+.2f}' if d is not None else '—'}"
        )
    metrics_table = "\n".join(metrics_lines)

    # Collect all key differences and improvement summaries
    all_differences: list[str] = []
    all_summaries: list[str] = []
    for i, tr in enumerate(test_results):
        scores = tr.get("scores", {})
        for d in scores.get("key_differences", []):
            if d and d not in all_differences:
                all_differences.append(d)
        s = scores.get("improvement_summary", "")
        if s:
            all_summaries.append(f"Test {i+1}: {s}")

    improved = summary.get("improved_count", 0)
    total = summary.get("total_cases", 0)

    system_prompt = """You are an expert prompt engineer. You will be given evaluation data comparing an original system prompt against an enhanced version, tested on real inputs.

Your job is to analyse the evaluation holistically and generate a concise, prioritised list of actionable improvement suggestions for the enhanced prompt.

Output ONLY valid JSON. No markdown, no explanation outside the JSON.

{
  "overall_analysis": "<2-3 sentence summary of what the evaluation reveals overall>",
  "suggestions": [
    {
      "priority": "HIGH" | "MEDIUM" | "LOW",
      "title": "<short title>",
      "suggestion": "<concrete, actionable instruction for improving the enhanced prompt>",
      "rationale": "<why this matters based on the metrics and differences observed>"
    }
  ]
}

Rules:
- Prioritise by impact: HIGH = fixes a consistent regression or large delta gap, MEDIUM = addresses a pattern seen in multiple tests, LOW = polish or edge-case improvement
- Suggestions must be specific and actionable (e.g. "Add a fallback instruction for X scenario" not "improve clarity")
- Do not suggest things already clearly working well
- Maximum 6 suggestions, minimum 2
- Base everything strictly on the data provided"""

    user_message = f"""## Evaluation Summary
Improved in {improved}/{total} test case(s).

## Metric Scores (averages across all test cases)
{metrics_table}

## Key Differences Observed
{chr(10).join(f'- {d}' for d in all_differences) if all_differences else 'None noted.'}

## Per-Test Improvement Summaries
{chr(10).join(all_summaries) if all_summaries else 'None noted.'}

## Original Prompt (first 800 chars)
{original_prompt[:800]}{'...' if len(original_prompt) > 800 else ''}

## Enhanced Prompt (first 800 chars)
{enhanced_prompt[:800]}{'...' if len(enhanced_prompt) > 800 else ''}"""

    with st.spinner("Generating overall suggestions..."):
        try:
            from src.chat.utils.llm_models import TrueFoundryLLM, get_truefoundry_llm
            from langchain.schema import SystemMessage, HumanMessage
            llm = TrueFoundryLLM(model=get_truefoundry_llm(
                model_name=st.session_state.model_name.strip() if st.session_state.model_name else None,
                max_tokens=st.session_state.max_tokens,
                temperature=0.2,
            ))
            raw = llm.generate([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
            st.session_state.trace_suggestions_result = result
        except json.JSONDecodeError as exc:
            st.error(f"LLM returned non-JSON: {exc}\n\nRaw: {raw[:300]}")
        except Exception as exc:
            st.error(f"Suggestion generation failed: {exc}")


def run_deepeval_prompt_metrics() -> None:
    """Run DeepEval reference-free metrics on trace judge evaluation results."""
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

    with st.spinner("Running DeepEval metrics (this may take a moment)..."):
        try:
            from src.chat.utils.llm_models import TrueFoundryLLM, get_truefoundry_llm
            from src.chat.graph.deepeval_prompt_metrics import run_deepeval_prompt_metrics as _run
            llm = TrueFoundryLLM(model=get_truefoundry_llm(
                model_name=st.session_state.model_name.strip() if st.session_state.model_name else None,
                max_tokens=st.session_state.max_tokens,
                temperature=0.0,
                reasoning_effort=st.session_state.get("reasoning_effort") or None,
            ))
            result = _run(
                test_cases=test_cases,
                llm=llm,
                geval_criteria=geval_criteria,
                prompt_instructions=prompt_instructions,
            )
            st.session_state.trace_deepeval_metrics_result = result
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
