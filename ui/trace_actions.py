"""Orchestration actions for the Trace Evaluation tab."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import requests
import streamlit as st

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trace.trace_parser import TraceInput, fetch_live_spans, load_spans_from_file, parse_spans_to_inputs

from .api_client import post_chat
from .extractors import extract_enhanced_prompt, extract_recommendations, extract_test_evaluation_result


@st.cache_data
def load_floqast_prompts() -> list[dict]:
    """Load prompts from prompts-floqast.json, return sorted list of {name, fqn, latest_version_fqn}."""
    try:
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts-floqast.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = []
        for entry in data.get("data", []):
            latest = entry.get("latest_version") or {}
            prompts.append({
                "name": entry.get("name", ""),
                "fqn": entry.get("fqn", ""),
                "latest_version_fqn": latest.get("fqn", ""),
            })
        return sorted(prompts, key=lambda p: p["name"])
    except Exception:
        return []


def load_trace_inputs(
    source: str,
    uploaded_content: str | None = None,
) -> None:
    """Load spans from file or upload, parse into TraceInput, store in session state.

    Args:
        source: One of "traces.json", "upload"
        uploaded_content: Raw JSON string when source == "upload"
    """
    with st.spinner("Loading traces..."):
        try:
            if source == "traces.json":
                path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "trace",
                    "traces.json",
                )
                spans = load_spans_from_file(path)
            elif source == "upload" and uploaded_content:
                spans = json.loads(uploaded_content)
            else:
                st.error("Invalid trace source configuration.")
                return

            inputs = parse_spans_to_inputs(spans)
            if not inputs:
                skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
                if not spans:
                    st.warning("No spans found in the file.")
                else:
                    st.warning(
                        f"Loaded {len(spans)} span(s) but none passed parsing. "
                        f"Skip reasons: {skip_reasons or 'unknown'}"
                    )
                st.session_state.trace_inputs = []
                st.session_state.trace_selected_indices = []
                return

            fqns = sorted({ti.prompt_fqn for ti in inputs if ti.prompt_fqn})
            st.session_state.trace_inputs = inputs
            st.session_state.trace_selected_indices = []
            skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
            skipped = len(spans) - len(inputs)
            skip_note = f"  ({skipped} skipped: {skip_reasons})" if skipped else ""
            st.success(
                f"Loaded {len(inputs)} traces from {len(spans)} spans  |  "
                f"{len(fqns)} prompt FQN(s) found.{skip_note}"
            )
        except FileNotFoundError:
            st.error("traces.json not found. Run trace/fetch_trace.py first or upload a file.")
        except Exception as exc:
            st.error(f"Failed to load traces: {exc}")


def fetch_live_trace_inputs(
    hours: int = 24,
    limit: int = 200,
    fqn_filter: str | None = None,
    tfy_host: str = "",
    tfy_api_key: str = "",
) -> None:
    """Fetch live ChatCompletion spans from TrueFoundry.

    Args:
        hours: How many hours back to fetch (default 24).
        limit: Max spans to fetch — caps SDK pagination for fast response (default 200).
        fqn_filter: Optional prompt FQN substring to filter client-side.
        tfy_host: TrueFoundry tenant base URL (overrides trace/.env).
        tfy_api_key: TrueFoundry API key (overrides trace/.env).
    """
    with st.spinner(f"Fetching live traces (last {hours // 24}d, max {limit} spans)..."):
        try:
            spans = fetch_live_spans(
                hours=hours,
                limit=limit,
                prompt_fqn_filter=fqn_filter or None,
                tfy_host=tfy_host or None,
                tfy_api_key=tfy_api_key or None,
            )
            if not spans:
                st.warning("No ChatCompletion spans found in the given time range.")
                st.session_state.trace_inputs = []
                st.session_state.trace_selected_indices = []
                return

            inputs = parse_spans_to_inputs(spans)
            if not inputs:
                skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
                st.warning(
                    f"Fetched {len(spans)} span(s) but none passed parsing. "
                    f"Skip reasons: {skip_reasons or 'unknown'}"
                )
                st.session_state.trace_inputs = []
                st.session_state.trace_selected_indices = []
                return

            fqns = sorted({ti.prompt_fqn for ti in inputs if ti.prompt_fqn})
            st.session_state.trace_inputs = inputs
            st.session_state.trace_selected_indices = []
            skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
            skipped = len(spans) - len(inputs)
            skip_note = f"  ({skipped} skipped: {skip_reasons})" if skipped else ""
            st.success(
                f"Fetched {len(inputs)} traces from {len(spans)} spans  |  "
                f"{len(fqns)} prompt FQN(s) found.{skip_note}"
            )
        except Exception as exc:
            st.error(f"Failed to fetch live traces: {exc}")


def run_trace_pipeline() -> None:
    """Run the full automated pipeline on selected trace(s):
    Step 1 — get_recommendation on the system prompt.
    Step 2 — apply_recommendation to produce an enhanced prompt.
    Step 3 — run llm_judge using the same model from the trace to compare quality.
    """
    selected_indices = st.session_state.get("trace_selected_indices", [])
    all_inputs: list[TraceInput] = st.session_state.get("trace_inputs", [])

    if not selected_indices:
        st.error("Select at least one trace row before running the pipeline.")
        return

    selected_inputs = [all_inputs[i] for i in selected_indices if i < len(all_inputs)]
    if not selected_inputs:
        st.error("No valid trace rows selected.")
        return

    original_sys = selected_inputs[0].system_prompt.strip()
    if not original_sys:
        st.error("The selected trace has no system prompt to enhance.")
        return

    # Use the model configured in the sidebar — not the trace's model
    trace_model = (st.session_state.get("model_name") or "").strip() or None

    reasoning = st.session_state.get("reasoning_effort", "low")
    common = {
        "sessionId": st.session_state.get("session_id", "123"),
        "systemPrompt": original_sys,
        "modelName": trace_model,
        "maxTokens": st.session_state.get("max_tokens", 15000),
        "temperature": st.session_state.get("temperature", 0.1),
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "recommendations": None,
    }

    # ── Step 1: Get recommendations ─────────────────────────────────────────
    with st.spinner("Step 1/3 — Analyzing prompt and fetching recommendations..."):
        try:
            rec_data = post_chat({**common, "type": "validation"}, include_grid_header=True)
            recommendations = extract_recommendations(rec_data)
            if not recommendations:
                st.error("No recommendations returned. Cannot proceed with pipeline.")
                return
            st.session_state.trace_pipeline_recommendations = recommendations
        except Exception as exc:
            st.error(f"Step 1 failed (get_recommendation): {exc}")
            return

    # ── Step 2: Apply recommendations ───────────────────────────────────────
    with st.spinner("Step 2/3 — Applying recommendations to enhance prompt..."):
        try:
            enh_data = post_chat(
                {**common, "type": "validation", "recommendations": recommendations},
                include_grid_header=False,
            )
            enhanced_prompt = extract_enhanced_prompt(enh_data)
            if not enhanced_prompt:
                st.error("No enhanced prompt returned. Cannot proceed with pipeline.")
                return
            st.session_state.trace_original_system_prompt = original_sys
            st.session_state.trace_enhanced_system_prompt = enhanced_prompt
        except Exception as exc:
            st.error(f"Step 2 failed (apply_recommendation): {exc}")
            return

    # ── Step 3: LLM judge comparison ────────────────────────────────────────
    test_cases = [
        {
            "test_case_id": str(i),
            "test_case_name": f"Trace {ti.span_id[:8]} — {ti.user_message[:40]}",
            "data": {"input": ti.user_message},
            "expected_output": "",
        }
        for i, ti in enumerate(selected_inputs)
    ]

    with st.spinner("Step 3/3 — Running LLM judge to compare original vs enhanced..."):
        try:
            judge_data = post_chat({
                "sessionId": st.session_state.get("session_id", "123"),
                "type": "llm_judge",
                "systemPrompt": original_sys,
                "enhancedSystemPrompt": enhanced_prompt,
                "modelName": trace_model,
                "maxTokens": st.session_state.get("max_tokens", 15000),
                "temperature": st.session_state.get("temperature", 0.1),
                "reasoningEffort": reasoning if reasoning != "none" else None,
                "recommendations": None,
                "testCases": test_cases,
            }, include_grid_header=False)
            result = extract_test_evaluation_result(judge_data)
            st.session_state.trace_llm_judge_result = result
            st.session_state.trace_llm_judge_api_debug = judge_data
            st.success("Pipeline complete. Scroll down to see results.")
        except Exception as exc:
            st.error(f"Step 3 failed (llm_judge): {exc}")


def build_trace_examples_for_recommendation(
    trace_inputs: list[TraceInput],
    selected_indices: list[int],
) -> list[dict]:
    """Convert selected TraceInput objects into (input, output) pairs for behavioral recommendation.

    Args:
        trace_inputs: All loaded TraceInput objects
        selected_indices: Indices into trace_inputs to include
    Returns:
        List of {"input": user_message, "output": trace_output} dicts
    """
    examples = []
    for idx in selected_indices:
        if idx < len(trace_inputs):
            ti = trace_inputs[idx]
            examples.append({"input": ti.user_message, "output": ti.trace_output})
    return examples


def build_test_cases_from_traces(
    selected_inputs: list[TraceInput],
    expected_mode: str,
) -> list[dict]:
    """Convert selected TraceInput objects into the test case format for /chat endpoint.

    Args:
        selected_inputs: TraceInput objects chosen by the user
        expected_mode: "use_trace_output" | "no_expected_output"
    """
    test_cases = []
    for i, ti in enumerate(selected_inputs):
        tc: dict = {
            "test_case_id": str(i),
            "test_case_name": f"Trace {ti.span_id[:8]} — {ti.user_message[:40]}",
            "data": {"input": ti.user_message},
        }
        if expected_mode == "use_trace_output" and ti.trace_output:
            tc["expected_output"] = ti.trace_output
        else:
            tc["expected_output"] = ""
        test_cases.append(tc)
    return test_cases


def run_llm_judge_on_traces() -> None:
    """Run LLM-as-judge comparing original vs enhanced prompt on selected trace inputs."""
    selected_indices = st.session_state.get("trace_selected_indices", [])
    all_inputs: list[TraceInput] = st.session_state.get("trace_inputs", [])

    if not selected_indices:
        st.error("Select at least one trace row before running evaluation.")
        return

    original_sys = st.session_state.get("trace_original_system_prompt", "").strip()
    enhanced_sys = st.session_state.get("trace_enhanced_system_prompt", "").strip()
    user_tpl = st.session_state.get("trace_user_prompt_template", "").strip()

    if not original_sys:
        st.error("Please paste the original system prompt.")
        return
    if not enhanced_sys:
        st.error("Please paste the enhanced system prompt.")
        return

    selected_inputs = [all_inputs[i] for i in selected_indices if i < len(all_inputs)]
    if not selected_inputs:
        st.error("No valid trace rows selected.")
        return

    test_cases = [
        {
            "test_case_id": str(i),
            "test_case_name": f"Trace {ti.span_id[:8]} — {ti.user_message[:40]}",
            "data": {"input": ti.user_message},
            "expected_output": "",
        }
        for i, ti in enumerate(selected_inputs)
    ]

    reasoning = st.session_state.get("reasoning_effort", "low")
    payload = {
        "sessionId": st.session_state.get("session_id", "123"),
        "type": "llm_judge",
        "systemPrompt": original_sys,
        "enhancedSystemPrompt": enhanced_sys,
        "modelName": (st.session_state.get("model_name") or "").strip() or None,
        "maxTokens": st.session_state.get("max_tokens", 15000),
        "temperature": st.session_state.get("temperature", 0.1),
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "recommendations": None,
        "testCases": test_cases,
    }
    if user_tpl:
        payload["userPromptTemplate"] = user_tpl

    with st.spinner(f"Running LLM judge on {len(test_cases)} trace input(s)..."):
        try:
            data = post_chat(payload, include_grid_header=False)
            result = extract_test_evaluation_result(data)
            st.session_state.trace_llm_judge_result = result
            st.session_state.trace_llm_judge_api_debug = data
            st.success("LLM judge evaluation complete.")
        except requests.RequestException as exc:
            st.error(f"LLM judge evaluation failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def run_trace_evaluation() -> None:
    """Run DeepEval on both original and enhanced prompt using selected trace inputs."""
    selected_indices = st.session_state.get("trace_selected_indices", [])
    all_inputs: list[TraceInput] = st.session_state.get("trace_inputs", [])

    if not selected_indices:
        st.error("Select at least one trace row before running evaluation.")
        return

    selected_inputs = [all_inputs[i] for i in selected_indices if i < len(all_inputs)]
    if not selected_inputs:
        st.error("No valid trace rows selected.")
        return

    expected_mode = st.session_state.get("trace_expected_output_mode", "use_trace_output")
    test_cases = build_test_cases_from_traces(selected_inputs, expected_mode)

    # Validate original prompt
    orig_mode = st.session_state.get("trace_eval_original_prompt_mode", "TFY Prompt FQN")
    if orig_mode == "TFY Prompt FQN":
        if not st.session_state.get("trace_eval_original_fqn", "").strip():
            st.error("Please enter an original Prompt FQN.")
            return
        orig_fields = {"promptFQN": st.session_state["trace_eval_original_fqn"].strip()}
    else:
        if not st.session_state.get("trace_eval_original_system_prompt", "").strip():
            st.error("Please enter a system prompt for the original prompt.")
            return
        orig_fields = {"systemPrompt": st.session_state["trace_eval_original_system_prompt"].strip()}
        user_tpl = st.session_state.get("trace_eval_original_user_template", "").strip()
        if user_tpl:
            orig_fields["userPromptTemplate"] = user_tpl

    # Validate enhanced prompt
    enhanced_sys = st.session_state.get("trace_eval_enhanced_system_prompt", "").strip()
    if not enhanced_sys:
        st.error("Please enter the enhanced system prompt.")
        return
    enhanced_fields: dict = {"systemPrompt": enhanced_sys}
    enhanced_user_tpl = st.session_state.get("trace_eval_enhanced_user_template", "").strip()
    if enhanced_user_tpl:
        enhanced_fields["userPromptTemplate"] = enhanced_user_tpl

    reasoning = st.session_state.get("reasoning_effort", "low")
    base_payload = {
        "sessionId": st.session_state.get("session_id", "123"),
        "modelName": (st.session_state.get("model_name") or "").strip() or None,
        "maxTokens": st.session_state.get("max_tokens", 15000),
        "temperature": st.session_state.get("temperature", 0.1),
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "verify_tests",
        "isRAG": False,
        "testCases": test_cases,
        "recommendations": None,
    }

    with st.spinner("Running trace evaluation on original prompt..."):
        try:
            orig_data = post_chat({**base_payload, **orig_fields}, include_grid_header=True)
            orig_result = extract_test_evaluation_result(orig_data)
            st.session_state.trace_eval_original_result = orig_result
            st.session_state.trace_eval_api_debug_original = orig_data
        except requests.RequestException as exc:
            st.error(f"Original prompt evaluation failed: {exc}")
            return
        except Exception as exc:
            st.error(f"Unexpected error (original): {exc}")
            return

    with st.spinner("Running trace evaluation on enhanced prompt..."):
        try:
            enh_data = post_chat({**base_payload, **enhanced_fields}, include_grid_header=True)
            enh_result = extract_test_evaluation_result(enh_data)
            st.session_state.trace_eval_enhanced_result = enh_result
            st.session_state.trace_eval_api_debug_enhanced = enh_data
        except requests.RequestException as exc:
            st.error(f"Enhanced prompt evaluation failed: {exc}")
            return
        except Exception as exc:
            st.error(f"Unexpected error (enhanced): {exc}")
            return

    # Compute delta
    delta = _compute_delta(orig_result, enh_result)
    st.session_state.trace_eval_delta = delta
    st.success("Trace evaluation complete.")


def _compute_delta(orig: dict | None, enh: dict | None) -> dict:
    if not orig or not enh:
        return {}

    def _pass_rate(result: dict) -> float:
        inner = result.get("results", result)
        all_tests = inner.get("all_tests", [])
        if not all_tests:
            return 0.0
        passed = sum(1 for t in all_tests if t.get("pass_status"))
        return passed / len(all_tests) * 100

    def _avg_correctness(result: dict) -> float:
        inner = result.get("results", result)
        metrics = inner.get("all_metrics", {})
        # Try common key names
        for key in ("Average GEval Score", "Average Correctness", "Average Score"):
            if key in metrics:
                return float(metrics[key])
        return 0.0

    orig_pass = _pass_rate(orig)
    enh_pass = _pass_rate(enh)
    orig_corr = _avg_correctness(orig)
    enh_corr = _avg_correctness(enh)

    return {
        "pass_rate_original": orig_pass,
        "pass_rate_enhanced": enh_pass,
        "pass_rate_delta": enh_pass - orig_pass,
        "avg_correctness_original": orig_corr,
        "avg_correctness_enhanced": enh_corr,
        "avg_correctness_delta": enh_corr - orig_corr,
    }
