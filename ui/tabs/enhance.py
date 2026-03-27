import json
import os

import pandas as pd
import requests
import streamlit as st

from ..actions import apply_recommendations, run_enhance_evaluation, generate_enhance_suggestions
from ..api_client import post_chat
from ..extractors import extract_enhanced_prompt, extract_test_evaluation_result
from ..components import render_diff, render_prompt_html, render_section_header, render_selected_recommendations_editor

def _load_global_css() -> str:
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style.css")
    with open(css_path) as f:
        return f"<style>{f.read()}</style>"

_FRAGMENT_CSS = _load_global_css()
from src.chat.graph.llm_judge_evaluator import (
    AVAILABLE_METRICS as _METRIC_DESCRIPTIONS,
    DEFAULT_METRICS,
    ALL_METRIC_KEYS,
)


def _load_models() -> list[str]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "models.json")) as f:
            return sorted(json.load(f).get("models", []))
    except Exception:
        return []

_MODELS = _load_models()


_PRIORITY_CONFIG = {
    "HIGH":   {"color": "#e74c3c", "icon": "🔴"},
    "MEDIUM": {"color": "#f39c12", "icon": "🟡"},
    "LOW":    {"color": "#27ae60", "icon": "🟢"},
}


def _render_overall_suggestions(result: dict, selected_key: str = "trace_suggestions_selected", on_apply=None) -> None:
    st.divider()
    st.write("#### Suggestions")

    analysis = result.get("overall_analysis", "")
    if analysis:
        st.info(analysis)

    suggestions = result.get("suggestions", [])
    if not suggestions:
        st.warning("No suggestions returned.")
        return

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    suggestions = sorted(suggestions, key=lambda s: priority_order.get(s.get("priority", "LOW"), 3))

    st.caption("Select suggestions to apply to the enhanced prompt, then click **Apply Selected**.")

    selected_texts: list[str] = []
    for i, s in enumerate(suggestions):
        priority = s.get("priority", "LOW").upper()
        cfg = _PRIORITY_CONFIG.get(priority, _PRIORITY_CONFIG["LOW"])
        title = s.get("title", "")
        suggestion_text = s.get("suggestion", "")
        rationale = s.get("rationale", "")

        col_check, col_card = st.columns([0.04, 0.96])
        checked = col_check.checkbox("Select", key=f"{selected_key}_{i}", label_visibility="collapsed")
        if checked:
            selected_texts.append(suggestion_text)

        col_card.markdown(
            f"""<div style="border-left: 4px solid {cfg['color']}; padding: 10px 14px; border-radius: 6px; margin-bottom: 6px;">
<span style="font-size:0.72rem; font-weight:700; color:{cfg['color']}; text-transform:uppercase; letter-spacing:0.05em;">{cfg['icon']} {priority}</span>
<div style="font-weight:600; margin: 4px 0 4px 0; font-size:0.93rem;">{title}</div>
<div style="font-size:0.88rem; margin-bottom:5px;">{suggestion_text}</div>
<div style="font-size:0.78rem; opacity:0.65;"><em>{rationale}</em></div>
</div>""",
            unsafe_allow_html=True,
        )

    st.session_state[selected_key] = selected_texts

    if selected_texts:
        st.caption(f"{len(selected_texts)} suggestion(s) selected")
        apply_fn = on_apply or _apply_suggestions_to_trace_prompt
        if st.button("Apply Selected → Refine Enhanced Prompt", key=f"{selected_key}_apply_btn", type="primary"):
            apply_fn(selected_texts)


def _apply_suggestions(
    suggestions: list[str],
    *,
    source_key: str,
    result_key: str,
    clear_keys: list[str],
    spinner_msg: str,
    success_msg: str,
    promote_to: str | None = None,
    promote_area_to: str | None = None,
    result_area_key: str | None = None,
    source_error_msg: str = "No prompt found.",
) -> None:
    """Apply selected suggestions via the validation API and store the refined prompt."""
    source_prompt = st.session_state.get(source_key, "").strip()
    if not source_prompt:
        st.error(source_error_msg)
        return

    if promote_to:
        st.session_state[promote_to] = source_prompt
    # Don't write promote_area_to / result_area_key directly here — those keys are
    # bound to already-rendered st.text_area widgets and Streamlit forbids mutating
    # them after instantiation.  Instead, stage the values under _pending_* keys and
    # flush them at the top of render_enhance_tab() before any widgets are created.
    if promote_area_to:
        st.session_state[f"_pending_{promote_area_to}"] = source_prompt

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        "systemPrompt": source_prompt,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "validation",
        "recommendations": suggestions,
    }
    with st.spinner(spinner_msg):
        try:
            data = post_chat(payload, include_grid_header=False)
            refined = extract_enhanced_prompt(data)
            if refined:
                st.session_state[result_key] = refined
                if result_area_key:
                    st.session_state[f"_pending_{result_area_key}"] = refined
                for key in clear_keys:
                    st.session_state[key] = None
                st.success(success_msg)
                st.rerun(scope="fragment")
            else:
                st.warning("API returned no refined prompt.")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def _apply_suggestions_to_trace_prompt(suggestions: list[str]) -> None:
    _apply_suggestions(
        suggestions,
        source_key="trace_enhanced_system_prompt",
        result_key="trace_enhanced_system_prompt",
        clear_keys=["trace_llm_judge_result", "trace_suggestions_result", "trace_deepeval_metrics_result"],
        spinner_msg="Applying suggestions to generate refined prompt...",
        success_msg="Refined prompt applied. Original → previous enhanced. Run the judge again to measure improvement.",
        promote_to="trace_original_system_prompt",
        source_error_msg="No enhanced prompt found — run the pipeline or judge first.",
    )


def _apply_suggestions_to_enhance_prompt(suggestions: list[str]) -> None:
    _apply_suggestions(
        suggestions,
        source_key="enhance_eval_enh_prompt",
        result_key="enhance_eval_enh_prompt",
        clear_keys=["enhance_eval_judge_result", "enhance_eval_suggestions_result"],
        spinner_msg="Applying suggestions to generate refined enhanced prompt...",
        success_msg="Refined prompt applied. Previous enhanced is now original. Run evaluation again to measure improvement.",
        promote_to="enhance_eval_orig_prompt",
        promote_area_to="enhance_eval_orig_area",
        result_area_key="enhance_eval_enh_area",
        source_error_msg="No enhanced prompt found — run LLM Judge evaluation first.",
    )


@st.fragment
def render_enhance_tab() -> None:
    # Re-inject global CSS on every fragment rerun so pt-section-header and
    # other style.css classes don't disappear when a button/radio triggers
    # a fragment-scoped re-render.
    st.markdown(_FRAGMENT_CSS, unsafe_allow_html=True)

    # Flush any pending text-area values staged by _apply_suggestions.
    # Must run BEFORE the corresponding st.text_area widgets are instantiated;
    # Streamlit raises StreamlitAPIException if a widget-bound key is mutated
    # after the widget has been created in the same script run.
    for _area_key in ("enhance_eval_orig_area", "enhance_eval_enh_area"):
        _pk = f"_pending_{_area_key}"
        if _pk in st.session_state:
            st.session_state[_area_key] = st.session_state.pop(_pk)

    st.markdown(
        """<div style="margin-bottom:20px;">
  <h2 style="margin:0 0 4px 0;font-size:1.3rem;font-weight:700;color:#1e293b;">
    Enhance Prompt
  </h2>
  <p style="margin:0;font-size:0.875rem;color:#64748b;line-height:1.5;">
    Apply selected recommendations to produce an enhanced prompt, then evaluate
    original vs enhanced side-by-side using LLM-as-judge.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

    render_section_header("📄", "Original Prompt", "View the current prompt before enhancement", step=1)
    with st.expander("Show Original Prompt", expanded=False):
        render_prompt_html(
            st.session_state.original_prompt,
            label="Original Prompt",
            max_height=320,
        )

    render_section_header("✏️", "Edit & Apply Recommendations", "Review and tweak the recommendations, then apply", step=2)
    render_selected_recommendations_editor()

    if st.button(
        "⚡ Apply Recommendations & Enhance",
        width="stretch",
        type="primary",
        key="apply_recs_enhance_btn",
    ):
        apply_recommendations()

    st.divider()

    if st.session_state.enhanced_prompt:
        view_mode = st.radio(
            "View",
            ("Enhanced Prompt", "Diff View"),
            horizontal=True,
            key="enhance_view_mode",
        )
        if view_mode == "Enhanced Prompt":
            render_prompt_html(
                st.session_state.enhanced_prompt,
                label="Enhanced Prompt",
                max_height=520,
            )
            st.download_button(
                label="Save Enhanced Prompt",
                data=st.session_state.enhanced_prompt,
                file_name="enhanced_prompt.txt",
                mime="text/plain",
                width="stretch",
                key="download_enhanced_prompt_tab",
            )
        else:
            render_diff(st.session_state.original_prompt, st.session_state.enhanced_prompt, key="diff_enhance_a")
    else:
        render_diff(st.session_state.original_prompt, st.session_state.enhanced_prompt, key="diff_enhance_b")

    with st.expander("Debug: Last API Response", expanded=False):
        st.json(st.session_state.api_response_debug)

    # ── Evaluate Original vs Enhanced ───────────────────────────────────────
    st.divider()
    render_section_header("⚖️", "Evaluate Original vs Enhanced",
                          "Enter both prompts, add test inputs, then run the LLM judge to compare quality", step=3)

    # Auto-populate from session whenever the source prompt changes
    orig = st.session_state.get("original_prompt", "")
    enh = st.session_state.get("enhanced_prompt", "")
    if orig and orig != st.session_state.get("_last_synced_orig_prompt"):
        st.session_state.enhance_eval_orig_area = orig
        st.session_state["_last_synced_orig_prompt"] = orig
    if enh and enh != st.session_state.get("_last_synced_enh_prompt"):
        st.session_state.enhance_eval_enh_area = enh
        st.session_state["_last_synced_enh_prompt"] = enh

    # After suggestion iterations, the rolled-forward prompts live in
    # enhance_eval_orig/enh_prompt.  Prefer those over the top-level
    # original_prompt / enhanced_prompt so the buttons always reflect the
    # current iteration state, not the stale first-run values.
    col_btn_orig, col_btn_enh = st.columns(2)
    with col_btn_orig:
        if st.button("Use session original prompt", key="eval_use_orig_btn", width="stretch"):
            val = (
                st.session_state.get("enhance_eval_orig_prompt", "").strip()
                or st.session_state.get("original_prompt", "")
            )
            st.session_state.enhance_eval_orig_prompt = val
            st.session_state.enhance_eval_orig_area = val
    with col_btn_enh:
        if st.button("Use session enhanced prompt", key="eval_use_enh_btn", width="stretch"):
            val = (
                st.session_state.get("enhance_eval_enh_prompt", "").strip()
                or st.session_state.get("enhanced_prompt", "")
            )
            st.session_state.enhance_eval_enh_prompt = val
            st.session_state.enhance_eval_enh_area = val

    col_orig, col_enh = st.columns(2)
    with col_orig:
        st.text_area(
            "Original System Prompt",
            height=250,
            placeholder="Paste original system prompt here...",
            key="enhance_eval_orig_area",
        )
        st.session_state.enhance_eval_orig_prompt = st.session_state.get("enhance_eval_orig_area", "")
    with col_enh:
        st.text_area(
            "Enhanced System Prompt",
            height=250,
            placeholder="Paste enhanced system prompt here...",
            key="enhance_eval_enh_area",
        )
        st.session_state.enhance_eval_enh_prompt = st.session_state.get("enhance_eval_enh_area", "")

    # ── Multiple Test Inputs ─────────────────────────────────────────────────
    render_section_header("🧩", "Test Inputs",
                          'Add test inputs manually or upload JSON — e.g. ["q1","q2"] or [{"input":"q1"}]')

    uploaded_json = st.file_uploader(
        "Upload test cases JSON",
        type=["json"],
        key="enhance_eval_json_upload",
        help='Simple JSON array — e.g. ["question 1", "question 2"] or [{"input": "question 1"}]',
        label_visibility="collapsed",
    )
    if uploaded_json is not None:
        # Use name+size as a fingerprint so we only process each file once,
        # even though the uploader widget keeps the file across reruns.
        fingerprint = f"{uploaded_json.name}:{uploaded_json.size}"
        if st.session_state.get("_enhance_json_fingerprint") != fingerprint:
            try:
                raw = json.loads(uploaded_json.read().decode("utf-8"))
                if not isinstance(raw, list) or not raw:
                    st.error("JSON must be a non-empty array.")
                else:
                    parsed_inputs = []
                    for item in raw:
                        if isinstance(item, str):
                            parsed_inputs.append(item.strip())
                        elif isinstance(item, dict) and "input" in item:
                            parsed_inputs.append(str(item["input"]).strip())
                        else:
                            st.warning(f"Skipping unrecognised item: {item!r}")

                    parsed_inputs = [x for x in parsed_inputs if x]
                    if parsed_inputs:
                        for idx, val in enumerate(parsed_inputs):
                            st.session_state[f"enhance_eval_input_{idx}"] = val
                        old_count = st.session_state.get("enhance_eval_input_count", 1)
                        for idx in range(len(parsed_inputs), old_count):
                            st.session_state.pop(f"enhance_eval_input_{idx}", None)
                        st.session_state.enhance_eval_input_count = len(parsed_inputs)
                        st.session_state["_enhance_json_fingerprint"] = fingerprint
                        st.rerun(scope="fragment")
                    else:
                        st.error("No valid inputs found in the uploaded JSON.")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
        else:
            st.success(f"✓ {st.session_state.get('enhance_eval_input_count', 0)} test input(s) loaded from **{uploaded_json.name}**")

    count = st.session_state.get("enhance_eval_input_count", 1)

    for i in range(count):
        col_input, col_remove = st.columns([11, 1])
        with col_input:
            st.text_area(
                f"Test Input {i + 1}",
                height=100,
                placeholder="Paste the user input you want to test both prompts against...",
                key=f"enhance_eval_input_{i}",
            )
        with col_remove:
            st.write("")
            st.write("")
            if count > 1 and st.button("✕", key=f"rm_input_{i}", help="Remove this test case"):
                for j in range(i, count - 1):
                    st.session_state[f"enhance_eval_input_{j}"] = st.session_state.get(
                        f"enhance_eval_input_{j + 1}", ""
                    )
                st.session_state.pop(f"enhance_eval_input_{count - 1}", None)
                st.session_state.enhance_eval_input_count = count - 1
                st.rerun(scope="fragment")

    if st.button("＋ Add Test Case", key="add_test_input_btn"):
        st.session_state.enhance_eval_input_count = count + 1
        st.rerun(scope="fragment")

    # ── Metric Picker ────────────────────────────────────────────────────────
    render_section_header("📏", "Evaluation Metrics", "Select the metrics to evaluate both prompts against")

    _METRIC_GROUPS = [
        ("General Quality", [
            ("clarity",           "Clarity"),
            ("completeness",      "Completeness"),
            ("accuracy",          "Accuracy"),
            ("conciseness",       "Conciseness"),
            ("professional_tone", "Professional Tone"),
        ]),
        ("Guardrails / Classification", [
            ("output_format_compliance", "Output Format Compliance"),
            ("hallucination_avoidance",  "Hallucination Avoidance"),
        ]),
        ("Conversational", [
            ("answer_relevance",             "Answer Relevance"),
            ("prompt_instruction_adherence", "Prompt Instruction Adherence"),
        ]),
    ]
    _default_metrics_set = set(DEFAULT_METRICS)

    # Seed checkbox defaults on first render only
    for key in ALL_METRIC_KEYS:
        ck = f"metric_{key}"
        if ck not in st.session_state:
            st.session_state[ck] = key in _default_metrics_set

    for group_label, group_metrics in _METRIC_GROUPS:
        st.markdown(
            f'<p style="font-size:0.78rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.06em;color:#6366f1;margin:12px 0 6px 0;">{group_label}</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(group_metrics))
        for col, (key, label) in zip(cols, group_metrics):
            col.checkbox(label, key=f"metric_{key}")

    selected_metrics = [k for k in ALL_METRIC_KEYS if st.session_state.get(f"metric_{k}", False)]
    if not selected_metrics:
        st.warning("Select at least one metric.")
    st.session_state.enhance_eval_selected_metrics = selected_metrics

    # ── Per-Model Configuration ──────────────────────────────────────────────
    with st.expander("Model Configuration", expanded=False):
        col_orig_m, col_enh_m = st.columns(2)
        with col_orig_m:
            st.caption("**Original Prompt**")
            st.info(f"Uses sidebar config — model: `{st.session_state.get('model_name') or 'default'}`")
        with col_enh_m:
            st.caption("**Enhanced Prompt** — override (leave blank to use same as original)")
            enh_model_options = [""] + _MODELS
            st.selectbox("Model", enh_model_options, key="enhance_eval_enh_model_name",
                         placeholder="Same as original")
            st.slider("Temperature", 0.0, 2.0, step=0.1, key="enhance_eval_enh_temperature")
            st.number_input("Max Tokens", 1, 100000, step=1000, key="enhance_eval_enh_max_tokens")

            # Reasoning effort — show only if selected model supports it
            _enh_model_lower = (st.session_state.get("enhance_eval_enh_model_name") or "").lower()
            _enh_supports = any(p in _enh_model_lower for p in (
                "gemini-2.5", "gemini-3", "claude-opus-4", "claude-sonnet-4",
                "claude-3-7", "o4-mini", "o4-preview", "o3-mini", "/o3", "/o1",
                "o1-mini", "deepseek-r1", "qwen3", "grok-3-mini",
            ))
            if _enh_supports:
                _enh_options = ["none", "minimal", "low", "medium", "high"] if any(
                    p in _enh_model_lower for p in ("gemini-2.5", "gemini-3")
                ) else ["none", "low", "medium", "high"]
                st.selectbox("Reasoning Effort", _enh_options,
                             key="enhance_eval_enh_reasoning_effort")
            else:
                st.session_state.enhance_eval_enh_reasoning_effort = "none"

    # Show what enhanced config will be sent so user can confirm before running
    _enh_m = st.session_state.get("enhance_eval_enh_model_name") or ""
    if _enh_m:
        _enh_t = st.session_state.get("enhance_eval_enh_temperature", 0.1)
        _enh_k = st.session_state.get("enhance_eval_enh_max_tokens", 15000)
        _enh_e = st.session_state.get("enhance_eval_enh_reasoning_effort", "none")
        _enh_e_str = f", effort={_enh_e}" if _enh_e and _enh_e != "none" else ""
        st.caption(f"Enhanced model override: `{_enh_m}` · temp={_enh_t} · max_tokens={_enh_k}{_enh_e_str}")
    else:
        st.caption("Enhanced model: same as sidebar config")

    if selected_metrics:
        st.caption(f"Will evaluate: {', '.join(m.replace('_', ' ').title() for m in selected_metrics)}")
        with st.expander("Metric Descriptions", expanded=False):
            for m in selected_metrics:
                desc = _METRIC_DESCRIPTIONS.get(m, "")
                if desc:
                    st.markdown(f"**{m.replace('_', ' ').title()}** — {desc}")

    if st.button("▶ Run Evaluation", width="stretch", key="enhance_eval_run_btn",
                 type="primary", disabled=not selected_metrics):
        valid_inputs = [
            st.session_state.get(f"enhance_eval_input_{i}", "").strip()
            for i in range(st.session_state.get("enhance_eval_input_count", 1))
        ]
        valid_inputs = [inp for inp in valid_inputs if inp]
        if not valid_inputs:
            st.error("Please enter at least one test input before running evaluation.")
        else:
            st.session_state.enhance_eval_uploaded_tests = [
                {
                    "test_case_id": str(i),
                    "test_case_name": f"test case {i + 1}",
                    "data": {"input": inp},
                    "expected_output": "",
                }
                for i, inp in enumerate(valid_inputs)
            ]
            run_enhance_evaluation()

    judge_result = st.session_state.get("enhance_eval_judge_result")
    if judge_result:
        st.divider()
        summary = judge_result.get("summary", {})
        test_results = judge_result.get("test_results", [])

        # Use metrics_requested stored in the result (set at evaluation time).
        # Fall back to inferring from judge response keys only if missing.
        _NON_METRIC = {"overall", "improved", "improvement_summary", "key_differences",
                       "prompt_recommendations", "error", "raw"}
        first_orig = next(
            (r.get("scores", {}).get("original", {}) for r in test_results
             if "error" not in r.get("scores", {})), {}
        )
        metrics = (
            judge_result.get("metrics_requested")
            or judge_result.get("metrics_used")
            or [k for k in ALL_METRIC_KEYS if k in first_orig]
            or [k for k in first_orig if k not in _NON_METRIC]
        )

        # ── Summary metrics ──────────────────────────────────────────────────
        render_section_header("📊", "Score Summary", "Average scores across all test cases for original vs enhanced")
        _orig_model = judge_result.get("original_model") or st.session_state.get("model_name") or "default"
        _enh_model_used = judge_result.get("enhanced_model") or st.session_state.get("enhance_eval_enh_model_name") or _orig_model
        st.caption(f"Original model: `{_orig_model}` · Enhanced model: `{_enh_model_used}`")
        improved = summary.get("improved_count", 0)
        total = summary.get("total_cases", 0)
        st.caption(f"Improved in {improved}/{total} test case(s)")

        avg_orig = summary.get("avg_original", {})
        avg_enh = summary.get("avg_enhanced", {})
        avg_delta = summary.get("avg_delta", {})

        all_metrics = metrics + ["overall"]
        summary_rows = []
        for m in all_metrics:
            orig_v = avg_orig.get(m)
            enh_v = avg_enh.get(m)
            delta_v = avg_delta.get(m)
            summary_rows.append({
                "Metric": m.replace("_", " ").title(),
                "Original (avg)": f"{orig_v:.3f}" if orig_v is not None else "—",
                "Enhanced (avg)": f"{enh_v:.3f}" if enh_v is not None else "—",
                "Δ": f"{delta_v:+.3f}" if delta_v is not None else "—",
            })
        st.dataframe(
            pd.DataFrame(summary_rows).set_index("Metric"),
            width="stretch",
        )

        with st.expander("Metric Descriptions", expanded=False):
            for m in metrics:
                desc = _METRIC_DESCRIPTIONS.get(m, "")
                if desc:
                    st.markdown(f"**{m.replace('_', ' ').title()}** — {desc}")

        # ── Per-test results ─────────────────────────────────────────────────
        for i, tr in enumerate(test_results):
            orig_lat = tr.get("original_latency_s")
            enh_lat = tr.get("enhanced_latency_s")
            lat_parts = []
            if orig_lat is not None:
                lat_parts.append(f"Original: ⏱ {orig_lat}s")
            if enh_lat is not None:
                lat_parts.append(f"Enhanced: ⏱ {enh_lat}s")
            lat_str = f" · {' | '.join(lat_parts)}" if lat_parts else ""

            scores = tr.get("scores", {})
            improved_flag = scores.get("improved", False)
            _orig_overall = scores.get("original", {}).get("overall")
            _enh_overall = scores.get("enhanced", {}).get("overall")
            # Derive outcome from actual scores so tied-at-high (e.g. both 1.0)
            # doesn't show ❌.  Resolution: improved > tied >= regressed.
            _TIED_THRESHOLD = 0.005  # scores within 0.005 are considered tied
            if _orig_overall is not None and _enh_overall is not None:
                _score_delta = _enh_overall - _orig_overall
                if _score_delta > _TIED_THRESHOLD:
                    _outcome = "improved"
                elif _score_delta >= -_TIED_THRESHOLD:
                    _outcome = "tied"
                else:
                    _outcome = "regressed"
            else:
                # Fall back to LLM boolean when scores are unavailable
                _outcome = "improved" if improved_flag else "tied"

            if "error" in scores:
                tc_badge = "⚠️"
            elif tr.get("original_output", "").strip() == tr.get("enhanced_output", "").strip():
                tc_badge = "🟰"
            elif _outcome == "improved":
                tc_badge = "✅"
            elif _outcome == "tied":
                tc_badge = "🟰"
            else:
                tc_badge = "❌"

            with st.expander(f"{tc_badge} Test Case {i + 1}{lat_str}", expanded=False):
                test_input = tr.get("input", "")
                if test_input:
                    with st.expander("Input", expanded=False):
                        st.text_area("Input", value=test_input, height=100,
                                     disabled=True, key=f"judge_input_{i}", label_visibility="collapsed")

                if "error" in scores:
                    st.error(f"Judge error: {scores.get('error')}")
                else:
                    summary_text = scores.get("improvement_summary", "")
                    orig_out = tr.get("original_output", "")
                    enh_out = tr.get("enhanced_output", "")
                    identical = orig_out.strip() == enh_out.strip()
                    if identical:
                        verdict = "🟰 Identical outputs"
                    elif _outcome == "improved":
                        verdict = "✅ Improved"
                    elif _outcome == "tied":
                        verdict = "🟰 Maintained"
                    else:
                        verdict = "❌ Regressed"
                    st.markdown(f"{verdict} — {summary_text}")
                    diffs = scores.get("key_differences", [])
                    if diffs:
                        with st.expander("Key Differences", expanded=True):
                            for d in diffs:
                                st.markdown(f"- {d}")

                orig_score = scores.get("original", {}).get("overall")
                enh_score = scores.get("enhanced", {}).get("overall")
                orig_label = f"Original Output  ({_orig_model})" + (f"  —  overall {orig_score:.2f}" if orig_score is not None else "")
                enh_label = f"Enhanced Output  ({_enh_model_used})" + (f"  —  overall {enh_score:.2f}" if enh_score is not None else "")
                col_orig_out, col_enh_out = st.columns(2)
                with col_orig_out:
                    with st.expander(orig_label, expanded=False):
                        st.text_area("Original Output", value=tr.get("original_output", ""), height=200,
                                     disabled=True, key=f"judge_orig_out_{i}", label_visibility="collapsed")
                with col_enh_out:
                    with st.expander(enh_label, expanded=False):
                        st.text_area("Enhanced Output", value=tr.get("enhanced_output", ""), height=200,
                                     disabled=True, key=f"judge_enh_out_{i}", label_visibility="collapsed")

                # ── Per-test metric scores table ──────────────────────────────
                orig_scores_detail = scores.get("original", {})
                enh_scores_detail = scores.get("enhanced", {})
                if orig_scores_detail or enh_scores_detail:
                    score_keys = [k for k in (list(orig_scores_detail.keys()) or list(enh_scores_detail.keys()))
                                  if k not in {"error", "raw"}]
                    if score_keys:
                        rows = []
                        for k in score_keys:
                            ov = orig_scores_detail.get(k)
                            ev = enh_scores_detail.get(k)
                            delta = (ev - ov) if (ov is not None and ev is not None) else None
                            rows.append({
                                "Metric": k.replace("_", " ").title(),
                                "Original": f"{ov:.3f}" if ov is not None else "—",
                                "Enhanced": f"{ev:.3f}" if ev is not None else "—",
                                "Δ": f"{delta:+.3f}" if delta is not None else "—",
                            })
                        with st.expander("Metric Scores", expanded=True):
                            st.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")

        with st.expander("Debug: API Response", expanded=False):
            st.json(st.session_state.get("enhance_eval_api_debug_original", {}))

    # ── LLM Suggestions ──────────────────────────────────────────────────────
    st.divider()
    render_section_header("🤖", "LLM Suggestions",
                          "AI-generated, priority-ranked suggestions based on all judge scores and deltas", step=4)
    has_judge = bool(st.session_state.get("enhance_eval_judge_result"))
    if not has_judge:
        st.info("Run LLM Judge evaluation above first to enable suggestion generation.")
    if st.button("Generate Suggestions", width="stretch", key="enhance_gen_suggestions_btn",
                 type="primary", disabled=not has_judge):
        st.session_state.enhance_eval_suggestions_result = None
        generate_enhance_suggestions()

    enhance_suggestions = st.session_state.get("enhance_eval_suggestions_result")
    if enhance_suggestions:
        _render_overall_suggestions(
            enhance_suggestions,
            selected_key="enhance_eval_suggestions_selected",
            on_apply=_apply_suggestions_to_enhance_prompt,
        )

    # ── Arena Evaluation ─────────────────────────────────────────────────────
    st.divider()
    render_section_header("🏆", "Arena Comparison (DeepEval)",
                          "Head-to-head: original vs enhanced prompt — DeepEval picks the winner per test case", step=5)

    arena_criteria = st.text_input(
        "Criteria",
        value=st.session_state.get("arena_eval_criteria", ""),
        key="arena_criteria_input",
        help="Describe what makes one response better than the other.",
    )
    st.session_state.arena_eval_criteria = arena_criteria

    arena_orig = st.session_state.get("enhance_eval_orig_prompt", "").strip()
    arena_enh = st.session_state.get("enhance_eval_enh_prompt", "").strip()
    arena_inputs = [
        st.session_state.get(f"enhance_eval_input_{i}", "").strip()
        for i in range(st.session_state.get("enhance_eval_input_count", 1))
    ]
    arena_inputs = [x for x in arena_inputs if x]
    arena_test_cases = [
        {"data": {"input": inp}, "expected_output": ""}
        for inp in arena_inputs
    ]

    arena_disabled = not (arena_orig and arena_enh and arena_test_cases)
    if arena_disabled:
        st.info("Fill in both prompts and at least one test input above to enable Arena Compare.")

    if st.button("Run Arena Compare", width="stretch", key="arena_run_btn", disabled=arena_disabled):
        with st.spinner(f"Running Arena comparison on {len(arena_test_cases)} test case(s)..."):
            try:
                reasoning = st.session_state.reasoning_effort
                payload = {
                    "sessionId": st.session_state.session_id,
                    "systemPrompt": arena_orig,
                    "enhancedSystemPrompt": arena_enh,
                    "modelName": st.session_state.get("model_name"),
                    "maxTokens": st.session_state.get("max_tokens"),
                    "temperature": st.session_state.get("temperature"),
                    "reasoningEffort": reasoning if reasoning != "none" else None,
                    "type": "arena_comparison",
                    "recommendations": None,
                    "testCases": arena_test_cases,
                    "arenaCriteria": arena_criteria,
                }
                data = post_chat(payload, include_grid_header=False)
                arena_result = extract_test_evaluation_result(data)
                if arena_result:
                    st.session_state.arena_eval_result = arena_result
                    st.success("Arena comparison complete.")
                else:
                    st.warning("Arena comparison returned no result.")
            except Exception as e:
                st.error(f"Arena comparison failed: {e}")

    arena_result = st.session_state.get("arena_eval_result")
    if arena_result:
        _render_arena_results(arena_result)


def _render_arena_results(result: dict) -> None:
    wins = result.get("wins", {})
    win_rate = result.get("win_rate", 0.0)
    per_test = result.get("per_test", [])
    total = len(per_test)

    enhanced_wins = wins.get("Enhanced", 0)
    original_wins = wins.get("Original", 0)

    if enhanced_wins > original_wins:
        headline = f"🏆 Enhanced won **{enhanced_wins} / {total}** tests ({win_rate:.0%})"
    elif original_wins > enhanced_wins:
        headline = f"⚠️ Original won **{original_wins} / {total}** tests ({1 - win_rate:.0%})"
    else:
        headline = f"🤝 Tied — **{enhanced_wins} / {total}** wins each"

    st.markdown(headline)

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Wins", original_wins)
    col2.metric("Enhanced Wins", enhanced_wins)
    col3.metric("Inconclusive", wins.get("Inconclusive", total - enhanced_wins - original_wins))

    st.write("#### Per-Test Results")
    for i, t in enumerate(per_test):
        winner = t.get("winner", "Unknown")
        if winner == "Enhanced":
            badge = "🏆 Enhanced"
        elif winner == "Original":
            badge = "❌ Original"
        else:
            badge = f"⚠️ {winner}"

        with st.expander(f"Test {i + 1} — {badge}", expanded=False):
            st.markdown(f"**Reason:** {t.get('reason', '—')}")
            col_o, col_e = st.columns(2)
            with col_o:
                st.write("**Original Output**")
                st.text_area(
                    "orig", value=t.get("original_output", ""), height=150,
                    disabled=True, key=f"arena_orig_out_{i}", label_visibility="collapsed",
                )
            with col_e:
                st.write("**Enhanced Output**")
                st.text_area(
                    "enh", value=t.get("enhanced_output", ""), height=150,
                    disabled=True, key=f"arena_enh_out_{i}", label_visibility="collapsed",
                )
            with st.expander("Input", expanded=False):
                st.text(t.get("input", ""))
