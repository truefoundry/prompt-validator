import json
import os

import pandas as pd
import streamlit as st

from ..actions import generate_overall_suggestions, run_deepeval_prompt_metrics
from ..components import render_diff, render_section_header
from ..trace_actions import (
    fetch_live_trace_inputs,
    load_trace_inputs,
    run_llm_judge_on_traces,
    run_trace_pipeline,
)
from .enhance import _render_overall_suggestions
from src.chat.graph.llm_judge_evaluator import DEFAULT_METRICS, ALL_METRIC_KEYS, AVAILABLE_METRICS


_JUDGE_METRICS = DEFAULT_METRICS + ["overall"]

_TRACE_METRIC_GROUPS = [
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

_METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "geval": "GEval (Quality)",
    "prompt_alignment": "Prompt Alignment",
    "pii": "PII Leakage",
}

# For pii: lower score = better (less pii)
_LOWER_IS_BETTER = {"pii"}

_INPUT_MODES = ("TFY Prompt FQN", "Paste Prompt Text")

_TRACE_SOURCE_OPTIONS = ("Fetch Live Traces", "Use traces.json", "Upload file")


def _load_models() -> list[str]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "models.json")) as f:
            return sorted(json.load(f).get("models", []))
    except Exception:
        return []


_MODELS = _load_models()



def _render_prompt_input(mode_key: str, fqn_key: str, sys_key: str, user_tpl_key: str, *, fqn_widget_key: str | None = None) -> None:
    """Render either a Prompt FQN text input or raw-text paste areas based on the selected mode."""
    st.session_state[mode_key] = st.radio(
        "Prompt Input Method",
        _INPUT_MODES,
        index=_INPUT_MODES.index(st.session_state.get(mode_key, _INPUT_MODES[0])),
        horizontal=True,
        key=f"{mode_key}_radio",
    )

    if st.session_state[mode_key] == "TFY Prompt FQN":
        kwargs = {"placeholder": "chat_prompt:truefoundry/new-repo/prompt-name:1"}
        if fqn_widget_key:
            kwargs["key"] = fqn_widget_key
        else:
            kwargs["value"] = st.session_state[fqn_key]
        st.session_state[fqn_key] = st.text_input("Prompt FQN", **kwargs)
    else:
        st.session_state[sys_key] = st.text_area(
            "System Prompt",
            value=st.session_state.get(sys_key, ""),
            height=200,
            placeholder="Paste your system prompt here...",
            key=f"{sys_key}_area",
        )
        st.session_state[user_tpl_key] = st.text_area(
            "User Prompt Template (optional)",
            value=st.session_state.get(user_tpl_key, ""),
            height=120,
            placeholder="Paste user prompt template here. Use {{input}} for variable injection.",
            help="Use {{input}} as a placeholder for test case inputs. Leave empty to send test inputs as plain user messages.",
            key=f"{user_tpl_key}_area",
        )


def _render_deepeval_prompt_metrics(result: dict) -> None:
    summary = result.get("summary", {})
    per_test = result.get("per_test", [])
    metrics_run = result.get("metrics_run", [])

    st.write("#### DeepEval Metrics Results")

    # Summary table
    header = st.columns([2, 1, 1, 1])
    header[0].markdown("**Metric**")
    header[1].markdown("**Original avg**")
    header[2].markdown("**Enhanced avg**")
    header[3].markdown("**Delta**")

    for key in metrics_run:
        s = summary.get(key, {})
        orig = s.get("original_avg")
        enh = s.get("enhanced_avg")
        delta = s.get("delta")
        lower_better = key in _LOWER_IS_BETTER

        cols = st.columns([2, 1, 1, 1])
        cols[0].write(_METRIC_LABELS.get(key, key))
        cols[1].write(f"{orig:.3f}" if orig is not None else "—")
        cols[2].write(f"{enh:.3f}" if enh is not None else "—")

        if delta is not None:
            improved = delta < 0 if lower_better else delta > 0
            regressed = delta > 0 if lower_better else delta < 0
            delta_str = f"{delta:+.3f}"
            if improved:
                cols[3].markdown(f"**:green[{delta_str}]**")
            elif regressed:
                cols[3].markdown(f"**:red[{delta_str}]**")
            else:
                cols[3].write(delta_str)
        else:
            cols[3].write("—")

    # Per-test breakdown
    if per_test:
        with st.expander(f"Per-test breakdown ({len(per_test)} cases)", expanded=False):
            for i, tc in enumerate(per_test):
                st.markdown(f"**Test {i + 1}:** {tc['input'][:100]}{'…' if len(tc['input']) > 100 else ''}")
                for key in metrics_run:
                    scores = tc["scores"].get(key, {})
                    orig_s = scores.get("original")
                    enh_s = scores.get("enhanced")
                    orig_r = scores.get("original_reason", "")
                    enh_r = scores.get("enhanced_reason", "")
                    label = _METRIC_LABELS.get(key, key)
                    c1, c2 = st.columns(2)
                    c1.caption(f"**{label} — Original:** {f'{orig_s:.3f}' if orig_s is not None else '—'}")
                    if orig_r:
                        c1.caption(orig_r)
                    c2.caption(f"**{label} — Enhanced:** {f'{enh_s:.3f}' if enh_s is not None else '—'}")
                    if enh_r:
                        c2.caption(enh_r)
                st.divider()


def _render_llm_judge_results(result: dict) -> None:
    """Render LLM judge comparison results: aggregate metrics, per-test details, recommendations."""
    summary = result.get("summary", {})
    test_results = result.get("test_results", [])

    _NON_METRIC = {"overall", "improved", "improvement_summary", "key_differences",
                   "prompt_recommendations", "error", "raw", "reasoning", "correctness_analysis"}

    # Derive actual metric list from the result (respects what was selected at eval time)
    first_orig = next(
        (r.get("scores", {}).get("original", {}) for r in test_results
         if "error" not in r.get("scores", {})), {}
    )
    metrics = (
        result.get("metrics_requested")
        or result.get("metrics_used")
        or [k for k in ALL_METRIC_KEYS if k in first_orig]
        or [k for k in first_orig if k not in _NON_METRIC]
    )
    all_metrics = metrics + ["overall"]

    # ── Aggregate metrics ──────────────────────────────────────────────────
    if summary:
        st.write("#### Aggregate Metrics")
        avg_orig = summary.get("avg_original", {})
        avg_enh = summary.get("avg_enhanced", {})
        avg_delta = summary.get("avg_delta", {})

        improved = summary.get("improved_count", 0)
        total = summary.get("total_cases", 0)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Cases", total)
        col_b.metric("Improved", improved)
        col_c.metric("Improvement Rate", f"{improved / total * 100:.0f}%" if total else "—")

        rows = []
        for m in all_metrics:
            o = avg_orig.get(m)
            e = avg_enh.get(m)
            d = avg_delta.get(m)
            rows.append({
                "Metric": m.replace("_", " ").title(),
                "Original": f"{o:.3f}" if o is not None else "—",
                "Enhanced": f"{e:.3f}" if e is not None else "—",
                "Δ Delta": f"{d:+.3f}" if d is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")

    # ── Prompt recommendations (aggregated across all test cases) ───────────
    all_recs: list[str] = []
    for tr in test_results:
        recs = tr.get("scores", {}).get("prompt_recommendations", [])
        if isinstance(recs, list):
            all_recs.extend(recs)

    if all_recs:
        st.divider()
        st.write("#### LLM Suggestions for Enhanced Prompt")
        seen: set[str] = set()
        unique_recs = [r for r in all_recs if r not in seen and not seen.add(r)]  # type: ignore[func-returns-value]
        for i, rec in enumerate(unique_recs[:4], 1):
            st.markdown(f"**{i}.** {rec}")

    # ── Per-test breakdown ─────────────────────────────────────────────────
    if test_results:
        st.divider()
        st.write("#### Per-Test Comparison")
        _TIED_THRESHOLD = 0.005
        for i, tr in enumerate(test_results):
            scores = tr.get("scores", {})
            orig_overall = scores.get("original", {}).get("overall")
            enh_overall = scores.get("enhanced", {}).get("overall")
            improved_flag = scores.get("improved", False)

            # Score-based outcome (same logic as enhance tab)
            if orig_overall is not None and enh_overall is not None:
                _delta = enh_overall - orig_overall
                if _delta > _TIED_THRESHOLD:
                    _outcome = "improved"
                elif _delta >= -_TIED_THRESHOLD:
                    _outcome = "tied"
                else:
                    _outcome = "regressed"
            else:
                _outcome = "improved" if improved_flag else "tied"

            if "error" in scores:
                badge = "⚠️"
            elif tr.get("original_output", "").strip() == tr.get("enhanced_output", "").strip():
                badge = "🟰"
            elif _outcome == "improved":
                badge = "✅"
            elif _outcome == "tied":
                badge = "🟰"
            else:
                badge = "❌"

            # Latency
            orig_lat = tr.get("original_latency_s")
            enh_lat = tr.get("enhanced_latency_s")
            lat_parts = []
            if orig_lat is not None:
                lat_parts.append(f"Original: ⏱ {orig_lat}s")
            if enh_lat is not None:
                lat_parts.append(f"Enhanced: ⏱ {enh_lat}s")
            lat_str = f" · {' | '.join(lat_parts)}" if lat_parts else ""

            tc_name = tr.get("test_case_name", tr.get("test_case_id", f"Test {i + 1}"))
            score_str = (
                f" | Orig: {orig_overall:.2f} → Enh: {enh_overall:.2f}"
                if orig_overall is not None and enh_overall is not None else ""
            )
            label = f"{badge} {tc_name}{score_str}{lat_str}"

            with st.expander(label, expanded=False):
                if tr.get("input"):
                    with st.expander("Input", expanded=False):
                        st.text_area("Input", value=tr.get("input", ""), height=200,
                                     disabled=True, key=f"trace_input_{tr.get('test_case_id')}_{i}",
                                     label_visibility="collapsed")

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

                    # ── Correctness Analysis ──────────────────────────────
                    ca = scores.get("correctness_analysis")
                    if ca:
                        with st.expander("Correctness Analysis", expanded=True):
                            c_orig = ca.get("original_correctness_score")
                            c_enh  = ca.get("enhanced_correctness_score")
                            c_delta = (c_enh - c_orig) if (c_orig is not None and c_enh is not None) else None
                            col_ca, col_cb, col_cc = st.columns(3)
                            col_ca.metric("Original Correctness", f"{c_orig:.2f}" if c_orig is not None else "—")
                            col_cb.metric("Enhanced Correctness", f"{c_enh:.2f}" if c_enh is not None else "—",
                                          delta=f"{c_delta:+.2f}" if c_delta is not None else None)
                            col_cc.metric("Δ Correctness", f"{c_delta:+.2f}" if c_delta is not None else "—")
                            verdict_text = ca.get("correctness_verdict", "")
                            if verdict_text:
                                st.caption(verdict_text)
                            orig_gaps = ca.get("original_gaps", [])
                            enh_gaps  = ca.get("enhanced_gaps", [])
                            if orig_gaps or enh_gaps:
                                gap_col_o, gap_col_e = st.columns(2)
                                with gap_col_o:
                                    if orig_gaps:
                                        st.markdown("**Original gaps:**")
                                        for g in orig_gaps:
                                            st.markdown(f"- {g}")
                                with gap_col_e:
                                    if enh_gaps:
                                        st.markdown("**Enhanced gaps:**")
                                        for g in enh_gaps:
                                            st.markdown(f"- {g}")

                    # ── CoT Reasoning ─────────────────────────────────────
                    reasoning = scores.get("reasoning")
                    if reasoning:
                        with st.expander("CoT Reasoning", expanded=False):
                            for rkey, rlabel in [
                                ("task_intent", "Task Intent"),
                                ("expected_response_profile", "Expected Response Profile"),
                                ("response_a_correctness", "Response A Correctness"),
                                ("response_b_correctness", "Response B Correctness"),
                            ]:
                                if reasoning.get(rkey):
                                    st.markdown(f"**{rlabel}**")
                                    st.markdown(reasoning[rkey])

                col_o, col_e = st.columns(2)
                orig_score = scores.get("original", {}).get("overall")
                enh_score = scores.get("enhanced", {}).get("overall")
                orig_label = "Original Output" + (f"  —  overall {orig_score:.2f}" if orig_score is not None else "")
                enh_label = "Enhanced Output" + (f"  —  overall {enh_score:.2f}" if enh_score is not None else "")
                with col_o:
                    with st.expander(orig_label, expanded=False):
                        st.text_area("orig", value=tr.get("original_output", ""), height=200,
                                     disabled=True, key=f"trace_orig_out_{tr.get('test_case_id')}_{i}",
                                     label_visibility="collapsed")
                with col_e:
                    with st.expander(enh_label, expanded=False):
                        st.text_area("enh", value=tr.get("enhanced_output", ""), height=200,
                                     disabled=True, key=f"trace_enh_out_{tr.get('test_case_id')}_{i}",
                                     label_visibility="collapsed")

                # Per-test metric scores — use actual keys from the result
                orig_scores_detail = scores.get("original", {})
                enh_scores_detail = scores.get("enhanced", {})
                if orig_scores_detail or enh_scores_detail:
                    score_keys = [k for k in (list(orig_scores_detail.keys()) or list(enh_scores_detail.keys()))
                                  if k not in _NON_METRIC]
                    if score_keys:
                        metric_rows = []
                        for k in score_keys:
                            ov = orig_scores_detail.get(k)
                            ev = enh_scores_detail.get(k)
                            delta = (ev - ov) if (ov is not None and ev is not None) else None
                            metric_rows.append({
                                "Metric": k.replace("_", " ").title(),
                                "Original": f"{ov:.3f}" if ov is not None else "—",
                                "Enhanced": f"{ev:.3f}" if ev is not None else "—",
                                "Δ": f"{delta:+.3f}" if delta is not None else "—",
                            })
                        with st.expander("Metric Scores", expanded=True):
                            st.dataframe(pd.DataFrame(metric_rows).set_index("Metric"), width="stretch")

                if scores.get("improvement_summary"):
                    st.info(scores["improvement_summary"])
                if scores.get("key_differences"):
                    st.markdown("**Key Differences:**")
                    for d in scores["key_differences"]:
                        st.markdown(f"- {d}")


@st.fragment
def render_trace_eval_tab() -> None:
    # Flush pending original prompt update BEFORE the text area widget is created.
    # _apply_suggestions stages the promoted value here so the key-bound widget
    # picks it up on the next render (Streamlit ignores value= for key-bound widgets).
    if "_pending_trace_orig_sys_area" in st.session_state:
        st.session_state["trace_orig_sys_area"] = st.session_state.pop("_pending_trace_orig_sys_area")

    st.markdown(
        """<div style="margin-bottom:20px;">
  <h2 style="margin:0 0 4px 0;font-size:1.3rem;font-weight:700;color:#1e293b;">
    Trace Evaluation
  </h2>
  <p style="margin:0;font-size:0.875rem;color:#64748b;line-height:1.5;">
    Load real production traces, group by prompt, then compare original vs enhanced
    prompt quality using LLM-as-judge at scale.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Section A: Load Traces ──────────────────────────────────────────────
    render_section_header("📡", "Load Traces", "Fetch live traces, load from file, or use traces.json", step=1)

    source_label = st.radio(
        "Trace source",
        _TRACE_SOURCE_OPTIONS,
        horizontal=True,
        key="trace_source_radio",
    )

    uploaded_content: str | None = None
    if source_label == "Fetch Live Traces":
        col_host, col_key = st.columns(2)
        st.session_state.trace_tfy_host = col_host.text_input(
            "TFY Host",
            value=st.session_state.get("trace_tfy_host", ""),
            placeholder="https://your-tenant.truefoundry.com",
            key="trace_tfy_host_input",
        )
        st.session_state.trace_tfy_api_key = col_key.text_input(
            "TFY API Key",
            value=st.session_state.get("trace_tfy_api_key", ""),
            placeholder="Enter your TrueFoundry API key",
            type="password",
            key="trace_tfy_api_key_input",
        )
        col_hrs, col_limit, col_fqn = st.columns([1, 1, 3])
        live_days = col_hrs.number_input(
            "Days back", min_value=1, max_value=90, value=1, step=1, key="trace_live_days"
        )
        live_limit = col_limit.number_input(
            "Max spans", min_value=10, max_value=2000, value=200, step=50, key="trace_live_limit"
        )
        live_fqn = col_fqn.text_input(
            "Prompt FQN filter (optional)",
            placeholder="e.g. prompt-registry/...",
            key="trace_live_fqn",
        )
        col_email, col_dest = st.columns(2)
        live_email = col_email.text_input(
            "Email filter (optional)",
            placeholder="e.g. user@example.com",
            key="trace_live_email",
        )
        live_dest = col_dest.text_input(
            "Data routing destination",
            value="default",
            placeholder="default",
            key="trace_live_dest",
        )
    elif source_label == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload traces JSON",
            type=["json"],
            key="trace_file_uploader",
        )
        if uploaded_file:
            uploaded_content = uploaded_file.read().decode("utf-8")

    _SOURCE_MAP = {"Use traces.json": "traces.json", "Upload file": "upload"}

    _missing_creds = (
        source_label == "Fetch Live Traces"
        and not (
            st.session_state.get("trace_tfy_host", "").strip()
            and st.session_state.get("trace_tfy_api_key", "").strip()
        )
    )
    if _missing_creds:
        st.caption("Enter TFY Host and API Key above to enable live trace fetching.")

    if st.button("Load Traces", width="stretch", key="trace_load_btn", disabled=_missing_creds):
        if source_label == "Fetch Live Traces":
            fetch_live_trace_inputs(
                hours=int(st.session_state.get("trace_live_days", 1)) * 24,
                limit=int(st.session_state.get("trace_live_limit", 200)),
                fqn_filter=st.session_state.get("trace_live_fqn", "").strip() or None,
                tfy_host=st.session_state.get("trace_tfy_host", "").strip(),
                tfy_api_key=st.session_state.get("trace_tfy_api_key", "").strip(),
                email_filter=st.session_state.get("trace_live_email", "").strip() or None,
                data_routing_destination=st.session_state.get("trace_live_dest", "default").strip() or "default",
            )
        else:
            load_trace_inputs(
                source=_SOURCE_MAP[source_label],
                uploaded_content=uploaded_content,
            )

    trace_inputs: list = st.session_state.get("trace_inputs", [])
    if not trace_inputs:
        return

    # ── Section B: Group by Prompt Group ───────────────────────────────────
    st.divider()
    render_section_header("🗂️", "Select Prompt Group", "Pick a prompt group and select traces to evaluate", step=2)

    # Build group options with counts and system prompt previews for display labels
    group_counts: dict[str, int] = {}
    group_key_to_sys: dict[str, str] = {}
    for ti in trace_inputs:
        gk = getattr(ti, "group_key", None) or ti.prompt_fqn or "(no group)"
        group_counts[gk] = group_counts.get(gk, 0) + 1
        if gk not in group_key_to_sys and getattr(ti, "system_prompt", ""):
            group_key_to_sys[gk] = ti.system_prompt

    def _group_label(gk: str, count: int) -> str:
        if not gk.startswith("auto:"):
            return f"{gk}  ({count} traces)"
        preview = (group_key_to_sys.get(gk, "") or "no system prompt")[:60].replace("\n", " ")
        return f"[Auto] {preview}…  ({count} traces)"

    group_options = sorted(group_counts.keys())
    group_display = [_group_label(gk, group_counts[gk]) for gk in group_options]
    group_display_map = dict(zip(group_display, group_options))

    selected_display = st.selectbox(
        "Select Prompt Group",
        options=group_display,
        key="trace_fqn_selectbox",
    )
    selected_group_key = group_display_map.get(selected_display, "")
    st.session_state.trace_selected_fqn = selected_group_key

    # Filter traces to selected group
    filtered = [
        ti for ti in trace_inputs
        if (getattr(ti, "group_key", None) or ti.prompt_fqn or "(no group)") == selected_group_key
    ]

    # Auto-fill original system prompt from the first trace that has one
    # Use getattr for backward compat with TraceInput objects loaded before system_prompt was added
    fqn_sys_prompt = next(
        (getattr(ti, "system_prompt", "") for ti in filtered if getattr(ti, "system_prompt", "")),
        "",
    )

    # Build an O(1) id → index map once rather than calling list.index() per row (O(n) each).
    _trace_id_to_idx: dict[int, int] = {id(ti): i for i, ti in enumerate(trace_inputs)}

    display_rows = filtered[:30]
    all_display_indices = [_trace_id_to_idx[id(ti)] for ti in display_rows]
    # Reverse map for O(1) "which display row number is this original index?" lookups.
    _display_orig_to_row: dict[int, int] = {
        orig_idx: row_num for row_num, orig_idx in enumerate(all_display_indices, 1)
    }

    # Build display DataFrame with truncated text for the table view
    def _trunc(s: str, n: int = 100) -> str:
        s = (s or "").replace("\n", " ")
        return s[:n] + "…" if len(s) > n else s

    table_data = []
    for i, ti in enumerate(display_rows):
        orig_idx = _trace_id_to_idx[id(ti)]
        table_data.append({
            "#": i + 1,
            "User Input": _trunc(ti.user_message or "", 120),
            "Assistant Response": _trunc(ti.trace_output or "", 100),
            "Model": _trunc(ti.model_name or "(unknown)", 40),
            "Span ID": (ti.span_id or "")[:20],
            "_orig_idx": orig_idx,
        })

    df_display = pd.DataFrame(table_data)

    st.dataframe(
        df_display.drop(columns=["_orig_idx"]),
        width="stretch",
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "User Input": st.column_config.TextColumn("User Input", width="large"),
            "Assistant Response": st.column_config.TextColumn("Assistant Response", width="large"),
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Span ID": st.column_config.TextColumn("Span ID", width="medium"),
        },
    )

    # Multiselect for picking traces by row number — instant, no rerender per click
    row_options = [f"#{r['#']}  {_trunc(display_rows[r['#']-1].user_message or '', 80)}" for r in table_data]
    row_option_to_orig = {opt: table_data[i]["_orig_idx"] for i, opt in enumerate(row_options)}

    # Apply pending select-all / clear BEFORE the widget is instantiated
    if st.session_state.pop("_trace_select_all_pending", False):
        st.session_state["trace_multiselect"] = row_options
    elif st.session_state.pop("_trace_clear_pending", False):
        st.session_state["trace_multiselect"] = []
    elif "trace_multiselect" not in st.session_state:
        # First render only: seed from previously selected indices
        prev_selected_orig = set(st.session_state.get("trace_selected_indices", []))
        st.session_state["trace_multiselect"] = [opt for opt, idx in row_option_to_orig.items() if idx in prev_selected_orig]

    col_multi, col_btns = st.columns([3, 1])
    with col_multi:
        chosen_options = st.multiselect(
            "Select traces to evaluate",
            options=row_options,
            key="trace_multiselect",
            placeholder="Click to pick traces, or use Select All →",
        )
    with col_btns:
        st.write("")  # vertical align
        st.write("")
        if st.button("Select All", key="trace_sel_all", width="stretch"):
            st.session_state["_trace_select_all_pending"] = True
            st.rerun(scope="fragment")
        if st.button("Clear", key="trace_clear_sel", width="stretch"):
            st.session_state["_trace_clear_pending"] = True
            st.rerun(scope="fragment")

    selected_indices = [row_option_to_orig[opt] for opt in chosen_options]
    st.session_state.trace_selected_indices = selected_indices
    st.caption(f"{len(selected_indices)} selected  |  showing up to 30 of {len(filtered)} traces")

    # Details for selected traces
    if selected_indices:
        with st.expander(f"Selected trace details ({len(selected_indices)})", expanded=False):
            for orig_idx in selected_indices:
                ti = trace_inputs[orig_idx]
                row_num = _display_orig_to_row.get(orig_idx, orig_idx)
                st.markdown(f"**Trace #{row_num}** — `{ti.span_id or 'no span id'}`")
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.caption("User Input")
                    st.text(ti.user_message or "(none)")
                with dcol2:
                    st.caption("Assistant Response")
                    st.text(ti.trace_output or "(none)")
                sys_p = getattr(ti, "system_prompt", "") or ""
                if sys_p:
                    st.caption("System Prompt")
                    st.text(sys_p[:400] + ("…" if len(sys_p) > 400 else ""))
                st.divider()

    # ── Section C: Prompt Configuration ────────────────────────────────────
    st.divider()
    render_section_header("⚙️", "Configure Prompts",
                          f"Group: {selected_group_key} — original system prompt auto-filled from trace data", step=3)

    # Auto-populate original system prompt when group changes and traces have a system prompt
    last_fqn = st.session_state.get("_trace_last_autofilled_fqn", "")
    if fqn_sys_prompt and selected_group_key != last_fqn:
        st.session_state.trace_original_system_prompt = fqn_sys_prompt
        st.session_state["trace_orig_sys_area"] = fqn_sys_prompt
        st.session_state._trace_last_autofilled_fqn = selected_group_key

    col_orig, col_enh = st.columns(2)
    with col_orig:
        st.write("**Original System Prompt**")
        if not fqn_sys_prompt:
            st.caption(
                "This prompt group has no separate system prompt in traces "
                "(prompt is embedded in the user message). Paste the system prompt manually."
            )
        st.session_state.trace_original_system_prompt = st.text_area(
            "Original system prompt",
            value=st.session_state.get("trace_original_system_prompt", ""),
            height=260,
            placeholder="Paste the original system prompt here...",
            key="trace_orig_sys_area",
            label_visibility="collapsed",
        )

    with col_enh:
        st.write("**Enhanced System Prompt**")
        session_enhanced = st.session_state.get("enhanced_prompt", "")
        if session_enhanced:
            if st.button("Use session enhanced prompt", key="trace_use_session_enh"):
                st.session_state.trace_enhanced_system_prompt = session_enhanced
                st.rerun(scope="fragment")
        else:
            st.caption("No enhanced prompt in session yet — go to the **Enhance** tab to generate one, or paste below.")
        st.session_state.trace_enhanced_system_prompt = st.text_area(
            "Enhanced system prompt",
            value=st.session_state.get("trace_enhanced_system_prompt", ""),
            height=260,
            placeholder="Paste the enhanced system prompt here...",
            label_visibility="collapsed",
        )

    # ── Model Configuration override (for enhanced prompt) ───────────────────
    with st.expander("Model Configuration", expanded=False):
        col_orig_m, col_enh_m = st.columns(2)
        with col_orig_m:
            st.caption("**Original Prompt**")
            st.info(f"Uses sidebar config — model: `{st.session_state.get('model_name') or 'default'}`")
        with col_enh_m:
            st.caption("**Enhanced Prompt** — override (leave blank to use same as original)")
            enh_model_options = [""] + _MODELS
            st.selectbox("Model", enh_model_options, key="trace_eval_enh_model_name",
                         placeholder="Same as original")
            st.slider("Temperature", 0.0, 2.0, value=0.1, step=0.1, key="trace_eval_enh_temperature")
            st.number_input("Max Tokens", 1, 100000, value=15000, step=1000, key="trace_eval_enh_max_tokens")

            _enh_model_lower = (st.session_state.get("trace_eval_enh_model_name") or "").lower()
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
                             key="trace_eval_enh_reasoning_effort")
            else:
                st.session_state.trace_eval_enh_reasoning_effort = "none"

    _enh_m = st.session_state.get("trace_eval_enh_model_name") or ""
    if _enh_m:
        _enh_t = st.session_state.get("trace_eval_enh_temperature", 0.1)
        _enh_k = st.session_state.get("trace_eval_enh_max_tokens", 15000)
        _enh_e = st.session_state.get("trace_eval_enh_reasoning_effort", "none")
        _enh_e_str = f", effort={_enh_e}" if _enh_e and _enh_e != "none" else ""
        st.caption(f"Enhanced model override: `{_enh_m}` · temp={_enh_t} · max_tokens={_enh_k}{_enh_e_str}")
    else:
        st.caption("Enhanced model: same as sidebar config")

    st.session_state.trace_user_prompt_template = st.text_input(
        "User Prompt Template (optional — use {{input}} for injection)",
        value=st.session_state.get("trace_user_prompt_template", ""),
        placeholder="e.g.  Answer this question: {{input}}",
        key="trace_user_tpl_input",
    )

    # ── Metric Picker ────────────────────────────────────────────────────────
    st.divider()
    render_section_header("📏", "Evaluation Metrics", "Select the metrics to evaluate both prompts against")

    _default_metrics_set = set(DEFAULT_METRICS)
    for key in ALL_METRIC_KEYS:
        ck = f"trace_metric_{key}"
        if ck not in st.session_state:
            st.session_state[ck] = key in _default_metrics_set

    for group_label, group_metrics in _TRACE_METRIC_GROUPS:
        st.markdown(
            f'<p style="font-size:0.78rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.06em;color:#6366f1;margin:12px 0 6px 0;">{group_label}</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(group_metrics))
        for col, (key, label) in zip(cols, group_metrics):
            col.checkbox(label, key=f"trace_metric_{key}")

    trace_selected_metrics = [k for k in ALL_METRIC_KEYS if st.session_state.get(f"trace_metric_{k}", False)]
    if not trace_selected_metrics:
        st.warning("Select at least one metric.")
    st.session_state.trace_eval_selected_metrics = trace_selected_metrics

    # ── Section D: Run ──────────────────────────────────────────────────────
    st.divider()
    render_section_header("▶", "Run Evaluation", "Choose Auto Pipeline (3-step) or run the LLM judge manually", step=4)
    if not selected_indices:
        st.info("Select trace rows above, then run the pipeline or evaluation.")

    st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#1e293b;margin:0 0 4px 0;">Auto Pipeline</p>', unsafe_allow_html=True)
    st.caption(
        "Fetches recommendations for the selected trace's system prompt, applies them to produce "
        "an enhanced prompt, then runs LLM-as-judge using the **same model from the trace** to "
        "compare output quality."
    )
    if st.button(
        f"Run Full Pipeline on {len(selected_indices)} selected trace(s)",
        width="stretch",
        key="trace_run_pipeline_btn",
        disabled=not selected_indices,
        type="primary",
    ):
        run_trace_pipeline()

    # Show pipeline recommendations if available
    pipeline_recs = st.session_state.get("trace_pipeline_recommendations", [])
    if pipeline_recs:
        with st.expander(f"Recommendations applied ({len(pipeline_recs)})", expanded=False):
            for i, rec in enumerate(pipeline_recs, 1):
                st.markdown(f"**{i}.** {rec}")

    # Show original vs current enhanced diff — always compares against the real original
    pipeline_original = st.session_state.get("trace_original_system_prompt", "")
    pipeline_enhanced = st.session_state.get("trace_enhanced_system_prompt", "")
    if pipeline_original and pipeline_enhanced and pipeline_original != pipeline_enhanced:
        st.write("#### Diff: Original → Current Enhanced")
        render_diff(pipeline_original, pipeline_enhanced, key="diff_trace_pipeline")

    st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#1e293b;margin:12px 0 4px 0;">Manual Judge</p>', unsafe_allow_html=True)
    st.caption("Use the prompts configured above (original + enhanced) to run the LLM judge directly.")
    if st.button(
        f"Run LLM Judge on {len(selected_indices)} selected trace(s)",
        width="stretch",
        key="trace_run_judge_btn",
        disabled=not selected_indices,
    ):
        run_llm_judge_on_traces()

    # ── Section E: Results ──────────────────────────────────────────────────
    judge_result = st.session_state.get("trace_llm_judge_result")
    if judge_result:
        st.divider()
        st.write("### Results")
        _render_llm_judge_results(judge_result)
        with st.expander("Debug: API Response", expanded=False):
            st.json(st.session_state.get("trace_llm_judge_api_debug", {}))

    # ── Section F: LLM Suggestions ──────────────────────────────────────────
    has_judge = bool(st.session_state.get("trace_llm_judge_result"))

    st.divider()
    st.write("### LLM Suggestions")
    st.caption(
        "One LLM call analyses all trace metrics, score deltas, and key differences "
        "to produce holistic, priority-ranked suggestions for improving the enhanced prompt."
    )
    if not has_judge:
        st.info("Run LLM Judge on traces first to enable suggestion generation.")
    if st.button("Generate Suggestions", width="stretch", key="trace_gen_suggestions_btn",
                 type="primary", disabled=not has_judge):
        st.session_state.trace_suggestions_result = None
        generate_overall_suggestions()

    trace_suggestions = st.session_state.get("trace_suggestions_result")
    if trace_suggestions:
        _render_overall_suggestions(trace_suggestions)

    # ── Section G: DeepEval Metrics ─────────────────────────────────────────
    st.divider()
    st.write("### DeepEval Metrics")
    st.caption("Reference-free metrics (AnswerRelevancy, GEval, PromptAlignment, PII) on every trace output. Requires LLM Judge to have been run above.")

    st.session_state.trace_deepeval_metrics_geval_criteria = st.text_input(
        "GEval Criteria",
        value=st.session_state.get("trace_deepeval_metrics_geval_criteria", ""),
        key="trace_deepeval_geval_criteria",
        help="Custom quality criteria for GEval. Leave blank to skip GEval.",
    )
    st.session_state.trace_deepeval_metrics_prompt_instructions = st.text_area(
        "Prompt Instructions for Alignment Check (one per line)",
        value=st.session_state.get("trace_deepeval_metrics_prompt_instructions", ""),
        height=80,
        key="trace_deepeval_prompt_instructions",
        placeholder="e.g.\nAlways respond in JSON.\nDo not use bullet points.\nLeave blank to skip PromptAlignment.",
        help="Instructions to check output alignment against. Leave blank to skip.",
    )

    if not has_judge:
        st.info("Run LLM Judge on traces first to enable DeepEval metrics.")
    if st.button("Run DeepEval Metrics", width="stretch", key="trace_deepeval_run_btn",
                 disabled=not has_judge, type="primary"):
        st.session_state.trace_deepeval_metrics_result = None
        run_deepeval_prompt_metrics()

    trace_metrics_result = st.session_state.get("trace_deepeval_metrics_result")
    if trace_metrics_result:
        _render_deepeval_prompt_metrics(trace_metrics_result)
