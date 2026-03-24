import streamlit as st

from ..actions import generate_overall_suggestions, run_deepeval_prompt_metrics
from ..components import render_diff
from ..trace_actions import (
    fetch_live_trace_inputs,
    load_floqast_prompts,
    load_trace_inputs,
    run_llm_judge_on_traces,
    run_trace_pipeline,
)
from .enhance import _render_overall_suggestions


_JUDGE_METRICS = ["clarity", "completeness", "accuracy", "conciseness", "professional_tone", "overall"]

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


def _get_tfy_tenant() -> str:
    """Extract tenant name from prompt FQN stored in session."""
    fqn = st.session_state.get("prompt_fqn", "")
    if fqn and "/" in fqn:
        parts = fqn.split("/")
        if len(parts) >= 2:
            return parts[1]
    return ""


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
    import pandas as pd

    summary = result.get("summary", {})
    test_results = result.get("test_results", [])

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
        for m in _JUDGE_METRICS:
            o = avg_orig.get(m)
            e = avg_enh.get(m)
            d = avg_delta.get(m)
            rows.append({
                "Metric": m.replace("_", " ").title(),
                "Original": f"{o:.3f}" if o is not None else "—",
                "Enhanced": f"{e:.3f}" if e is not None else "—",
                "Δ Delta": f"{d:+.3f}" if d is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

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
        for tr in test_results:
            scores = tr.get("scores", {})
            orig_overall = scores.get("original", {}).get("overall")
            enh_overall = scores.get("enhanced", {}).get("overall")
            improved_flag = scores.get("improved", False)
            badge = "✅" if improved_flag else "➖"
            label = (
                f"{badge} {tr.get('test_case_name', tr.get('test_case_id', '?'))}  "
                f"| Orig: {orig_overall:.2f} → Enh: {enh_overall:.2f}"
                if orig_overall is not None and enh_overall is not None
                else f"{badge} {tr.get('test_case_name', tr.get('test_case_id', '?'))}"
            )
            with st.expander(label, expanded=False):
                with st.expander("Input", expanded=False):
                    st.text_area("Input", value=tr.get("input", ""), height=200,
                                 disabled=True, key=f"trace_input_{tr.get('test_case_id')}",
                                 label_visibility="collapsed")
                col_o, col_e = st.columns(2)
                with col_o:
                    with st.expander("Original Output", expanded=False):
                        st.text_area("Original Output", value=tr.get("original_output", ""), height=200,
                                     disabled=True, key=f"trace_orig_out_{tr.get('test_case_id')}",
                                     label_visibility="collapsed")
                with col_e:
                    with st.expander("Enhanced Output", expanded=False):
                        st.text_area("Enhanced Output", value=tr.get("enhanced_output", ""), height=200,
                                     disabled=True, key=f"trace_enh_out_{tr.get('test_case_id')}",
                                     label_visibility="collapsed")

                if "error" not in scores:
                    metric_rows = []
                    for m in _JUDGE_METRICS:
                        o = scores.get("original", {}).get(m)
                        e = scores.get("enhanced", {}).get(m)
                        metric_rows.append({
                            "Metric": m.replace("_", " ").title(),
                            "Original": f"{o:.2f}" if o is not None else "—",
                            "Enhanced": f"{e:.2f}" if e is not None else "—",
                        })
                    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")

                if scores.get("improvement_summary"):
                    st.info(scores["improvement_summary"])
                if scores.get("key_differences"):
                    st.markdown("**Key Differences:**")
                    for d in scores["key_differences"]:
                        st.markdown(f"- {d}")


def render_trace_eval_tab() -> None:
    st.subheader("Trace Evaluation")
    st.caption(
        "Load real production traces, group by prompt, then compare original vs enhanced prompt quality using LLM-as-judge."
    )

    # ── Section A: Load Traces ──────────────────────────────────────────────
    st.write("### 1. Load Traces")

    source_label = st.radio(
        "Trace source",
        _TRACE_SOURCE_OPTIONS,
        horizontal=True,
        key="trace_source_radio",
    )

    uploaded_content: str | None = None
    if source_label == "Fetch Live Traces":
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
    elif source_label == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload traces JSON",
            type=["json"],
            key="trace_file_uploader",
        )
        if uploaded_file:
            uploaded_content = uploaded_file.read().decode("utf-8")

    _SOURCE_MAP = {"Use traces.json": "traces.json", "Upload file": "upload"}

    if st.button("Load Traces", width="stretch", key="trace_load_btn"):
        if source_label == "Fetch Live Traces":
            fetch_live_trace_inputs(
                hours=int(st.session_state.get("trace_live_days", 1)) * 24,
                limit=int(st.session_state.get("trace_live_limit", 200)),
                fqn_filter=st.session_state.get("trace_live_fqn", "").strip() or None,
            )
        else:
            load_trace_inputs(
                source=_SOURCE_MAP[source_label],
                uploaded_content=uploaded_content,
            )

    trace_inputs: list = st.session_state.get("trace_inputs", [])
    if not trace_inputs:
        return

    # ── Section B: Group by Prompt FQN ─────────────────────────────────────
    st.divider()
    st.write("### 2. Select Prompt Group")

    # Build FQN options with counts
    fqn_counts: dict[str, int] = {}
    for ti in trace_inputs:
        fqn = ti.prompt_fqn or "(no FQN)"
        fqn_counts[fqn] = fqn_counts.get(fqn, 0) + 1

    fqn_options = sorted(fqn_counts.keys())
    fqn_display = [f"{fqn}  ({fqn_counts[fqn]} traces)" for fqn in fqn_options]
    fqn_display_map = dict(zip(fqn_display, fqn_options))

    selected_display = st.selectbox(
        "Group by Prompt FQN",
        options=fqn_display,
        key="trace_fqn_selectbox",
    )
    selected_fqn = fqn_display_map.get(selected_display, "")
    st.session_state.trace_selected_fqn = selected_fqn

    # Filter traces to selected FQN
    filtered = [ti for ti in trace_inputs if (ti.prompt_fqn or "(no FQN)") == selected_fqn]

    # Auto-fill original system prompt from the first trace that has one
    # Use getattr for backward compat with TraceInput objects loaded before system_prompt was added
    fqn_sys_prompt = next(
        (getattr(ti, "system_prompt", "") for ti in filtered if getattr(ti, "system_prompt", "")),
        "",
    )

    import pandas as pd

    display_rows = filtered[:30]
    all_display_indices = [trace_inputs.index(ti) for ti in display_rows]

    # Build display DataFrame with truncated text for the table view
    def _trunc(s: str, n: int = 100) -> str:
        s = (s or "").replace("\n", " ")
        return s[:n] + "…" if len(s) > n else s

    table_data = []
    for i, ti in enumerate(display_rows):
        orig_idx = trace_inputs.index(ti)
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

    prev_selected_orig = set(st.session_state.get("trace_selected_indices", []))
    prev_selected_options = [opt for opt, idx in row_option_to_orig.items() if idx in prev_selected_orig]

    col_multi, col_btns = st.columns([3, 1])
    with col_multi:
        chosen_options = st.multiselect(
            "Select traces to evaluate",
            options=row_options,
            default=prev_selected_options,
            key="trace_multiselect",
            placeholder="Click to pick traces, or use Select All →",
        )
    with col_btns:
        st.write("")  # vertical align
        st.write("")
        if st.button("Select All", key="trace_sel_all", width="stretch"):
            st.session_state.trace_selected_indices = all_display_indices
            st.rerun()
        if st.button("Clear", key="trace_clear_sel", width="stretch"):
            st.session_state.trace_selected_indices = []
            st.rerun()

    selected_indices = [row_option_to_orig[opt] for opt in chosen_options]
    st.session_state.trace_selected_indices = selected_indices
    st.caption(f"{len(selected_indices)} selected  |  showing up to 30 of {len(filtered)} traces")

    # Details for selected traces
    if selected_indices:
        with st.expander(f"Selected trace details ({len(selected_indices)})", expanded=False):
            for orig_idx in selected_indices:
                ti = trace_inputs[orig_idx]
                st.markdown(f"**Trace #{all_display_indices.index(orig_idx) + 1 if orig_idx in all_display_indices else orig_idx}** — `{ti.span_id or 'no span id'}`")
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
    st.write("### 3. Configure Prompts")
    st.caption(
        f"Prompt group: `{selected_fqn}`  — original system prompt auto-filled from trace data."
    )

    # Auto-populate original system prompt when FQN changes and traces have a system prompt
    last_fqn = st.session_state.get("_trace_last_autofilled_fqn", "")
    if fqn_sys_prompt and selected_fqn != last_fqn:
        st.session_state.trace_original_system_prompt = fqn_sys_prompt
        st.session_state._trace_last_autofilled_fqn = selected_fqn

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
                st.rerun()
        else:
            st.caption("No enhanced prompt in session yet — go to the **Enhance** tab to generate one, or paste below.")
        st.session_state.trace_enhanced_system_prompt = st.text_area(
            "Enhanced system prompt",
            value=st.session_state.get("trace_enhanced_system_prompt", ""),
            height=260,
            placeholder="Paste the enhanced system prompt here...",
            label_visibility="collapsed",
        )

    st.session_state.trace_user_prompt_template = st.text_input(
        "User Prompt Template (optional — use {{input}} for injection)",
        value=st.session_state.get("trace_user_prompt_template", ""),
        placeholder="e.g.  Answer this question: {{input}}",
        key="trace_user_tpl_input",
    )

    # ── Section D: Run ──────────────────────────────────────────────────────
    st.divider()
    if not selected_indices:
        st.info("Select trace rows above, then run the pipeline or evaluation.")

    st.write("#### Auto Pipeline")
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

    # Show original vs enhanced diff after pipeline runs
    pipeline_original = st.session_state.get("trace_original_system_prompt", "")
    pipeline_enhanced = st.session_state.get("trace_enhanced_system_prompt", "")
    if pipeline_enhanced and pipeline_recs:
        st.write("#### Enhanced Prompt")
        render_diff(pipeline_original, pipeline_enhanced, key="diff_trace_pipeline")

    st.write("#### Manual Judge")
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

    # ── Section F: DeepEval Metrics ─────────────────────────────────────────
    has_judge = bool(st.session_state.get("trace_llm_judge_result"))

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

    # ── Section G: LLM Suggestions ──────────────────────────────────────────
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
