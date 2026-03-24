import streamlit as st

from ..actions import apply_recommendations, run_enhance_evaluation
from ..api_client import post_chat
from ..extractors import extract_enhanced_prompt
from ..components import render_diff, render_prompt_html, render_selected_recommendations_editor


_PRIORITY_CONFIG = {
    "HIGH":   {"color": "#e74c3c", "icon": "🔴"},
    "MEDIUM": {"color": "#f39c12", "icon": "🟡"},
    "LOW":    {"color": "#27ae60", "icon": "🟢"},
}


def _render_overall_suggestions(result: dict, selected_key: str = "trace_suggestions_selected") -> None:
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
        checked = col_check.checkbox("", key=f"{selected_key}_{i}", label_visibility="collapsed")
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
        if st.button("Apply Selected → Refine Enhanced Prompt", key=f"{selected_key}_apply_btn", type="primary"):
            _apply_suggestions_to_trace_prompt(selected_texts)


def _apply_suggestions_to_trace_prompt(suggestions: list[str]) -> None:
    """Apply selected suggestions: enhanced prompt becomes new original, suggestions applied on top."""
    import requests

    current_enhanced = st.session_state.get("trace_enhanced_system_prompt", "").strip()
    if not current_enhanced:
        st.error("No enhanced prompt found — run the pipeline or judge first.")
        return

    # Promote enhanced → original so the next iteration starts from it
    st.session_state.trace_original_system_prompt = current_enhanced

    reasoning = st.session_state.reasoning_effort
    payload = {
        "sessionId": st.session_state.session_id,
        "systemPrompt": current_enhanced,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": "validation",
        "recommendations": suggestions,
    }
    with st.spinner("Applying suggestions to generate refined prompt..."):
        try:
            data = post_chat(payload, include_grid_header=False)
            refined = extract_enhanced_prompt(data)
            if refined:
                st.session_state.trace_enhanced_system_prompt = refined
                # Clear previous results so user re-runs fresh
                st.session_state.trace_llm_judge_result = None
                st.session_state.trace_suggestions_result = None
                st.session_state.trace_deepeval_metrics_result = None
                st.success("Refined prompt applied. Original → previous enhanced. Run the judge again to measure improvement.")
                st.rerun()
            else:
                st.warning("API returned no refined prompt.")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def render_enhance_tab() -> None:
    st.subheader("Enhance Prompt")

    with st.expander("Original Prompt (Read-only)", expanded=False):
        render_prompt_html(
            st.session_state.original_prompt,
            label="Original Prompt",
            max_height=320,
        )

    render_selected_recommendations_editor()

    if st.button("Apply Recommendations & Enhance", width="stretch"):
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
    st.write("### Evaluate Original vs Enhanced")
    st.caption("Enter both prompts, upload test inputs, then run DeepEval to compare output quality.")

    # Auto-populate from session when the text area hasn't been manually edited yet
    if st.session_state.get("original_prompt") and not st.session_state.get("enhance_eval_orig_area"):
        st.session_state.enhance_eval_orig_area = st.session_state.original_prompt
    if st.session_state.get("enhanced_prompt") and not st.session_state.get("enhance_eval_enh_area"):
        st.session_state.enhance_eval_enh_area = st.session_state.enhanced_prompt

    col_btn_orig, col_btn_enh = st.columns(2)
    with col_btn_orig:
        if st.button("Use session original prompt", key="eval_use_orig_btn", width="stretch"):
            val = st.session_state.get("original_prompt", "")
            st.session_state.enhance_eval_orig_prompt = val
            st.session_state.enhance_eval_orig_area = val
    with col_btn_enh:
        if st.button("Use session enhanced prompt", key="eval_use_enh_btn", width="stretch"):
            val = st.session_state.get("enhanced_prompt", "")
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
    st.write("**Test Inputs**")
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
                st.rerun()

    if st.button("＋ Add Test Case", key="add_test_input_btn"):
        st.session_state.enhance_eval_input_count = count + 1
        st.rerun()



    if st.button("Run Evaluation", width="stretch", key="enhance_eval_run_btn"):
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
        metrics = ["clarity", "completeness", "accuracy", "conciseness", "professional_tone", "overall"]

        # ── Summary metrics ──────────────────────────────────────────────────
        st.write("#### Score Summary")
        improved = summary.get("improved_count", 0)
        total = summary.get("total_cases", 0)
        st.caption(f"Improved in {improved}/{total} test case(s)")

        avg_orig = summary.get("avg_original", {})
        avg_enh = summary.get("avg_enhanced", {})
        avg_delta = summary.get("avg_delta", {})

        header_cols = st.columns([2, 1, 1, 1])
        header_cols[0].markdown("**Metric**")
        header_cols[1].markdown("**Original**")
        header_cols[2].markdown("**Enhanced**")
        header_cols[3].markdown("**Delta**")
        for m in metrics:
            cols = st.columns([2, 1, 1, 1])
            orig_v = avg_orig.get(m)
            enh_v = avg_enh.get(m)
            delta_v = avg_delta.get(m)
            cols[0].write(m.replace("_", " ").title())
            cols[1].write(f"{orig_v:.3f}" if orig_v is not None else "—")
            cols[2].write(f"{enh_v:.3f}" if enh_v is not None else "—")
            delta_str = (f"{delta_v:+.3f}" if delta_v is not None else "—")
            cols[3].markdown(
                f"**:green[{delta_str}]**" if (delta_v or 0) > 0
                else (f"**:red[{delta_str}]**" if (delta_v or 0) < 0 else delta_str)
            )

        # ── Per-test results ─────────────────────────────────────────────────
        for i, tr in enumerate(test_results):
            st.divider()
            st.write(f"**Test Case {i + 1}**")
            scores = tr.get("scores", {})

            if "error" in scores:
                st.error(f"Judge error: {scores.get('error')}")
            else:
                improved_flag = scores.get("improved", False)
                st.markdown(
                    f"{'✅ Improved' if improved_flag else '❌ Not improved'} — {scores.get('improvement_summary', '')}"
                )
                diffs = scores.get("key_differences", [])
                if diffs:
                    with st.expander("Key Differences", expanded=True):
                        for d in diffs:
                            st.markdown(f"- {d}")

            col_orig_out, col_enh_out = st.columns(2)
            with col_orig_out:
                st.write("**Original Output**")
                st.text_area("Original Output", value=tr.get("original_output", ""), height=200,
                             disabled=True, key=f"judge_orig_out_{i}", label_visibility="collapsed")
            with col_enh_out:
                st.write("**Enhanced Output**")
                st.text_area("Enhanced Output", value=tr.get("enhanced_output", ""), height=200,
                             disabled=True, key=f"judge_enh_out_{i}", label_visibility="collapsed")

        with st.expander("Debug: API Response", expanded=False):
            st.json(st.session_state.get("enhance_eval_api_debug_original", {}))

    # ── Arena Evaluation ─────────────────────────────────────────────────────
    st.divider()
    st.write("### Arena Comparison (DeepEval)")
    st.caption(
        "Uses DeepEval's `ArenaGEval` to pick a winner per test case between "
        "original and enhanced prompt. Requires both prompts to be set above."
    )

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
        from src.chat.graph.arena_evaluator import run_arena_comparison
        with st.spinner(f"Running Arena comparison on {len(arena_test_cases)} test case(s)..."):
            try:
                arena_result = run_arena_comparison(
                    original_prompt=arena_orig,
                    enhanced_prompt=arena_enh,
                    test_cases=arena_test_cases,
                    criteria=arena_criteria,
                    model_name=st.session_state.get("model_name"),
                    max_tokens=st.session_state.get("max_tokens"),
                    temperature=st.session_state.get("temperature"),
                )
                st.session_state.arena_eval_result = arena_result
                st.success("Arena comparison complete.")
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
