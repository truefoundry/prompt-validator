import difflib

import streamlit as st


def render_recommendation_checkboxes() -> None:
    recommendations: list[str] = st.session_state.recommendations
    if not recommendations:
        st.info("No recommendations to display. Fetch recommendations first.")
        return

    st.write("### Select Recommendations to Apply")
    selected_values: list[str] = []
    for index, recommendation in enumerate(recommendations):
        key = f"rec_checkbox_{index}_{abs(hash(recommendation))}"
        previous_selected = recommendation in st.session_state.selected_recommendations
        checked = st.checkbox(recommendation, value=previous_selected, key=key)
        if checked:
            selected_values.append(recommendation)

    st.session_state.selected_recommendations = selected_values


def render_selected_recommendations_editor() -> list[str]:
    selected = st.session_state.selected_recommendations
    selected_signature = "||".join(selected)
    if (
        not st.session_state.editable_recommendations
        or st.session_state.last_selected_signature != selected_signature
    ):
        st.session_state.editable_recommendations = list(selected)
        st.session_state.last_selected_signature = selected_signature

    st.write("### Selected Recommendations (Editable)")
    editable_text = st.text_area(
        "Edit selected recommendations (one per line)",
        value="\n".join(st.session_state.editable_recommendations),
        height=160,
        help="You can update the recommendation text before applying.",
    )

    updated = [line.strip() for line in editable_text.splitlines() if line.strip()]
    st.session_state.editable_recommendations = updated
    return updated


def render_scores(title: str, empty_info_text: str) -> None:
    st.write(title)
    if isinstance(st.session_state.total_score, int):
        st.metric("Total Score", st.session_state.total_score)

    criteria_scores = st.session_state.criteria_scores
    explanations = st.session_state.explanations

    if not criteria_scores:
        st.info(empty_info_text)
        return

    for criterion, score in criteria_scores.items():
        label = str(criterion).replace("_", " ").title()
        explanation = (
            explanations.get(criterion, "")
            if isinstance(explanations, dict)
            else ""
        )
        col_score, col_explanation = st.columns([1, 4])
        with col_score:
            st.metric(label, score)
        with col_explanation:
            if explanation:
                st.caption(explanation)


def render_explanations(title: str, empty_info_text: str) -> None:
    st.write(title)
    explanations = st.session_state.explanations
    if not isinstance(explanations, dict) or not explanations:
        st.info(empty_info_text)
        return

    for criterion, explanation in explanations.items():
        with st.expander(str(criterion).replace("_", " ").title(), expanded=False):
            st.write(explanation)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_diff(original: str, enhanced: str) -> None:
    st.write("### Diff: Original vs Enhanced")

    if not original and not enhanced:
        st.info("Apply recommendations to see the diff here.")
        return

    if not enhanced:
        st.info("Enhanced prompt not available yet.")
        return

    original_lines = original.splitlines() if original else []
    enhanced_lines = enhanced.splitlines()

    diff = list(difflib.ndiff(original_lines, enhanced_lines))

    html_parts = [
        '<div style="border:1px solid #d0d0d0; border-radius:6px; '
        'overflow-y:auto; max-height:520px; font-family:monospace; font-size:13px;">'
    ]

    for line in diff:
        if line.startswith("- "):
            content = _escape(line[2:])
            html_parts.append(
                f'<div style="background:#ffeef0; color:#b31d28; '
                f'padding:4px 12px; border-left:4px solid #e74c3c; margin:0;">'
                f'<span style="font-weight:bold; margin-right:6px;">-</span>{content}</div>'
            )
        elif line.startswith("+ "):
            content = _escape(line[2:])
            html_parts.append(
                f'<div style="background:#e6ffed; color:#22863a; '
                f'padding:4px 12px; border-left:4px solid #2ecc71; margin:0;">'
                f'<span style="font-weight:bold; margin-right:6px;">+</span>{content}</div>'
            )
        elif line.startswith("? "):
            continue
        else:
            content = _escape(line[2:])
            html_parts.append(
                f'<div style="padding:4px 12px; color:inherit; margin:0;">'
                f'<span style="margin-right:6px; visibility:hidden;">+</span>{content}</div>'
            )

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.download_button(
        label="Save Enhanced Prompt",
        data=enhanced,
        file_name="enhanced_prompt.txt",
        mime="text/plain",
        use_container_width=True,
    )


def render_deepeval_results(result: dict) -> None:
    inner = result.get("results", result)  # unwrap {"results": ..., "recommendation_result": ...}
    all_metrics = inner.get("all_metrics", {})
    all_tests = inner.get("all_tests", [])

    num_tests = all_metrics.get("Number of Tests", len(all_tests))
    num_errors = all_metrics.get("Number of Tests with Errors", 0)
    num_passed = sum(1 for t in all_tests if t.get("pass_status"))
    pass_rate = f"{num_passed / num_tests * 100:.1f}%" if num_tests else "—"

    st.write("#### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tests", num_tests)
    c2.metric("Passed", num_passed)
    c3.metric("Pass Rate", pass_rate)
    c4.metric("Errors", num_errors)

    avg_keys = [k for k in all_metrics if k.startswith("Average ")]
    if avg_keys:
        cols = st.columns(len(avg_keys))
        for col, key in zip(cols, avg_keys):
            col.metric(key, f"{all_metrics[key]:.3f}")

    if not all_tests:
        return

    st.divider()
    st.write("#### Test Cases")
    for t in all_tests:
        tid = t.get("test_case_id", "")
        passed = t.get("pass_status")
        icon = "✅" if passed else "❌"
        input_text = (
            t.get("input")
            or t.get("data", {}).get("input", "")
            or t.get("test_case_name", "")
        )
        label = (
            f"{icon}  Test {tid} — {str(input_text)[:80]}{'…' if len(str(input_text)) > 80 else ''}"
            if input_text
            else f"{icon}  Test {tid}"
        )
        with st.expander(label, expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Test Case ID:** `{tid}`")
            c2.markdown(f"**Status:** {'✅ Passed' if passed else '❌ Failed'}")

            name = t.get("test_case_name", "")
            if name:
                st.markdown(f"**Test Case Name:** {name}")

            if input_text:
                st.markdown("**Input:**")
                st.code(input_text, language=None)

            st.markdown("**Expected Output:**")
            st.code(t.get("expected_output") or "—", language=None)

            st.markdown("**Actual Output:**")
            st.code(t.get("actual_output") or "—", language=None)

            metric_values = t.get("metric_values", {})
            if metric_values:
                st.markdown("**Metric Results:**")
                for metric_name, md in metric_values.items():
                    m_icon = "✅" if md.get("success") else "❌"
                    score = md.get("score")
                    score_str = f"{score:.3f}" if score is not None else "—"
                    st.markdown(
                        f"- **{metric_name}**: {score_str} {m_icon} "
                        f"&nbsp;*(threshold: {md.get('threshold', '—')})*"
                    )
                    reason = md.get("reason", "")
                    if reason:
                        st.caption(f"  Reason: {reason}")

    recommendation = result.get("recommendation_result", "")
    if recommendation:
        st.divider()
        st.write("#### LLM Commentary")
        st.info(recommendation)


def render_exact_match_results(result: dict) -> None:
    inner = result.get("results", result)  # unwrap {"results": ..., "recommendation_result": ...}
    all_metrics = inner.get("all_metrics", {})
    all_tests = inner.get("all_tests", [])

    st.write("#### Summary")
    num_tests = all_metrics.get("Number of Tests", len(all_tests))
    num_passed = sum(1 for t in all_tests if t.get("pass_status"))
    num_failed = num_tests - num_passed

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Tests", num_tests)
    c2.metric("Passed", num_passed)
    c3.metric("Failed", num_failed)
    c4.metric("Accuracy", f"{all_metrics.get('Accuracy', 0):.3f}")
    c5.metric("Avg Precision", f"{all_metrics.get('Average Precision', 0):.3f}")
    c6.metric("Avg Recall", f"{all_metrics.get('Average Recall', 0):.3f}")
    c7.metric("Avg F1", f"{all_metrics.get('Average F1 Score', 0):.3f}")

    classwise = all_metrics.get("Classwise Metrics", {})
    if classwise:
        st.divider()
        st.write("#### Classwise Metrics")
        class_rows = [
            {
                "Class": label,
                "Tests": data.get("Number of Tests", 0),
                "Accuracy": round(data.get("Accuracy", 0), 3),
                "Precision": round(data.get("Precision", 0), 3),
                "Recall": round(data.get("Recall", 0), 3),
                "F1 Score": round(data.get("F1 Score", 0), 3),
            }
            for label, data in classwise.items()
        ]
        st.dataframe(class_rows, use_container_width=True, hide_index=True)

    if not all_tests:
        return

    st.divider()
    st.write("#### Test Cases")
    for t in all_tests:
        tid = t.get("test_case_id", "")
        passed = t.get("pass_status")
        icon = "✅" if passed else "❌"
        input_text = (
            t.get("input")
            or t.get("data", {}).get("input", "")
        )
        label = f"{icon}  Test {tid} — {str(input_text)[:80]}{'…' if len(str(input_text)) > 80 else ''}"
        with st.expander(label, expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Test Case ID:** `{tid}`")
            c2.markdown(f"**Status:** {'✅ Passed' if passed else '❌ Failed'}")

            name = t.get("test_case_name", "")
            if name:
                st.markdown(f"**Test Case Name:** {name}")

            if input_text:
                st.markdown("**Input:**")
                st.code(input_text, language=None)

            st.markdown("**Expected Output:**")
            st.code(t.get("expected_output") or "—", language=None)

            st.markdown("**Actual Output:**")
            st.code(t.get("actual_output") or "—", language=None)

    recommendation = result.get("recommendation_result", "")
    if recommendation:
        st.divider()
        st.write("#### LLM Commentary")
        st.info(recommendation)
