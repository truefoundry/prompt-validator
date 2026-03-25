import difflib
import re

import streamlit as st


def render_section_header(
    icon: str,
    title: str,
    description: str = "",
    step: int | None = None,
) -> None:
    """Render a styled section header with optional step number and description."""
    step_html = ""
    if step is not None:
        step_html = f'<span class="pt-step">{step}</span>'
    desc_html = f'<p class="pt-desc">{description}</p>' if description else ""
    st.markdown(
        f"""<div class="pt-section-header">
  {step_html}
  <span class="pt-icon">{icon}</span>
  <div><p class="pt-title">{title}</p>{desc_html}</div>
</div>""",
        unsafe_allow_html=True,
    )

_PROMPT_CONTAINER_CSS = """\
<style>
.prompt-box {
    border: 1px solid rgba(150,150,150,0.25);
    border-radius: 8px;
    overflow-y: auto;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 13px;
    line-height: 1.65;
    padding: 0;
    background: rgba(30,30,30,0.35);
}
.prompt-box .prompt-line {
    display: flex;
    padding: 0 16px;
    min-height: 20px;
    border-bottom: 1px solid rgba(150,150,150,0.06);
}
.prompt-box .prompt-line:hover {
    background: rgba(255,255,255,0.03);
}
.prompt-box .line-no {
    flex-shrink: 0;
    width: 38px;
    text-align: right;
    padding-right: 14px;
    color: rgba(150,150,150,0.35);
    user-select: none;
    font-size: 11px;
    line-height: 1.65;
}
.prompt-box .line-content {
    flex: 1;
    white-space: pre-wrap;
    word-break: break-word;
    color: #e0e0e0;
}
.prompt-box .xml-tag { color: #7ec8e3; }
.prompt-box .xml-attr-name { color: #c9a0dc; }
.prompt-box .xml-attr-value { color: #ce9178; }
.prompt-box .tpl-var { color: #dcdcaa; font-weight: 600; }
.prompt-box .md-heading { color: #569cd6; font-weight: 700; }
.prompt-box strong.md-bold { color: #e0e0e0; font-weight: 700; }
.prompt-box em.md-italic { color: #c9d1d9; font-style: italic; }
.prompt-box code.md-code {
    background: rgba(110,118,129,0.2);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 12px;
    color: #79c0ff;
}
.prompt-box .md-bullet { color: #6a9955; font-weight: 700; }
.prompt-box .comment-line { color: #6a9955; font-style: italic; }
.prompt-box .empty-line { min-height: 20px; }
.prompt-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: rgba(80,80,80,0.25);
    border-bottom: 1px solid rgba(150,150,150,0.15);
    border-radius: 8px 8px 0 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 12px;
    color: rgba(200,200,200,0.7);
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.prompt-header .line-count {
    font-size: 11px;
    opacity: 0.6;
    text-transform: none;
}
</style>
"""

_TAG_RE = re.compile(
    r'(&lt;/?)'                     # opening < or </
    r'([\w:.-]+)'                   # tag name
    r'((?:\s+[\w:.-]+\s*=\s*'       # optional attributes
    r'(?:&quot;[^&]*?&quot;|'
    r"&apos;[^&]*?&apos;|"
    r'&amp;\w+;|[^\s&>]*))*)'
    r'(\s*/?&gt;)',                  # closing > or />
    re.DOTALL,
)
_ATTR_RE = re.compile(
    r'([\w:.-]+)(\s*=\s*)((?:&quot;[^&]*?&quot;|&apos;[^&]*?&apos;|[^\s&]+))'
)
_TPL_VAR_RE = re.compile(r'(\{\{[\w.\-\s|]+?\}\}|\{[\w.\-]+?\})')
_MD_BOLD_RE = re.compile(r'\*\*([^*]+?)\*\*|__([^_]+?)__')
_MD_ITALIC_RE = re.compile(r'(?<!\*)\*([^*]+?)\*(?!\*)|(?<!_)_([^_]+?)_(?!_)')
_MD_CODE_RE = re.compile(r'`([^`]+?)`')
_MD_BULLET_RE = re.compile(r'^(\s*(?:[-*]|\d+\.)\s)')


def _highlight(text: str) -> str:
    """Apply syntax highlighting + markdown rendering to an HTML-escaped prompt line."""
    text = _TAG_RE.sub(_highlight_tag, text)
    text = _TPL_VAR_RE.sub(r'<span class="tpl-var">\1</span>', text)
    # Render markdown as actual HTML formatting
    text = _MD_BOLD_RE.sub(lambda m: f'<strong class="md-bold">{m.group(1) or m.group(2)}</strong>', text)
    text = _MD_ITALIC_RE.sub(lambda m: f'<em class="md-italic">{m.group(1) or m.group(2)}</em>', text)
    text = _MD_CODE_RE.sub(r'<code class="md-code">\1</code>', text)
    m_bullet = _MD_BULLET_RE.match(text)
    if m_bullet:
        text = f'<span class="md-bullet">{m_bullet.group(1)}</span>{text[m_bullet.end():]}'
    return text


def _highlight_tag(m: re.Match) -> str:
    open_bracket = m.group(1)
    tag_name = m.group(2)
    attrs_raw = m.group(3)
    close_bracket = m.group(4)

    attrs_highlighted = _ATTR_RE.sub(
        r'<span class="xml-attr-name">\1</span>\2<span class="xml-attr-value">\3</span>',
        attrs_raw,
    )
    return (
        f'<span class="xml-tag">{open_bracket}{tag_name}</span>'
        f'{attrs_highlighted}'
        f'<span class="xml-tag">{close_bracket}</span>'
    )


def render_prompt_html(prompt_text: str, *, label: str = "Prompt", max_height: int = 480, key: str | None = None) -> None:
    """Render a prompt string as a syntax-highlighted, read-only HTML block."""
    if not prompt_text:
        st.info(f"No {label.lower()} available yet.")
        return

    lines = prompt_text.splitlines()
    line_count = len(lines)

    html_lines: list[str] = []
    for i, raw_line in enumerate(lines, start=1):
        escaped = _escape(raw_line)
        if not escaped.strip():
            html_lines.append(
                f'<div class="prompt-line empty-line">'
                f'<span class="line-no">{i}</span>'
                f'<span class="line-content"> </span></div>'
            )
            continue

        is_heading = raw_line.lstrip().startswith("#")
        highlighted = _highlight(escaped)
        if is_heading:
            highlighted = f'<span class="md-heading">{highlighted}</span>'

        html_lines.append(
            f'<div class="prompt-line">'
            f'<span class="line-no">{i}</span>'
            f'<span class="line-content">{highlighted}</span></div>'
        )

    html = (
        f'{_PROMPT_CONTAINER_CSS}'
        f'<div class="prompt-box" style="max-height:{max_height}px;">'
        f'<div class="prompt-header">'
        f'<span>{_escape(label)}</span>'
        f'<span class="line-count">{line_count} line{"s" if line_count != 1 else ""}</span>'
        f'</div>'
        f'{"".join(html_lines)}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation_checkboxes() -> None:
    recommendations: list[str] = st.session_state.recommendations
    if not recommendations:
        st.info("No recommendations to display. Fetch recommendations first.")
        return

    render_section_header("💡", "Select Recommendations to Apply",
                          f"{len(recommendations)} suggestion(s) found — check the ones you want to apply")
    selected_values: list[str] = []
    for index, recommendation in enumerate(recommendations):
        key = f"rec_checkbox_{index}_{abs(hash(recommendation))}"
        previous_selected = recommendation in st.session_state.selected_recommendations
        col_cb, col_text = st.columns([0.05, 0.95])
        with col_cb:
            checked = st.checkbox("", value=previous_selected, key=key, label_visibility="collapsed")
        with col_text:
            bg = "#f5f3ff" if previous_selected else "#fafafa"
            border = "#c4b5fd" if previous_selected else "#e2e8f0"
            num_color = "#6366f1"
            st.markdown(
                f"""<div style="background:{bg};border:1.5px solid {border};border-radius:8px;
padding:10px 14px;margin-bottom:4px;display:flex;align-items:flex-start;gap:10px;
cursor:pointer;transition:all 0.15s;">
  <span style="background:#6366f115;color:{num_color};font-size:0.72rem;font-weight:700;
               padding:2px 7px;border-radius:20px;flex-shrink:0;margin-top:1px;">
    #{index + 1}
  </span>
  <span style="font-size:0.875rem;color:#334155;line-height:1.5;">{recommendation}</span>
</div>""",
                unsafe_allow_html=True,
            )
        if checked:
            selected_values.append(recommendation)

    st.session_state.selected_recommendations = selected_values
    if selected_values:
        st.markdown(
            f'<p style="font-size:0.8rem;color:#6366f1;font-weight:600;margin:6px 0 0 0;">'
            f'✓ {len(selected_values)} recommendation(s) selected</p>',
            unsafe_allow_html=True,
        )


def render_selected_recommendations_editor() -> list[str]:
    selected = st.session_state.selected_recommendations
    selected_signature = "||".join(selected)
    if (
        not st.session_state.editable_recommendations
        or st.session_state.last_selected_signature != selected_signature
    ):
        st.session_state.editable_recommendations = list(selected)
        st.session_state.last_selected_signature = selected_signature

    count = len(st.session_state.editable_recommendations)
    st.markdown(
        f'<p style="font-size:0.82rem;font-weight:600;color:#334155;margin:0 0 6px 0;">'
        f'Selected Recommendations'
        f'{"" if not count else f" <span style=\'background:#6366f115;color:#6366f1;padding:1px 8px;border-radius:20px;font-size:0.72rem;font-weight:700;\'>{count}</span>"}'
        f'</p>',
        unsafe_allow_html=True,
    )
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
    render_section_header("📊", title.lstrip("#").strip())

    criteria_scores = st.session_state.criteria_scores
    explanations = st.session_state.explanations

    if isinstance(st.session_state.total_score, int):
        total = st.session_state.total_score
        color = "#10b981" if total >= 70 else "#f59e0b" if total >= 40 else "#ef4444"
        label_text = "Excellent" if total >= 70 else "Needs Work" if total >= 40 else "Needs Improvement"
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,{color}18,{color}28);
border:1px solid {color}50;border-radius:14px;padding:16px 22px;
display:flex;align-items:center;gap:20px;margin:12px 0 16px 0;">
  <div style="font-size:2.4rem;font-weight:800;color:{color};line-height:1;">{total}</div>
  <div>
    <p style="margin:0;font-size:0.82rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.06em;color:{color};">Total Score</p>
    <p style="margin:4px 0 0 0;font-size:0.78rem;color:#64748b;">{label_text}</p>
  </div>
  <div style="margin-left:auto;background:{color}20;border-radius:8px;
              padding:6px 14px;border:1px solid {color}40;">
    <span style="font-size:0.8rem;font-weight:600;color:{color};">{total}/100</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    if not criteria_scores:
        st.info(empty_info_text)
        return

    cols = st.columns(min(len(criteria_scores), 4))
    for i, (criterion, score) in enumerate(criteria_scores.items()):
        label = str(criterion).replace("_", " ").title()
        explanation = (
            explanations.get(criterion, "")
            if isinstance(explanations, dict)
            else ""
        )
        with cols[i % len(cols)]:
            st.metric(label, score)
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


@st.cache_data(show_spinner=False)
def _build_diff_html(original: str, enhanced: str) -> str:
    """Build the diff HTML string for two prompt texts.

    Pure function — cached by Streamlit so the difflib computation and HTML
    construction only run when *original* or *enhanced* actually changes.
    """
    original_lines = original.splitlines() if original else []
    enhanced_lines = enhanced.splitlines()

    diff = list(difflib.ndiff(original_lines, enhanced_lines))

    removed = sum(1 for l in diff if l.startswith("- "))
    added = sum(1 for l in diff if l.startswith("+ "))

    html_parts = [
        f'<div style="margin-bottom:8px;display:flex;gap:10px;align-items:center;">'
        f'<span style="background:rgba(248,81,73,0.15);color:#ff7b72;padding:3px 10px;border-radius:20px;'
        f'font-size:0.75rem;font-weight:600;border:1px solid rgba(248,81,73,0.4);">−{removed} removed</span>'
        f'<span style="background:rgba(63,185,80,0.15);color:#3fb950;padding:3px 10px;border-radius:20px;'
        f'font-size:0.75rem;font-weight:600;border:1px solid rgba(63,185,80,0.4);">+{added} added</span>'
        f'</div>',
        '<div style="border:1px solid #30363d; border-radius:10px; overflow:hidden; '
        'overflow-y:auto; max-height:520px; font-family:\'SFMono-Regular\',Consolas,monospace; '
        'font-size:0.82rem; line-height:1.7; background:#0d1117;">',
    ]

    for line in diff:
        if line.startswith("- "):
            content = _escape(line[2:])
            html_parts.append(
                f'<div style="background:rgba(248,81,73,0.1);color:#ff7b72;padding:3px 14px;'
                f'border-left:3px solid #f85149;margin:0;white-space:pre-wrap;word-break:break-all;">'
                f'<span style="font-weight:700;margin-right:8px;">−</span>{content}</div>'
            )
        elif line.startswith("+ "):
            content = _escape(line[2:])
            html_parts.append(
                f'<div style="background:rgba(46,160,67,0.1);color:#3fb950;padding:3px 14px;'
                f'border-left:3px solid #238636;margin:0;white-space:pre-wrap;word-break:break-all;">'
                f'<span style="font-weight:700;margin-right:8px;">+</span>{content}</div>'
            )
        elif line.startswith("? "):
            continue
        else:
            content = _escape(line[2:])
            html_parts.append(
                f'<div style="padding:3px 14px;color:#8b949e;margin:0;'
                f'white-space:pre-wrap;word-break:break-all;border-left:3px solid transparent;">'
                f'<span style="margin-right:8px;visibility:hidden;">+</span>{content}</div>'
            )

    html_parts.append("</div>")
    return "".join(html_parts)


def render_diff(original: str, enhanced: str, key: str = "download_enhanced_prompt_diff") -> None:
    render_section_header("🔀", "Diff: Original vs Enhanced", "Lines removed from original (red) and added in enhanced (green)")

    if not original and not enhanced:
        st.info("Apply recommendations to see the diff here.")
        return

    if not enhanced:
        st.info("Enhanced prompt not available yet.")
        return

    st.markdown(_build_diff_html(original, enhanced), unsafe_allow_html=True)

    st.download_button(
        label="Save Enhanced Prompt",
        data=enhanced,
        file_name="enhanced_prompt.txt",
        mime="text/plain",
        width="stretch",
        key=key,
    )


def render_deepeval_results(result: dict) -> None:
    inner = result.get("results", result)  # unwrap {"results": ..., "recommendation_result": ...}
    all_metrics = inner.get("all_metrics", {})
    all_tests = inner.get("all_tests", [])

    num_tests = all_metrics.get("Number of Tests", len(all_tests))
    num_errors = all_metrics.get("Number of Tests with Errors", 0)
    num_passed = sum(1 for t in all_tests if t.get("pass_status"))
    pass_rate_val = num_passed / num_tests * 100 if num_tests else 0
    pass_rate = f"{pass_rate_val:.1f}%"

    render_section_header("📋", "Test Results Summary")
    rate_color = "#10b981" if pass_rate_val >= 80 else "#f59e0b" if pass_rate_val >= 50 else "#ef4444"
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,{rate_color}12,{rate_color}20);
border:1px solid {rate_color}40;border-radius:12px;padding:14px 20px;margin:0 0 16px 0;
display:flex;align-items:center;gap:8px;">
  <span style="font-size:1.5rem;font-weight:800;color:{rate_color};">{pass_rate}</span>
  <span style="font-size:0.82rem;color:#64748b;">pass rate &nbsp;·&nbsp;
  <strong style="color:#1e293b;">{num_passed}</strong> of <strong style="color:#1e293b;">{num_tests}</strong> tests passed
  {f'&nbsp;·&nbsp; <strong style="color:#ef4444;">{num_errors} error(s)</strong>' if num_errors else ''}
  </span>
</div>""",
        unsafe_allow_html=True,
    )
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
        latency = t.get("latency_s")
        latency_str = f" · ⏱ {latency}s" if latency is not None else ""
        input_text = (
            t.get("input")
            or t.get("data", {}).get("input", "")
            or t.get("test_case_name", "")
        )
        label = (
            f"{icon}  Test {tid} — {str(input_text)[:80]}{'…' if len(str(input_text)) > 80 else ''}{latency_str}"
            if input_text
            else f"{icon}  Test {tid}{latency_str}"
        )
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Test Case ID:** `{tid}`")
            c2.markdown(f"**Status:** {'✅ Passed' if passed else '❌ Failed'}")
            if latency is not None:
                c3.markdown(f"**Latency:** `{latency}s`")

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
        st.dataframe(class_rows, width="stretch", hide_index=True)

    if not all_tests:
        return

    st.divider()
    st.write("#### Test Cases")
    for t in all_tests:
        tid = t.get("test_case_id", "")
        passed = t.get("pass_status")
        icon = "✅" if passed else "❌"
        latency = t.get("latency_s")
        latency_str = f" · ⏱ {latency}s" if latency is not None else ""
        input_text = (
            t.get("input")
            or t.get("data", {}).get("input", "")
        )
        label = f"{icon}  Test {tid} — {str(input_text)[:80]}{'…' if len(str(input_text)) > 80 else ''}{latency_str}"
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Test Case ID:** `{tid}`")
            c2.markdown(f"**Status:** {'✅ Passed' if passed else '❌ Failed'}")
            if latency is not None:
                c3.markdown(f"**Latency:** `{latency}s`")

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
