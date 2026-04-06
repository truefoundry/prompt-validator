import streamlit as st

from ..actions import fetch_recommendations
from ..components import (
    render_prompt_html,
    render_recommendation_checkboxes,
    render_scores,
    render_section_header,
)
from ._shared import _render_prompt_input


def render_recommendations_tab() -> None:
    # ── Tab header ───────────────────────────────────────────────────────────
    st.markdown(
        """<div style="margin-bottom:20px;">
  <h2 style="margin:0 0 4px 0;font-size:1.3rem;font-weight:700;color:#1e293b;">
    Get Recommendations
  </h2>
  <p style="margin:0;font-size:0.875rem;color:#64748b;line-height:1.5;">
    Fetch your prompt from TrueFoundry or paste it directly, then let the AI evaluate it
    and generate targeted improvement recommendations.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Step 1: Input ────────────────────────────────────────────────────────
    render_section_header("📝", "Prompt Input", "Choose a prompt FQN or paste text directly", step=1)
    _render_prompt_input("prompt_input_mode", "prompt_fqn", "system_prompt_text", "user_prompt_template_text")

    st.button(
        "🚀 Fetch Prompt & Get Recommendations",
        width="stretch",
        type="primary",
        key="fetch_recs_btn",
        on_click=fetch_recommendations,
    )

    st.divider()

    # ── Step 2: Scores ───────────────────────────────────────────────────────
    render_scores("Evaluation Scores", "No criteria scores available yet. Fetch recommendations first.")

    st.divider()

    # ── Step 3: Prompt + Recommendations ────────────────────────────────────
    col_left, col_right = st.columns([2, 3])
    with col_left:
        render_section_header("📄", "Original Prompt", "Read-only syntax-highlighted view")
        if st.session_state.original_prompt:
            render_prompt_html(
                st.session_state.original_prompt,
                label="Original Prompt",
                max_height=500,
            )
        else:
            st.markdown(
                """<div style="background:#f8fafc;border:1.5px dashed #cbd5e1;border-radius:10px;
padding:32px 24px;text-align:center;color:#94a3b8;">
  <div style="font-size:2rem;margin-bottom:8px;">📄</div>
  <p style="margin:0;font-size:0.875rem;font-weight:500;">Original prompt will appear here</p>
  <p style="margin:6px 0 0 0;font-size:0.78rem;">after you fetch recommendations above</p>
</div>""",
                unsafe_allow_html=True,
            )
    with col_right:
        render_recommendation_checkboxes()
