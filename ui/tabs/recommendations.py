import streamlit as st

from ..actions import fetch_recommendations
from ..components import render_prompt_html, render_recommendation_checkboxes, render_scores
from ._shared import _render_prompt_input


def render_recommendations_tab() -> None:
    st.subheader("Get Recommendations")

    _render_prompt_input("prompt_input_mode", "prompt_fqn", "system_prompt_text", "user_prompt_template_text")

    if st.button("Fetch Prompt & Get Recommendations", width="stretch"):
        fetch_recommendations()

    st.divider()
    render_scores("### Scores From Endpoint 1", "No criteria scores available yet.")

    st.divider()
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.write("### Original Prompt")
        if st.session_state.original_prompt:
            render_prompt_html(
                st.session_state.original_prompt,
                label="Original Prompt",
                max_height=480,
            )
        else:
            st.info("Original prompt will appear here after API call.")
            st.caption(
                "Original prompt text is not present in current endpoint-1 response; "
                "you will still get recommendations and scores."
            )
    with col_right:
        render_recommendation_checkboxes()
