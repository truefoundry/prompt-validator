import os
import sys

import streamlit as st

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ui.state import init_session_state
from ui.tabs import (
    render_deepeval_tab,
    render_enhance_tab,
    render_exact_match_tab,
    render_recommendations_tab,
    render_sidebar,
    render_trace_eval_tab,
)


def main() -> None:
    st.set_page_config(page_title="Prompt Tuner UI", layout="wide")
    init_session_state()
    st.markdown(
        "<style>.stTextArea textarea:disabled { -webkit-text-fill-color: white !important; }</style>",
        unsafe_allow_html=True,
    )

    st.title("Prompt Tuner UI")
    st.caption("Fetch recommendations and apply selected improvements to enhance prompts.")

    render_sidebar()

    tab_recommendations, tab_enhance, tab_deepeval, tab_exact, tab_trace = st.tabs([
        "Get Recommendations",
        "Enhance Prompt",
        "Verify Tests (DeepEval)",
        "Verify Tests (Exact Match)",
        "Trace Evaluation(Suggestion)",
    ])

    with tab_recommendations:
        render_recommendations_tab()

    with tab_enhance:
        render_enhance_tab()

    with tab_deepeval:
        render_deepeval_tab()

    with tab_exact:
        render_exact_match_tab()

    with tab_trace:
        render_trace_eval_tab()


if __name__ == "__main__":
    main()
