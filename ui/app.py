import base64
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

def _load_css() -> str:
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path) as f:
        return f"<style>{f.read()}</style>"


# Legacy alias kept so the rest of the file can stay unchanged
_GLOBAL_CSS = _load_css()


def _load_logo_b64() -> str:
    try:
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifact", "tfy-logo.jpeg",
        )
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def main() -> None:
    st.set_page_config(
        page_title="Prompt Tuner",
        page_icon="artifact/tfy-logo.jpeg",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # ── Hero header ──────────────────────────────────────────────────────────
    logo_b64 = _load_logo_b64()
    logo_html = (
        f'<img src="data:image/jpeg;base64,{logo_b64}" '
        'style="width:52px;height:52px;border-radius:12px;object-fit:cover;flex-shrink:0;">'
        if logo_b64
        else '<div style="width:52px;height:52px;border-radius:12px;background:#6366f1;'
             'display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🔧</div>'
    )
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#312e81 100%);
     border-radius:16px;padding:22px 28px;margin:8px 0 20px 0;
     display:flex;align-items:center;gap:18px;
     box-shadow:0 6px 32px rgba(15,23,42,0.25);">
  {logo_html}
  <div style="flex:1;min-width:0;">
    <h1 style="color:white;margin:0 0 4px 0;font-size:1.65rem;font-weight:700;
               letter-spacing:-0.02em;line-height:1.2;">Prompt Tuner</h1>
    <p style="color:#94a3b8;margin:0;font-size:0.875rem;line-height:1.5;">
      Evaluate, enhance, and test your LLM prompts with AI-powered recommendations.
    </p>
  </div>
  <div style="display:flex;gap:8px;flex-shrink:0;">
    <span style="background:rgba(99,102,241,0.2);color:#a5b4fc;padding:5px 12px;
                 border-radius:20px;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;
                 border:1px solid rgba(99,102,241,0.35);text-transform:uppercase;">
      AI Tools
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    render_sidebar()

    tab_recommendations, tab_enhance, tab_deepeval, tab_exact, tab_trace = st.tabs([
        "🔍 Recommendations",
        "✨ Enhance",
        "🧪 DeepEval Tests",
        "🎯 Exact Match",
        "📊 Trace Eval",
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

    # ── Debug: Session State ─────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        if st.toggle("🔍 Debug Session State", value=False, key="debug_session_toggle"):
            st.caption("All session state keys and values:")
            debug_data = {}
            for k, v in sorted(st.session_state.items()):
                if k == "debug_session_toggle":
                    continue
                if isinstance(v, (list, dict)) and len(str(v)) > 200:
                    debug_data[k] = f"{type(v).__name__}[{len(v)}] = {str(v)[:200]}…"
                elif isinstance(v, str) and len(v) > 200:
                    debug_data[k] = v[:200] + "…"
                else:
                    debug_data[k] = v
            st.json(debug_data)


if __name__ == "__main__":
    main()
