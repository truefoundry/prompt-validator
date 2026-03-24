"""Shared helpers used across multiple tab modules."""
import streamlit as st

_INPUT_MODES = ("TFY Prompt FQN", "Paste Prompt Text")


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
