import streamlit as st


def _is_paste_mode(mode_key: str) -> bool:
    return st.session_state.get(mode_key) == "Paste Prompt Text"


def _build_prompt_payload(mode_key: str, fqn_key: str, sys_key: str, user_tpl_key: str) -> dict:
    """Return the prompt-identification fields for the API payload.

    In FQN mode the dict contains ``promptFQN``; in paste mode it contains
    ``systemPrompt`` and optionally ``userPromptTemplate``.
    """
    if _is_paste_mode(mode_key):
        result: dict = {"systemPrompt": st.session_state.get(sys_key, "").strip()}
        user_tpl = st.session_state.get(user_tpl_key, "").strip()
        if user_tpl:
            result["userPromptTemplate"] = user_tpl
        return result
    return {"promptFQN": st.session_state.get(fqn_key, "").strip()}


def _validate_prompt_input(mode_key: str, fqn_key: str, sys_key: str) -> bool:
    """Validate that at least one prompt source has been provided. Returns True when valid."""
    if _is_paste_mode(mode_key):
        if not st.session_state.get(sys_key, "").strip():
            st.error("Please enter a system prompt before proceeding.")
            return False
    else:
        if not st.session_state.get(fqn_key, "").strip():
            st.error("Please enter a Prompt FQN before proceeding.")
            return False
    return True


def _build_base_payload(type_: str) -> dict:
    reasoning = st.session_state.reasoning_effort
    return {
        "sessionId": st.session_state.session_id,
        "modelName": st.session_state.model_name.strip() if st.session_state.model_name else None,
        "maxTokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "reasoningEffort": reasoning if reasoning != "none" else None,
        "type": type_,
    }
