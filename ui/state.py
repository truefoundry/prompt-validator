import streamlit as st

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME

def init_session_state() -> None:
    defaults = {
        "base_url": DEFAULT_BASE_URL,
        "session_id": "123",
        "model_name": DEFAULT_MODEL_NAME,
        "max_tokens": 15000,
        "temperature": 0.1,
        "reasoning_effort": "low",
        # Recommendations / Enhance tab
        "prompt_input_mode": "TFY Prompt FQN",
        "prompt_fqn": "",
        "system_prompt_text": "",
        "user_prompt_template_text": "",
        "original_prompt": "",
        "recommendations": [],
        "selected_recommendations": [],
        "editable_recommendations": [],
        "enhanced_prompt": "",
        "total_score": None,
        "criteria_scores": {},
        "explanations": {},
        "last_selected_signature": "",
        "api_response_debug": {},
        # DeepEval tab
        "deepeval_input_mode": "TFY Prompt FQN",
        "deepeval_prompt_fqn": "",
        "deepeval_system_prompt": "",
        "deepeval_user_prompt_template": "",
        "deepeval_is_rag": False,
        "deepeval_result": None,
        "deepeval_api_debug": {},
        "deepeval_uploaded_tests": None,
        # Exact Match tab
        "exact_match_input_mode": "TFY Prompt FQN",
        "exact_match_prompt_fqn": "",
        "exact_match_system_prompt": "",
        "exact_match_user_prompt_template": "",
        "exact_match_result": None,
        "exact_match_api_debug": {},
        "exact_match_uploaded_tests": None,
        # Enhance tab evaluation
        "enhance_eval_orig_prompt": "",
        "enhance_eval_enh_prompt": "",
        "enhance_eval_orig_area": "",
        "enhance_eval_enh_area": "",
        "enhance_eval_input_count": 1,
        "enhance_eval_uploaded_tests": None,
        "enhance_eval_judge_result": None,
        "enhance_eval_api_debug_original": {},
        "enhance_eval_judge_recs_selected": [],
        # Metric picker
        "enhance_eval_selected_metrics": ["clarity", "completeness", "accuracy", "conciseness", "professional_tone"],
        # Enhanced prompt model override
        "enhance_eval_enh_model_name": "",
        "enhance_eval_enh_temperature": 0.1,
        "enhance_eval_enh_max_tokens": 15000,
        "enhance_eval_enh_reasoning_effort": "none",
        # Trace credentials
        "trace_tfy_host": "",
        "trace_tfy_api_key": "",
        # Trace Evaluation tab
        "trace_inputs": [],
        "trace_applications": [],
        "trace_selected_indices": [],
        "trace_selected_fqn": "",
        "_trace_last_autofilled_fqn": "",
        "trace_original_system_prompt": "",
        "trace_enhanced_system_prompt": "",
        "trace_user_prompt_template": "",
        "trace_llm_judge_result": None,
        "trace_llm_judge_api_debug": {},
        "trace_pipeline_recommendations": [],
        # Behavioral recommendations (F1)
        "rec_trace_examples": None,
        "behavioral_recommendations": [],
        # Arena Evaluation (Enhance tab)
        "arena_eval_result": None,
        "arena_eval_criteria": "Choose the response that is more accurate, complete, and professional.",
        # Enhance tab — suggestions
        "enhance_eval_suggestions_result": None,
        "enhance_eval_suggestions_selected": [],
        # Trace eval — suggestions + DeepEval metrics
        "trace_suggestions_result": None,
        "trace_suggestions_selected": [],
        "trace_deepeval_metrics_result": None,
        "trace_deepeval_metrics_geval_criteria": "Assess the overall quality, relevance, and completeness of the response given the user input.",
        "trace_deepeval_metrics_prompt_instructions": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
