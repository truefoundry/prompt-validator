from enum import Enum


class RequestType(Enum):
    VALIDATION = "validation"
    VERIFY_TESTS = "verify_tests"
    VERIFY_TESTS_EXACT = "verify_tests_exact"
    LLM_JUDGE = "llm_judge"
    GET_BEHAVIORAL_RECOMMENDATIONS = "get_behavioral_recommendations"
    GENERATE_SUGGESTIONS = "generate_suggestions"
    ARENA_COMPARISON = "arena_comparison"
    DEEPEVAL_PROMPT_METRICS = "deepeval_prompt_metrics"
