from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

# Request types that do not require a prompt FQN or system prompt.
_PROMPT_NOT_REQUIRED_TYPES = {"deepeval_prompt_metrics"}


class PromptRecommendationRequest(BaseModel):
    """
    Payload for the Prompt Validation service.
    Accepts either a TFY Prompt FQN **or** raw prompt text (system + optional user template).
    """

    session_id: str = Field(
        ..., alias="sessionId", min_length=1, description="Session ID for the prompt validation conversation."
    )
    prompt_fqn: Optional[str] = Field(
        default=None, alias="promptFQN", description="Prompt FQN for the prompt that needs to be validated."
    )
    system_prompt: Optional[str] = Field(
        default=None,
        alias="systemPrompt",
        description="Raw system prompt text (alternative to promptFQN).",
    )
    user_prompt_template: Optional[str] = Field(
        default=None,
        alias="userPromptTemplate",
        description="Raw user prompt template text. May contain {{input}} for variable injection.",
    )
    enhanced_system_prompt: Optional[str] = Field(
        default=None,
        alias="enhancedSystemPrompt",
        description="Enhanced system prompt for LLM-as-judge comparison.",
    )
    is_rag: Optional[bool] = Field(False, alias="isRAG", description="Whether the prompt is a RAG prompt.")
    model_name: Optional[str] = Field(
        None,
        alias="modelName",
        description="Optional model name to use for this request.",
    )
    max_tokens: Optional[int] = Field(
        None,
        alias="maxTokens",
        description="Maximum number of tokens in the model response.",
    )
    temperature: Optional[float] = Field(
        None,
        description="Controls randomness. Lower values are more deterministic.",
    )
    reasoning_effort: Optional[str] = Field(
        None,
        alias="reasoningEffort",
        description="Reasoning effort level: 'high', 'medium', 'low', or 'no' (Gemini only).",
    )
    type: str = Field(..., description="The type of input request")
    recommendations: Optional[List[str]] = Field(..., description="The list of recommendations for each section in order to improve the prompt.")
    test_cases: Optional[List[dict]] = Field(
        default=None,
        alias="testCases",
        description="Optional test cases to use instead of fetching from TrueFoundry.",
    )
    trace_examples: Optional[List[dict]] = Field(
        default=None,
        alias="traceExamples",
        description="Real (input, output) pairs from production used for behavioral recommendation analysis.",
    )
    judge_system_prompt_override: Optional[str] = Field(
        default=None,
        alias="judgeSystemPromptOverride",
        description="Optional override for the LLM judge system prompt.",
    )
    # ── Fields for generate_suggestions ──────────────────────────────────────
    judge_result: Optional[dict] = Field(
        default=None,
        alias="judgeResult",
        description="Full LLM-judge result dict used by the generate_suggestions handler.",
    )
    # ── Fields for llm_judge ─────────────────────────────────────────────────
    judge_metrics: Optional[List[str]] = Field(
        default=None,
        alias="judgeMetrics",
        description="List of metric keys to evaluate. Defaults to the 5 general metrics.",
    )
    enhanced_model_name: Optional[str] = Field(
        default=None,
        alias="enhancedModelName",
        description="Model to use for the enhanced prompt responses (falls back to modelName if not set).",
    )
    enhanced_temperature: Optional[float] = Field(
        default=None,
        alias="enhancedTemperature",
        description="Temperature for enhanced prompt model (falls back to temperature if not set).",
    )
    enhanced_max_tokens: Optional[int] = Field(
        default=None,
        alias="enhancedMaxTokens",
        description="Max tokens for enhanced prompt model (falls back to maxTokens if not set).",
    )
    enhanced_reasoning_effort: Optional[str] = Field(
        default=None,
        alias="enhancedReasoningEffort",
        description="Reasoning effort for enhanced prompt model (falls back to reasoningEffort if not set).",
    )
    # ── Fields for arena_comparison ───────────────────────────────────────────
    arena_criteria: Optional[str] = Field(
        default=None,
        alias="arenaCriteria",
        description="Natural-language criteria describing what makes one response better than another.",
    )
    # ── Fields for deepeval_prompt_metrics ───────────────────────────────────
    geval_criteria: Optional[str] = Field(
        default=None,
        alias="gevalCriteria",
        description="Custom quality criteria for GEval. Leave blank to skip GEval.",
    )
    prompt_instructions: Optional[List[str]] = Field(
        default=None,
        alias="promptInstructions",
        description="Instruction strings for PromptAlignmentMetric. Leave empty to skip.",
    )

    @model_validator(mode="after")
    def require_fqn_or_raw_text(self):
        if self.type not in _PROMPT_NOT_REQUIRED_TYPES and not self.prompt_fqn and not self.system_prompt:
            raise ValueError("Either promptFQN or systemPrompt must be provided.")
        return self
