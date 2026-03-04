from pydantic import BaseModel, Field, create_model, model_validator
from typing import List, Optional


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

    @model_validator(mode="after")
    def require_fqn_or_raw_text(self):
        if not self.prompt_fqn and not self.system_prompt:
            raise ValueError("Either promptFQN or systemPrompt must be provided.")
        return self
