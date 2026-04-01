from pydantic import BaseModel, Field, create_model
from typing import List, Optional


class PromptRecommendationRequest(BaseModel):
    """
    Payload for the Prompt Validation service.
    """

    session_id: str = Field(
        ..., alias="sessionId", min_length=1, description="Session ID for the prompt validation conversation."
    )
    prompt_fqn: str = Field(
        ..., alias="promptFQN", description="Prompt ID for the prompt that needs to be validated."
    )
    is_rag: Optional[bool] = Field(False, alias="isRAG", description="Whether the prompt is a RAG prompt.")
    type: str = Field(..., description="The type of input request")
    recommendations: Optional[List[str]] = Field(..., description="The list of recommendations for each section in order to improve the prompt.")
