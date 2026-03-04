import json
from pydantic import BaseModel, Field, create_model, model_validator
from pydantic.alias_generators import to_camel
from typing import List, Optional


class CriteriaScores(BaseModel):
    # Score for each scoring criteria
    clarity_and_specificity: int
    structure_and_organization: int
    output_specification: int
    contextual_guidance: int
    error_handling: int

class Explanations(BaseModel):
    # Explanation of the score otained for each criteria
    clarity_and_specificity: str
    structure_and_organization: str
    output_specification: str
    contextual_guidance: str
    error_handling: str

class EvaluationResult(BaseModel):
    total_score: int = Field(..., description="Total score for the prompt.")
    criteria_scores: CriteriaScores = Field(..., description="Score for each criteria for the prompt.")
    explanations: Explanations = Field(..., description="The explanation of the score for each criteria.")
    recommendations: List[str] = Field(..., description="The list of recommendations in order to improve the prompt.")

class PromptMessage(BaseModel):
    role: str = Field(..., description="The role of the message.")
    content: str = Field(..., description="The content of the message.")

class PromptMessageList(BaseModel):
    prompt_messages_list: List[PromptMessage] = Field(..., description="The list of prompt messages.")

    @model_validator(mode='before')
    @classmethod
    def validate_input(cls, data):
        if isinstance(data, list):
            return {"prompt_messages_list": data}
        return data

class Content(BaseModel):
    """
    Response from the agent.
    """
    eval_result: Optional[EvaluationResult] = Field(..., description="The result of prompt evaluation.")
    final_prompt_result: Optional[PromptMessageList]= Field(..., description="The result of incorporating recommendations into the prompt.")
    test_evaluation_result: Optional[dict] = Field(..., description="The evaluation result for the test cases.")
    class Config:
        alias_generator = to_camel
        populate_by_name = True

class PromptRecommendationResponse(BaseModel):
    """
    Response from the agent.
    """
    prompt_fqn: Optional[str] = Field(None, description="Prompt FQN for the prompt that was validated (None when raw text was used).")
    session_id: str = Field(..., description="The session id of the chat.")
    content: Content = Field(..., description="The response from the agent.")
    status_code: str = Field(..., description="The status code of the response.")
    status_description: str = Field(
        ..., description="The description of the status code."
    )

    class Config:
        alias_generator = to_camel
        populate_by_name = True