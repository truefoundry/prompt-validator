from typing import Any

from fastapi import FastAPI, status, HTTPException

from src.chat.models.prompt_recommendation_request import PromptRecommendationRequest
from src.chat.models.prompt_recommendation_response import PromptRecommendationResponse
from src.chat.service.chat_service import ChatService
from src.common.service.logging.logger import error

app = FastAPI()
SUCCESS_STATUS_CODE = "0000"
SUCCESS_STATUS_DESCRIPTION = "Success"


@app.post("/chat", status_code=status.HTTP_200_OK, response_model=PromptRecommendationResponse)
async def chat(request: PromptRecommendationRequest) -> tuple[PromptRecommendationResponse, Any] | PromptRecommendationResponse:
    """
    Args:
        request (PromptRecommendationRequest): The request to the chat service.

    Returns:
        PromptRecommendationResponse: The response from the chat service.
    """
    try:
        response = await ChatService.get_chat_response(request)
    except Exception as e:
        error(f"Error fetching agent response: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch agent response"
        ) from e
    return response
