import time
import traceback
from typing import Any

from fastapi import FastAPI, status, HTTPException

from src.chat.models.prompt_recommendation_request import PromptRecommendationRequest
from src.chat.models.prompt_recommendation_response import PromptRecommendationResponse
from src.chat.service.chat_service import ChatService
from src.common.service.logging.logger import error, info

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
    prompt_source = f"fqn={request.prompt_fqn}" if request.prompt_fqn else f"system_prompt_len={len(request.system_prompt or '')}"
    info(
        f"[REQUEST] type={request.type} | session={request.session_id} | "
        f"model={request.model_name or 'default'} | {prompt_source} | "
        f"recs={len(request.recommendations or [])} | "
        f"test_cases={len(request.test_cases or [])} | "
        f"trace_examples={len(request.trace_examples or [])} | "
        f"judge_override={'yes' if request.judge_system_prompt_override else 'no'}"
    )
    t0 = time.time()
    try:
        response = await ChatService.get_chat_response(request)
        elapsed = round(time.time() - t0, 2)
        info(
            f"[RESPONSE] type={request.type} | session={request.session_id} | "
            f"status={response.status_code} | elapsed={elapsed}s"
        )
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        error(
            f"[ERROR] type={request.type} | session={request.session_id} | "
            f"elapsed={elapsed}s | {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to fetch agent response"
        ) from e
    return response
