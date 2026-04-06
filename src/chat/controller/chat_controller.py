import time
import traceback
from typing import Any, Optional

from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel

from src.chat.models.prompt_recommendation_request import PromptRecommendationRequest
from src.chat.models.prompt_recommendation_response import PromptRecommendationResponse
from src.chat.service.chat_service import ChatService
from src.common.service.logging.logger import error, info

app = FastAPI()
SUCCESS_STATUS_CODE = "0000"
SUCCESS_STATUS_DESCRIPTION = "Success"


class FetchTracesRequest(BaseModel):
    tfy_host: str
    tfy_api_key: str
    hours: int = 24
    limit: int = 200
    fqn_filter: Optional[str] = None
    email_filter: Optional[str] = None
    data_routing_destination: str = "default"


class TraceRecord(BaseModel):
    span_id: str
    trace_id: str
    timestamp: str
    system_prompt: str
    user_message: str
    trace_output: str
    model_name: str
    application: str
    latency_ms: float
    cost_usd: float
    prompt_fqn: str
    group_key: str = ""


class FetchTracesResponse(BaseModel):
    status_code: str
    status_description: str
    traces: list[TraceRecord]
    total_spans: int
    skip_reasons: dict
    sample_span_names: list[str] = []


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


@app.post("/traces/fetch", status_code=status.HTTP_200_OK, response_model=FetchTracesResponse)
async def fetch_traces(request: FetchTracesRequest) -> FetchTracesResponse:
    """Fetch live ChatCompletion spans from a TrueFoundry tenant and return parsed trace inputs."""
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from trace.trace_parser import fetch_live_spans, parse_spans_to_inputs

    info(
        f"[TRACES] Fetching spans | host={request.tfy_host} | "
        f"hours={request.hours} | limit={request.limit} | fqn_filter={request.fqn_filter}"
    )
    t0 = time.time()
    try:
        spans = fetch_live_spans(
            hours=request.hours,
            limit=request.limit,
            prompt_fqn_filter=request.fqn_filter or None,
            tfy_host=request.tfy_host,
            tfy_api_key=request.tfy_api_key,
            email_filter=request.email_filter or None,
            data_routing_destination=request.data_routing_destination,
        )
        inputs = parse_spans_to_inputs(spans)
        skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
        sample_span_names = getattr(parse_spans_to_inputs, "sample_span_names", [])
        elapsed = round(time.time() - t0, 2)
        info(
            f"[TRACES] Fetched {len(inputs)} traces from {len(spans)} spans | elapsed={elapsed}s"
            + (f" | skip_reasons={skip_reasons}" if skip_reasons else "")
            + (f" | sample_span_names={sample_span_names}" if sample_span_names and not inputs else "")
        )
        return FetchTracesResponse(
            status_code=SUCCESS_STATUS_CODE,
            status_description=SUCCESS_STATUS_DESCRIPTION,
            traces=[TraceRecord(**vars(ti)) for ti in inputs],
            total_spans=len(spans),
            skip_reasons=skip_reasons,
            sample_span_names=sample_span_names,
        )
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        error(f"[TRACES] fetch_traces failed | elapsed={elapsed}s | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
