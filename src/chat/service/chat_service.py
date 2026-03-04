import asyncio
import traceback
import json
import re
from typing import List, Type
from langchain_core.messages import AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.types import StateSnapshot
from pydantic import BaseModel

from src.chat.models.prompt_recommendation_request import PromptRecommendationRequest
from src.chat.models.prompt_recommendation_response import PromptRecommendationResponse, Content, EvaluationResult, PromptMessageList
from src.chat.utils.constants import RequestType
from src.chat.utils.chat_utils import print_event
# from src.chat.utils.langfuse_handler_util import LangfuseHandler
from src.chat.graph.primary_graph import _build_state_graph
from src.common.config.app_config import get_application_config
from src.common.service.logging.logger import error, info

CONFIG = get_application_config()


def _sanitize_json_like_output(raw_text: str) -> str:
    """Sanitize common LLM JSON formatting issues before parsing."""
    sanitized = raw_text.strip()

    # Remove fenced markdown blocks like ```json ... ```
    sanitized = re.sub(r"^\s*```(?:json)?\s*", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*```\s*$", "", sanitized)

    # Remove trailing commas before closing objects/arrays.
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    return sanitized.strip()


def _parse_with_sanitizer(raw_text: str, model_cls: Type[BaseModel]) -> BaseModel | None:
    """Parse model output with strict parser first, then sanitized JSON fallback."""
    parser = PydanticOutputParser(pydantic_object=model_cls)
    try:
        return parser.parse(raw_text)
    except Exception as primary_error:
        try:
            sanitized = _sanitize_json_like_output(raw_text)
            parsed_json = json.loads(sanitized)
            return model_cls.model_validate(parsed_json)
        except Exception as fallback_error:
            error(f"Error parsing output: {primary_error}")
            error(f"Fallback parsing failed after sanitization: {fallback_error}")
            return None


async def _get_last_ai_message(events):
    """
    Retrieve the last AI message from a list of events, ensuring the message
    does not include any tool calls.

    Args:
        events (list): A list of event dictionaries, where each event can contain messages.

    Returns:
        str or None: The content of the last AI message without tool calls, concatenated if content is a list;
                     otherwise, returns None if no such message exists.
    """
    # Iterate over events in reverse to find the last message
    for event in reversed(events):
        if event.get("messages"):
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                try:
                    content = last_message.content
                    # Flatten list and send as string
                    if isinstance(content, list):
                        content = " ".join(
                            [str(item) for sublist in content for item in sublist]
                        )
                    # Flatten dict and send as string
                    elif isinstance(content, dict):
                        content = " ".join(
                            [str(v) for k, v in content.items() if isinstance(v, str)]
                        )
                    elif not isinstance(content, str):
                        content = str(content)
                    return content.strip()
                except Exception as e:
                    error(f"Error while processing last AI message: {e}")
    return None


class ChatService:
    @staticmethod
    async def get_chat_response(request: PromptRecommendationRequest) -> PromptRecommendationResponse | None:
        """
        Get chat response
        Args:
            request (PromptRecommendationRequest): The request to the chat service.
        Returns:
            PromptRecommendationResponse: The response from the chat service.
        """
        try:
            # Build the state graph
            graph = await _build_state_graph(request)

            # Graph Configuration
            configuration = _get_graph_configuration(request)

            # print the graph
            # signal(graph.get_graph().draw_mermaid())

            """
            Conversation can be classified into types: Interrupted and New
            Interrupted conversation:
            An interrupted conversation is when the user is asked a question and the user responds with a yes or no.
            Example: User is asked if they would like to refill a metformin prescription and they respond with a yes or no.
            Request will always comprise of a context. This context will be used to determine the next action.

            New conversation:
            A new conversation is when the user initiates a conversation with the chatbot.
            Example: User asks the chatbot to refill a prescription.
            Request will not have a context.
            """
            events = await _resume_a_new_conversation(request, configuration, graph)

            """
            Streaming the graph with the user input and configuration generates a list of conversational events.
            Events are a list of dictionaries containing messages, tool calls, and other information.
            AI message in the last event is the response to the user.
            Example:
            events = [
            {
                "messages": [
                HumanMessage(content="refill my prescriptions.", id='...'),
                AIMessage(content="Let me find prescriptions for refills",  tools=["ToRefill"] id='...')
                ToolMessage(content="Searching for prescriptions to refill", tool_call_id='...'),
                AIMessage(content="You can refill metformin 40 mg", id='...'),
                ],
            }
            """
            chat_response = await _process_events_and_build_response(
                request, events, graph, configuration
            )

            return chat_response

        except Exception as e:
            error(f"Error: {e}")
            raise e


def _get_graph_configuration(request: PromptRecommendationRequest) -> dict:
    """
    Get the graph configuration for the given request.

    Args:
        request (PromptRecommendationRequest): The request to the chat service.

    Returns:
        dict: The graph configuration.
    """

    return {
        # "callbacks": [LangfuseHandler.get_handler()]
        # if CONFIG.get("LANGFUSE_ENABLED")
        # else [],
        "configurable": {
            "thread_id": request.session_id,
            "prompt_fqn": request.prompt_fqn or "pasted_prompt",
        },
        "recursion_limit": 15,
    }


async def _resume_a_new_conversation(request, configuration, graph):
    """
    Resume a new conversation with the chatbot.
    Args:
        request (PromptRecommendationRequest): The request to the chat service.
        configuration (dict): The graph configuration.
        graph (CompiledGraph): The compiled state graph.
    Returns:
        list: The list of events from the graph.
    """
    events = []
    prompt_label = request.prompt_fqn or "pasted_prompt"
    async for event in graph.astream(
        {"messages": ("user", prompt_label), "request": request.model_dump()},
        config=configuration,
        stream_mode="values",
    ):
        events.append(event)
    return events


async def _process_events_and_build_response(request, events, graph, configuration):
    """Process events and build the chat response."""
    _printed = set()
    last_message = None

    try:
        for event in events:
            print_event(event, _printed)
            last_message = await _get_last_ai_message(events)

    except Exception as e:
        traceback.print_exc()
        error(f"process events and build response failed: ", e)
        last_message = (
            f"I am sorry, I dont have sufficient information to fulfill your request."
        )

    """
    After processing the events, capture the state snapshot to determine the next action.
    If the next action requires human intervention, generate a human-in-loop message.

    Example: Before submitting the refill request,
    the chatbot will ask the user if they would like to refill the prescription by confirming yes or no.    
    """
    if not last_message:
        return PromptRecommendationResponse(
            session_id=request.session_id,
            status_code="5013",
            status_description="The provided parameters are invalid. Please refer to the developer portal for correct specifications.",
            content=Content(
                eval_result=None,
                final_prompt_result=None,
                test_evaluation_result=None
            ),
            prompt_fqn=request.prompt_fqn
        )
    eval_result, prompt_result, test_evaluation_result = None, None, None
    # Return the last message from the graph, usually for Uninterrupted flows
    if request.type == RequestType.VALIDATION.value:
        if not request.recommendations:
            eval_result = _parse_with_sanitizer(last_message, EvaluationResult)
        else:
            prompt_result = _parse_with_sanitizer(last_message, PromptMessageList)
    elif request.type in [RequestType.VERIFY_TESTS.value, RequestType.VERIFY_TESTS_EXACT.value]:
        try:
            test_evaluation_result = json.loads(last_message)
        except Exception:
            try:
                test_evaluation_result = json.loads(_sanitize_json_like_output(last_message))
            except Exception as e:
                error(f"Error parsing test evaluation result: {e}")
                test_evaluation_result = None

    return PromptRecommendationResponse(
        session_id=request.session_id,
        status_code="0000",
        status_description="Success.",
        content=Content(
            eval_result=eval_result,
            final_prompt_result=prompt_result,
            test_evaluation_result=test_evaluation_result
        ),
        prompt_fqn=request.prompt_fqn
    )
