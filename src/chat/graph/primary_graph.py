import base64
import json   
import asyncio
from typing import Optional, List
import csv
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from truefoundry.ml import ArtifactPath

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from langchain.output_parsers import PydanticOutputParser
from openai import OpenAI

from src.chat.models.state_model import State
from src.chat.models.prompt_recommendation_request import PromptRecommendationRequest
from src.chat.utils.constants import RequestType
from src.chat.utils.llm_models import get_recommendation_response_schema
from src.chat.utils.checkpointer_factory_util import CheckpointerFactory
from src.common.config.app_config import get_application_config
from src.common.util.method_stats_util import method_exec_stats
from src.common.service.llm_prompt.prompt_service import PromptService
from src.common.service.logging.logger import error, info
from src.chat.graph.exact_match_evaluator import ExactMatchEvaluator
from src.chat.graph.deepeval_evaluator import DeepEvalPromptEvaluator

CONFIG = get_application_config()
in_memory_checkpointer = MemorySaver()


async def validate_fn(state, prompt_to_validate):
    info("Entering validate_fn")
    if not state['request']['recommendations']:
        return prompt_to_validate[0]['manifest'].messages
    else:
        return {"prompt_messages_list":prompt_to_validate[0]['manifest'].messages,
                "recommendations": state['request']['recommendations']}

async def validator(state: State):
    prompt_to_validate = await PromptService.get_prompt_details([state['request']['prompt_fqn']])
    # If the prompt was not found
    if not prompt_to_validate:
        return {"messages": ["Prompt not found"]}

    response_schema = None
    if state['request']['type'] == RequestType.VALIDATION.value:
        if not state['request']['recommendations']:
        # if the recommendation list is empty
            # Get prompt ID from config
            prompt_template_id = CONFIG.get("assistants.get_recommendation.prompt_template_id")
            response_schema = get_recommendation_response_schema()
        else:
            prompt_template_id = CONFIG.get("assistants.apply_recommendation.prompt_template_id")
        input_ = await validate_fn(state, prompt_to_validate)

    elif state['request']['type'] == RequestType.VERIFY_TESTS.value:
        # Use DeepEval metrics for evaluation
        prompt_template_id = CONFIG.get("assistants.prompt_test_eval_recommendation.prompt_template_id")
        evaluator = DeepEvalPromptEvaluator(model_name=state["request"].get("model_name"))
        input_ = await evaluator.verify_tests(state)
    
    elif state['request']['type'] == RequestType.VERIFY_TESTS_EXACT.value:
        # Use exact matching for evaluation
        prompt_template_id = CONFIG.get("assistants.prompt_test_eval_exact_recommendation.prompt_template_id")
        evaluator = ExactMatchEvaluator(model_name=state["request"].get("model_name"))
        input_ = await evaluator.verify_tests(state)

    # Get prompt response
    new_message = await PromptService.get_prompt_response(
        prompt_template_id,
        {"input": str(input_)},
        model_name=state["request"].get("model_name"),
        response_schema=response_schema,
    )
    info(f"new_message: {new_message}")
    if state['request']['type'] in [RequestType.VERIFY_TESTS.value, RequestType.VERIFY_TESTS_EXACT.value]:
        response_json = {"results": input_, "recommendation_result": new_message}
        new_message = AIMessage(json.dumps(response_json))
        with open("src/chat/data/test_results/test_results.json", "w") as f:
            json.dump(response_json, f)
        prompt_slug = state["request"]["prompt_fqn"].split("/")[-1].split(":")[0]
        evaluator.tf_client.log_artifact(
            ml_repo=CONFIG.get("application_details.ml_repo"),
            name=f"{prompt_slug}_test_results",
            artifact_paths=[ArtifactPath(src="src/chat/data/test_results/test_results.json", dest="test_results.json")])
    else:
        new_message = AIMessage(new_message)
    return {"messages": [new_message]}


@method_exec_stats
async def _build_state_graph(request: PromptRecommendationRequest):
    """Build and compile the state graph with all configured agents.

    Returns:
        Optional[CompiledGraph]: The compiled state graph or None if compilation fails.
    """
    try:
        # Get the state agent.
        builder = StateGraph(State)
        builder.add_node(validator)
        builder.add_edge(START, "validator")
        builder.add_edge("validator", END)

        # Compile the graph with the checkpointer
        memory_storage_type = CONFIG.get("checkpointer.type")

        try:
            if memory_storage_type in {"in_memory", None, ""}:
                return builder.compile(
                    checkpointer=in_memory_checkpointer
                )

            async with await CheckpointerFactory.create_checkpointer() as checkpointer:
                return builder.compile(
                    checkpointer=checkpointer
                )
        except Exception as e:
            error(f"Failed to compile graph: {str(e)}")
            raise e

    except Exception as e:
        error(f"Failed to build state graph: {str(e)}")
        raise e
