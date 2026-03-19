import base64
import json   
import asyncio
from types import SimpleNamespace
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
from src.chat.graph.llm_judge_evaluator import LLMJudgeEvaluator

CONFIG = get_application_config()
in_memory_checkpointer = MemorySaver()

_BEHAVIORAL_REC_PROMPT = """You are a prompt engineer analyzing behavioral gaps in an AI system prompt.
You are given:
1. The current system prompt
2. Real (input, output) pairs from production

Your task: Identify specific instructions that are MISSING or INCORRECT in the system prompt
that would have caused the model to produce better responses for the given examples.

Rules:
- Each recommendation must be a concrete directive (e.g. "Always...", "When X occurs, do Y")
- Ground each in a specific example — do not write generic advice
- Focus only on behaviors the prompt could fix — not model capability limits
- Return 3-6 recommendations as a JSON array of strings

Quality standards for your recommendations:
- Write high-quality, general-purpose instructions that work correctly for all valid inputs, not just the observed examples
- Do not suggest hard-coding values or workarounds that only address specific test inputs
- Recommend instructions that implement actual logic solving the problem generally
- Focus on principled, robust directives that follow best practices and are maintainable and extendable
- If a trace output is fundamentally unreasonable or the task is infeasible for a prompt to fix, flag it rather than suggesting a workaround
- Do not recommend helper scripts or prompt hacks — only real instructional improvements"""


async def validate_fn(state, prompt_to_validate):
    info("Entering validate_fn")
    if not state['request']['recommendations']:
        return prompt_to_validate[0]['manifest'].messages
    else:
        return {"prompt_messages_list":prompt_to_validate[0]['manifest'].messages,
                "recommendations": state['request']['recommendations']}

def _build_synthetic_prompt(state):
    """Build a synthetic prompt structure from raw text that matches the TFY format."""
    msgs = [SimpleNamespace(role="system", content=state["request"]["system_prompt"])]
    user_tpl = state["request"].get("user_prompt_template")
    if user_tpl:
        msgs.append(SimpleNamespace(role="user", content=user_tpl))
    return [{"manifest": SimpleNamespace(messages=msgs)}]


def _get_prompt_slug(state):
    """Derive a prompt slug from FQN or fall back to a default for pasted prompts."""
    fqn = state["request"].get("prompt_fqn")
    if fqn:
        return fqn.split("/")[-1].split(":")[0]
    return "pasted_prompt"


async def validator(state: State):
    prompt_fqn = state["request"].get("prompt_fqn")

    if prompt_fqn:
        prompt_to_validate = await PromptService.get_prompt_details([prompt_fqn])
        if not prompt_to_validate:
            return {"messages": ["Prompt not found"]}
    else:
        prompt_to_validate = _build_synthetic_prompt(state)

    response_schema = None
    if state['request']['type'] == RequestType.VALIDATION.value:
        if not state['request']['recommendations']:
            prompt_template_id = CONFIG.get("assistants.get_recommendation.prompt_template_id")
            response_schema = get_recommendation_response_schema()
        else:
            prompt_template_id = CONFIG.get("assistants.apply_recommendation.prompt_template_id")
        input_ = await validate_fn(state, prompt_to_validate)

    elif state['request']['type'] == RequestType.VERIFY_TESTS.value:
        prompt_template_id = CONFIG.get("assistants.prompt_test_eval_recommendation.prompt_template_id")
        evaluator = DeepEvalPromptEvaluator(
            model_name=state["request"].get("model_name"),
            max_tokens=state["request"].get("max_tokens"),
            temperature=state["request"].get("temperature"),
            reasoning_effort=state["request"].get("reasoning_effort"),
        )
        input_ = await evaluator.verify_tests(state)
    
    elif state['request']['type'] == RequestType.VERIFY_TESTS_EXACT.value:
        prompt_template_id = CONFIG.get("assistants.prompt_test_eval_exact_recommendation.prompt_template_id")
        evaluator = ExactMatchEvaluator(
            model_name=state["request"].get("model_name"),
            max_tokens=state["request"].get("max_tokens"),
            temperature=state["request"].get("temperature"),
            reasoning_effort=state["request"].get("reasoning_effort"),
        )
        input_ = await evaluator.verify_tests(state)

    elif state['request']['type'] == RequestType.GET_BEHAVIORAL_RECOMMENDATIONS.value:
        system_prompt = state["request"].get("system_prompt", "")
        if not system_prompt and prompt_to_validate:
            # Extract system prompt text from loaded TFY prompt structure
            messages = prompt_to_validate[0]['manifest'].messages
            sys_msgs = [m for m in messages if getattr(m, 'role', None) == 'system']
            if sys_msgs:
                system_prompt = sys_msgs[0].content
        trace_examples = state["request"].get("trace_examples") or []
        info(f"[BEHAVIORAL_RECS] system_prompt_len={len(system_prompt)} | trace_examples={len(trace_examples)}")
        lines = [f"System Prompt:\n{system_prompt}\n\nTrace Examples:"]
        for idx, ex in enumerate(trace_examples, 1):
            lines.append(f"\nExample {idx}:\nInput: {ex.get('input', '')}\nOutput: {ex.get('output', '')}")
        formatted_content = "\n".join(lines)
        info(f"[BEHAVIORAL_RECS] Calling LLM | content_len={len(formatted_content)}")
        raw = await PromptService.get_prompt_response_from_text(
            system_prompt=_BEHAVIORAL_REC_PROMPT,
            user_prompt_template=None,
            data={"input": formatted_content},
            model_name=state["request"].get("model_name"),
            max_tokens=state["request"].get("max_tokens"),
            temperature=state["request"].get("temperature"),
            reasoning_effort=state["request"].get("reasoning_effort"),
        )
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            recs = json.loads(cleaned.strip())
            if not isinstance(recs, list):
                recs = []
        except (json.JSONDecodeError, IndexError):
            error(f"[BEHAVIORAL_RECS] Failed to parse LLM JSON response | raw={raw[:200]}")
            recs = []
        info(f"[BEHAVIORAL_RECS] Parsed {len(recs)} recommendations")
        new_message = AIMessage(json.dumps({"behavioral_recommendations": recs}))
        return {"messages": [new_message]}

    elif state['request']['type'] == RequestType.LLM_JUDGE.value:
        override = state["request"].get("judge_system_prompt_override")
        info(f"[LLM_JUDGE] Starting | test_cases={len(state['request'].get('test_cases') or [])} | judge_override={'yes' if override else 'no'}")
        evaluator = LLMJudgeEvaluator(
            model_name=state["request"].get("model_name"),
            max_tokens=state["request"].get("max_tokens"),
            temperature=state["request"].get("temperature"),
            reasoning_effort=state["request"].get("reasoning_effort"),
            judge_system_prompt_override=override,
        )
        input_ = await evaluator.verify_tests(state)
        new_message = AIMessage(json.dumps(input_))
        return {"messages": [new_message]}

    new_message = await PromptService.get_prompt_response(
        prompt_template_id,
        {"input": str(input_)},
        model_name=state["request"].get("model_name"),
        response_schema=response_schema,
        max_tokens=state["request"].get("max_tokens"),
        temperature=state["request"].get("temperature"),
        reasoning_effort=state["request"].get("reasoning_effort"),
    )
    info(f"new_message: {new_message}")
    if state['request']['type'] in [RequestType.VERIFY_TESTS.value, RequestType.VERIFY_TESTS_EXACT.value]:
        response_json = {"results": input_, "recommendation_result": new_message}
        new_message = AIMessage(json.dumps(response_json))
        os.makedirs("src/chat/data/test_results", exist_ok=True)
        with open("src/chat/data/test_results/test_results.json", "w") as f:
            json.dump(response_json, f)
        prompt_slug = _get_prompt_slug(state)
        if prompt_fqn:
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
