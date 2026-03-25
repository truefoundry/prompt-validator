import json
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from truefoundry.ml import ArtifactPath

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

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
_BEHAVIORAL_REC_PROMPT = (_PROMPTS_DIR / "behavioral_recommendations_system.txt").read_text()
_GENERATE_SUGGESTIONS_SYSTEM_PROMPT = (_PROMPTS_DIR / "generate_suggestions_system.txt").read_text()


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def validate_fn(state, prompt_to_validate):
    info("Entering validate_fn")
    if not state['request']['recommendations']:
        return prompt_to_validate[0]['manifest'].messages
    else:
        return {"prompt_messages_list": prompt_to_validate[0]['manifest'].messages,
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


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences from a raw LLM JSON response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


async def _fetch_prompt(state):
    """Fetch prompt from TFY or build synthetic from pasted text. Returns None on failure."""
    req = state["request"]
    prompt_fqn = req.get("prompt_fqn")
    if prompt_fqn:
        info(f"[VALIDATOR] Fetching prompt from TFY | fqn={prompt_fqn}")
        prompt_to_validate = await PromptService.get_prompt_details([prompt_fqn])
        if not prompt_to_validate:
            error(f"[VALIDATOR] Prompt not found | fqn={prompt_fqn}")
            return None
        info(f"[VALIDATOR] Prompt fetched OK | fqn={prompt_fqn}")
        return prompt_to_validate
    else:
        info(f"[VALIDATOR] Using pasted system prompt | len={len(req.get('system_prompt') or '')}")
        return _build_synthetic_prompt(state)


async def _call_prompt_service(state, prompt_template_id, input_, response_schema, evaluator=None):
    """Call PromptService, build AIMessage, and optionally log artifact for test evaluations."""
    req = state["request"]
    prompt_fqn = req.get("prompt_fqn")

    info(f"[VALIDATOR] Calling PromptService | template_id={prompt_template_id} | model={req.get('model_name')}")
    new_message = await PromptService.get_prompt_response(
        prompt_template_id,
        {"input": str(input_)},
        model_name=req.get("model_name"),
        response_schema=response_schema,
        max_tokens=req.get("max_tokens"),
        temperature=req.get("temperature"),
        reasoning_effort=req.get("reasoning_effort"),
    )
    info(f"[VALIDATOR] PromptService response_len={len(new_message) if new_message else 0}")

    if req["type"] in [RequestType.VERIFY_TESTS.value, RequestType.VERIFY_TESTS_EXACT.value]:
        response_json = {"results": input_, "recommendation_result": new_message}
        os.makedirs("src/chat/data/test_results", exist_ok=True)
        with open("src/chat/data/test_results/test_results.json", "w") as f:
            json.dump(response_json, f)
        if prompt_fqn and evaluator:
            prompt_slug = _get_prompt_slug(state)
            try:
                evaluator.tf_client.log_artifact(
                    ml_repo=CONFIG.get("application_details.ml_repo"),
                    name=f"{prompt_slug}_test_results",
                    artifact_paths=[ArtifactPath(src="src/chat/data/test_results/test_results.json", dest="test_results.json")])
            except Exception as _artifact_err:
                error(f"[VALIDATOR] Failed to log artifact (non-fatal) | {type(_artifact_err).__name__}: {_artifact_err}")
        return {"messages": [AIMessage(json.dumps(response_json))]}

    return {"messages": [AIMessage(new_message)]}


# ── Request type handlers ──────────────────────────────────────────────────────

async def _handle_validation(state, prompt_to_validate):
    req = state["request"]
    response_schema = None
    if not req["recommendations"]:
        prompt_template_id = CONFIG.get("assistants.get_recommendation.prompt_template_id")
        response_schema = get_recommendation_response_schema()
    else:
        prompt_template_id = CONFIG.get("assistants.apply_recommendation.prompt_template_id")
    input_ = await validate_fn(state, prompt_to_validate)
    return await _call_prompt_service(state, prompt_template_id, input_, response_schema)


async def _handle_verify_tests(state, prompt_to_validate):
    prompt_template_id = CONFIG.get("assistants.prompt_test_eval_recommendation.prompt_template_id")
    evaluator = DeepEvalPromptEvaluator(
        model_name=state["request"].get("model_name"),
        max_tokens=state["request"].get("max_tokens"),
        temperature=state["request"].get("temperature"),
        reasoning_effort=state["request"].get("reasoning_effort"),
    )
    input_ = await evaluator.verify_tests(state)
    return await _call_prompt_service(state, prompt_template_id, input_, response_schema=None, evaluator=evaluator)


async def _handle_verify_tests_exact(state, prompt_to_validate):
    prompt_template_id = CONFIG.get("assistants.prompt_test_eval_exact_recommendation.prompt_template_id")
    evaluator = ExactMatchEvaluator(
        model_name=state["request"].get("model_name"),
        max_tokens=state["request"].get("max_tokens"),
        temperature=state["request"].get("temperature"),
        reasoning_effort=state["request"].get("reasoning_effort"),
    )
    input_ = await evaluator.verify_tests(state)
    return await _call_prompt_service(state, prompt_template_id, input_, response_schema=None, evaluator=evaluator)


async def _handle_behavioral_recommendations(state, prompt_to_validate):
    req = state["request"]
    system_prompt = req.get("system_prompt", "")
    if not system_prompt and prompt_to_validate:
        messages = prompt_to_validate[0]['manifest'].messages
        sys_msgs = [m for m in messages if getattr(m, 'role', None) == 'system']
        if sys_msgs:
            system_prompt = sys_msgs[0].content

    trace_examples = req.get("trace_examples") or []
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
        model_name=req.get("model_name"),
        max_tokens=req.get("max_tokens"),
        temperature=req.get("temperature"),
        reasoning_effort=req.get("reasoning_effort"),
    )
    try:
        recs = json.loads(_strip_json_fences(raw))
        if not isinstance(recs, list):
            recs = []
    except (json.JSONDecodeError, IndexError):
        error(f"[BEHAVIORAL_RECS] Failed to parse LLM JSON response | raw={raw[:200]}")
        recs = []

    info(f"[BEHAVIORAL_RECS] Parsed {len(recs)} recommendations")
    return {"messages": [AIMessage(json.dumps({"behavioral_recommendations": recs}))]}


async def _handle_llm_judge(state, prompt_to_validate):
    req = state["request"]
    override = req.get("judge_system_prompt_override")
    info(f"[LLM_JUDGE] Starting | test_cases={len(req.get('test_cases') or [])} | judge_override={'yes' if override else 'no'}")
    evaluator = LLMJudgeEvaluator(
        model_name=req.get("model_name"),
        max_tokens=req.get("max_tokens"),
        temperature=req.get("temperature"),
        reasoning_effort=req.get("reasoning_effort"),
        judge_system_prompt_override=override,
    )
    result = await evaluator.verify_tests(state)
    return {"messages": [AIMessage(json.dumps(result))]}


async def _handle_generate_suggestions(state, prompt_to_validate):
    req = state["request"]
    judge_result: dict = req.get("judge_result") or {}
    original_prompt: str = req.get("system_prompt") or ""
    enhanced_prompt: str = req.get("enhanced_system_prompt") or ""

    summary = judge_result.get("summary", {})
    test_results = judge_result.get("test_results", [])
    metrics = ["clarity", "completeness", "accuracy", "conciseness", "professional_tone", "overall"]

    avg_orig = summary.get("avg_original", {})
    avg_enh = summary.get("avg_enhanced", {})
    avg_delta = summary.get("avg_delta", {})
    metrics_lines = ["Metric | Original | Enhanced | Delta", "--- | --- | --- | ---"]
    for m in metrics:
        o, e, d = avg_orig.get(m), avg_enh.get(m), avg_delta.get(m)
        metrics_lines.append(
            f"{m.replace('_', ' ').title()} | "
            f"{f'{o:.2f}' if o is not None else '—'} | "
            f"{f'{e:.2f}' if e is not None else '—'} | "
            f"{f'{d:+.2f}' if d is not None else '—'}"
        )
    metrics_table = "\n".join(metrics_lines)

    all_differences: list[str] = []
    all_summaries: list[str] = []
    for i, tr in enumerate(test_results):
        scores = tr.get("scores", {})
        for d in scores.get("key_differences", []):
            if d and d not in all_differences:
                all_differences.append(d)
        s = scores.get("improvement_summary", "")
        if s:
            all_summaries.append(f"Test {i + 1}: {s}")

    improved = summary.get("improved_count", 0)
    total = summary.get("total_cases", 0)
    orig_trunc = original_prompt[:800] + ("..." if len(original_prompt) > 800 else "")
    enh_trunc = enhanced_prompt[:800] + ("..." if len(enhanced_prompt) > 800 else "")

    user_message = (
        f"## Evaluation Summary\nImproved in {improved}/{total} test case(s).\n\n"
        f"## Metric Scores (averages across all test cases)\n{metrics_table}\n\n"
        f"## Key Differences Observed\n"
        f"{chr(10).join(f'- {d}' for d in all_differences) if all_differences else 'None noted.'}\n\n"
        f"## Per-Test Improvement Summaries\n"
        f"{chr(10).join(all_summaries) if all_summaries else 'None noted.'}\n\n"
        f"## Original Prompt (first 800 chars)\n{orig_trunc}\n\n"
        f"## Enhanced Prompt (first 800 chars)\n{enh_trunc}"
    )

    info(f"[GENERATE_SUGGESTIONS] Calling LLM | judge_tests={len(test_results)} | model={req.get('model_name') or 'default'}")
    raw = await PromptService.get_prompt_response_from_text(
        system_prompt=_GENERATE_SUGGESTIONS_SYSTEM_PROMPT,
        user_prompt_template=None,
        data={"input": user_message},
        model_name=req.get("model_name"),
        max_tokens=req.get("max_tokens"),
        temperature=0.2,
        reasoning_effort=req.get("reasoning_effort"),
    )
    try:
        result = json.loads(_strip_json_fences(raw))
    except (json.JSONDecodeError, IndexError, TypeError):
        error(f"[GENERATE_SUGGESTIONS] Failed to parse JSON | raw={str(raw)[:200]}")
        result = {"overall_analysis": "Failed to parse suggestions.", "suggestions": []}

    info(f"[GENERATE_SUGGESTIONS] Done | suggestions={len(result.get('suggestions', []))}")
    return {"messages": [AIMessage(json.dumps(result))]}


async def _handle_arena_comparison(state, prompt_to_validate):
    from src.chat.graph.arena_evaluator import run_arena_comparison
    req = state["request"]
    info(
        f"[ARENA_COMPARISON] Starting | test_cases={len(req.get('test_cases') or [])} | "
        f"criteria_len={len(req.get('arena_criteria') or '')} | model={req.get('model_name') or 'default'}"
    )
    result = await asyncio.to_thread(
        run_arena_comparison,
        original_prompt=req.get("system_prompt") or "",
        enhanced_prompt=req.get("enhanced_system_prompt") or "",
        test_cases=req.get("test_cases") or [],
        criteria=req.get("arena_criteria") or "",
        model_name=req.get("model_name"),
        max_tokens=req.get("max_tokens"),
        temperature=req.get("temperature"),
    )
    info(f"[ARENA_COMPARISON] Done | wins={result.get('wins')}")
    return {"messages": [AIMessage(json.dumps(result))]}


async def _handle_deepeval_prompt_metrics(state, prompt_to_validate):
    from src.chat.graph.deepeval_prompt_metrics import run_deepeval_prompt_metrics
    from src.chat.utils.llm_models import TrueFoundryLLM, get_truefoundry_llm
    req = state["request"]
    llm = TrueFoundryLLM(model=get_truefoundry_llm(
        model_name=req.get("model_name"),
        max_tokens=req.get("max_tokens"),
        temperature=0.0,
        reasoning_effort=req.get("reasoning_effort"),
    ))
    info(
        f"[DEEPEVAL_METRICS] Starting | test_cases={len(req.get('test_cases') or [])} | "
        f"geval={bool(req.get('geval_criteria'))} | model={req.get('model_name') or 'default'}"
    )
    result = await asyncio.to_thread(
        run_deepeval_prompt_metrics,
        test_cases=req.get("test_cases") or [],
        llm=llm,
        geval_criteria=req.get("geval_criteria") or "",
        prompt_instructions=req.get("prompt_instructions") or [],
    )
    info(f"[DEEPEVAL_METRICS] Done | metrics_run={result.get('metrics_run')}")
    return {"messages": [AIMessage(json.dumps(result))]}


# ── Dispatch table ─────────────────────────────────────────────────────────────

_HANDLERS = {
    RequestType.VALIDATION.value:                     _handle_validation,
    RequestType.VERIFY_TESTS.value:                   _handle_verify_tests,
    RequestType.VERIFY_TESTS_EXACT.value:             _handle_verify_tests_exact,
    RequestType.GET_BEHAVIORAL_RECOMMENDATIONS.value: _handle_behavioral_recommendations,
    RequestType.LLM_JUDGE.value:                      _handle_llm_judge,
    RequestType.GENERATE_SUGGESTIONS.value:           _handle_generate_suggestions,
    RequestType.ARENA_COMPARISON.value:               _handle_arena_comparison,
    RequestType.DEEPEVAL_PROMPT_METRICS.value:        _handle_deepeval_prompt_metrics,
}


# ── Graph node ─────────────────────────────────────────────────────────────────

async def validator(state: State):
    req = state["request"]
    info(
        f"[VALIDATOR] type={req.get('type')} | model={req.get('model_name') or 'default'} | "
        f"reasoning_effort={req.get('reasoning_effort')} | "
        f"prompt_fqn={req.get('prompt_fqn') or 'pasted'} | "
        f"recs={len(req.get('recommendations') or [])} | "
        f"test_cases={len(req.get('test_cases') or [])} | "
        f"trace_examples={len(req.get('trace_examples') or [])}"
    )

    prompt_to_validate = await _fetch_prompt(state)
    if prompt_to_validate is None:
        return {"messages": ["Prompt not found"]}

    handler = _HANDLERS.get(req.get("type"))
    if not handler:
        error(f"[VALIDATOR] Unknown request type: {req.get('type')}")
        return {"messages": ["Unknown request type"]}

    return await handler(state, prompt_to_validate)


# ── Graph builder ──────────────────────────────────────────────────────────────

@method_exec_stats
async def _build_state_graph(request: PromptRecommendationRequest):
    """Build and compile the state graph with all configured agents."""
    try:
        builder = StateGraph(State)
        builder.add_node(validator)
        builder.add_edge(START, "validator")
        builder.add_edge("validator", END)

        memory_storage_type = CONFIG.get("checkpointer.type")
        try:
            if memory_storage_type in {"in_memory", None, ""}:
                return builder.compile(checkpointer=in_memory_checkpointer)
            async with await CheckpointerFactory.create_checkpointer() as checkpointer:
                return builder.compile(checkpointer=checkpointer)
        except Exception as e:
            error(f"Failed to compile graph: {str(e)}")
            raise e

    except Exception as e:
        error(f"Failed to build state graph: {str(e)}")
        raise e
