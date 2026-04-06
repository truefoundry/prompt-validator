"""Async eval pipeline runner: orchestrates 5 steps per golden-set prompt.

Pipeline per prompt:
  1. get_recommendation      → scores + recommendations
  2. rec_quality_judge       → are the recommendations correct PE advice?
  3. apply_recommendation    → enhanced prompt
  4. application_quality_judge → did the enhanced prompt apply the recs?
  5. llm_judge               → did the enhanced prompt produce better outputs?
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import httpx

from eval.config import REQUEST_TIMEOUT_S
from eval.pipeline.judges import evaluate_application_quality, evaluate_rec_quality
from eval.pipeline.validator import (
    validate_application_quality,
    validate_improvement,
    validate_keyword_coverage,
    validate_rec_quality,
    validate_score_bounds,
)

logger = logging.getLogger(__name__)


def load_golden_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_test_cases(test_inputs: list[dict], prompt_id: str) -> list[dict]:
    """Convert golden-set test_inputs to the API test_cases format."""
    return [
        {
            "data": {"input": item["input"]},
            "test_case_id": f"{prompt_id}_tc_{i + 1}",
            "test_case_name": f"{prompt_id} test {i + 1}",
        }
        for i, item in enumerate(test_inputs)
    ]


def _extract_enhanced_prompt(apply_response: dict) -> str | None:
    """Extract the enhanced system prompt text from the apply_recommendation response."""
    try:
        fp = apply_response.get("content", {}).get("finalPromptResult") or {}
        messages = fp.get("promptMessagesList") or fp.get("prompt_messages_list") or []
        system_messages = [m["content"] for m in messages if m.get("role") == "system"]
        return "\n\n".join(system_messages) if system_messages else None
    except (AttributeError, TypeError, KeyError):
        return None


async def _post_chat(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict,
) -> dict:
    """POST /chat and return the parsed JSON response."""
    response = await client.post(
        f"{base_url.rstrip('/')}/chat",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


async def get_recommendation(
    client: httpx.AsyncClient,
    prompt_entry: dict,
    session_id: str,
    base_url: str,
    model_name: str | None,
) -> dict:
    """Call POST /chat with type=get_recommendation."""
    payload: dict = {
        "sessionId": session_id,
        "type": "validation",
        "systemPrompt": prompt_entry["system_prompt"],
        "recommendations": None,
    }
    if model_name:
        payload["modelName"] = model_name

    t0 = time.monotonic()
    data = await _post_chat(client, base_url, payload)
    latency = round(time.monotonic() - t0, 2)

    eval_result = data.get("content", {}).get("evalResult")
    return {
        "status": "ok",
        "eval_result": eval_result,
        "error": None,
        "latency_s": latency,
    }


async def apply_recommendation(
    client: httpx.AsyncClient,
    prompt_entry: dict,
    recommendations: list[str],
    session_id: str,
    base_url: str,
    model_name: str | None,
) -> dict:
    """Call POST /chat with type=apply_recommendation."""
    payload: dict = {
        "sessionId": session_id,
        "type": "validation",
        "systemPrompt": prompt_entry["system_prompt"],
        "recommendations": recommendations,
    }
    if model_name:
        payload["modelName"] = model_name

    t0 = time.monotonic()
    data = await _post_chat(client, base_url, payload)
    latency = round(time.monotonic() - t0, 2)

    enhanced_prompt = _extract_enhanced_prompt(data)
    return {
        "status": "ok",
        "enhanced_prompt": enhanced_prompt,
        "error": None,
        "latency_s": latency,
    }


async def run_llm_judge(
    client: httpx.AsyncClient,
    prompt_entry: dict,
    enhanced_prompt: str,
    session_id: str,
    base_url: str,
    model_name: str | None,
) -> dict:
    """Call POST /chat with type=llm_judge."""
    test_cases = _build_test_cases(prompt_entry.get("test_inputs", []), prompt_entry["id"])
    payload: dict = {
        "sessionId": session_id,
        "type": "llm_judge",
        "systemPrompt": prompt_entry["system_prompt"],
        "enhancedSystemPrompt": enhanced_prompt,
        "recommendations": None,
        "testCases": test_cases,
    }
    if model_name:
        payload["modelName"] = model_name

    t0 = time.monotonic()
    data = await _post_chat(client, base_url, payload)
    latency = round(time.monotonic() - t0, 2)

    judge_result = data.get("content", {}).get("testEvaluationResult")
    return {
        "status": "ok",
        "judge_result": judge_result,
        "error": None,
        "latency_s": latency,
    }


def _build_validation(
    rec_step: dict,
    rec_quality_step: dict,
    apply_step: dict,
    app_quality_step: dict,
    judge_step: dict,
    prompt_entry: dict,
) -> dict:
    """Run all validators and return consolidated validation result."""
    eval_result = rec_step.get("eval_result") or {}
    recommendations_text = eval_result.get("recommendations") or []
    judge_result = judge_step.get("judge_result") or {}
    expected_ranges = prompt_entry.get("expected_score_ranges", {})
    expected_keywords = prompt_entry.get("expected_recommendation_keywords", [])

    score_val = validate_score_bounds(eval_result, expected_ranges)
    keyword_val = validate_keyword_coverage(recommendations_text, expected_keywords)
    improvement_val = validate_improvement(judge_result)
    rec_quality_val = validate_rec_quality(rec_quality_step)
    app_quality_val = validate_application_quality(app_quality_step)

    return {
        "score_bounds": score_val,
        "score_bounds_pass": score_val["all_pass"],
        "keyword_coverage": keyword_val["coverage"],
        "keyword_coverage_pass": keyword_val["pass"],
        "keyword_details": keyword_val,
        "rec_quality_score": rec_quality_val["score"],
        "rec_quality_pass": rec_quality_val["pass"],
        "rec_quality_details": rec_quality_val,
        "application_quality_score": app_quality_val["score"],
        "application_quality_pass": app_quality_val["pass"],
        "application_quality_details": app_quality_val,
        "improvement_delta": improvement_val["improvement_delta"],
        "regression": improvement_val["regression"],
        "improvement_details": improvement_val,
    }


async def run_single_prompt(
    client: httpx.AsyncClient,
    prompt_entry: dict,
    semaphore: asyncio.Semaphore,
    base_url: str,
    model_name: str | None,
    intermediate_path: str,
) -> dict:
    """Run the full 3-step pipeline for one golden-set prompt."""
    prompt_id = prompt_entry["id"]
    session_id = f"eval_{prompt_id}_{uuid.uuid4().hex[:8]}"

    _skipped_judge = {"status": "skipped", "overall_score": None, "verdict": None,
                      "per_rec": [], "error": None}
    result: dict = {
        "prompt_id": prompt_id,
        "domain": prompt_entry.get("domain"),
        "complexity": prompt_entry.get("complexity"),
        "intentional_weaknesses": prompt_entry.get("intentional_weaknesses", []),
        "tags": prompt_entry.get("tags", []),
        "expected_score_ranges": prompt_entry.get("expected_score_ranges", {}),
        "get_recommendation":      {"status": "skipped", "eval_result": None, "error": None, "latency_s": 0.0},
        "rec_quality_judge":       {**_skipped_judge, "missed_weaknesses": [], "issues": []},
        "apply_recommendation":    {"status": "skipped", "enhanced_prompt": None, "error": None, "latency_s": 0.0},
        "application_quality_judge": {**_skipped_judge, "regressions": [], "new_issues": []},
        "llm_judge":               {"status": "skipped", "judge_result": None, "error": None, "latency_s": 0.0},
        "validation": {},
    }

    try:
        async with semaphore:
            # Step 1 — get_recommendation
            # Step 1 — get_recommendation
            try:
                logger.info("[EVAL] %s | step=get_recommendation", prompt_id)
                result["get_recommendation"] = await get_recommendation(
                    client, prompt_entry, session_id, base_url, model_name
                )
            except Exception as exc:
                logger.error("[EVAL] %s | step=get_recommendation | error=%s", prompt_id, exc)
                result["get_recommendation"] = {
                    "status": "error", "eval_result": None, "error": str(exc), "latency_s": 0.0
                }
                _append_intermediate(intermediate_path, result)
                return result

            recommendations = (
                (result["get_recommendation"].get("eval_result") or {}).get("recommendations") or []
            )

            # Step 2 — rec_quality_judge (did the recommendations give correct PE advice?)
            if recommendations:
                try:
                    logger.info("[EVAL] %s | step=rec_quality_judge", prompt_id)
                    t0 = time.monotonic()
                    result["rec_quality_judge"] = await evaluate_rec_quality(
                        original_prompt=prompt_entry["system_prompt"],
                        recommendations=recommendations,
                        model_name=model_name,
                    )
                    result["rec_quality_judge"]["latency_s"] = round(time.monotonic() - t0, 2)
                except Exception as exc:
                    logger.error("[EVAL] %s | step=rec_quality_judge | error=%s", prompt_id, exc)
                    result["rec_quality_judge"]["status"] = "error"
                    result["rec_quality_judge"]["error"] = str(exc)

            # Step 3 — apply_recommendation
            if recommendations:
                try:
                    logger.info("[EVAL] %s | step=apply_recommendation", prompt_id)
                    result["apply_recommendation"] = await apply_recommendation(
                        client, prompt_entry, recommendations, session_id, base_url, model_name
                    )
                except Exception as exc:
                    logger.error("[EVAL] %s | step=apply_recommendation | error=%s", prompt_id, exc)
                    result["apply_recommendation"] = {
                        "status": "error", "enhanced_prompt": None, "error": str(exc), "latency_s": 0.0
                    }

            enhanced_prompt = result["apply_recommendation"].get("enhanced_prompt")

            # Step 4 — application_quality_judge (were the recs correctly applied?)
            if enhanced_prompt and recommendations:
                try:
                    logger.info("[EVAL] %s | step=application_quality_judge", prompt_id)
                    t0 = time.monotonic()
                    result["application_quality_judge"] = await evaluate_application_quality(
                        original_prompt=prompt_entry["system_prompt"],
                        recommendations=recommendations,
                        enhanced_prompt=enhanced_prompt,
                        model_name=model_name,
                    )
                    result["application_quality_judge"]["latency_s"] = round(time.monotonic() - t0, 2)
                except Exception as exc:
                    logger.error("[EVAL] %s | step=application_quality_judge | error=%s", prompt_id, exc)
                    result["application_quality_judge"]["status"] = "error"
                    result["application_quality_judge"]["error"] = str(exc)

            # Step 5 — llm_judge (did the enhanced prompt produce better outputs?)
            if enhanced_prompt and prompt_entry.get("test_inputs"):
                try:
                    logger.info("[EVAL] %s | step=llm_judge", prompt_id)
                    result["llm_judge"] = await run_llm_judge(
                        client, prompt_entry, enhanced_prompt, session_id, base_url, model_name
                    )
                except Exception as exc:
                    logger.error("[EVAL] %s | step=llm_judge | error=%s", prompt_id, exc)
                    result["llm_judge"] = {
                        "status": "error", "judge_result": None, "error": str(exc), "latency_s": 0.0
                    }

            result["validation"] = _build_validation(
                result["get_recommendation"],
                result["rec_quality_judge"],
                result["apply_recommendation"],
                result["application_quality_judge"],
                result["llm_judge"],
                prompt_entry,
            )

    except Exception as exc:
        logger.error("[EVAL] %s | unhandled error=%s", prompt_id, exc)
        result["get_recommendation"]["error"] = str(exc)

    _append_intermediate(intermediate_path, result)
    logger.info(
        "[EVAL] %s | done | score_bounds_pass=%s | keyword_pass=%s | improvement_delta=%s",
        prompt_id,
        result["validation"].get("score_bounds_pass"),
        result["validation"].get("keyword_coverage_pass"),
        result["validation"].get("improvement_delta"),
    )
    return result


def _append_intermediate(path: str, result: dict) -> None:
    """Append one result as an NDJSON line for crash recovery."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as exc:
        logger.warning("Failed to write intermediate result: %s", exc)


async def run_all(
    golden_set: list[dict],
    base_url: str,
    model_name: str | None,
    concurrency: int,
    intermediate_path: str,
) -> list[dict]:
    """Run the full pipeline for all prompts with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            run_single_prompt(client, entry, semaphore, base_url, model_name, intermediate_path)
            for entry in golden_set
        ]
        return list(await asyncio.gather(*tasks))
