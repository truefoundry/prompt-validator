"""Pipeline logic: API calls and result extraction for the trace eval CLI."""

from __future__ import annotations

import requests


# ── HTTP helpers ────────────────────────────────────────────────────────────

def post_chat(backend_url: str, payload: dict, include_grid_header: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if include_grid_header:
        headers["x-grid"] = "8091"
    resp = requests.post(f"{backend_url}/chat", json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected response type: {type(data)}")
    return data


# ── Extractors ───────────────────────────────────────────────────────────────

def extract_recommendations(data: dict) -> list[str]:
    content = data.get("content", {})
    if isinstance(content, dict):
        eval_result = content.get("eval_result") or content.get("evalResult") or {}
        if isinstance(eval_result, dict):
            recs = eval_result.get("recommendations", [])
            if isinstance(recs, list):
                return [str(r).strip() for r in recs if str(r).strip()]
    return []


def extract_total_score(data: dict) -> int | None:
    content = data.get("content", {})
    if isinstance(content, dict):
        eval_result = content.get("eval_result") or content.get("evalResult") or {}
        if isinstance(eval_result, dict):
            score = eval_result.get("total_score")
            if isinstance(score, int):
                return score
    return None


def extract_criteria_scores(data: dict) -> dict:
    content = data.get("content", {})
    if isinstance(content, dict):
        eval_result = content.get("eval_result") or content.get("evalResult") or {}
        if isinstance(eval_result, dict):
            scores = eval_result.get("criteria_scores") or eval_result.get("criteriaScores") or {}
            if isinstance(scores, dict):
                return scores
    return {}


def extract_enhanced_prompt(data: dict) -> str:
    content = data.get("content", {})
    if isinstance(content, dict):
        final = content.get("final_prompt_result") or content.get("finalPromptResult") or {}
        if isinstance(final, dict):
            msgs = final.get("prompt_messages_list") or final.get("promptMessagesList") or []
            if isinstance(msgs, list):
                parts = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "")).strip()
                    text = str(m.get("content", "")).strip()
                    if text:
                        parts.append(f"{role}: {text}" if role else text)
                return "\n\n".join(parts)
    return ""


def extract_judge_result(data: dict) -> dict | None:
    content = data.get("content", {})
    if isinstance(content, dict):
        result = content.get("test_evaluation_result") or content.get("testEvaluationResult")
        return result if isinstance(result, dict) else None
    return None


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_get_recommendations(
    backend_url: str,
    system_prompt: str,
    model_name: str | None,
    session_id: str,
    max_tokens: int,
    temperature: float,
) -> tuple[list[str], int | None, dict]:
    """Step 1: Analyze prompt and fetch recommendations.

    Returns:
        (recommendations, total_score, raw_response)
    """
    payload = {
        "sessionId": session_id,
        "type": "validation",
        "systemPrompt": system_prompt,
        "modelName": model_name,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "reasoningEffort": None,
        "recommendations": None,
    }
    data = post_chat(backend_url, payload, include_grid_header=True)
    return extract_recommendations(data), extract_total_score(data), data


def step_apply_recommendations(
    backend_url: str,
    system_prompt: str,
    recommendations: list[str],
    model_name: str | None,
    session_id: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict]:
    """Step 2: Apply recommendations to produce an enhanced prompt.

    Returns:
        (enhanced_prompt, raw_response)
    """
    payload = {
        "sessionId": session_id,
        "type": "validation",
        "systemPrompt": system_prompt,
        "modelName": model_name,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "reasoningEffort": None,
        "recommendations": recommendations,
    }
    data = post_chat(backend_url, payload, include_grid_header=False)
    return extract_enhanced_prompt(data), data


def step_llm_judge(
    backend_url: str,
    original_prompt: str,
    enhanced_prompt: str,
    user_message: str,
    span_id: str,
    model_name: str | None,
    session_id: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict | None, dict]:
    """Step 3: Run LLM-as-judge comparing original vs enhanced prompt.

    Returns:
        (judge_result, raw_response)
    """
    test_cases = [
        {
            "test_case_id": "0",
            "test_case_name": f"Trace {span_id[:8]} — {user_message[:40]}",
            "data": {"input": user_message},
            "expected_output": "",
        }
    ]
    payload = {
        "sessionId": session_id,
        "type": "llm_judge",
        "systemPrompt": original_prompt,
        "enhancedSystemPrompt": enhanced_prompt,
        "modelName": model_name,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "reasoningEffort": None,
        "recommendations": None,
        "testCases": test_cases,
    }
    data = post_chat(backend_url, payload)
    return extract_judge_result(data), data
