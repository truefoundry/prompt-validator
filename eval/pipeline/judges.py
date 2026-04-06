"""Direct LLM judge calls for recommendation quality and application quality."""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
_prompt_cache: dict[str, str] = {}


def _load_prompt(filename: str) -> str:
    if filename not in _prompt_cache:
        with open(os.path.join(_PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
            _prompt_cache[filename] = f.read()
    return _prompt_cache[filename]


def _parse_judge_json(raw: str) -> dict:
    """Parse JSON from LLM output with progressively more aggressive cleanup."""
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract outermost {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        extracted = cleaned[start : end + 1]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            cleaned = extracted

    # Attempt 3: remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 4: replace smart quotes with straight quotes
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    return json.loads(cleaned)


async def _call_llm(system_prompt: str, user_message: str, model_name: str | None) -> str:
    """Single async LLM call using get_truefoundry_llm."""
    from langchain.schema import HumanMessage, SystemMessage
    from src.chat.utils.llm_models import get_truefoundry_llm

    llm = get_truefoundry_llm(model_name=model_name, max_tokens=2000, temperature=0.0, reasoning_effort="low")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = await llm.ainvoke(messages)
    return response.content


async def evaluate_rec_quality(
    original_prompt: str,
    recommendations: list[str],
    model_name: str | None = None,
) -> dict:
    """Judge whether the generated recommendations are correct PE advice.

    Returns:
        {
          "status": "ok" | "error",
          "overall_score": float,        # 0.0–1.0
          "verdict": str,                # good | mixed | poor
          "per_rec": list[dict],
          "missed_weaknesses": list[str],
          "issues": list[str],
          "error": str | None,
        }
    """
    try:
        from src.chat.utils.llm_models import get_truefoundry_llm  # noqa: F401 — import check
    except ImportError as exc:
        return {"status": "error", "overall_score": None, "verdict": None,
                "per_rec": [], "missed_weaknesses": [], "issues": [],
                "error": f"Import failed: {exc}"}

    rec_list = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(recommendations))
    user_message = (
        f"## Original System Prompt\n\n{original_prompt}\n\n"
        f"## Recommendations Generated\n\n{rec_list}"
    )

    try:
        raw = await _call_llm(_load_prompt("rec_quality_system.txt"), user_message, model_name)
        result = _parse_judge_json(raw)
        return {
            "status": "ok",
            "overall_score": result.get("overall_score"),
            "verdict": result.get("verdict"),
            "per_rec": result.get("per_rec", []),
            "missed_weaknesses": result.get("missed_weaknesses", []),
            "issues": result.get("issues", []),
            "error": None,
        }
    except Exception as exc:
        logger.error("evaluate_rec_quality failed: %s", exc)
        return {"status": "error", "overall_score": None, "verdict": None,
                "per_rec": [], "missed_weaknesses": [], "issues": [], "error": str(exc)}


async def evaluate_application_quality(
    original_prompt: str,
    recommendations: list[str],
    enhanced_prompt: str,
    model_name: str | None = None,
) -> dict:
    """Judge whether the enhanced prompt correctly applied the recommendations.

    Returns:
        {
          "status": "ok" | "error",
          "overall_score": float,        # 0.0–1.0
          "verdict": str,                # fully_applied | partially_applied | not_applied
          "per_rec": list[dict],
          "regressions": list[str],
          "new_issues": list[str],
          "error": str | None,
        }
    """
    try:
        from src.chat.utils.llm_models import get_truefoundry_llm  # noqa: F401 — import check
    except ImportError as exc:
        return {"status": "error", "overall_score": None, "verdict": None,
                "per_rec": [], "regressions": [], "new_issues": [],
                "error": f"Import failed: {exc}"}

    rec_list = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(recommendations))
    user_message = (
        f"## Original System Prompt\n\n{original_prompt}\n\n"
        f"## Recommendations\n\n{rec_list}\n\n"
        f"## Enhanced System Prompt\n\n{enhanced_prompt}"
    )

    try:
        raw = await _call_llm(
            _load_prompt("application_quality_system.txt"), user_message, model_name
        )
        result = _parse_judge_json(raw)
        return {
            "status": "ok",
            "overall_score": result.get("overall_score"),
            "verdict": result.get("verdict"),
            "per_rec": result.get("per_rec", []),
            "regressions": result.get("regressions", []),
            "new_issues": result.get("new_issues", []),
            "error": None,
        }
    except Exception as exc:
        logger.error("evaluate_application_quality failed: %s", exc)
        return {"status": "error", "overall_score": None, "verdict": None,
                "per_rec": [], "regressions": [], "new_issues": [], "error": str(exc)}
