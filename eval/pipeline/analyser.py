"""Single LLM meta-analysis call over the full eval report."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "prompts", "eval_analysis_system.txt")
)
_prompt_cache: str | None = None


def _load_analysis_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is None:
        with open(_ANALYSIS_PROMPT_PATH, "r", encoding="utf-8") as f:
            _prompt_cache = f.read()
    return _prompt_cache


async def analyse_report(report: dict, model_name: str | None = None) -> str:
    """Run a single LLM meta-analysis call over the full eval report.

    Reuses get_truefoundry_llm() from src/chat/utils/llm_models.py.
    Gracefully skips if CONFIGBASEPATH is not set or the import fails.

    Returns markdown analysis string, or a descriptive error message.
    """
    try:
        from langchain.schema import HumanMessage, SystemMessage
        from src.chat.utils.llm_models import get_truefoundry_llm
    except ImportError as exc:
        msg = f"Cannot import src modules (CONFIGBASEPATH may not be set): {exc}"
        logger.warning(msg)
        return f"[Analysis skipped] {msg}"

    try:
        llm = get_truefoundry_llm(model_name=model_name, max_tokens=4000, temperature=0.1, reasoning_effort="low")
        messages = [
            SystemMessage(content=_load_analysis_prompt()),
            HumanMessage(
                content=f"Here is the evaluation report:\n\n{json.dumps(report, indent=2)}"
            ),
        ]
        response = await llm.ainvoke(messages)
        return response.content
    except Exception as exc:
        logger.error("analyse_report failed: %s", exc)
        return f"[Analysis failed] {exc}"
