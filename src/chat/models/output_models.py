"""Pydantic output models for LLM-structured responses.

Used for:
- Post-parse validation after JSON extraction
- Generating JSON schemas passed to get_truefoundry_llm for OpenAI structured output
"""

from typing import Literal
from pydantic import BaseModel


# ── Generate Suggestions ──────────────────────────────────────────────────────

class Suggestion(BaseModel):
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    title: str
    suggestion: str
    rationale: str


class GenerateSuggestionsOutput(BaseModel):
    overall_analysis: str
    suggestions: list[Suggestion]


# ── LLM Judge ─────────────────────────────────────────────────────────────────

class LLMJudgeSideScores(BaseModel):
    """Per-side (original / enhanced) metric scores. Keys are metric names + 'overall'."""
    # Defined as extra-fields-allowed because metric keys are dynamic
    model_config = {"extra": "allow"}


class LLMJudgeOutput(BaseModel):
    original: LLMJudgeSideScores
    enhanced: LLMJudgeSideScores
    improved: bool
    improvement_summary: str
    key_differences: list[str]
