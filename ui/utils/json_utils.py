"""Shared JSON parsing utilities."""
import json


def strip_markdown_fences(raw: str) -> str:
    """Strip markdown code fences from LLM output before JSON parsing."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    return json.loads(strip_markdown_fences(raw))
