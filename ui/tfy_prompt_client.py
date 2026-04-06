import asyncio
from typing import Any

from src.common.service.llm_prompt.prompt_service import PromptService


def _format_prompt_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
        else:
            # TrueFoundry SDK can return typed message objects instead of dicts.
            role = str(getattr(message, "role", "")).strip()
            content = str(getattr(message, "content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}" if role else content)
    return "\n\n".join(lines)


def get_prompt_text_by_fqn(prompt_fqn: str) -> str:
    """Fetch raw prompt template content using backend PromptService path."""
    if not prompt_fqn:
        return ""

    try:
        prompt_details = asyncio.run(PromptService.get_prompt_details([prompt_fqn]))
    except RuntimeError:
        # Fallback for environments where an event loop is already active.
        loop = asyncio.new_event_loop()
        try:
            prompt_details = loop.run_until_complete(PromptService.get_prompt_details([prompt_fqn]))
        finally:
            loop.close()

    if not prompt_details or not isinstance(prompt_details, list):
        return ""

    prompt_entry = prompt_details[0]
    if not isinstance(prompt_entry, dict):
        return ""

    manifest = prompt_entry.get("manifest")
    messages = getattr(manifest, "messages", None) if manifest is not None else None
    if not isinstance(messages, list):
        return ""

    return _format_prompt_messages(messages)
