"""Arena evaluation service using DeepEval's ArenaGEval.

Compares an original system prompt against an enhanced system prompt on a set
of test cases. For each test case, an LLM judge picks a winner and provides
reasoning. Returns per-test results and aggregate win counts.
"""
from __future__ import annotations

from typing import Any

from deepeval.metrics import ArenaGEval
from deepeval.test_case import ArenaTestCase, Contestant, LLMTestCase, LLMTestCaseParams
from deepeval.prompt import Prompt
from langchain.schema import SystemMessage, HumanMessage


def _make_llm(model_name: str | None, max_tokens: int | None, temperature: float | None):
    from src.chat.utils.llm_models import TrueFoundryLLM, get_truefoundry_llm
    return TrueFoundryLLM(model=get_truefoundry_llm(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    ))


def _generate_output(llm, system_prompt: str, user_input: str) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
    return llm.generate(messages)


def run_arena_comparison(
    original_prompt: str,
    enhanced_prompt: str,
    test_cases: list[dict],
    criteria: str,
    model_name: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Run ArenaGEval comparison between original and enhanced prompt.

    Args:
        original_prompt: The original system prompt text.
        enhanced_prompt: The enhanced/optimized system prompt text.
        test_cases: List of dicts with keys data.input (and optionally expected_output).
        criteria: Natural-language description of what makes a better response.
        model_name: Optional model override.
        max_tokens: Optional max tokens override.
        temperature: Optional temperature override.

    Returns:
        {
            "wins": {"Original": int, "Enhanced": int},
            "win_rate": float,   # Enhanced win fraction 0.0–1.0
            "per_test": [
                {
                    "input": str,
                    "winner": str,
                    "reason": str,
                    "original_output": str,
                    "enhanced_output": str,
                }
            ]
        }
    """
    if not original_prompt.strip():
        raise ValueError("original_prompt must not be empty")
    if not enhanced_prompt.strip():
        raise ValueError("enhanced_prompt must not be empty")
    if not test_cases:
        raise ValueError("At least one test case is required")
    if not criteria.strip():
        raise ValueError("criteria must not be empty")

    llm = _make_llm(model_name, max_tokens, temperature)

    prompt_orig = Prompt(alias="Original", text_template=original_prompt)
    prompt_enh = Prompt(alias="Enhanced", text_template=enhanced_prompt)

    arena_metric = ArenaGEval(
        name="Prompt Quality",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        criteria=criteria,
        model=llm,
        async_mode=False,
    )

    wins: dict[str, int] = {"Original": 0, "Enhanced": 0, "Inconclusive": 0}
    per_test: list[dict] = []

    for tc in test_cases:
        tc_input = tc.get("data", {}).get("input") or tc.get("input", "")
        expected = tc.get("expected_output", "")
        if not tc_input:
            continue

        orig_output = _generate_output(llm, original_prompt, tc_input)
        enh_output = _generate_output(llm, enhanced_prompt, tc_input)

        arena_tc = ArenaTestCase(contestants=[
            Contestant(
                name="Original",
                hyperparameters={"prompt": prompt_orig},
                test_case=LLMTestCase(
                    input=tc_input,
                    actual_output=orig_output,
                    expected_output=expected or None,
                ),
            ),
            Contestant(
                name="Enhanced",
                hyperparameters={"prompt": prompt_enh},
                test_case=LLMTestCase(
                    input=tc_input,
                    actual_output=enh_output,
                    expected_output=expected or None,
                ),
            ),
        ])

        try:
            arena_metric.measure(arena_tc, _show_indicator=False)
            winner = getattr(arena_metric, "winner", None) or "Unknown"
            reason = getattr(arena_metric, "reason", "") or ""
            # ArenaGEval uses masked names ($Jack$/$Jill$) internally — if the LLM
            # returns the masked name instead of a verdict the unmasking step fails
            # and the raw mask leaks out. Treat those as inconclusive.
            if winner.startswith("$") or winner not in ("Original", "Enhanced"):
                winner = "Inconclusive"
                reason = reason or "Judge returned an inconclusive result."
        except Exception as e:
            winner = "Inconclusive"
            reason = f"Evaluation error: {e}"

        if winner in wins:
            wins[winner] += 1

        per_test.append({
            "input": tc_input,
            "winner": winner,
            "reason": reason,
            "original_output": orig_output,
            "enhanced_output": enh_output,
        })

    total = len(per_test)
    win_rate = wins["Enhanced"] / total if total > 0 else 0.0

    return {
        "wins": wins,
        "win_rate": win_rate,
        "per_test": per_test,
    }
