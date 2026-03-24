"""LLM-as-a-Judge evaluator: compares original vs enhanced prompt outputs."""

import asyncio
import json

from src.chat.graph.base_evaluator import PromptEvaluator
from src.common.service.llm_prompt.prompt_service import PromptService
from src.common.service.logging.logger import info

_JUDGE_SYSTEM_PROMPT = """You are an expert prompt engineer and evaluator. You are given a user input, and two AI responses — Response A from the original system prompt and Response B from an enhanced system prompt.

Your job: score both responses on quality metrics and identify the key observable differences between them.

### Scoring Rubric (0.0 to 1.0 scale)
- 0.0: Fails to meet any requirements or is completely irrelevant.
- 0.5: Meets basic requirements and is acceptable.
- 1.0: Exceeds all expectations, providing exceptional quality and insight.

### Metrics
Score each response on these metrics:
- clarity: How clear, well-structured, and easy to understand is the response?
- completeness: How thoroughly does it address the user's request?
- accuracy: How factually correct and relevant is the content? For creative or subjective tasks, evaluate internal consistency, adherence to constraints, and relevance instead of factual truth.
- conciseness: Is it appropriately concise without losing important detail?
- professional_tone: How professional and contextually appropriate is the tone?

### Calculation and Logic Rules
- The 'overall' score MUST be the exact mathematical average of the five individual metrics. Do not provide a subjective total.
- Fallback: If Response A and Response B are identical, assign identical scores to both and note this in the 'improvement_summary'.

### Key Differences
Identify the most notable observable differences between Response A and Response B in terms of style, structure, content coverage, or phrasing. These are factual comparisons, not judgements.

### Output Constraints
Return ONLY valid JSON. Do NOT include markdown code fences (e.g., ```json), conversational filler, or any text before or after the JSON object.

Format:
{
  "original": {
    "clarity": 0.0,
    "completeness": 0.0,
    "accuracy": 0.0,
    "conciseness": 0.0,
    "professional_tone": 0.0,
    "overall": 0.0
  },
  "enhanced": {
    "clarity": 0.0,
    "completeness": 0.0,
    "accuracy": 0.0,
    "conciseness": 0.0,
    "professional_tone": 0.0,
    "overall": 0.0
  },
  "improved": true,
  "improvement_summary": "One sentence summary of whether and how quality improved.",
  "key_differences": [
    "Response A uses full month names while Response B uses abbreviations.",
    "Response B includes all subsidiary facets in the summary; Response A omits them."
  ]
}"""

_JUDGE_USER_TEMPLATE = """User Input:
{input}

Response A (Original Prompt):
{original_output}

Response B (Enhanced Prompt):
{enhanced_output}"""


class LLMJudgeEvaluator(PromptEvaluator):
    """Evaluator that uses an LLM as a judge to compare original vs enhanced outputs."""

    def __init__(self, *args, judge_system_prompt_override: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._judge_system_prompt_override = judge_system_prompt_override

    async def get_test_responses_both(
        self,
        all_tests: list[dict],
        original_system_prompt: str,
        enhanced_system_prompt: str,
        user_prompt_template: str | None,
    ) -> list[dict]:
        """Get LLM responses for both prompts in parallel for each test case.

        Uses a semaphore to cap concurrent LLM calls and avoid rate-limit retries
        that make large test sets slower than controlled parallelism.
        """
        sem = asyncio.Semaphore(5)

        async def get_pair(test: dict) -> dict:
            async with sem:
                orig, enh = await asyncio.gather(
                    PromptService.get_prompt_response_from_text(
                        system_prompt=original_system_prompt,
                        user_prompt_template=user_prompt_template,
                        data=test["data"],
                        model_name=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        reasoning_effort=self.reasoning_effort,
                    ),
                    PromptService.get_prompt_response_from_text(
                        system_prompt=enhanced_system_prompt,
                        user_prompt_template=user_prompt_template,
                        data=test["data"],
                        model_name=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        reasoning_effort=self.reasoning_effort,
                    ),
                )
            test["original_output"] = orig
            test["enhanced_output"] = enh
            return test

        return list(await asyncio.gather(*[get_pair(t) for t in all_tests]))

    async def _judge_single(self, test: dict, judge_system_prompt_override: str | None = None) -> dict:
        """Run the LLM judge on a single test case."""
        judge_prompt = judge_system_prompt_override or self._judge_system_prompt_override or _JUDGE_SYSTEM_PROMPT
        using_override = judge_prompt is not _JUDGE_SYSTEM_PROMPT
        info(f"[JUDGE_SINGLE] test_id={test.get('test_case_id')} | override={'yes' if using_override else 'no'} | input_len={len(str(test.get('data', {}).get('input', '')))}")
        user_msg = _JUDGE_USER_TEMPLATE.format(
            input=test["data"]["input"],
            original_output=test["original_output"],
            enhanced_output=test["enhanced_output"],
        )
        raw = await PromptService.get_prompt_response_from_text(
            system_prompt=judge_prompt,
            user_prompt_template=None,
            data={"input": user_msg},
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0.0,
            reasoning_effort=self.reasoning_effort,
        )
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
            scores = json.loads(cleaned.strip())
            info(f"[JUDGE_SINGLE] test_id={test.get('test_case_id')} | orig_overall={scores.get('original', {}).get('overall')} | enh_overall={scores.get('enhanced', {}).get('overall')} | improved={scores.get('improved')}")
        except (json.JSONDecodeError, IndexError):
            info(f"[JUDGE_SINGLE] test_id={test.get('test_case_id')} | parse_error=true | raw_len={len(raw)}")
            scores = {"error": "Failed to parse judge response", "raw": raw}

        return {
            "test_case_id": test.get("test_case_id"),
            "test_case_name": test.get("test_case_name", ""),
            "input": test["data"]["input"],
            "original_output": test["original_output"],
            "enhanced_output": test["enhanced_output"],
            "scores": scores,
        }

    async def evaluate_tests(self, all_tests: list[dict], is_rag: bool = False, judge_system_prompt_override: str | None = None) -> list[dict]:
        sem = asyncio.Semaphore(5)

        async def _judge_with_sem(t):
            async with sem:
                return await self._judge_single(t, judge_system_prompt_override=judge_system_prompt_override)

        return list(await asyncio.gather(*[_judge_with_sem(t) for t in all_tests]))

    def get_final_report(self, all_tests: list[dict], evaluation_results: list[dict]) -> dict:
        metrics = ["clarity", "completeness", "accuracy", "conciseness", "professional_tone", "overall"]
        agg_orig = {m: [] for m in metrics}
        agg_enh = {m: [] for m in metrics}
        improved_count = 0

        for r in evaluation_results:
            scores = r.get("scores", {})
            if "error" in scores:
                continue
            for m in metrics:
                orig_val = scores.get("original", {}).get(m)
                enh_val = scores.get("enhanced", {}).get(m)
                if orig_val is not None:
                    agg_orig[m].append(orig_val)
                if enh_val is not None:
                    agg_enh[m].append(enh_val)
            if scores.get("improved"):
                improved_count += 1

        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else None

        return {
            "test_results": evaluation_results,
            "summary": {
                "total_cases": len(evaluation_results),
                "improved_count": improved_count,
                "avg_original": {m: avg(agg_orig[m]) for m in metrics},
                "avg_enhanced": {m: avg(agg_enh[m]) for m in metrics},
                "avg_delta": {
                    m: round(avg(agg_enh[m]) - avg(agg_orig[m]), 3)
                    if avg(agg_enh[m]) is not None and avg(agg_orig[m]) is not None
                    else None
                    for m in metrics
                },
            },
        }

    async def verify_tests(self, state: dict) -> dict:
        original_system_prompt = state["request"].get("system_prompt", "")
        enhanced_system_prompt = state["request"].get("enhanced_system_prompt", "")
        user_prompt_template = state["request"].get("user_prompt_template")

        provided_tests = state["request"].get("test_cases")
        if not provided_tests:
            return {}

        info(f"LLM Judge: {len(provided_tests)} test cases")
        all_tests = await self.get_test_responses_both(
            provided_tests, original_system_prompt, enhanced_system_prompt, user_prompt_template
        )
        info("LLM Judge: responses collected for both prompts")

        evaluation_results = await self.evaluate_tests(all_tests, judge_system_prompt_override=self._judge_system_prompt_override)
        info("LLM Judge: evaluation complete")

        return self.get_final_report(all_tests, evaluation_results)
