"""LLM-as-a-Judge evaluator: compares original vs enhanced prompt outputs."""

import asyncio
import json

from src.chat.graph.base_evaluator import PromptEvaluator
from src.chat.utils.llm_models import build_judge_schema
from src.common.service.llm_prompt.prompt_service import PromptService
from src.common.service.logging.logger import info

# ── Metric catalogue ─────────────────────────────────────────────────────────

# Short description used in the UI tooltip / metric picker
AVAILABLE_METRICS: dict[str, str] = {
    # General quality (default)
    "clarity":                      "How clear, well-structured, and easy to understand is the response?",
    "completeness":                  "How thoroughly does it address every aspect of the user's request?",
    "accuracy":                      "How factually correct and internally consistent is the content?",
    "conciseness":                   "Is it appropriately concise — no padding — without losing important detail?",
    "professional_tone":             "How professional and contextually appropriate is the tone?",
    # Guardrails / Classification
    "output_format_compliance":      "Does the response strictly follow the required output format or schema?",
    "hallucination_avoidance":       "Did it avoid fabricating facts, policies, names, or services?",
    # Conversational
    "answer_relevance":              "Is the response directly and fully relevant to what the user asked?",
    "prompt_instruction_adherence":  "Did the model follow every rule and instruction in the system prompt?",
}

# Detailed per-metric scoring guide used inside the judge prompt
# Each entry: description + 5-point anchor rubric
_METRIC_GUIDE: dict[str, dict] = {
    "clarity": {
        "description": (
            "Measures how clear, well-structured, and easy to read the response is. "
            "Consider logical flow, appropriate use of formatting (bullets, headers, numbered steps), "
            "sentence readability, and absence of ambiguous or confusing phrasing."
        ),
        "anchors": {
            1.0: "Exceptionally clear. Logical structure throughout; appropriate formatting used where helpful; zero ambiguous phrasing; immediately actionable.",
            0.75: "Mostly clear with minor structural gaps or isolated ambiguous phrasing that does not impede understanding.",
            0.5: "Understandable but noticeably cluttered or poorly organised; reader must re-read parts to grasp the meaning.",
            0.25: "Hard to follow. Disorganised, excessively verbose, or contains significant ambiguous language.",
            0.0: "Incomprehensible or incoherent; so poorly structured it conveys no clear meaning.",
        },
    },
    "completeness": {
        "description": (
            "Measures how thoroughly the response addresses the user's full request. "
            "Every sub-question, constraint, and edge case the user raised should be covered. "
            "Missing information that a reasonable user would expect counts against this score."
        ),
        "anchors": {
            1.0: "Every aspect of the request is addressed; no gaps; all sub-questions answered; relevant edge cases acknowledged.",
            0.75: "Fully addresses the main request but misses minor sub-points or nuances.",
            0.5: "Addresses the core request but omits one or more notable aspects explicitly raised by the user.",
            0.25: "Only partially answers the request; several key points missing or glossed over.",
            0.0: "Fails to address the user's request; entirely off-topic or contains no substantive answer.",
        },
    },
    "accuracy": {
        "description": (
            "Measures factual correctness, internal consistency, and relevance of all claims. "
            "For subjective or creative tasks where ground truth is unavailable, evaluate whether the response "
            "is internally consistent and free from contradictions. Do not penalise for reasonable opinions."
        ),
        "anchors": {
            1.0: "All statements are factually correct and internally consistent; no unsupported or contradictory claims.",
            0.75: "Mostly accurate with at most one minor inaccuracy or unverifiable detail that does not affect the core answer.",
            0.5: "Generally correct but contains one notable factual error or an unsupported assertion that could mislead.",
            0.25: "Multiple factual errors or significant internal inconsistencies that undermine the response.",
            0.0: "Factually wrong throughout, self-contradictory, or predominantly fabricated.",
        },
    },
    "conciseness": {
        "description": (
            "Measures whether the response is appropriately scoped — no unnecessary padding, filler phrases, "
            "preamble, or repetition — while still including all important detail. "
            "Penalise both excessive verbosity and responses so terse they omit necessary context."
        ),
        "anchors": {
            1.0: "Perfectly scoped. Every sentence adds value; no filler words, preamble, or repetition.",
            0.75: "Slightly verbose in places but no significant padding; core content is tight.",
            0.5: "Noticeably padded; contains repetition or filler that should be cut, though core content is present.",
            0.25: "Very verbose; key points buried in excessive text; would benefit from significant trimming.",
            0.0: "Extremely bloated or repetitive to the point of obscuring the answer, or uselessly terse.",
        },
    },
    "professional_tone": {
        "description": (
            "Measures how well the tone matches the context (formal, semi-formal, or conversational as appropriate). "
            "Consider register consistency, absence of slang or casual language in formal contexts, "
            "confidence of language, and politeness."
        ),
        "anchors": {
            1.0: "Tone perfectly calibrated to the context; polished, confident, consistent register throughout.",
            0.75: "Appropriate tone with minor register inconsistencies that do not distract.",
            0.5: "Mostly appropriate but contains tonal shifts, overly casual phrasing, or mildly unprofessional language.",
            0.25: "Noticeably inappropriate tone; casual where formal is required, or stiff and robotic in a conversational context.",
            0.0: "Completely inappropriate tone; rude, dismissive, unprofessional, or wildly mismatched to the context.",
        },
    },
    "output_format_compliance": {
        "description": (
            "Measures whether the response strictly follows the output format or schema specified in the system prompt "
            "(e.g. JSON, Markdown table, numbered list, specific field names). "
            "Use the system prompt as the ground truth for what format is required."
        ),
        "anchors": {
            1.0: "Output perfectly matches the required format; all required fields present, correct types, valid structure.",
            0.75: "Format mostly correct with one minor deviation (e.g. extra field, slight schema mismatch).",
            0.5: "Recognisable attempt at the format but with notable deviations — missing required fields or wrong structure.",
            0.25: "Significant format violations; partially structured but largely non-compliant.",
            0.0: "Completely ignores the required format; returns unstructured prose or an entirely wrong schema.",
        },
    },
    "hallucination_avoidance": {
        "description": (
            "Measures whether the response avoids fabricating facts, people, policies, product names, or services. "
            "Every claim should be either verifiable from the user input/context or clearly flagged as uncertain. "
            "Penalise confident assertions of unverifiable or invented details."
        ),
        "anchors": {
            1.0: "Every claim is grounded; no invented facts, names, policies, or services; uncertainty is flagged appropriately.",
            0.75: "Mostly grounded with at most one unverifiable minor claim that does not affect the core answer.",
            0.5: "Contains one clear hallucination or fabricated detail that could mislead the user.",
            0.25: "Multiple fabricated facts or policies presented as real.",
            0.0: "Predominantly fabricated; invents significant facts, people, services, or events with confidence.",
        },
    },
    "answer_relevance": {
        "description": (
            "Measures how directly and fully relevant the response is to the specific question asked. "
            "Penalise responses that answer a related but different question, add excessive tangential information, "
            "or fail to engage with the specific intent behind the user's input."
        ),
        "anchors": {
            1.0: "Directly answers the question; every sentence is on-topic; no tangential or off-topic information.",
            0.75: "Mostly on-topic with minor tangents or a slightly indirect answer.",
            0.5: "Partially relevant; addresses the topic but drifts or answers a related but different question.",
            0.25: "Loosely related to the question; mostly off-topic or answers a different question entirely.",
            0.0: "Completely irrelevant; does not address the user's question at all.",
        },
    },
    "prompt_instruction_adherence": {
        "description": (
            "Measures whether the model followed every explicit rule and instruction in the system prompt. "
            "Use the system prompt as the authoritative specification. "
            "Each violated or ignored instruction counts against this score."
        ),
        "anchors": {
            1.0: "Every explicit instruction in the system prompt is followed precisely; zero rule violations.",
            0.75: "Follows most instructions with one minor oversight or partial adherence to a rule.",
            0.5: "Follows the main instructions but misses or partially violates one notable directive.",
            0.25: "Violates multiple instructions or ignores a key constraint from the system prompt.",
            0.0: "Ignores the system prompt entirely; output contradicts or disregards the given instructions.",
        },
    },
}

DEFAULT_METRICS: list[str] = [
    "clarity", "completeness", "accuracy", "conciseness", "professional_tone"
]

# Ordered for display (ALL_METRIC_KEYS is the canonical order used by the UI)
ALL_METRIC_KEYS: list[str] = list(AVAILABLE_METRICS.keys())


def _format_metric_block(key: str) -> str:
    """Format a single metric as a detailed scoring block for the judge prompt."""
    guide = _METRIC_GUIDE.get(key)
    if not guide:
        return f"**{key}**: {AVAILABLE_METRICS.get(key, '')}"

    anchors = guide["anchors"]
    anchor_lines = "\n".join(
        f"  - {score:.2f}: {text}" for score, text in sorted(anchors.items(), reverse=True)
    )
    return (
        f"**{key}**\n"
        f"{guide['description']}\n"
        f"Scoring anchors:\n{anchor_lines}"
    )


def build_judge_prompt(metrics: list[str]) -> str:
    """Build a judge system prompt dynamically from selected metric keys."""
    valid = [m for m in metrics if m in AVAILABLE_METRICS]
    if not valid:
        valid = DEFAULT_METRICS

    metric_blocks = "\n\n".join(_format_metric_block(m) for m in valid)
    json_fields = "\n".join(f'    "{m}": 0.0,' for m in valid)
    exact_keys = ", ".join(f'"{m}"' for m in valid)

    return f"""You are an expert prompt engineer and evaluator. You are given:
- The original and enhanced system prompts that were used to generate the responses
- A user input
- Response A generated by the original system prompt
- Response B generated by the enhanced system prompt

Your job: score both responses on every specified metric and identify the key observable differences between them.

### How to Score
- Use the 0.0–1.0 scale. You may use any value in this range (e.g. 0.6, 0.85); do not restrict yourself to the anchor values.
- The anchor values (1.0, 0.75, 0.5, 0.25, 0.0) are reference points — interpolate between them for intermediate quality.
- Score each response independently against the metric definition. Do not score relative to the other response.
- Be calibrated: a score of 0.5 means genuinely acceptable, not mediocre. Reserve 0.0 and 1.0 for truly extreme cases.

### Metrics

{metric_blocks}

### Calculation and Logic Rules
- The 'overall' score MUST be the exact mathematical average of all metric scores. Do not provide a subjective total.
- Use the system prompts as ground truth when evaluating output_format_compliance and prompt_instruction_adherence.
- If Response A and Response B are identical, assign identical scores to both and note this in 'improvement_summary'.
- Set "improved": true ONLY when the enhanced overall score is meaningfully higher than the original (difference > 0.05). Set "improved": false when scores are tied or the enhanced regressed. "Not improved" does not mean "bad" — a tie at high scores is a valid outcome.

### Key Differences
Identify the most notable observable differences between Response A and Response B in terms of style, structure, content coverage, or phrasing. State facts — do not editorialize.

### Output Constraints
Return ONLY valid JSON. Do NOT include markdown code fences, conversational filler, or any text outside the JSON object.
CRITICAL: The JSON keys inside "original" and "enhanced" MUST be EXACTLY: {exact_keys} plus "overall". Do NOT add, rename, or omit any metric keys.

Format:
{{
  "original": {{
{json_fields}
    "overall": 0.0
  }},
  "enhanced": {{
{json_fields}
    "overall": 0.0
  }},
  "improved": true,
  "improvement_summary": "One sentence summary of whether and how quality improved.",
  "key_differences": [
    "Concrete factual difference between Response A and Response B.",
    "Another observable difference."
  ]
}}"""


_JUDGE_USER_TEMPLATE = """Original System Prompt:
{original_system_prompt}

Enhanced System Prompt:
{enhanced_system_prompt}

---

User Input:
{input}

Response A (Original Prompt Output):
{original_output}

Response B (Enhanced Prompt Output):
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
        enhanced_model_name: str | None = None,
        enhanced_temperature: float | None = None,
        enhanced_max_tokens: int | None = None,
        enhanced_reasoning_effort: str | None = None,
    ) -> list[dict]:
        """Get LLM responses for both prompts in parallel for each test case.

        Original prompt uses self.model_name/temperature/etc.
        Enhanced prompt uses enhanced_* overrides when provided, falling back to self.* values.
        Both fire concurrently via asyncio.gather — different models do not serialise execution.
        """
        enh_model = enhanced_model_name or self.model_name
        enh_temp = enhanced_temperature if enhanced_temperature is not None else self.temperature
        enh_tokens = enhanced_max_tokens or self.max_tokens
        enh_effort = enhanced_reasoning_effort or self.reasoning_effort

        sem = asyncio.Semaphore(5)

        async def _timed_call(coro):
            t = asyncio.get_running_loop().time()
            result = await coro
            return result, round(asyncio.get_running_loop().time() - t, 2)

        async def get_pair(test: dict) -> dict:
            async with sem:
                (orig, orig_lat), (enh, enh_lat) = await asyncio.gather(
                    _timed_call(PromptService.get_prompt_response_from_text(
                        system_prompt=original_system_prompt,
                        user_prompt_template=user_prompt_template,
                        data=test["data"],
                        model_name=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        reasoning_effort=self.reasoning_effort,
                    )),
                    _timed_call(PromptService.get_prompt_response_from_text(
                        system_prompt=enhanced_system_prompt,
                        user_prompt_template=user_prompt_template,
                        data=test["data"],
                        model_name=enh_model,
                        max_tokens=enh_tokens,
                        temperature=enh_temp,
                        reasoning_effort=enh_effort,
                    )),
                )
            test["original_output"] = orig
            test["enhanced_output"] = enh
            test["original_latency_s"] = orig_lat
            test["enhanced_latency_s"] = enh_lat
            return test

        return list(await asyncio.gather(*[get_pair(t) for t in all_tests]))

    async def _judge_single(
        self,
        test: dict,
        metrics: list[str],
        original_system_prompt: str = "",
        enhanced_system_prompt: str = "",
    ) -> dict:
        """Run the LLM judge on a single test case."""
        judge_prompt = self._judge_system_prompt_override or build_judge_prompt(metrics)
        info(f"[JUDGE_SINGLE] test_id={test.get('test_case_id')} | metrics={metrics} | input_len={len(str(test.get('data', {}).get('input', '')))}")

        user_msg = _JUDGE_USER_TEMPLATE.format(
            original_system_prompt=original_system_prompt or "(not provided)",
            enhanced_system_prompt=enhanced_system_prompt or "(not provided)",
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
            response_schema=build_judge_schema(metrics),
        )
        try:
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
            "original_latency_s": test.get("original_latency_s"),
            "enhanced_latency_s": test.get("enhanced_latency_s"),
        }

    async def evaluate_tests(
        self,
        all_tests: list[dict],
        is_rag: bool = False,
        metrics: list[str] | None = None,
        original_system_prompt: str = "",
        enhanced_system_prompt: str = "",
    ) -> list[dict]:
        active_metrics = metrics or DEFAULT_METRICS
        sem = asyncio.Semaphore(5)

        async def _judge_with_sem(t):
            async with sem:
                return await self._judge_single(
                    t,
                    metrics=active_metrics,
                    original_system_prompt=original_system_prompt,
                    enhanced_system_prompt=enhanced_system_prompt,
                )

        return list(await asyncio.gather(*[_judge_with_sem(t) for t in all_tests]))

    def get_final_report(
        self,
        all_tests: list[dict],
        evaluation_results: list[dict],
        requested_metrics: list[str] | None = None,
    ) -> dict:
        # Use requested metrics as authoritative; infer from response only as fallback
        if requested_metrics:
            metrics = [m for m in requested_metrics if m in AVAILABLE_METRICS]
        else:
            first_scores = next(
                (r.get("scores", {}) for r in evaluation_results if "error" not in r.get("scores", {})),
                {},
            )
            metrics = [k for k in first_scores.get("original", {}).keys() if k != "overall"]
        if not metrics:
            metrics = DEFAULT_METRICS

        agg_orig = {m: [] for m in metrics + ["overall"]}
        agg_enh = {m: [] for m in metrics + ["overall"]}
        improved_count = 0

        for r in evaluation_results:
            scores = r.get("scores", {})
            if "error" in scores:
                continue
            for m in metrics + ["overall"]:
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

        display_metrics = metrics + ["overall"]
        return {
            "test_results": evaluation_results,
            "metrics_used": metrics,
            "metrics_requested": metrics,
            "summary": {
                "total_cases": len(evaluation_results),
                "improved_count": improved_count,
                "avg_original": {m: avg(agg_orig[m]) for m in display_metrics},
                "avg_enhanced": {m: avg(agg_enh[m]) for m in display_metrics},
                "avg_delta": {
                    m: round(avg(agg_enh[m]) - avg(agg_orig[m]), 3)
                    if avg(agg_enh[m]) is not None and avg(agg_orig[m]) is not None
                    else None
                    for m in display_metrics
                },
            },
        }

    async def verify_tests(self, state: dict) -> dict:
        original_system_prompt = state["request"].get("system_prompt", "")
        enhanced_system_prompt = state["request"].get("enhanced_system_prompt", "")
        user_prompt_template = state["request"].get("user_prompt_template")
        metrics = state["request"].get("judge_metrics") or DEFAULT_METRICS

        enhanced_model_name = state["request"].get("enhanced_model_name")
        enhanced_temperature = state["request"].get("enhanced_temperature")
        enhanced_max_tokens = state["request"].get("enhanced_max_tokens")
        enhanced_reasoning_effort = state["request"].get("enhanced_reasoning_effort")

        provided_tests = state["request"].get("test_cases")
        if not provided_tests:
            return {}

        info(f"LLM Judge: {len(provided_tests)} test cases | metrics={metrics} | enh_model={enhanced_model_name or 'same as original'}")
        all_tests = await self.get_test_responses_both(
            provided_tests,
            original_system_prompt,
            enhanced_system_prompt,
            user_prompt_template,
            enhanced_model_name=enhanced_model_name,
            enhanced_temperature=enhanced_temperature,
            enhanced_max_tokens=enhanced_max_tokens,
            enhanced_reasoning_effort=enhanced_reasoning_effort,
        )
        info("LLM Judge: responses collected for both prompts")

        evaluation_results = await self.evaluate_tests(
            all_tests,
            metrics=metrics,
            original_system_prompt=original_system_prompt,
            enhanced_system_prompt=enhanced_system_prompt,
        )
        info("LLM Judge: evaluation complete")

        report = self.get_final_report(all_tests, evaluation_results, requested_metrics=metrics)
        report["original_model"] = self.model_name or "default"
        report["enhanced_model"] = enhanced_model_name or self.model_name or "default"
        return report
