"""DeepEval-based prompt quality metrics evaluator.

Runs reference-free DeepEval metrics on original vs enhanced prompt outputs
to compare prompt quality without needing ground truth.

Uses async execution (asyncio.gather + semaphore) to parallelise across all
test cases and outputs — mirroring DeepEval's internal evaluate() architecture.
"""
from __future__ import annotations

import asyncio
from typing import Any

_MAX_CONCURRENT = 5  # concurrent metric.a_measure() calls


def _safe_measure_sync(metric, test_case) -> tuple[float | None, str]:
    try:
        metric.measure(test_case, _show_indicator=False)
        score = getattr(metric, "score", None)
        reason = getattr(metric, "reason", "") or ""
        return float(score) if score is not None else None, reason
    except Exception as e:
        return None, f"Error: {e}"


async def _safe_measure_async(metric, test_case, sem: asyncio.Semaphore) -> tuple[float | None, str]:
    async with sem:
        try:
            await metric.a_measure(test_case, _show_indicator=False)
            score = getattr(metric, "score", None)
            reason = getattr(metric, "reason", "") or ""
            return float(score) if score is not None else None, reason
        except Exception as e:
            return None, f"Error: {e}"


def _build_metrics(llm, geval_criteria: str, prompt_instructions: list[str]) -> list[tuple[str, Any]]:
    from deepeval.test_case import LLMTestCaseParams
    from deepeval.metrics import AnswerRelevancyMetric, GEval
    from deepeval.metrics.prompt_alignment.prompt_alignment import PromptAlignmentMetric
    from deepeval.metrics.pii_leakage.pii_leakage import PIILeakageMetric

    metrics: list[tuple[str, Any]] = []

    metrics.append(("answer_relevancy", AnswerRelevancyMetric(
        model=llm, async_mode=True, include_reason=True,
    )))

    if geval_criteria.strip():
        metrics.append(("geval", GEval(
            name="Quality",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            criteria=geval_criteria.strip(),
            model=llm,
            async_mode=True,
            _include_g_eval_suffix=False,
        )))

    if prompt_instructions:
        metrics.append(("prompt_alignment", PromptAlignmentMetric(
            prompt_instructions=prompt_instructions,
            model=llm,
            async_mode=True,
            include_reason=True,
        )))

    metrics.append(("pii", PIILeakageMetric(
        model=llm, async_mode=True, include_reason=True,
    )))

    return metrics


def run_deepeval_prompt_metrics(
    test_cases: list[dict],
    llm,
    geval_criteria: str,
    prompt_instructions: list[str],
) -> dict[str, Any]:
    """Run DeepEval reference-free metrics on original vs enhanced prompt outputs.

    Executes all (test_case × output × metric) combinations concurrently using
    asyncio.gather + semaphore — same approach as DeepEval's built-in evaluate().

    Args:
        test_cases: List of dicts with keys: input, original_output, enhanced_output.
        llm: TrueFoundryLLM instance.
        geval_criteria: Natural-language criteria for GEval. Leave blank to skip.
        prompt_instructions: List of instruction strings for PromptAlignmentMetric. Leave empty to skip.

    Returns:
        {
            "per_test": [{"input", "scores": {metric_key: {original, enhanced, original_reason, enhanced_reason}}}],
            "summary": {metric_key: {original_avg, enhanced_avg, delta}},
            "metrics_run": [str]
        }
    """
    from deepeval.test_case import LLMTestCase

    metrics = _build_metrics(llm, geval_criteria, prompt_instructions)
    metrics_run = [key for key, _ in metrics]

    valid_cases = [
        tc for tc in test_cases
        if tc.get("input") and tc.get("original_output") and tc.get("enhanced_output")
    ]

    async def _run_all() -> list[dict]:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        results: list[dict] = []

        # Build all tasks: for each test case, measure every metric on orig and enh output
        for tc in valid_cases:
            inp = tc["input"]
            orig_tc = LLMTestCase(input=inp, actual_output=tc["original_output"])
            enh_tc = LLMTestCase(input=inp, actual_output=tc["enhanced_output"])

            # Schedule all metric × output combos concurrently
            orig_tasks = [_safe_measure_async(metric, orig_tc, sem) for _, metric in metrics]
            enh_tasks = [_safe_measure_async(metric, enh_tc, sem) for _, metric in metrics]

            all_results = await asyncio.gather(*orig_tasks, *enh_tasks)

            orig_results = all_results[:len(metrics)]
            enh_results = all_results[len(metrics):]

            scores: dict[str, dict] = {}
            for (key, _), (o_score, o_reason), (e_score, e_reason) in zip(metrics, orig_results, enh_results):
                scores[key] = {
                    "original": o_score,
                    "enhanced": e_score,
                    "original_reason": o_reason,
                    "enhanced_reason": e_reason,
                }
            results.append({"input": inp, "scores": scores})

        return results

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop (e.g. Jupyter/Streamlit with async runner)
            import nest_asyncio
            nest_asyncio.apply()
        per_test = loop.run_until_complete(_run_all())
    except RuntimeError:
        per_test = asyncio.run(_run_all())

    # Aggregate summaries
    summary: dict[str, dict] = {}
    for key in metrics_run:
        orig_vals = [t["scores"][key]["original"] for t in per_test if t["scores"][key]["original"] is not None]
        enh_vals = [t["scores"][key]["enhanced"] for t in per_test if t["scores"][key]["enhanced"] is not None]
        orig_avg = round(sum(orig_vals) / len(orig_vals), 3) if orig_vals else None
        enh_avg = round(sum(enh_vals) / len(enh_vals), 3) if enh_vals else None
        delta = round(enh_avg - orig_avg, 3) if orig_avg is not None and enh_avg is not None else None
        summary[key] = {"original_avg": orig_avg, "enhanced_avg": enh_avg, "delta": delta}

    return {"per_test": per_test, "summary": summary, "metrics_run": metrics_run}
