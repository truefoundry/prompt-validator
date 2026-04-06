"""DeepEval-based prompt quality metrics evaluator.

Runs reference-free DeepEval metrics on original vs enhanced prompt outputs
to compare prompt quality without needing ground truth.

Parallelism strategy:
- All test cases fire concurrently via asyncio.gather (outer level).
- Within each case, all metrics on orig + enh fire concurrently (inner gather).
- Each metric call runs via loop.run_in_executor(_THREAD_POOL, ...) in a dedicated
  ThreadPoolExecutor with _MAX_WORKERS threads — critical under nest_asyncio +
  Streamlit's running loop where await metric.a_measure() serialises despite being
  awaited, and asyncio.to_thread may queue if the default pool is exhausted.
- A semaphore caps total concurrent threads across all cases to avoid overwhelming
  the LLM API.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import time
from typing import Any

from src.common.service.logging.logger import error, info

_MAX_CONCURRENT = 60   # semaphore: max simultaneous metric.measure() calls
_MAX_WORKERS    = 120  # ThreadPoolExecutor size — must be >= _MAX_CONCURRENT
_THREAD_POOL    = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)


async def _safe_measure_thread(metric, test_case, sem: asyncio.Semaphore) -> tuple[float | None, str]:
    """Run metric.measure() in a dedicated thread pool so it never blocks the event loop."""
    async with sem:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _THREAD_POOL,
                functools.partial(metric.measure, test_case, _show_indicator=False),
            )
            score = getattr(metric, "score", None)
            reason = getattr(metric, "reason", "") or ""
            return float(score) if score is not None else None, reason
        except Exception as e:
            error(f"[DeepEvalMetrics] measure failed | metric={type(metric).__name__} | {type(e).__name__}: {e}")
            return None, f"Error: {e}"


def _build_metrics(llm, geval_criteria: str, prompt_instructions: list[str]) -> list[tuple[str, Any]]:
    from deepeval.test_case import LLMTestCaseParams
    from deepeval.metrics import AnswerRelevancyMetric, GEval
    from deepeval.metrics.prompt_alignment.prompt_alignment import PromptAlignmentMetric
    from deepeval.metrics.pii_leakage.pii_leakage import PIILeakageMetric

    metrics: list[tuple[str, Any]] = []

    # async_mode=False — we drive parallelism via asyncio.to_thread, not DeepEval's
    # internal async runner (which conflicts with nest_asyncio under Streamlit).
    metrics.append(("answer_relevancy", AnswerRelevancyMetric(
        model=llm, async_mode=False, include_reason=True,
    )))

    if geval_criteria.strip():
        metrics.append(("geval", GEval(
            name="Quality",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            criteria=geval_criteria.strip(),
            model=llm,
            async_mode=False,
            _include_g_eval_suffix=False,
        )))

    if prompt_instructions:
        metrics.append(("prompt_alignment", PromptAlignmentMetric(
            prompt_instructions=prompt_instructions,
            model=llm,
            async_mode=False,
            include_reason=True,
        )))

    metrics.append(("pii", PIILeakageMetric(
        model=llm, async_mode=False, include_reason=True,
    )))

    return metrics


def run_deepeval_prompt_metrics(
    test_cases: list[dict],
    llm,
    geval_criteria: str,
    prompt_instructions: list[str],
) -> dict[str, Any]:
    """Run DeepEval reference-free metrics on original vs enhanced prompt outputs.

    All test cases run concurrently. Within each case, all metrics on orig and enh
    outputs fire concurrently via a dedicated ThreadPoolExecutor (_THREAD_POOL).
    A semaphore of _MAX_CONCURRENT caps total in-flight threads across all cases.

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

    # Build one set of metrics just to get the keys — actual per-case instances
    # are created fresh inside _evaluate_one to avoid shared state race conditions.
    probe_metrics = _build_metrics(llm, geval_criteria, prompt_instructions)
    metrics_run = [key for key, _ in probe_metrics]
    info(f"[DeepEvalMetrics] Starting | metrics={metrics_run} | total_cases={len(test_cases)} | concurrency={_MAX_CONCURRENT} | workers={_MAX_WORKERS}")

    valid_cases = [
        tc for tc in test_cases
        if tc.get("input") and tc.get("original_output") and tc.get("enhanced_output")
    ]
    info(f"[DeepEvalMetrics] Valid cases={len(valid_cases)} (skipped {len(test_cases) - len(valid_cases)} incomplete)")

    async def _run_all() -> list[dict]:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def _evaluate_one(idx: int, tc: dict) -> dict:
            inp = tc["input"]
            info(f"[DeepEvalMetrics] Evaluating case {idx+1}/{len(valid_cases)} | input_len={len(inp)}")
            t_case = time.time()
            orig_tc = LLMTestCase(input=inp, actual_output=tc["original_output"])
            enh_tc = LLMTestCase(input=inp, actual_output=tc["enhanced_output"])

            # Fresh metric instances per test case — metric objects store score/reason
            # as instance state; sharing across threads causes race conditions.
            local_metrics = _build_metrics(llm, geval_criteria, prompt_instructions)

            orig_tasks = [_safe_measure_thread(metric, orig_tc, sem) for _, metric in local_metrics]
            enh_tasks  = [_safe_measure_thread(metric, enh_tc,  sem) for _, metric in local_metrics]

            all_results  = await asyncio.gather(*orig_tasks, *enh_tasks)
            orig_results = all_results[:len(local_metrics)]
            enh_results  = all_results[len(local_metrics):]

            scores: dict[str, dict] = {}
            for (key, _), (o_score, o_reason), (e_score, e_reason) in zip(local_metrics, orig_results, enh_results):
                scores[key] = {
                    "original": o_score,
                    "enhanced": e_score,
                    "original_reason": o_reason,
                    "enhanced_reason": e_reason,
                }
                info(f"[DeepEvalMetrics] case={idx+1} | metric={key} | orig={o_score} | enh={e_score}")
            info(f"[DeepEvalMetrics] case={idx+1} done | elapsed={round(time.time()-t_case,2)}s")
            return {"input": inp, "scores": scores}

        # All cases fire at once — semaphore caps total concurrent threads.
        return list(await asyncio.gather(*[_evaluate_one(idx, tc) for idx, tc in enumerate(valid_cases)]))

    t_start = time.time()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        per_test = loop.run_until_complete(_run_all())
    except RuntimeError:
        per_test = asyncio.run(_run_all())
    info(f"[DeepEvalMetrics] All cases done | elapsed={round(time.time()-t_start,2)}s")

    # Aggregate summaries
    summary: dict[str, dict] = {}
    for key in metrics_run:
        orig_vals = [t["scores"][key]["original"] for t in per_test if t["scores"][key]["original"] is not None]
        enh_vals  = [t["scores"][key]["enhanced"]  for t in per_test if t["scores"][key]["enhanced"]  is not None]
        orig_avg = round(sum(orig_vals) / len(orig_vals), 3) if orig_vals else None
        enh_avg  = round(sum(enh_vals)  / len(enh_vals),  3) if enh_vals  else None
        delta    = round(enh_avg - orig_avg, 3) if orig_avg is not None and enh_avg is not None else None
        summary[key] = {"original_avg": orig_avg, "enhanced_avg": enh_avg, "delta": delta}

    return {"per_test": per_test, "summary": summary, "metrics_run": metrics_run}
