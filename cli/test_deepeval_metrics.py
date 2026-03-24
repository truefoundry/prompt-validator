"""CLI timing test for DeepEval metrics — 4 metrics, async parallel execution."""
import sys, os, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trace.trace_parser import load_spans_from_file, parse_spans_to_inputs

TRACE_FILE = os.path.join(os.path.dirname(__file__), "../trace/testset_fq4_variance_v2.json")

spans = load_spans_from_file(TRACE_FILE)
inputs = parse_spans_to_inputs(spans)[:1]

tc_input  = inputs[0].user_message or ""
tc_output = inputs[0].trace_output or ""
print(f"Input length: {len(tc_input)} chars | Output length: {len(tc_output)} chars\n")

from src.chat.utils.llm_models import TrueFoundryLLM, get_truefoundry_llm
llm = TrueFoundryLLM(model=get_truefoundry_llm(max_tokens=4000, temperature=0.0, reasoning_effort="low"))

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.metrics.prompt_alignment.prompt_alignment import PromptAlignmentMetric
from deepeval.metrics.pii_leakage.pii_leakage import PIILeakageMetric

metrics = [
    ("AnswerRelevancy", AnswerRelevancyMetric(model=llm, async_mode=True, include_reason=True)),
    ("GEval",           GEval(name="Quality",
                              evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                              criteria="Assess overall quality, accuracy and completeness of the response given the user input.",
                              model=llm, async_mode=True, _include_g_eval_suffix=False)),
    ("PromptAlignment", PromptAlignmentMetric(
                              prompt_instructions=["Write in paragraph form.", "Keep explanations concise (3-5 sentences)."],
                              model=llm, async_mode=True, include_reason=True)),
    ("PII",             PIILeakageMetric(model=llm, async_mode=True, include_reason=True)),
]

tc = LLMTestCase(input=tc_input, actual_output=tc_output)

# ── Async parallel run (all 4 metrics at once) ────────────────────────────────
print("Running all 4 metrics in PARALLEL (async_mode=True) ...")
t_start = time.perf_counter()

async def run_all():
    sem = asyncio.Semaphore(5)
    async def measure(name, metric):
        async with sem:
            t0 = time.perf_counter()
            try:
                await metric.a_measure(tc, _show_indicator=False)
                elapsed = time.perf_counter() - t0
                score = getattr(metric, "score", "?")
                reason = (getattr(metric, "reason", "") or "")[:80]
                return name, elapsed, score, None, reason
            except Exception as e:
                elapsed = time.perf_counter() - t0
                return name, elapsed, None, str(e)[:80], ""

    return await asyncio.gather(*[measure(n, m) for n, m in metrics])

results = asyncio.run(run_all())
total_parallel = time.perf_counter() - t_start

print(f"\n{'Metric':<20} {'Time':>6}  {'Score':>6}  Status")
print("-" * 55)
for row in results:
    name, elapsed, score, err, reason = row
    if err:
        print(f"{name:<20} {elapsed:>5.1f}s  {'—':>6}  ❌  {err}")
    else:
        print(f"{name:<20} {elapsed:>5.1f}s  {str(score):>6}  ✅  {reason}")

print()
print(f"Wall time (parallel, 1 output):         {total_parallel:.1f}s")
print(f"1 test case, 2 outputs (orig+enh):      {total_parallel*2:.1f}s  (upper bound — can overlap too)")
print(f"20 cases × 2 outputs (with semaphore=5): ~{total_parallel*2*20/5:.0f}s  ({total_parallel*2*20/5/60:.1f} min)")
